---
title: 'Weighted random fallback flattens to uniform under high error rates'
description: |
  When a weighted-random fallback rejects samples and retries without replacement, high error rates cause low-weight models to be selected far more often than their weights suggest.
pubDate: 'Feb 17 2026'
slug: weighted-fallback-flattening
---

AI inference gateways often route requests across multiple upstream providers. When one of them returns a 502 or a 429, it's common to retry the request against a different provider rather than propagate the error[>0]. In the [Doubleword Control Layer](https://github.com/doublewordai/control-layer) we support two fallback strategies.

[>0]: Inference providers have historically been unreliable enough that this is less of an edge case and more of a design requirement.

The first is priority fallback: an ordered list of providers, tried in sequence. Provider A fails, try B. B fails, try C.

The second is weighted random. Each provider gets a weight, and for each request you sample one proportional to those weights. If it fails, you remove it from the pool, re-normalize the weights over whatever's left, and sample again. This repeats until something succeeds or you exhaust the pool[>1].

[>1]: Simplified from the actual [onwards](https://github.com/doublewordai/onwards) implementation, which also handles rate limits, concurrency limits, and configurable status code matching. The core selection logic is the same.

```rust
while !remaining.is_empty() {
    let total: u32 = remaining.iter().map(|(_, w)| w).sum();
    let r: u32 = rng.random_range(0..total);

    let mut cumulative = 0;
    let mut selected = 0;
    for (pos, (_, weight)) in remaining.iter().enumerate() {
        cumulative += weight;
        if r < cumulative {
            selected = pos;
            break;
        }
    }

    let (idx, _) = remaining.remove(selected);
    order.push(idx);
}
```

Weighted random is appealing if you want to spread load across providers, maybe because of rate limits or because you want to balance cost across multiple accounts. The weights give you a knob to control the distribution. We noticed recently, though, that when providers fail at high rates across the board (not one provider going down, but general flakiness), the distribution you actually get diverges from the one you configured. The "remove and re-normalize" step is the culprit.

Suppose you have three models, A, B, and C, with weights 0.7, 0.2, and 0.1. Under normal conditions, roughly 70% of requests go to A, 20% to B, 10% to C. Now suppose each provider fails independently with some probability $p$. The failure probability is the same regardless of whether a model was picked first, second, or third. So you might expect the distribution of *successful* selections to still reflect the configured weights. It doesn't.

## Simulation

We can model this directly. For each trial: sample a model proportional to weights, flip a coin with probability $p$ of failure. On failure, remove that model and sample again from what's left. Repeat until something succeeds or the pool is empty.

```python
import numpy as np
from collections import Counter

def simulate(weights: list[float], p_fail: float, n_trials: int, rng: np.random.Generator) -> Counter:
    counts: Counter = Counter()
    n = len(weights)
    w = np.array(weights, dtype=float)

    for _ in range(n_trials):
        available = np.ones(n, dtype=bool)

        while available.any():
            probs = w * available
            probs /= probs.sum()

            choice = rng.choice(n, p=probs)
            available[choice] = False

            if rng.random() >= p_fail:
                counts[choice] += 1
                break
    return counts
```

Running this with weights $[0.7, 0.2, 0.1]$ across a range of failure rates, 500,000 trials each[>_sim]:

[>_sim]: These frequencies are conditioned on at least one model succeeding. Trials where all three models fail are excluded and the remaining frequencies re-normalized. This is equivalent to assuming that fully-failed requests are resubmitted until they eventually succeed.

| $p$ | A | B | C |
| :-- | :-- | :-- | :-- |
| 0.0 | 0.7001 | 0.1996 | 0.1003 |
| 0.1 | 0.6535 | 0.2274 | 0.1191 |
| 0.3 | 0.5610 | 0.2697 | 0.1693 |
| 0.5 | 0.4797 | 0.2981 | 0.2222 |
| 0.7 | 0.4103 | 0.3181 | 0.2716 |
| 0.9 | 0.3561 | 0.3288 | 0.3151 |

At $p = 0$, the frequencies match the weights exactly. As $p$ increases, the distribution flattens: by $p = 0.9$, a 7:2:1 configured weight ratio has become roughly 1:1:1.

## Tracing through the paths

We can derive this exactly for the three-model case. The intuition first: removal is conditional on being drawn, and being drawn is proportional to weight. When a retry happens (because the first draw failed), the model that was drawn and removed is A 70% of the time, B 20% of the time, C 10% of the time. So conditional on reaching a second draw, A is absent from the retry pool far more often than B or C are. The higher $p$ is, the more often the process reaches these retry rounds, and the more the final outcome is shaped by retry pools that are disproportionately missing the heavy model.

Let $w_i$ be the weights (summing to 1) and $p$ the failure probability. Model $i$ can be selected on the first draw (probability $w_i$, succeeds with probability $1-p$), the second draw (some $j$ was drawn and failed first, then $i$ is drawn from the reduced pool), or the third draw ($i$ is the last model standing). Each path picks up a factor of $p$ per prior failure and a factor of $(1-p)$ for the final success. Summing over all paths:

$$P_i = (1-p)\Big[w_i + p \cdot w_i \sum_{j \neq i} \frac{w_j}{1 - w_j} + p^2 \sum_{j \neq i} \sum_{k \neq i, j} \frac{w_j \cdot w_k}{1 - w_j}\Big]$$

The $\frac{w_j}{1 - w_j}$ terms are the re-normalized draw probabilities after removing earlier models. The thing to notice is that the first two terms contain $w_i$ but the third term doesn't: when $i$ is the last model left, it's selected with probability 1 regardless of its weight.

Write $f_i(p) = c_1 + p \cdot c_2 + p^2 \cdot c_3$ for the bracket. At $p = 0$, $f_i(0) = c_1 = w_i$, which varies across models (that's the configured distribution). As $p \to 1$, $f_i(p) \to c_1 + c_2 + c_3$. But $c_1$, $c_2$, $c_3$ partition the ways $i$ can be drawn (first, second, or third), and since there are only three models, $i$ is always drawn eventually. So $c_1 + c_2 + c_3 = 1$ for every model, $f_i(p) \to 1$ for everyone, and the normalized distribution $P_i / \sum_j P_j$ converges to $\frac{1}{3}$.

## Conclusion

Weighted random fallback with sampling without replacement flattens the configured weight distribution under high, uncorrelated error rates. The effect is continuous: at low error rates the distortion is small, at high error rates the distribution approaches uniform. It's arguably not a terrible property. If everything is failing at the same rate, you probably don't have strong preferences about which provider handles the request, and spreading load evenly across whatever happens to succeed is reasonable.

We've since added sampling with replacement as a configuration option in onwards, which preserves the configured weights regardless of error rate (the same provider can be retried). The tradeoff is that you might waste a retry slot on a provider that just failed, but the distribution you configure is the distribution you get.
