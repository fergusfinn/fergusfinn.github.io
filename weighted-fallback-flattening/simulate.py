"""
Priority inversion in weighted-random fallback.

Models are sampled proportional to weights. Each sample is rejected with
probability p (iid, uncorrelated across models). On rejection we sample
without replacement from the remaining models until we get a success or
exhaust the pool — then start fresh.

At high error rates the heavy-weight models act as "shields": they're
drawn first, rejected, and removed, concentrating the remaining probability
mass on the lighter models. The result is that the empirical selection
frequency of low-weight models *exceeds* their nominal weight.
"""

import numpy as np
from collections import Counter


def simulate(weights: list[float], p_fail: float, n_trials: int, rng: np.random.Generator) -> Counter:
    """Run n_trials of the fallback process, return counts of successful selections."""
    counts: Counter = Counter()
    n = len(weights)
    w = np.array(weights, dtype=float)

    for _ in range(n_trials):
        available = np.ones(n, dtype=bool)
        success = False

        while available.any():
            # normalise over available models
            probs = w * available
            probs /= probs.sum()

            choice = rng.choice(n, p=probs)
            available[choice] = False

            if rng.random() >= p_fail:  # success
                counts[choice] += 1
                success = True
                break

        # if every model failed, nothing is counted (request dropped)
        if not success:
            counts["dropped"] += 1

    return counts


def run_experiment():
    rng = np.random.default_rng(42)

    models = ["A", "B", "C"]
    weights = [0.7, 0.2, 0.1]
    n_trials = 500_000
    fail_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"Models:  {models}")
    print(f"Weights: {weights}")
    print(f"Trials:  {n_trials:,}")
    print()

    header = f"{'p_fail':>8}  " + "  ".join(f"{m:>10}" for m in models) + f"  {'dropped':>10}"
    print(header)
    print("-" * len(header))

    for p in fail_rates:
        counts = simulate(weights, p, n_trials, rng)
        parts = []
        for i, m in enumerate(models):
            freq = counts[i] / n_trials
            parts.append(f"{freq:>10.4f}")
        drop_freq = counts["dropped"] / n_trials
        parts.append(f"{drop_freq:>10.4f}")
        print(f"{p:>8.1f}  " + "  ".join(parts))

    # show the ratio of empirical frequency to nominal weight
    print()
    print("Ratio of empirical frequency to nominal weight (excluding drops):")
    header2 = f"{'p_fail':>8}  " + "  ".join(f"{m:>10}" for m in models)
    print(header2)
    print("-" * len(header2))

    for p in fail_rates:
        counts = simulate(weights, p, n_trials, rng)
        total_success = sum(counts[i] for i in range(len(models)))
        if total_success == 0:
            continue
        parts = []
        for i, m in enumerate(models):
            empirical = counts[i] / total_success
            ratio = empirical / weights[i]
            parts.append(f"{ratio:>10.3f}")
        print(f"{p:>8.1f}  " + "  ".join(parts))


if __name__ == "__main__":
    run_experiment()
