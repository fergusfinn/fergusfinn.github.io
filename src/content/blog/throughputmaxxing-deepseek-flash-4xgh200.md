---
title: 'Throughputmaxxing DeepSeek-Flash on a 4×GH200'
description: 'Pushing DeepSeek-V4-Flash serving throughput on a single 4×GH200 node: baseline, CUDA graphs, memory utilization, TP shape.'
pubDate: 'May 26 2026'
index: true
draft: true
---

It turns out inference optimization is still hard for agents:

> Across 15 frontier agent configurations, agents reliably
> improve over a naïve PyTorch baseline (up to 8.08×) and often match or exceed serving engines with
> default settings (4.05× for vLLM), but still fall below a simple hyperparameter search under the same
> time budget (up to 11.53×) [>1]

[>1]: From this recent [paper](https://inferencebench.ai/assets/paper.pdf).

Given that novel maths discoveries are [not so hard](https://openai.com/index/model-disproves-discrete-geometry-conjecture/), we probably shouldn't assume this will last forever.

This article then is an offering to our new AI overlords: one last echo of the age of centaurs, before the horses gallop off into the horizon of fully automated research.

Doubleword was recently named as one of six companies in the first wave of UK Sovereign AI investments[>2], which has given us access to a meaningful allocation on the AI Research Resource, including time on [Isambard-AI](https://blogs.nvidia.com/blog/isambard-ai/), the UK's national AI supercomputing facility. This post is a worklog of pushing that capacity as hard as it will go on a single node DeepSeek-Flash serving deployment, starting from `vllm serve` with sensible defaults.

[>2]: [Our first investments](https://sovereignai.gov.uk/post/our-first-investments), UK Sovereign AI.

## Make it work

First, make it work. Let's start with TP=4[>3].

[>3]: We'll see why this isn't perfect for DeepSeek with MLA later.

DeepSeek has a native FP8 KV cache[>4], so enable that. We're focused on throughput at all costs. We want tool calling and reasoning to be parsed out properly.

[>4]: DeepSeek-V4 [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf): "Both paths use FP8 storage for most KV entries and BF16 only for the RoPE dimensions."

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 128000 \
  --max-num-batched-tokens 8192 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

Benchmark is `vllm bench serve`, closed-loop with concurrency tracking `max-num-seqs` so the queue is always full, ISL/OSL = 1024/1024, fp8 KV.

**4408 output tok/s, 1102 per GPU.** Reference point.

## Tune it

Pin TP workers to their NUMA nodes. Small but free.

```diff
+ --numa-bind
```

**4805 output tok/s, 1201 per GPU. +9%.**

Now push the shape. vLLM caps CUDA-graph capture at batch 512 by default, so the decode batch falls off into eager. Raise the CG cap, raise `max-num-seqs` and bench concurrency together, push gmu to the practical ceiling.

```diff
- --gpu-memory-utilization 0.92
+ --gpu-memory-utilization 0.97
+ --max-num-seqs 1536
+ --compilation-config '{"max_cudagraph_capture_size": 1536}'
```

**7624 output tok/s, 1906 per GPU. +59% on NUMA, +73% on §1.** Diminishing returns on this shape.

## Change the shape

DeepSeek-V4 has an interesting shape for parallelism, which we've been ignoring so far. It uses Multi-head Latent Attention[>5], original to DeepSeek, an attention variant that reduces the amount of stored KV cache by compressing the KV heads into a single shared "latent" vector.

[>5]: DeepSeek-V3 [technical report](https://arxiv.org/abs/2412.19437); Doubleword, [tensor network attention](https://blog.doubleword.ai/tensor-network-attention#multi-head-latent-attention).

The problem is that for tensor parallel attention, the head dimension is the useful dimension to shard over — we usually compute different attention heads on different accelerators. With only one head, naive tensor parallelism for MLA has to replicate the KV cache on each accelerator. The KV cache then has to live in N copies, where N is the parallelism degree.

For DeepSeek, the much cleaner shape is data-parallel attention. If you replicate your attention layers on each accelerator, then each can have its own KV cache. Side effect: every per-token kernel (elementwise ops, FP8 quant, sampling) now runs on 1/N of the batch per rank, instead of the full global batch on every rank as it would under TP.

The drawback is that the weights get replicated. But for high-throughput or long-context inference, that's much better (remember, we're only discussing the attention weights here) than replicating the KV cache.

```diff
- --tensor-parallel-size 4
+ --tensor-parallel-size 1
+ --data-parallel-size 4
- --gpu-memory-utilization 0.97
+ --gpu-memory-utilization 0.92
```

**11532 output tok/s, 2883 per GPU. +51%.**

## Tune it, again

Same playbook on the new shape: push gmu and max decode batch toward the practical ceiling. gmu wins. Push it to 0.975.

```diff
- --gpu-memory-utilization 0.92
+ --gpu-memory-utilization 0.975
```

**13970 output tok/s, 3493 per GPU. +21% on §3. 3.17× on §1.**

## Worklog ends here

That's where the easy wins end. Past this number the levers stop being config knobs. MoE kernel choice and expert-routing imbalance, attention-side speculation, multi-node expert parallelism. Each is its own post.

Written by hand.
