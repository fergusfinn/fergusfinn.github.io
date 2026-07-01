---
title: 'Throughputmaxxing DeepSeek-V4-Flash on a single 4×GH200'
description: 'Single-node serving throughput for DeepSeek-V4-Flash: baseline, MLA/DP-attention, config squeeze, and MTP1 speculative decoding.'
pubDate: 'Jul 1 2026'
draft: true
---

> **Draft plan (agreed — not body copy; delete before publish).** The single-node
> story: pushing DeepSeek-V4-Flash throughput on one 4×GH200 by changing serving
> config only. Arc: baseline (`vllm serve` defaults) → the MLA/DP-attention shape
> change (measured, 4,852 → 11,489 output tok/s, 2.37×) → squeezing the remaining
> single-node knobs → speculative decoding with MTP1, measured on a *real* dataset
> (not `random`, which carries no acceptance signal). Steps 1–2 are done; the MTP1
> run is the only experiment still pending. No scale-out in this post.

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
  --max-num-seqs 2048 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --numa-bind
```

Benchmark is `vllm bench serve`, random ISL/OSL = 1024/1024, request rate = inf, max concurrency = 2048.

```bash
vllm bench serve \
  --backend openai-chat \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 4096 \
  --request-rate inf \
  --max-concurrency 2048 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos
```

The run completed 4096/4096 requests with 0 failures, and the KV cache reached 100% usage.

**4852 output tok/s, 1213 per GPU. 9789 peak output tok/s. 9724 total tok/s.** Reference point.

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
- --gpu-memory-utilization 0.92
+ --gpu-memory-utilization 0.95
+ --compilation-config '{"max_cudagraph_capture_size":1536}'
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 4096 \
  --request-rate inf \
  --max-concurrency 2048 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos
```

The run completed 4096/4096 requests with 0 failures, and the KV cache reached 53% usage.

**11489 output tok/s, 2872 per GPU. 21329 peak output tok/s. 23023 total tok/s. 2.37× on §1.**

## Squeeze the config

`[pending]` The remaining single-node knobs — gpu-mem-util, CUDA-graph capture sizing, `max-num-batched-tokens`. One open thread carried over from the bundle:

TODO: Claude thinks that the SWA buffer for deepseek is overprovisioned for a
benchmark of this shape. This kind of makes sense for deepseek given that its
long context. We could follow this up later.

## Speculative decoding

`[pending: acceptance rate + output tok/s with MTP1, on a real dataset — random/ignore-eos has no acceptance signal]`

Speculative decoding gives a conditional win at high throughput because experts are so hard to get compute bound, so extra speculation tokens can ride along for free. They're not free on the attention half though, for deepseek. If the bottleneck analysis so far has told us that there is likely to be such a win (i.e we're memory bound, not compute or comms bound, and cleanly that (i.e. you're not bound if you're not fully overlapped - comms probably isn't fully overlapped, so doesn't bind, whereas memory & compute are fully overlapped, so we're cleanly either one or the other)
