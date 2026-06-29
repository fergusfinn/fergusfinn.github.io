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

## Draft: Next steps

TODO: Claude thinks that the SWA buffer for deepseek is overprovisioned for a
benchmark of this shape. This kind of makes sense for deepseek given that its
long context. We could follow this up later. 

### Scale out

This post should continue by scaling out to multiple nodes, with dp attention
and expert parallelism, using the upstream PRs
https://github.com/uccl-project/uccl/pull/997 and
https://github.com/uccl-project/uccl/pull/998 to build a version of UCCL that
will support multi-node EP on CXI/slingshot. That then gets installed in vLLM
as a DeepEP shim, and the backend runs as DeepEP high throughput
(alternatively, maybe deepEP low-latency? i have experimented with getting cuda
graphs workig with both https://github.com/doublewordai/uccl/pulls, and have,
non-repeatably, seen higher e2e throughput per GPU with scale out using ht with
CUDA graphs.). 

We should be able to see a progression: 1 node, 2 node, 4 nodes, with each step
increasing the throughput per GPU. We achieve this increase in throughput per
GPU by being able to store a larger KV cache, and therefore run at a higher
batch size. 

We might break this down into separate experiemnts for prefill/decode, and it
might require getting CUDA graphs setup or doing some modelling of the comms
boundedness/benchmarking of the UCCL EP shape. It's possible that achieving the
proper scale out behaviour requires DBO. It almost certainly at decode requires
CUDA graphs.

The ultimate goal of this step, is this succession of increasing numbers, and a
profile for a representative step. From that profile, we ought to be able to
pick out the most expensive kernels, which we ought to be able to categorise
cleanly, as memory bound, compute bound, or comms bound.

There's an important ratchet that needs doing. We need to trade off KV cache
space (gmu), communications buffers, and the space used for CUDA graphs. Before
leaving this section, we want to be confident that we've got the largest KV
cache size that we can have, that throughput scales up linearly with
concurrency, that every pure decode step (at least) is cleanly CUDA graphed,
that there are no stalls in the profile. 

During this scale up we probably want to run different benchmarks - especially
ISL=1, OSL=1024, when focussing on decode performance, ISL=1024, OSL=1 when
focussing on prefill performance etc, to make sure that each phase of the
target (ISL=1024, OSL=1024) benchmark runs at max perf.

#### Findings

##### Eager mode

DeepSeek-V4-Flash with
host-native vLLM and the UCCL DeepEP high-throughput shim over CXI, at
ISL=OSL=1024, all with `--numa-bind` so each worker's host buffers stay local
to its GPU's NIC. Throughput below is output tokens/s (total, input+output, is
≈2× at ISL=OSL=1024).

One task per node (`srun --ntasks-per-node=1 --gres=gpu:4`), the same command
everywhere — DP ranks span all nodes, four GPUs each, and the per-node role
falls out of `$SLURM_NODEID`:

```bash
NNODES=$SLURM_JOB_NUM_NODES                          # 1, 2, or 4
MASTER=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
DP_LOCAL=4                                            # GPUs per node
DP_GLOBAL=$((NNODES * DP_LOCAL))
START_RANK=$((SLURM_NODEID * DP_LOCAL))
HEADLESS=$([ "$SLURM_NODEID" = 0 ] && echo "" || echo "--headless")

vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --enforce-eager \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DP_GLOBAL" \
  --data-parallel-size-local "$DP_LOCAL" \
  --data-parallel-address "$MASTER" \
  --data-parallel-rpc-port 29972 \
  --data-parallel-start-rank "$START_RANK" $HEADLESS \
  --enable-expert-parallel --all2all-backend deepep_high_throughput \
  --kv-cache-dtype fp8 --block-size 256 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --max-num-batched-tokens 8192 --max-num-seqs 1536 \
  --numa-bind --numa-bind-nodes 0 1 2 3 \
  --host 0.0.0.0 --port 8000
```

Rank 0 serves the HTTP endpoint; every other node runs the identical command
with `--headless` (no API server, just engine cores). The headless partner only
exists at ≥2 nodes — the single-node run is rank 0 alone. The benchmark is the
same `vllm bench serve` as above, with `--max-concurrency` set to 1024 (fixed)
or 1024×nodes (matched) and `--num-prompts` twice that.

Two questions.

**First: at a fixed total load, how much KV headroom does scaling out open
up?** Holding concurrency at 1024 across all three configurations and reading
the peak KV cache occupancy tells us how far we could push concurrency before KV
fills — the implied maximum is just 1024 divided by the peak fraction.

<!-- PRIOR NUMBERS (pre --numa-bind; "throughput" rows were total tokens, in+out):
Table 1 (fixed conc 1024): Peak KV 30.7/14.5/10.2%; implied max conc 3,336/7,075/10,057;
total tok/s 9,750/6,196/4,916; per-GPU 2,438/774/307; median TPOT 192/315/392 ms.
Table 2 (matched 256/GPU): per-GPU 2,438/1,367/1,010; retained 100/56/41%;
median TPOT 192/352/480 ms; Peak KV 30.7/23.2/22.3%. -->

| | 1 node (4 GPU) | 2 node (8 GPU) | 4 node (16 GPU) |
|---|---|---|---|
| Peak KV usage | 30.7% | 14.5% | 10.2% |
| Implied max concurrency | 3,336 | 7,077 | 10,059 |
| Output throughput (tok/s) | 5,152 | 4,775 | 4,601 |
| Per-GPU output (tok/s) | 1,288 | 597 | 288 |
| Median TPOT | 183 ms | 197 ms | 201 ms |

The headroom grows steeply, but at a fixed total concurrency the per-GPU
throughput goes down.

**Second: holding the per-GPU work constant, what does scaling out actually
cost?** Matching concurrency to GPU count — 1024, 2048, 4096, i.e. 256 requests
per GPU throughout — keeps the decode batch fixed and isolates the communication
tax.

| | 1 node @ 1024 | 2 node @ 2048 | 4 node @ 4096 |
|---|---|---|---|
| Per-GPU output (tok/s) | 1,288 | 1,078 | 877 |
| Retained vs 1 node | 100% | 84% | 68% |
| Median TPOT | 183 ms | 227 ms | 279 ms |
| Peak KV usage | 30.7% | 23.3% | 22.4% |

At matched batch the per-GPU throughput loss tracks 1/TPOT almost exactly, so
the tax lands in the decode step.

**Third: holding the KV-cache occupancy roughly constant, what is the per-GPU
throughput?** The first two questions either waste the extra KV (fixed total
load) or refuse to use it (fixed per-GPU batch). This one pushes concurrency on
each configuration until KV sits near saturation (~90%) — the operating point a
throughput deployment would actually run at — and reads the per-GPU output
there. KV per GPU grows with scale-out because the experts shard further (FP4
expert weights drop from ~52 GiB/GPU at EP4 to ~26 at EP8), so each step out
both frees KV and admits a larger decode batch.

| | 1 node | 2 node | 4 node |
|---|---|---|---|
| KV pool (GiB/GPU) | 35.3 | 46.7 | 48.3 |
| Concurrency (≈90% KV) | 4,096 | 11,264 | 23,040 |
| Peak KV usage | 86% | 89% | 87% |
| Output throughput (tok/s) | 22,500 | 28,000 | 19,000 |
| Per-GPU output (tok/s) | 5,625 | 3,500 | 1,190 |

The KV pool stops growing at four nodes (46.7 → 48.3 GiB/GPU) even though the
experts keep sharding to half the per-GPU weight — at EP16 the per-GPU prefill
all-to-all activation reservation grows enough to absorb what sharding frees
(TODO: setting a smaller max-num-batched-tokens gets this back under control, as would disagg prefill).
See [Disaggregated prefill](#disaggregated-prefill) for the measured breakdown.

##### CUDA Graphs

### Dual batch overlap

Dual batch overlap is a technique for overlapping comms and compute, by
splitting the running batch into two batches, and running the expert comms
overlapping with the compute of the other batch etc. This section would have
the picture of the overlap schedule from sglang or vLLM. 

This should give a throughput increase, and that throughput increase should
increase with increasing scale out, up to the point at which the comms
dominates over the compute, at which point the growing comms should start to
dominate. 

It might require careful tuning though, since comms uses SMs that are
partitioned off of the SMs that are required for doing the compute. 

Before doing any careful tuning, we need to have a CUDA graphed overlapped
trace, where we know that there's no host overhead, and we can see visually
that every comms step is cleanly overlapped with a corresponding compute step. 

### Disaggregated prefill

https://github.com/uccl-project/uccl/pull/999 this PR merged into UCCL main
gives support for disaggregated prefill. 

Disaggregated prefill has a throughput benefit for DPA. Chunked prefills route
randomly to DP ranks, and ranks with prefill chunks stall ranks without. Its
also possible to specialise i.e. kernels of each stage, and things like the EP
width of each.

In this section, we ought to demonstrate the throughput benefit of disagg
prefill. We need to prove it 'per-GPU', which is much harder. its only likely
to make sense once the comms group is as wide as it can be, since then we're no
longer KV cache bound on decode throughput. 

**Prefill activation eats decode KV (measured, 1/2/4-node sweep).**

- vLLM sizes the KV pool as `budget − weights − peak forward activation −
  non-torch`. The peak activation is set by a *prefill* step: up to
  `max_num_batched_tokens` tokens through the MoE all-to-all.
- That activation scales with `max_num_batched_tokens` and with EP rank count.
  At `max_num_batched_tokens=8192`: 8.8 GiB/GPU at EP8, 15.8 GiB at EP16 (~1
  GiB/rank). Non-torch (communication) memory stays flat at ~2.1 GiB.
- The token scaling is direct: at EP16, cutting `max_num_batched_tokens`
  8192→2048 drops peak activation 15.8 → 5.7 GiB and lifts the KV pool 48 → 60
  GiB/GPU — ~12 GiB of decode KV recovered by shrinking the prefill chunk.
- So per-GPU KV barely grew 2→4 nodes (46.7 → 48.3 GiB) even though the expert
  weights halved (26.2 → 17.6 GiB/GPU): the ~8.5 GiB that sharding freed went to
  peak activation (+7.0), not KV (+1.6).
- During decode this activation doesn't bind — decode batches are small and
  don't drive the wide all-to-all. On an aggregated deployment every decode rank
  still reserves it, so decode width is wasted on a prefill reservation that
  grows with EP width.
- Disaggregating prefill takes that reservation off the decode ranks: KV returns
  to decode, and the saving grows as the decode group scales out. Prefill and
  decode then also want different `max_num_batched_tokens` and EP width.

### Expert balancing, routing

The expert load is likely uneven in the benchmark. We want to load balance the
experts properly, and also do 'expert waterfill', where the least loaded
expert node runs the shared expert? 

Since we're throughput optimizing, this is also an interesting place to explore
reordering requests to minimise cross node comms, or similar. 

## Speculative decoding

Speculative decoding gives a conditional win at high throughput because experts
are so hard to get compute bound, so extra speculation tokens can ride along
for free. They're not free on the attention half though, for deepseek. If the
bottleneck analysis so far has told us that there is likely to be such a win
(i.e we're memory bound, not compute or comms bound, and cleanly that (i.e.
you're not bound if you're not fully overlapped - comms probably isn't fully
overlapped, so doesn't bind, whereas memory & compute are fully overlapped, so
we're cleanly either one or the other)

## Next

At this stage, we should have some wide model deployment, with $K$ decoding
ranks, and $M$ prefilling ranks, all sending KV cache to one another.

It should be reliable, and this blog should have shown ever increasing
throughputs for the benchmark. We should be able to show high MFU numbers for
the GPUs. We should have a good sense of what the bottleneck is in the model
(compute, memory, comms). Each step should be repeatable, and preferably we'll
have a profile for each step, as the throughput increases.

If we've successfully got to compute, great! we should think about offloading
some of the weights to get more KV cache maybe. 

Generally, we can try out some more exotic stuff. Fused MoE backends, SonicMoE,
mKernel and other fused compute/comms kernels. Thinking about the serving
structure: can we use the host RAM & NVLink C2C at all? 
