---
title: 'Scaling out: when wide expert parallelism pays'
description: 'When does adding nodes raise per-GPU throughput? The wide-EP scale-out story, on a model still to be chosen.'
pubDate: 'Jul 1 2026'
draft: true
---

> **Draft plan (agreed — not body copy; delete before publish).** The scale-out
> story: when does adding nodes raise *per-GPU* throughput? The mechanism — go
> wider, shard expert weights, free HBM, grow KV and batch, pay in scale-out comms
> — is the empirical companion to the *when-to-wideEP* post. The model is
> undecided: V4-Flash doesn't scale past a single node (scale-out BW ≪ intra-node,
> DBO halves the batch), so this needs a model where wide EP actually pays —
> **TBD**. Everything below is model-agnostic scale-out machinery, copied from the
> bundled draft as scratch to adapt; the model choice and the rising 1→2→4-node
> per-GPU curve are both pending. Stands alone for now.

## Scale out

This post should continue by scaling out to multiple nodes, with dp attention and expert parallelism, using the upstream PRs https://github.com/uccl-project/uccl/pull/997 and https://github.com/uccl-project/uccl/pull/998 to build a version of UCCL that will support multi-node EP on CXI/slingshot. That then gets installed in vLLM as a DeepEP shim, and the backend runs as DeepEP high throughput (alternatively, maybe deepEP low-latency? i have experimented with getting cuda graphs workig with both https://github.com/doublewordai/uccl/pulls, and have, non-repeatably, seen higher e2e throughput per GPU with scale out using ht with CUDA graphs.).

We should be able to see a progression: 1 node, 2 node, 4 nodes, with each step increasing the throughput per GPU. We achieve this increase in throughput per GPU by being able to store a larger KV cache, and therefore run at a higher batch size.

We might break this down into separate experiemnts for prefill/decode, and it might require getting CUDA graphs setup or doing some modelling of the comms boundedness/benchmarking of the UCCL EP shape. It's possible that achieving the proper scale out behaviour requires DBO. It almost certainly at decode requires CUDA graphs.

The ultimate goal of this step, is this succession of increasing numbers, and a profile for a representative step. From that profile, we ought to be able to pick out the most expensive kernels, which we ought to be able to categorise cleanly, as memory bound, compute bound, or comms bound.

There's an important ratchet that needs doing. We need to trade off KV cache space (gmu), communications buffers, and the space used for CUDA graphs. Before leaving this section, we want to be confident that we've got the largest KV cache size that we can have, that throughput scales up linearly with concurrency, that every pure decode step (at least) is cleanly CUDA graphed, that there are no stalls in the profile.

During this scale up we probably want to run different benchmarks - especially ISL=1, OSL=1024, when focussing on decode performance, ISL=1024, OSL=1 when focussing on prefill performance etc, to make sure that each phase of the target (ISL=1024, OSL=1024) benchmark runs at max perf.

## Dual batch overlap

Dual batch overlap is a technique for overlapping comms and compute, by splitting the running batch into two batches, and running the expert comms overlapping with the compute of the other batch etc. This section would have the picture of the overlap schedule from sglang or vLLM.

This should give a throughput increase, and that throughput increase should increase with increasing scale out, up to the point at which the comms dominates over the compute, at which point the growing comms should start to dominate.

It might require careful tuning though, since comms uses SMs that are partitioned off of the SMs that are required for doing the compute.

Before doing any careful tuning, we need to have a CUDA graphed overlapped trace, where we know that there's no host overhead, and we can see visually that every comms step is cleanly overlapped with a corresponding compute step.

## Expert balancing, routing

The expert load is likely uneven in the benchmark. We want to load balance the experts properly, and also do 'expert waterfill', where the least loaded expert node runs the shared expert?

Since we're throughput optimizing, this is also an interesting place to explore reordering requests to minimise cross node comms, or similar.

## Next

At this stage, we should have some wide model deployment, with $K$ decoding ranks, and $M$ prefilling ranks, all sending KV cache to one another.

It should be reliable, and this blog should have shown ever increasing throughputs for the benchmark. We should be able to show high MFU numbers for the GPUs. We should have a good sense of what the bottleneck is in the model (compute, memory, comms). Each step should be repeatable, and preferably we'll have a profile for each step, as the throughput increases.

If we've successfully got to compute, great! we should think about offloading some of the weights to get more KV cache maybe.

Generally, we can try out some more exotic stuff. Fused MoE backends, SonicMoE, mKernel and other fused compute/comms kernels. Thinking about the serving structure: can we use the host RAM & NVLink C2C at all?

## Appendix: V4-Flash scale-out sweep (scratch data)

`[Scratch — measured on V4-Flash, carried over from the bundled draft. This is the in-flight data behind "V4-Flash doesn't scale out, so switch models": per-GPU output falls as nodes are added. Kept for the numbers, not as narrative — rework against the real (different-model) curve, or cut, once that lands.]`

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
same `vllm bench serve` as in the single-node post, with `--max-concurrency` set to 1024 (fixed)
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
See the disaggregated-prefill post for the measured breakdown.
