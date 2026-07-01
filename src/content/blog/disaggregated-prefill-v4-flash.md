---
title: 'Disaggregating prefill on DeepSeek-V4-Flash'
description: 'Why disaggregated prefill helps a DP-attention deployment: prefill interference and the prefill-activation KV tax, and the per-GPU throughput payoff.'
pubDate: 'Jul 1 2026'
draft: true
---

> **Draft plan (agreed — not body copy; delete before publish).** Disaggregated
> prefill on DeepSeek-V4-Flash, treated as its own question. Two motivations, both
> worth the maths: under DP-attention a chunked prefill lands on a random rank and
> stalls its decode; and the prefill activation reservation (sized by a prefill
> step, growing with EP width) silently eats decode KV on every rank. The measured
> memory accounting is already in hand (~12 GiB/GPU of decode KV recovered by
> shrinking the prefill chunk). Pending: the payoff — end-to-end per-GPU
> throughput, disagg vs aggregated. Model stays V4-Flash; stands alone.

## Disaggregated prefill

https://github.com/uccl-project/uccl/pull/999 this PR merged into UCCL main gives support for disaggregated prefill.

Disaggregated prefill has a throughput benefit for DPA. Chunked prefills route randomly to DP ranks, and ranks with prefill chunks stall ranks without. Its also possible to specialise i.e. kernels of each stage, and things like the EP width of each.

In this section, we ought to demonstrate the throughput benefit of disagg prefill. We need to prove it 'per-GPU', which is much harder. its only likely to make sense once the comms group is as wide as it can be, since then we're no longer KV cache bound on decode throughput.

**Prefill activation eats decode KV (measured, 1/2/4-node sweep).**

- vLLM sizes the KV pool as `budget − weights − peak forward activation − non-torch`. The peak activation is set by a *prefill* step: up to `max_num_batched_tokens` tokens through the MoE all-to-all.
- That activation scales with `max_num_batched_tokens` and with EP rank count. At `max_num_batched_tokens=8192`: 8.8 GiB/GPU at EP8, 15.8 GiB at EP16 (~1 GiB/rank). Non-torch (communication) memory stays flat at ~2.1 GiB.
- The token scaling is direct: at EP16, cutting `max_num_batched_tokens` 8192→2048 drops peak activation 15.8 → 5.7 GiB and lifts the KV pool 48 → 60 GiB/GPU — ~12 GiB of decode KV recovered by shrinking the prefill chunk.
- So per-GPU KV barely grew 2→4 nodes (46.7 → 48.3 GiB) even though the expert weights halved (26.2 → 17.6 GiB/GPU): the ~8.5 GiB that sharding freed went to peak activation (+7.0), not KV (+1.6).
- During decode this activation doesn't bind — decode batches are small and don't drive the wide all-to-all. On an aggregated deployment every decode rank still reserves it, so decode width is wasted on a prefill reservation that grows with EP width.
- Disaggregating prefill takes that reservation off the decode ranks: KV returns to decode, and the saving grows as the decode group scales out. Prefill and decode then also want different `max_num_batched_tokens` and EP width.

**The payoff.** `[pending]` End-to-end per-GPU throughput, disaggregated vs aggregated, once the decode group is wide enough not to be the KV-bound loser.
