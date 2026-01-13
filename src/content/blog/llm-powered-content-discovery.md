---
title: 'LLM-Powered Content Discovery'
description: |
  Building a content discovery system using parallel primitives and BST-based ranking with LLM comparisons
pubDate: '13 Jan 2026'
index: false
---

<!--
STATUS: Skeleton - awaiting feedback

OUTLINE:
- [ ] Introduction - the index problem
- [ ] The Pipeline - high-level view
- [ ] Query Expansion - unfold in action
- [ ] Filtering and Summarization - LLM as quality gate
- [ ] Ranking via BST - pairwise comparisons
- [ ] Exemplar Learning - votes as feedback
- [ ] Performance Reality - actual numbers
- [ ] Conclusion - judgment not generation

NOTES:
- Third post in series after parallel-primitives and bst-expensive-comparisons
- Audience: technical generalists comfortable with algorithms
- Don't call it "Vibe News"
- Emphasize the "weird/surprising" content angle
-->

## Introduction

[The problem: content discovery when your ranking criterion is subjective and freeform. Keywords don't capture "weird programming projects, people building things the hard way." Embeddings get you similarity but not judgment. What if you could describe what you want in natural language and have the system find and rank content accordingly?]

[This post ties together the primitives from the previous two posts. We built fold/unfold for parallel coordination; we built a concurrent BST for expensive comparisons. Now we put them to work.]

## The Pipeline

[High-level view of the system. User writes a freeform description of their interests. The system expands this into search queries, fetches content, filters for quality, and ranks by relevance using pairwise LLM comparisons. The result is a personalized feed that updates in real-time as new content is discovered.]

```txt
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Interest  │────▶│  Query Expansion │────▶│     Search      │
│  Description    │     │  (unfold)        │     │   (parallel)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                        ┌─────────────────┐     ┌────────▼────────┐
                        │   Ranked Feed   │◀────│    Filter &     │
                        │   (BST output)  │     │   Summarize     │
                        └─────────────────┘     └─────────────────┘
                                 ▲                       │
                                 │              ┌────────▼────────┐
                                 └──────────────│   BST Ranking   │
                                                │ (pairwise LLM)  │
                                                └─────────────────┘
```

[Each stage runs in parallel where possible. The unfold expands breadth-first. Search workers hit multiple providers concurrently. Summarization runs in parallel across results. BST insertion is concurrent with optimistic locking. The whole pipeline streams results to the user as they're ranked.]

## Query Expansion

[The unfold primitive from the first post, now doing real work. Starting from "weird programming projects, unusual implementations," the LLM generates orthogonal search directions: different content types (projects vs discussions vs tutorials), different sources (GitHub vs HN vs blogs), different framings (how-to vs post-mortem vs comparison).]

[Show the prompt structure. Emphasize orthogonality: we want coverage, not variations on a theme. The tree expands to depth 3, yielding 50-100 specific searches from a single freeform description.]

[The "SEARCH" escape hatch: when a query is specific enough, stop expanding and execute. This prevents over-decomposition.]

## Filtering and Summarization

[Before content enters the ranking BST, it passes through an LLM filter. This is doing two jobs: quality gating and feature extraction.]

[Quality filters: index pages, SEO spam, marketing dressed as content, aggregators without added value, thin content, paywalled teasers. The prompt enumerates these explicitly so the LLM knows what to reject.]

[Feature extraction: a summary optimized for downstream ranking (the hook, concrete details, why it's interesting), a relevance score, and a "weirdness score" calibrated from 0-1. The weirdness score is a first-pass signal that feeds into comparisons later.]

[The gate: content must pass quality filters AND score above a relevance threshold. This prevents the BST from filling up with marginally-relevant results.]

## Ranking via BST

[The BST from the second post, now ranking real content. Each piece of content that passes filtering gets inserted into a tree ordered by "interestingness to this user." Insertion requires O(log n) pairwise comparisons, each asking: "Which of these is more relevant/surprising/valuable for someone interested in {description}?"]

[The comparison prompt includes: the user's interest description, the two items being compared (title, summary, age, weirdness label), and crucially, exemplars from the user's voting history.]

[The threaded linked list pays off here. Once content is ranked, we can iterate in sorted order without further comparisons. Min/max are O(1). The BST becomes an index: pay for the comparisons once during insertion, query the results for free afterward.]

## Exemplar Learning

[Users can upvote and downvote content. These votes become exemplars that shape future comparisons.]

[Positive exemplars: "The user has upvoted content similar to these - prioritize content in this style." Included in the comparison prompt with title and truncated summary.]

[Negative exemplars: "The user has DOWNVOTED content similar to these - AVOID and deprioritize." Same structure, opposite signal.]

[This is a form of few-shot learning baked into the ranking criterion. The LLM isn't just matching the description; it's learning the user's taste from examples. The comparison cache invalidates when the description changes (different hash), but exemplars don't invalidate the cache - they're guidance layered on top.]

## Performance

[What do the numbers actually look like?]

[Query expansion: depth-3 unfold with 3-5 children per node yields 50-100 leaf queries. Wall-clock time is O(depth) since each level expands in parallel.]

[Search: 10 concurrent workers, rate-limited per provider. 30-minute cache on queries. Deduplication by URL before summarization.]

[Summarization: 40 concurrent workers. Each item gets one LLM call for filtering + summary + scores.]

[Ranking: BST insertions batched 500 at a time. Comparisons batched 100 at a time with 2-second windows. For a 200-item leaderboard, roughly 200 * log(200) ≈ 1500 comparisons, but many hit cache. Cache hit rates after warmup: 60-80%.]

[End-to-end: a fresh feed with no cache takes several minutes to populate. Incremental updates (new content into existing feed) are much faster since the BST structure and comparison cache persist.]

## Conclusion

[The system uses LLMs for judgment, not generation. Every LLM call is a decision: should this query expand further? Is this content worth ranking? Is A more interesting than B? The output isn't generated text; it's a ranked index that encodes thousands of these micro-judgments.]

[The result is a materialized view of the LLM's taste, personalized to the user's description and refined by their feedback. Once built, you can query it without further LLM calls: what's the best thing? What's in the top 10? Iterate through everything in order. The comparisons are already paid for.]

[There's an obvious parallel to embeddings-based indexes. Both pre-compute something expensive (embedding vectors, pairwise comparisons) to make queries cheap. The difference is what gets pre-computed. Embeddings encode semantic similarity via fixed geometry. LLM comparisons encode judgment that can consider relevance, surprise, quality, timeliness, and whatever else the prompt asks for. The tradeoff is cost: embeddings are cheap to compute and compare, LLM comparisons are expensive. But for applications where nuanced judgment matters, the cost might be worth it.]
