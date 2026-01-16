---
title: 'Scaling Curation with LLM Comparisons'
description: |
  Building a content discovery system using parallel primitives and BST-based ranking with LLM comparisons
pubDate: '16 Jan 2026'
---

<!--
STATUS: Draft complete - ready for review

OUTLINE:
- [x] Introduction - the curation problem
- [x] Research - query expansion via unfold
- [x] Filtering - relevance scoring
- [x] Ranking - BST with pairwise comparisons
- [x] Exemplar Learning - votes as feedback
- [x] Conclusion - judgment not generation

NOTES:
- Third post in series after parallel-primitives and bst-expensive-comparisons
- Audience: technical generalists comfortable with algorithms
-->

## Introduction

I've been thinking about how content discovery actually works online. The dominant model is aggregation: crowds voting on things, with the most popular stuff floating to the top. Hacker News, Reddit, Lobsters - they all work this way, and it's genuinely effective when your interests happen to align with an existing community. You get a feed filtered by collective judgment, there's always something new, and the sheer scale of participation means obscure things surface that you'd never find on your own.

The alternative is curation, where an individual does the filtering for you[>1]. Someone reads widely and picks the best of what they found, or you follow writers whose taste you trust. You're borrowing someone's judgment rather than averaging a crowd's, which often produces better results - a good curator has judgment that a voting mechanism can't replicate.

[>1]: Some good curated sources: [The Browser](https://thebrowser.com/) for essays, [Five Books](https://fivebooks.com/) for expert book recommendations.

The catch is that you need the crowd or the curator to exist. There has to be enough people who share your particular interest that their votes create signal, or you need one person who cares enough to create that signal by themself. And for a lot of things, neither is there. I'd like a feed of interesting programming content that isn't about AI, for example[>_1]. That's a perfectly reasonable thing to want, and the content certainly exists, scattered across blogs and forums and newsletters.

[>_1]: Fig 1. The problem.
![Google results for "interesting programming content" - mostly AI articles](https://fergusfinn.com/blog-images/google-programming-content.png)

I've been experimenting with a different approach: you describe what you want in natural language, and coordinated LLM calls make the judgment calls a curator would make. Is this relevant? Is this quality? Is this more interesting than that? The judgment is personalized to your description, not averaged from a crowd's taste or limited by one person's reading. Curation's judgment at aggregation's scale, through coordination rather than crowds.

This post ties together the primitives from the [previous](https://fergusfinn.com/blog/parallel-primitives) [two](https://fergusfinn.com/blog/bst-expensive-comparisons) posts. We built fold/unfold for parallel coordination; we built a concurrent BST for expensive comparisons. Let's put them to work. Here's a concrete description of what we're looking for:

> Interesting programming content that isn't about AI or LLMs. People building things the hard way: writing their own compilers, emulators, operating systems, text editors. Deep investigations into why something broke or how something works under the hood. Weird constraints leading to creative solutions. Projects where someone clearly cared more about the craft than the outcome.

## Research

High-throughput, low-cost inference[>3] means we can afford to filter aggressively - evaluate thousands of candidates and keep the best. But first we need the candidates. Search APIs are how content comes in, and each query returns a narrow slice of the web. To use the filtering capacity we have, we need to fan out: turn one description into many diverse queries that together cover the territory.

[>3]: We've been building a batch API for exactly these use cases at [Doubleword](https://app.doubleword.ai). Results come back in minutes rather than milliseconds, but you can run thousands of calls for pennies.

We expand recursively. The LLM takes the description and generates 3-5 search queries - different angles on what we're looking for. Each of those expands again, branching out until the queries are specific enough to run. This is [unfold](https://fergusfinn.com/blog/parallel-primitives): build a tree by repeated expansion.

```txt
                     "Interesting programming content..."
                                      │
               ┌──────────────────────┼──────────────────────┐
               │                      │                      │
         "compilers              "debugging              "hobby OS
          from scratch"          war stories"             projects"
               │                      │                      │
          ┌────┴────┐            ┌────┴─────┐               SEARCH
          │         │            │          │
     "parsing     SEARCH   "postmortem    "tracking
      from                    writeups"     down
      scratch"                   │        heisenbugs"
          │                      │          │
       SEARCH                 SEARCH      SEARCH
```

The LLM decides when to stop expanding. If a query is already specific enough to search, it returns SEARCH instead of generating children. "Hobby OS projects" is concrete enough; "compilers from scratch" could use another level of decomposition.

The prompt is simple:

```
Expand into 3-5 different search queries, or reply SEARCH if specific enough.

Query: {current_query}
Already searched: {path}

Generate diverse queries (different angles, not variations). One per line:
```

The path shows queries already generated above this point in the tree, which helps avoid redundancy. The emphasis on "different angles, not variations" pushes the LLM toward breadth rather than rephrasing the same idea.

The tree expands breadth-first, with each level running in parallel. A depth-3 tree with 3-5 children per node yields 50-100 leaf queries from a single description. Wall-clock time is O(depth) - three sequential LLM calls - regardless of how wide the tree gets.

## Filtering

The searches return noise. Index pages, SEO spam, paywalled teasers with no real content. You could try to filter aggressively here - reject anything that doesn't look like quality content - but I think that's the wrong place to spend effort. The ranking step already makes quality judgments through pairwise comparisons; filtering just needs to keep the candidate pool roughly on-topic.

So filtering is a single LLM call per candidate: given the content and the description, how relevant is this, from 0 to 1? Anything above 0.2 passes through.

```txt
   content                       score
   ───────────────────────────────────────
   "Writing a Compiler in Go"     0.7   ✓
   "10 Best Coding Bootcamps"     0.05  ✗
   "How We Debug at Stripe"       0.6   ✓
   "Marketing Your SaaS"          0.1   ✗
   "My Hobby OS Project"          0.4   ✓
   "nginx default index"          0.0   ✗
```

## Ranking

Content that passes filtering gets inserted into a [BST ordered by pairwise LLM comparisons](https://fergusfinn.com/blog/bst-expensive-comparisons). Each insertion traverses down the tree, comparing the new item against existing nodes. After $O(\log n)$ comparisons, it finds its place in the ranking.

The comparison prompt is minimal[>4]:

[>4]: More on exemplars below.

```
Which is more interesting for someone into: {description}?

Exemplars (content they liked):
{exemplars}

A: {title_a} ({age_a})
{content_a}

B: {title_b} ({age_b})
{content_b}

Reply A, B, or EQUAL:
```

The leaderboard has a maximum size, say 200 items. When a new item is inserted and the tree exceeds that limit, the minimum gets evicted[>2]. The [threaded linked list](https://fergusfinn.com/blog/bst-expensive-comparisons#threaded-linked-list) makes this cheap: the minimum is always at the head, so eviction is just unlinking a node.

[>2]: We didn't cover deletion in the BST post. It's fiddly, especially with concurrent access, but the threaded linked list makes eviction straightforward.

Once built, the BST is an index you can read without further LLM calls. Iterate in sorted order, grab the top 10, find the minimum - the comparisons were paid for during insertion.

## Exemplar Learning

The description alone doesn't fully capture what you want[>_2]. "Interesting programming content that isn't about AI" leaves a lot of room for interpretation - the LLM has to guess at your taste. Votes provide a way to refine that.

[>_2]: Fig 2. Better.
![Ranked results showing programming content without AI](https://fergusfinn.com/blog-images/vibe-news-results.png)

When you upvote or downvote content, those votes become exemplars that shape future comparisons.

## Conclusion

What makes this work is that judgment is cheaper than generation. A yes/no relevance check, a pairwise comparison - these are short outputs, easy to batch, and the cost keeps dropping. At no point does the LLM write anything you actually read; it just decides what's worth reading and in what order. That's what curators do. The difference is that a human curator can only read so much, and their taste might not match yours.

Embeddings are fast and good enough for a lot of things. But for the cases where you actually care about the quality of the ranking - where the judgment is the product - it might be worth paying for.

---

This runs on [Doubleword's batch API](https://app.doubleword.ai). If you want to build something similar, that's where to start.
