---
title: 'Parallel Primitives for Multi-Agent LLMs'
description: |
  Exploring coordination patterns from parallel computing for multi-agent LLM systems
pubDate: 'Dec 31 2025'
---

## The Multi-Agent Coordination Problem

'Agent' is a fuzzy concept that's slowly starting to become concrete as the
capabilities emerge to make it meaningful. People have tried to outline a sharp
definition—here are a few examples from people who ought to know what they're
talking about:

> AI agents are software systems that use AI to pursue goals and complete tasks
> on behalf of users. They show reasoning, planning, and memory and have a
> level of autonomy to make decisions, learn, and adapt. (from Google, [here](https://cloud.google.com/discover/what-are-ai-agents?hl=en))

> Agents are systems that intelligently accomplish tasks—from simple goals to complex, open-ended workflows (from openAI, [here](https://platform.openai.com/docs/guides/agents))

> Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
> Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks. (from anthropic, [here](https://www.anthropic.com/engineering/building-effective-agents))

I want to work from a different definition here:

> An agent is an algorithm in which some of the logic is replaced by a call to
> an LLM.

This encompasses all of the above. There's a spectrum from pure LLM workflows,
to workloads with rigid scaffolding, in which calls to the LLM are statically
encoded in the algorithm structure. We can choose to let the structure of the
algorithm emerge as the application progresses, by letting the LLM 'make
choices'. Or we can encode the algorithm ahead of time, and use the LLM as our
'reasoning module'.

In principle, the same agent could be implemented anywhere along the spectrum.
You could imagine a terrible version of Claude Code as an LLM-only workflow,
keeping all the files in your codebase in context, storing its diffs in
context, searching across files by reasoning over its context. Or (as in
reality) - it can offload those parts of the algorithm's structure to
'tools'-ripgrep, filesystems, edit tools. There's no difference in kind, just
in structure (and performance). We offload those parts of the algorithm that
are sufficiently formulaic that the opportunity cost of using LLM context for
those jobs is too high.

Today's agents are mostly calls to the same LLM in a loop. The default mode is
sequential: one model, thinking step by step, accumulating context as it goes.
Sometimes we spawn 'subagents' - subroutines to accomplish certain tasks.
Sometimes this happens in parallel. Sometimes the LLM chooses to call a 'tool':
performing some deterministic, algorithmic step, and then viewing the results.

This paradigm works fine when the task fits in a single agent's capacity, but
there's a class of problems where it doesn't, where the work simply exceeds
what one agent can handle. Querying large datasets is a motivating example.
It's also relevant for deep research - where Anthropic has said, while
exploring multi-agent systems:

> Multi-agent systems work mainly because they help spend enough tokens to
> solve the problem (from Anthropic,
> [here](https://www.anthropic.com/engineering/multi-agent-research-system))

It's a powerful idea: trying to build algorithmic structures that let us point
more tokens at the same problem might help us to eke out more performance. But what
can we actually do that makes many tokens processed across many LLM calls more
powerful than the same number of tokens processed by a single LLM? What
primitives we have for getting many LLM calls to cooperate on a shared goal?

The natural analogy when you start thinking about multi-agent coordination is
human societies. People have explored this: agent architectures that mirror
markets, firms, hierarchies, voting - humans have developed a toolkit of social
technologies to get groups to accomplish things. But there's a mismatch in
aims. These social structures evolved to solve a particular problem, which is
aligning agents that have _independent goals_, by harnessing self-interest or
human foibles as a coordination mechanism. We've mostly shied away from imbuing
LLMs with self-interest, for pretty good reasons, which means we're actually
solving a different problem than the one capitalism, the firm, democracy, are
optimized for.

The better analogy, I think, is also a more humble one. It's also one that
computer scientists have been working on for decades: **how do you get many
obedient processors to cooperate efficiently on a shared task?**

This literature gives us primitives with well-understood properties, complexity
measures like [work and depth](https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms) that let us reason about efficiency, and patterns
that have been battle-tested across applications. We can harness it by treating
the LLM not as an independent agent with goals but as a "smart" processing
unit, one that can do fuzzy comparisons, semantic aggregation, complex
reasoning rather than just arithmetic, and then adapt the classical parallel
algorithms to use these capabilities[>1].

[>1]:
Agency doesn't have to disappear as a goal. these algorithmic workflows
can be called by orchestrator agents, or the LLM processing units can take
up larger and larger chunks of the problem as capabilities keep climbing.

## Two primitives for coordination

We can understand this kind of structured parallel coordination in terms of two
dual primitives:

### [Fold](<https://en.wikipedia.org/wiki/Fold_(higher-order_function)>): aggregating unlike items

A fold takes a collection of items and reduces them to a single result by repeatedly applying a combining function. The sequential version walks through a list, maintaining an accumulator, and at each step combines the accumulator with the next element. But the sequential version has $O(n)$ depth, which means $n$ round trips to the LLM if that's your combining function, and that's often too slow.

The parallel version achieves $O(\log n)$ depth by restructuring the computation as a tree. Instead of combining elements left-to-right, you pair them up and combine each pair in parallel, then pair up the results and combine those, and so on until you have a single result. This is sometimes called a parallel reduction[>2]. If you have $n$ items and a combining function, you can get down to $\log n$ sequential LLM calls by running $n/2$, then $n/4$, then $n/8$ calls in parallel at each level.

The catch is that this only works cleanly if your combining function is [associative](https://en.wikipedia.org/wiki/Associative_property): `combine(combine(a, b), c)` needs to equal `combine(a, combine(b, c))`. We'll come back to this.

```python
async def fold(items: list[T], combine: Callable[[T, T], Awaitable[T]]) -> T:
    """Reduce a list to a single value using parallel tree reduction."""
    if len(items) == 1:
        return items[0]

    # Pair up elements and combine each pair in parallel
    pairs = [(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
    combined = await asyncio.gather(*[combine(a, b) for a, b in pairs])

    # Handle odd element if present
    if len(items) % 2 == 1:
        combined.append(items[-1])

    # Recurse on the reduced list
    return await fold(combined, combine)
```

Each level of recursion halves the list, so we get $O(\log n)$ depth. At each level, all the `combine` calls run in parallel via `asyncio.gather`. The `combine` function is where the LLM does its work.

[>2]: The [prefix scan](https://en.wikipedia.org/wiki/Prefix_sum) is a related operation that computes all the partial reductions, not just the final result. It requires extra work to propagate intermediate values back through the tree. The parallel reduction we're describing here is simpler: you only need the final answer.

### [Unfold](https://en.wikipedia.org/wiki/Anamorphism): decomposing a single item

Unfold is the dual of fold: where fold takes many items and produces one, unfold takes one item and produces many. You start with a seed value and a decomposition function that splits it into subproblems, and you keep splitting until you hit base cases. The parallel version expands all children at each level simultaneously, which again gives you $O(\log n)$ depth if the tree is balanced[>3].

[Quicksort](https://en.wikipedia.org/wiki/Quicksort) is a good example. The decomposition function takes a list and a pivot, partitions the list into elements less than and greater than the pivot, and returns the two partitions. Unfold then recursively expands both partitions in parallel. The work is $O(n)$ comparisons at each level since everything gets compared to something, but because left and right subtrees expand simultaneously, the depth is $O(\log n)$ for a balanced partition.

The decomposition function is where the LLM does its work. For sorting, it's making comparisons against the pivot. For other applications, it might be breaking a complex question into sub-questions, or splitting a document into semantically coherent chunks, or generating the branches of a search tree.

```python
async def unfold(seed: T, decompose: Callable[[T], Awaitable[list[T] | None]]) -> list[T]:
    """Expand a seed into leaves via parallel tree expansion."""
    children = await decompose(seed)

    if children is None:
        # Base case: seed is a leaf
        return [seed]

    # Recursively unfold all children in parallel
    nested = await asyncio.gather(*[unfold(child, decompose) for child in children])

    # Flatten results, preserving order
    return [leaf for leaves in nested for leaf in leaves]
```

The `decompose` function returns `None` for base cases (leaves) or a list of
subproblems to expand further. All children at each level expand in parallel,
giving $O(\log n)$ depth for balanced trees.

[>3]:
"Balanced" here means the decomposition splits things roughly evenly at
each step. If your decomposition function consistently puts most items on
one side and few on the other, you end up with $O(n)$ depth instead of $O(log n)$. For sorting, this depends on pivot selection: a pivot near the median
gives balanced partitions, while a pivot near the minimum or maximum gives
degenerate ones.

### [Hylomorphism](<https://en.wikipedia.org/wiki/Hylomorphism_(computer_science)>): divide and conquer

A hylomorphism is an unfold followed by a fold: you decompose a problem into
subproblems, solve the base cases, and then aggregate the results back up[>4].
This is the structure of [divide-and-conquer](https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm) algorithms, and it turns out to be
surprisingly general.

[Mergesort](https://en.wikipedia.org/wiki/Merge_sort) fits this pattern well: unfold splits the list in half recursively
until you hit single elements, and then fold merges pairs back together using
the LLM comparator[>5]. Quicksort is a hylomorphism too, though the fold step
is just concatenation since the partitioning already establishes order.

The pattern generalizes beyond sorting. You could structure question-answering
over a large corpus this way, with unfold decomposing the question into
sub-questions or splitting the corpus into chunks, and fold aggregating partial
answers into a coherent response. Code review could work similarly, splitting
into files or functions and then combining per-file assessments. The
decomposition and aggregation functions change, but the computational structure
stays the same.

[>4]: The name comes from [category theory](https://en.wikipedia.org/wiki/Category_theory), where a hylomorphism is defined as the composition of an [anamorphism](https://en.wikipedia.org/wiki/Anamorphism) (unfold) and a [catamorphism](https://en.wikipedia.org/wiki/Catamorphism) (fold). You don't need the category theory to use the pattern, but the names are evocative once you know them.

[>5]: Sorting via mergesort is $O(n log n)$ because it's comparative. If we wanted $O(n)$ we could do something like [bucket sort](https://en.wikipedia.org/wiki/Bucket_sort), where the LLM classifies each item into a bucket in a single pass.

## Instantiations

### Summarization

Summarization is the clearest application of fold. You have a document too long
to fit in context, so you chunk it and fold the chunks into a single summary.
The combiner takes two text segments and produces a combined summary:

```python
async def summarize_combine(text_a: str, text_b: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"""Combine these two text segments into a single summary.

Instructions:
- Extract and merge the key factual claims from both segments
- Remove redundancy where the segments overlap
- Preserve specific details, names, and numbers
- For each key point, include a brief narrative synthesis

<segment_a>
{text_a}
</segment_a>

<segment_b>
{text_b}
</segment_b>

Combined summary:"""
        }]
    )
    return response.choices[0].message.content

chunks = split_into_chunks(document, max_tokens=2000)
summary = await fold(chunks, summarize_combine)
```

A few things to note about the prompt. We call them "text segments" rather than
"summaries" because the first level of the fold is combining raw chunks, not
summaries. The structured output format (key points with narrative synthesis)
makes the operation more associative: merging structured lists is less
order-dependent than merging prose. And the explicit instruction to preserve
details helps prevent information loss as you go up the tree.

By the end you have a single summary that's seen the whole document, even
though no single LLM call ever did.

### Search

Search works the same way structurally, but the combiner does something
different. Instead of compressing based on what's internally important, it
filters based on an external criterion: the query.

```python
async def search_combine(text_a: str, text_b: str, query: str, k: int) -> str:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"""Find the top {k} matches for this query across both text segments.

<query>{query}</query>

Instructions:
- Identify passages relevant to the query
- For each match, extract the key quote and explain its relevance to the query
- Rank by relevance and keep only the top {k}

<segment_a>
{text_a}
</segment_a>

<segment_b>
{text_b}
</segment_b>

Top {k} matches (format: quote, then relevance):"""
        }]
    )
    return response.choices[0].message.content

chunks = split_into_chunks(corpus, max_tokens=2000)
top_matches = await fold(chunks, lambda a, b: search_combine(a, b, query, k))
```

The query acts as a stable criterion across all levels of the fold. Unlike
summarization, where "what's important" might shift as context accumulates,
"relevant to the query" stays fixed. This makes the operation more naturally
associative: the same query drives every combination, so order matters less.

### Research Expansion

Unfold is natural for expanding a research query into a tree of web searches.
At each level, the LLM generates several search directions that cover different
facets of the question:

```python
async def expand_research(state: dict) -> list[dict] | None:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"""Expand this research direction into more specific searches.

<query>{state["query"]}</query>

<previous_searches>
{state["path"]}
</previous_searches>

Instructions:
- Generate 2-4 more specific search queries that explore different facets
- Each query should cover distinct ground (avoid overlap)
- Consider: different time periods, different perspectives, different subtopics
- If the query is already specific enough to search directly, respond: SEARCH

Response:"""
        }]
    )

    text = response.choices[0].message.content.strip()
    if text == "SEARCH":
        return None  # Leaf node: execute this search

    queries = [q.strip() for q in text.split("\n") if q.strip()]
    return [{"query": q, "path": state["path"] + [state["query"]]} for q in queries]

search_tree = await unfold(
    {"query": original_question, "path": []},
    expand_research
)
```

Say the original query is "What are the long-term economic effects of remote
work?" The first expansion might generate:

- "Remote work impact on commercial real estate prices 2020-2024"
- "Remote work effects on worker productivity studies"
- "Remote work migration patterns and regional economies"

Each of these expands in parallel. The real estate branch might generate
"Office vacancy rates major US cities post-pandemic" and "Commercial real
estate loan defaults remote work". The productivity branch generates "Remote
work productivity meta-analyses" and "Remote work employee monitoring software
effectiveness".

The `path` accumulates ancestor queries, so each branch knows what territory
has been claimed above it. Siblings expand in parallel without seeing each
other, but the parent generated them together with orthogonality in mind: the
instruction to "cover distinct ground" pushes the LLM to generate diverse
directions at each split. Eventually branches hit queries specific enough to
search directly, and the unfold returns all the leaves: 10-20 specific searches
that together cover the original question from multiple angles.

### Sorting

Sorting is the clearest hylomorphism: unfold to split the list, fold to merge
it back together. Mergesort makes the structure explicit:

```python
async def split(items: list[str]) -> list[list[str]] | None:
    if len(items) <= 1:
        return None  # Base case: already sorted
    mid = len(items) // 2
    return [items[:mid], items[mid:]]

async def merge(left: list[str], right: list[str], query: str) -> list[str]:
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""Which of these is more relevant to: {query}

<option_a>{left[i]}</option_a>
<option_b>{right[j]}</option_b>

Respond with just A or B."""
            }]
        )
        if response.choices[0].message.content.strip() == "A":
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result + left[i:] + right[j:]

# Unfold splits into singletons, fold merges back up
leaves = await unfold(items, split)
sorted_items = await fold(leaves, lambda a, b: merge(a, b, query))
```

The unfold is trivial here (just splitting in half), but the fold is doing real
work: each merge asks the LLM to compare items and interleave them in order.
The query provides the ranking criterion, like in search.

The contrast with embedding-based similarity search is worth noting. Both
approaches can use the same algorithms (heaps for top-k, quickselect,
mergesort), but embeddings do comparisons via cheap vector math while LLM
comparators require API calls. The tradeoff is cost vs quality: embeddings are
fast but the similarity metric is fixed, while LLMs can reason about relevance,
understand context, and apply judgment. Whether that's worth the cost depends
on the application[>6].

[>6]:
Embeddings also enable approximate nearest neighbor techniques like [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)
that achieve sublinear query time via pre-computed indexes. These don't
translate to LLM comparators: the comparison is query-dependent, you can't
pre-index, and there's no guarantee of transitivity to prune search.

### Deep Research as Full Hylomorphism

The research expansion example above only shows the unfold: expanding a
question into a tree of specific searches. But what happens after you execute
those searches? You fold the results back up.

```python
# Unfold: expand question into search tree
searches = await unfold(
    {"query": question, "path": []},
    expand_research
)

# Base case: execute each search
results = await asyncio.gather(*[execute_search(s["query"]) for s in searches])

# Fold: synthesize results back into a coherent answer
answer = await fold(results, synthesize_combine)
```

The synthesize combiner might look like the summarization combiner, but tuned
for research synthesis: combining findings from different sources, noting
agreements and contradictions, building toward an answer to the original
question.

This is a hylomorphism with LLM judgment on both sides. The unfold uses
judgment to decide how to explore (what directions to branch into, when a query
is specific enough). The fold uses judgment to decide what matters (which
findings are relevant, how they fit together). The full round-trip has LLMs
making substantive decisions at every level.

## What's missing

`fold` and `unfold` provide the scaffolding, but the actual work is in the
functions we pass to them.

A good combiner needs to be approximately associative, handle variable-quality
inputs without degrading catastrophically, and produce outputs structurally
similar to its inputs so the fold can keep going. In practice, many
applications enjoy a kind of fuzzy associativity: summarizing A and B then
combining with C probably lands close enough to summarizing B and C first. The
results end up in the same neighborhood regardless of evaluation order.

There's also a tradeoff in how much work we push to the LLM versus the
structure. LLMs don't need to be pure comparators operating on base cases—they
can sort lists of length $k$, or summarize chunks of size $m$, folding within
their own contexts. As capabilities improve, we can push more work into each
node and less into the network. The primitives stay the same but the granularity
shifts.

More on applying these ideas soon.
