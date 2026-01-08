---
title: 'Large-Scale Semantic Search Without Embeddings'
description: |
  Applying parallel primitives to search and rank 2.4 million arXiv papers using LLM judgments
pubDate: 'Jan 2026'
---

What would we do differently if LLM inference got 100x cheaper?

One answer: we'd stop using embeddings for search.

Embeddings are a compression. You run the model once per document, extract a vector, and store it. At query time, you do vector math instead of running the model again. It's fast because you've front-loaded the work, but the tradeoff is that the similarity metric is fixed at embedding time. You can't ask "relevant to X" for arbitrary X: you're stuck with whatever notion of similarity the embedding model learned.

If generative inference were cheap enough, you'd skip the compression. You'd just ask the model: "Is this document relevant to this query?" for every document, at query time. The judgment would be contextual, specific to what you're actually asking, not baked in ahead of time.

Inference isn't quite that cheap yet (though we're [working on it](https://batched.doubleword.ai/)). Embeddings aren't dead. But batched inference does close the gap enough to make this approach practical for a lot of use cases. The [OpenAI batch API](https://platform.openai.com/docs/guides/batch) gives you 50% off in exchange for higher latency, and the Doubleword [Batched](https://docs.doubleword.ai/batches) service takes off another factor of two.

At batch prices, you can start to afford things that would have been absurd at realtime rates: asking an LLM "is this relevant?" for every document in a corpus, ranking results with pairwise comparisons, treating the model as a general-purpose judgment function rather than a carefully-rationed oracle.

This post walks through building this from scratch: searching arXiv papers by relevance to a natural language query, using LLM judgments for both filtering and ranking.

If you'd like skip the details and see what this looks like in action, there's a live demo running at the [arXiv Observatory](https://arxiv-web.fly.dev). Sign in, and use large scale LLM inference to search and rank arXiv papers on any topic you like.

## The Approach

We have a dataset of arXiv paper abstracts, around 2.4 million of them, and a user shows up with a question: "What are the best papers on using attention mechanisms for time series forecasting?" We want to return the most relevant papers, ranked.

The direct approach is to ask an LLM to evaluate every abstract: show it the query and the abstract, ask "is this relevant?" At batch prices, 2.4 million yes/no judgments is tractable. But ranking is harder. If you wanted to rank all 2.4 million papers by relevance, you'd need pairwise comparisons, and O(n log n) comparisons on millions of papers is a different proposition entirely.

The solution is a two-pass approach. First, filter: ask the LLM "is this relevant?" for every paper, and discard the ones that aren't. Most papers are obviously irrelevant, so this pass is highly selective. Then, rank: use pairwise comparisons to sort the survivors. If the filter reduces 2.4 million papers to a few hundred candidates, the ranking pass becomes affordable.

## Setup

First, install the dependencies[>1]:

```bash
pip install duckdb autobatcher parfold
```

[>1]: [DuckDB](https://duckdb.org/) is an embedded analytical database that can query parquet files directly without loading them into memory. [autobatcher](https://github.com/doublewordai/autobatcher) wraps the OpenAI batch API to make it feel like normal async calls. [parfold](https://github.com/doublewordai/parfold) provides parallel primitives like the quicksort we'll use for ranking.

You'll also need an API key for a batched inference service. Sign up at [app.doubleword.ai](https://app.doubleword.ai) to get one, or configure autobatcher to use another OpenAI-compatible batch API.

Throughout this post we'll use `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`[>2], which is well-suited to these short judgment tasks. If you're using a different API, swap in whatever model you have access to.

[>2]: [Qwen3](https://huggingface.co/Qwen) is a family of open-weight models from Alibaba. The 30B-A3B variant is a mixture-of-experts model that activates only 3B parameters per token, making it fast and cheap to run. We host it on the Doubleword API at a fraction of the cost of comparable closed models.

The dataset is the [arXiv papers dataset](https://huggingface.co/datasets/nick007x/arxiv-papers) from Hugging Face. Download the parquet file:

```bash
wget https://huggingface.co/datasets/nick007x/arxiv-papers/resolve/main/train.parquet -O arxiv-metadata.parquet
```

We'll query it with DuckDB:

```python
import duckdb

def search_papers(query: str, limit: int = 1000) -> list[dict]:
    """Search for papers matching a keyword query."""
    con = duckdb.connect()

    sql = f"""
        SELECT arxiv_id, title, abstract, primary_subject, submission_date
        FROM 'arxiv-metadata.parquet'
        WHERE title ILIKE '%{query}%' OR abstract ILIKE '%{query}%'
        ORDER BY submission_date DESC
        LIMIT {limit}
    """

    result = con.execute(sql).fetchall()
    columns = ["arxiv_id", "title", "abstract", "primary_subject", "submission_date"]
    return [dict(zip(columns, row)) for row in result]
```

This gives us a crude first pass: keyword matching to pull out papers that mention the relevant terms. From here, we'll use LLM judgments to filter and rank.

## The Filter Pass

The filter asks a simple question for each paper: is this relevant to what the user is looking for?

```python
def format_paper(paper: dict) -> str:
    """Format a paper for the LLM to read."""
    abstract = paper.get("abstract", "")[:500]  # Truncate long abstracts
    return f"[{paper['arxiv_id']}] {paper['title']}\n{abstract}"

def make_relevance_prompt(user_interest: str, paper: str) -> str:
    """Create the prompt for relevance filtering."""
    return f"""Is this paper relevant to: {user_interest}

<paper>
{paper}
</paper>

Reply with ONLY "YES" or "NO"."""
```

The implementation uses `parfold.filter` to run all the relevance checks in parallel:

```python
from parfold import filter as pfilter

async def filter_papers(
    papers: list[dict],
    user_interest: str,
    client,
) -> list[dict]:
    """Filter papers by relevance using parallel LLM calls."""

    async def is_relevant(paper: dict) -> bool:
        prompt = make_relevance_prompt(user_interest, format_paper(paper))

        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )

        text = response.choices[0].message.content or ""
        return "YES" in text.upper() and "NO" not in text.upper()

    return await pfilter(papers, is_relevant)
```

Every paper gets checked independently, so the calls can all happen at once. If you have 1000 papers, you make 1000 LLM calls, but in wall-clock time it's a single round-trip (modulo batching, which we'll get to).

## The Sort Pass

Once we have relevant papers, we need to rank them. The approach is to use the LLM as a comparator: given two papers, which is more relevant to the query?

```python
def make_compare_prompt(user_interest: str, paper_a: str, paper_b: str) -> str:
    """Create the prompt for pairwise comparison."""
    return f"""Which paper is MORE relevant to: {user_interest}

<paper_a>
{paper_a}
</paper_a>

<paper_b>
{paper_b}
</paper_b>

Reply with ONLY "A" or "B"."""
```

This comparator slots into `parfold.quicksort`. The parallel structure comes from the partition step: all comparisons to the pivot happen in parallel, then the left and right partitions are sorted in parallel.

```python
from parfold import quicksort

async def rank_papers(
    papers: list[dict],
    user_interest: str,
    client,
) -> list[dict]:
    """Rank papers by relevance using parallel quicksort."""

    async def compare(a: dict, b: dict) -> int:
        prompt = make_compare_prompt(
            user_interest,
            format_paper(a),
            format_paper(b)
        )

        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )

        text = response.choices[0].message.content or ""
        winner = "A" if "A" in text.upper() and "B" not in text.upper() else "B"
        return -1 if winner == "A" else 1

    return await quicksort(papers, compare)
```

If you have k papers to rank, quicksort makes O(k log k) comparisons with O(log k) sequential depth. At each level of the recursion, all the comparisons happen in parallel.

## Batching

The code above has a problem. If you fire off 1000 parallel requests to the OpenAI API, you'll hit rate limits. Even if you don't, you're paying for 1000 separate HTTP round-trips, and latency adds up.

The OpenAI batch API solves this: you upload a file of requests, they process it, you download the results. It's cheaper (50% off) and avoids rate limits. But it breaks the async abstraction. You can't just `await client.chat.completions.create()` and get a response; you have to manage files, poll for completion, parse results.

We built [autobatcher](https://github.com/doublewordai/autobatcher) to bridge this gap. It's a drop-in replacement for `AsyncOpenAI` that collects requests, submits them as batches, and resolves the original futures when results come back:

```python
from autobatcher import BatchOpenAI

client = BatchOpenAI(
    base_url="https://api.doubleword.ai/v1",
    api_key=os.environ["DOUBLEWORD_API_KEY"],
    batch_size=100,            # submit after 100 requests accumulate
    batch_window_seconds=1.0,  # or after 1 second, whichever first
)

# Use exactly like AsyncOpenAI
response = await client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

From the caller's perspective, nothing changes. You write normal async code. Under the hood, requests accumulate and get batched together. When results come back, each waiting coroutine gets its response.

The Doubleword batch API also supports partial result streaming: you can poll for completed results before the whole batch finishes. autobatcher uses this to resolve futures as soon as their individual results are ready, rather than waiting for the entire batch. This matters for sorting, where you want to start the next level of recursion as soon as the current level's comparisons complete[>3].

[>3]: Partial result streaming is a Doubleword extension. The standard OpenAI batch API only returns results when the entire batch completes.

## The Full Pipeline

The full pipeline:

```python
async def search_arxiv(
    keyword_query: str,
    user_interest: str,
    initial_limit: int = 1000,
    top_k: int = 20,
) -> list[dict]:
    """
    Search arXiv papers by relevance to a natural language query.

    1. Keyword search to get initial candidates
    2. LLM filter to keep only relevant papers
    3. LLM sort to rank by relevance
    """
    client = BatchOpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
        batch_size=100,
        batch_window_seconds=1.0,
    )

    # Step 1: Keyword search
    candidates = search_papers(keyword_query, limit=initial_limit)
    print(f"Found {len(candidates)} candidates from keyword search")

    # Step 2: LLM filter
    relevant = await filter_papers(candidates, user_interest, client)
    print(f"Filtered to {len(relevant)} relevant papers")

    # Step 3: LLM sort
    ranked = await rank_papers(relevant, user_interest, client)

    await client.close()

    return ranked[:top_k]
```

## Results

The dataset contains 2.55 million arXiv papers. Let's search for papers on transformers for time series forecasting:

```python
import asyncio

results = asyncio.run(search_arxiv(
    keyword_query="time series forecasting",
    user_interest="Using attention mechanisms and transformers for time series forecasting",
    initial_limit=200,
    top_k=20,
))

for i, paper in enumerate(results, 1):
    print(f"{i}. [{paper['arxiv_id']}] {paper['title']}")
```

Here's what happens:

1. **Keyword search** pulls 200 candidates mentioning "time series forecasting"
2. **Filter pass**: 200 LLM calls, batched into a single request. The model marks 54 papers as relevant (27% pass rate). Wall-clock time: 17 seconds.
3. **Sort pass**: 362 pairwise comparisons to rank 54 papers. This is close to the theoretical O(n log n) ≈ 324 comparisons. The comparisons run across 11 batches as the quicksort recursion unfolds. Wall-clock time: 1 minute 55 seconds.

Total time: **2 minutes 14 seconds** to search, filter, and rank. The top results:

```
 1. [2509.18107] AdaMixT: Adaptive Weighted Mixture of Multi-Scale Expert Transformers
 2. [2509.04782] VARMA-Enhanced Transformer for Time Series Forecasting
 3. [2206.04038] Scaleformer: Iterative Multi-scale Refining Transformers
 4. [2410.04803] Timer-XL: Long-Context Transformers for Unified Time Series Forecasting
 5. [2506.05597] FaCTR: Factorized Channel-Temporal Representation Transformers
 6. [2212.02789] A K-variate Time Series Is Worth K Words
 7. [2405.03429] ReCycle: Residual Cyclic Transformers
 8. [2503.04118] TimeFound: A Foundation Model for Time Series Forecasting
 9. [2408.02279] DRFormer: Multi-Scale Transformer Utilizing Diverse Receptive Fields
10. [2308.04791] PETformer: Placeholder-enhanced Transformer
```

All transformer papers, and all designed specifically for time series
forecasting. The LLM understood our query in depth and ranked accordingly,
surfacing papers that a keyword search alone might have buried among thousands
of results.

## Where This Makes Sense

This approach isn't a wholesale replacement for embeddings. If you need
sub-second latency, or you're running the same similarity query millions of
times, embeddings are still the right choice because the upfront cost amortizes
over queries.

But there's a class of problems where the query is specific, the corpus is
bounded, and you care a lot about the quality of the answers: literature review,
due diligence, exploratory research. For these, the flexibility of asking an LLM
"is this what I'm looking for?" outweighs the cost. At batch prices, the search
we just ran costs less than a cent.

---

The code from this post is available at [github.com/doublewordai/arxiv-sorter](https://github.com/doublewordai/arxiv-sorter). The parallel primitives live in [parfold](https://github.com/doublewordai/parfold), and [autobatcher](https://github.com/doublewordai/autobatcher) handles the batch API plumbing. If you want to try it yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) for API access.
