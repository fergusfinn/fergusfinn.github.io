---
title: 'LLM powered data structures: A concurrent, lock-free binary search tree'
description: |
  A lock-free binary search tree optimized for expensive async comparisons, with threaded linked list for O(1) sorted iteration
pubDate: '13 Jan 2026'
---

<!--
STATUS: Draft in progress

OUTLINE:
- [x] Introduction - BST as alternative to quicksort for LLM sorting
- [x] BST insertion - the basic mechanics
- [x] Parallel insertion - items racing down the tree
- [x] Two-phase insertion - optimistic concurrency control
- [x] Threaded linked list - zero-comparison iteration
- [x] Prefix caching and argument order
- [x] Limitations
- [x] Conclusion

NOTES:
- Code snippets from parfold library, simplified
- Audience: technical generalists, no BST hand-holding needed
-->

In a [previous post](https://fergusfinn.com/blog/parallel-primitives-blog) I introduced parallel
primitives for LLM workflows, and in a [follow-up](https://fergusfinn.com/blog/arxiv-llm-search)
applied them to search and rank 2.4 million arXiv papers. The ranking step used
quicksort with an LLM comparator: To sort a list of $N$ items, we partition
them around a pivot, then recurse on the partitioned halves, and let the
parallelism emerge from the recursion structure.

I want to introduce a neat alternative that comes from thinking about data
structures rather than algorithms. Instead of designing a computation that
transforms an unsorted list into a sorted one, you can ask: what data structure
would make sorted order easy to maintain? One really nice answer is a binary
search tree. Insert items, and sorted order falls out as a side effect of the
invariants of the BST structure.

The parallelism story is different from quicksort. In quicksort, you get
parallelism at each level of recursion: all comparisons against the pivot
happen at once, then the two partitions sort in parallel. In a BST (as we'll
see), you get parallelism across insertions: multiple items traverse the tree
simultaneously, each racing down toward its insertion point. The total
comparison count is the same, $O(n \log n)$, but the shape of the parallelism is
different. The result is a kind of LLM powered index for your data[^1].

[^1]:
    There's an obvious similarity to embeddings powered vector indexes. The
    difference is, with embeddings, the intelligence is baked into the vectors; the
    similarity metric is just cosine distance, arithmetic that doesn't know what
    you're asking. With generative comparisons, the metric itself is intelligent.
    The tradeoff is that embeddings front-load the work, whereas for generative
    comparisons you have to pay at query time. But you get to ask arbitrary
    questions: "most relevant to X", "most persuasive", "most technically novel".

Let's walk through how to build one that works well when comparisons are expensive and async.

## BST insertion

A binary search tree maintains a simple invariant: for every node, everything in
its left subtree is smaller, and everything in its right subtree is larger[>2].
To insert a new item, you compare it against the root; if it's smaller, you
recurse into the left subtree, otherwise the right. When you hit an empty spot,
you've found your insertion point.

[>2]: ![Binary search tree](https://upload.wikimedia.org/wikipedia/commons/d/da/Binary_search_tree.svg)
A BST. Everything left of 8 is smaller; everything right is larger. The property holds recursively at every node.

```python
async def insert(self, value: T) -> None:
    node = self._root
    while node is not None:
        cmp = await self._compare(value, node.value)
        if cmp < 0:
            if node.left is None:
                node.left = Node(value)
                return
            node = node.left
        else:
            if node.right is None:
                node.right = Node(value)
                return
            node = node.right
```

The comparison function is async because, in our case, it might be an LLM call.
For a tree of depth $d$, insertion requires $d$ comparisons, one at each level
as you descend. A balanced tree has depth $O(\log n)$[>3], so insertion is
$O(\log n)$ comparisons.

[>3]: $n$ being the total number of elements in the tree.

## Parallel insertion

If each comparison is an LLM call, it might take 500ms. To insert $n$ items into
a tree of depth $\log n$, you need $O(n \log n)$ comparisons. Do them
sequentially and you're waiting a long time. For 1000 items in a balanced tree,
that's roughly 10,000 comparisons, or about 80 minutes of wall-clock time spent
waiting for LLM responses.

But most of those comparisons don't depend on each other. When item A is
comparing against the root, item B could be comparing against a node in the left
subtree, and item C somewhere in the right subtree. They're touching different
nodes, so there's no reason they can't proceed in parallel:

```txt
                      ┌───┐
                      │ 5 │ ← A comparing here
                      └───┘
                     ╱     ╲
                    ╱       ╲
                ┌───┐       ┌───┐
    B here →    │ 2 │       │ 8 │    ← C comparing here
                └───┘       └───┘
               ╱     ╲           ╲
              ╱       ╲           ╲
          ┌───┐     ┌───┐       ┌───┐
          │ 1 │     │ 3 │       │ 9 │
          └───┘     └───┘       └───┘
```

The interface is just `asyncio.gather` over the insertions:

```python
tree = BST(llm_compare)
await asyncio.gather(*[tree.insert(item) for item in items])
```

When you fire off all the insertions at once, each item starts traversing the
tree independently, racing down toward its insertion point.

Each insertion does $O(\log n)$ comparisons as it descends, and there are $n$
insertions, so we're still doing $O(n \log n)$ comparisons total. But the shape
of the parallelism is different from quicksort. In quicksort, you get parallelism
within each partition: all $k$ elements compare against the pivot simultaneously,
then you recurse. In the BST, you get parallelism across insertions: all $n$
items are traversing the tree at once, but each one is doing its own sequential
chain of comparisons down from root to leaf.

There's a problem lurking here, though. What happens when two items race down
the same path and both try to insert at the same spot? If A and B both decide
they belong as the left child of some node, one of them is going to have a bad
time. We need some kind of concurrency control.

## Two-phase insertion with optimistic concurrency

The naive fix is a global lock: acquire it before traversing, release it after
inserting. But that serializes everything. We'd be back to doing one comparison
at a time, waiting for each LLM call to complete before starting the next.

The insight is that comparisons don't modify the tree. They just read a node's
value and decide left or right. The only mutation happens at the end, when we
link the new node as someone's child. So we can split insertion into two phases:
a lock-free traversal where we do all the expensive comparisons, and a brief
locked phase where we do the pointer write.

```python
async def insert(self, value: T) -> None:
    while retries < max_retries:
        # Phase 1: Traverse to find insertion point (no lock)
        node = self._root
        parent = None
        go_left = False

        while node is not None:
            saved_version = node.version
            cmp = await self._compare(value, node.value)  # expensive!

            if node.version != saved_version:
                break  # tree changed, restart

            parent = node
            go_left = cmp < 0
            node = node.left if go_left else node.right
        else:
            # Phase 2: Link new node (with lock)
            async with self._link_lock:
                if go_left and parent.left is None:
                    parent.left = Node(value)
                    return
                elif not go_left and parent.right is None:
                    parent.right = Node(value)
                    return
            # slot was taken, retry
        retries += 1
```

The `node.version` field is the key. Each node has a version counter that gets
bumped whenever its children change. After doing an expensive comparison, we
check if the version changed while we were waiting. If it did, the tree
structure might have shifted under us: maybe a new child appeared exactly where
we were about to insert. Rather than try to recover, we restart from the root.

This is optimistic concurrency control. We assume conflicts are rare, proceed
without locking, and pay a retry cost when we're wrong. There's a tension here:
optimistic locking is usually attractive when retries are cheap, but retrying
here means throwing away $O(\log n)$ LLM calls and starting over. The saving grace
is that conflicts require two items to reach the same insertion point at nearly
the same instant. With comparisons taking hundreds of milliseconds, that window
is tiny, and items on different paths don't conflict at all. In practice, with
hundreds of concurrent insertions, retries are rare. And if you do retry, you can cache the comparison results
from your first attempt - the retry just replays the cached answers until it
reaches the point where the tree changed.

The lock itself is minimal: just the pointer assignment to link the new node.
Everything expensive happens outside the lock.

## Threaded linked list for free iteration

Once you've built the tree, you'll want to access the sorted results. With a
plain BST, finding the minimum means walking down the left spine[>4]. Threading
a linked list through the nodes makes min O(1), and iteration becomes simple
pointer chasing instead of tree traversal. Each node gets
`prev` and `next` pointers to its in-order predecessor and successor:

```python
@dataclass
class Node(Generic[T]):
    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None
    prev: Node[T] | None = None   # in-order predecessor
    next: Node[T] | None = None   # in-order successor
```

[>4]: This is O(log n) cheap pointer hops, not O(log n) expensive comparisons.
The threading is a nice-to-have since we're already doing pointer updates
during insertion.

The tree also keeps `_head` and `_tail` pointers to the smallest and largest
nodes. When we insert a new node, we thread it into the list as part of the
linking phase:

```python
if go_left:
    parent.left = new_node
    # new_node comes just before parent in sorted order
    new_node.next = parent
    new_node.prev = parent.prev
    if parent.prev:
        parent.prev.next = new_node
    else:
        self._head = new_node
    parent.prev = new_node
```

The key insight is that in a BST, a node's in-order predecessor is always the
parent you most recently went left from. When you insert as a left child, you
slot in just before your parent. When you insert as a right child, you slot in
just after. So we can maintain the threading during insertion with just a few
pointer updates, all inside the lock we already hold.

Now iteration is just pointer chasing:

```python
def __iter__(self) -> Iterator[T]:
    node = self._head
    while node is not None:
        yield node.value
        node = node.next
```

And min/max are $O(1)$:

```python
@property
def min(self) -> T | None:
    return self._head.value if self._head else None
```

## Prefix caching and argument order

Most LLM providers offer prefix caching: prompts sharing a common prefix reuse
cached computation. For comparison prompts, this creates a tradeoff in how you
order the arguments, especially when comparing large items.

**Item-first ordering**: When inserting item X, the comparisons are (X, root),
(X, child), (X, grandchild), and so on. These all share X's description as a
prefix. As X descends the tree, each comparison benefits from having X already
cached. You get good temporal locality within a single insertion.

**Node-first ordering**: The comparisons become (root, X), (root, Y), (root, Z)
for different insertions. All comparisons against the same node share that
node's description as a prefix, and you get good sharing _across_ concurrent insertions
hitting popular nodes.

The performance difference comes down to cache size. Node-first wins if the
cache is large enough to hold the whole tree: every node's prefix stays warm,
and comparisons against it keep hitting. Item-first wins if the cache churns:
an item's own comparisons happen in quick succession as it descends, so its
prefix stays warm even if older nodes have been evicted.

## Limitations

**Tree balance.** If you insert items in already-sorted order, the tree
degenerates into a linked list with depth $O(n)$. Every insertion traverses the
whole spine, and you lose all parallelism. The fix is either to randomize
insertion order or use a self-balancing tree. The implementation here doesn't
self-balance, so shuffle your inputs.

**Comparison transitivity.** BSTs assume transitive comparisons: if A < B and
B < C, then A < C. LLM comparisons don't always satisfy this - they can be
inconsistent, especially for items that are close in rank. The tree will still
produce _some_ ordering, but it might not match what you'd get from a different
insertion order. If you need robust rankings despite inconsistent comparisons,
consider voting across multiple comparisons or using a different ranking
algorithm.

**No deletion.** We didn't cover deletion, and some design choices (like full
restart on version conflict) are conservative in ways that would matter more
if we did.

## Conclusion

We built a BST optimized for expensive async comparisons: parallel insertion
with optimistic concurrency control, and a threaded linked list for $O(1)$ access
to sorted results. The implementation lives in
[parfold](https://github.com/doubleword/parfold)[^2].

[^2]:
    It's in python! Because we're calling LLMs for each comparison, we don't
    care about cache locality or the number of instructions we're using or
    threading, &c., so python + async/await works fine.

The result is less a sorting algorithm than an index. During
construction, you're asking the LLM "is A better than B?" for hundreds of pairs,
and the tree structure encodes all those answers into something you can query
later without making further calls. The BST becomes a materialized view of the
LLM's judgment - you pay for the comparisons once, and then you can access
min, max, or the full sorted order as many times as you like, for free.
