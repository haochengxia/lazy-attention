"""LazyRoute: query-adaptive sparse decode over reusable KV blocks.

Two pieces, both behind `LAZY_SPARSE` (see `lazy/utils/variants.py`):

    descriptors.py  min/max boxes over cached keys, one per (layer, physical
                    block, KV head), written when a document's blocks are
                    filled and read by the router.
    router.py       per decode step, scores the cached pages against the
                    current query and compacts the packed block table down to
                    the pages worth reading.

Design rule R1 governs both: **sparsity is a read-time view**. Allocation,
hashing, eviction, preemption and the persistent packed block table are never
modified -- each step gathers a subset of rows into a walk table and hands that
to the unchanged decode kernel.
"""
