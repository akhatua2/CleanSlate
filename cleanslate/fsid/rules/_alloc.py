"""Shared per-doc quota allocator for retrieval rules (bm25, igm)."""


def allocate_per_doc_quota(
    available: list[int], cap: int, per_doc_floor: int,
) -> list[int]:
    """Upfront per-doc window quota for one song.

    Docs are listed in score-desc order. Phase 1 gives each doc a floor of
    `per_doc_floor` windows (bounded by its availability) so every hit gets
    represented. Phase 2 distributes any remaining budget round-robin in score
    order so top docs take the overflow first. Quota is always bounded by
    availability and sums to `min(cap, sum(available))`.
    """
    n = len(available)
    if n == 0 or cap <= 0:
        return [0] * n
    quota = [0] * n
    remaining = min(cap, sum(available))
    for i in range(n):
        if remaining == 0:
            break
        take = min(per_doc_floor, available[i], remaining)
        quota[i] = take
        remaining -= take
    while remaining > 0:
        progressed = False
        for i in range(n):
            if remaining == 0:
                break
            if quota[i] < available[i]:
                quota[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quota
