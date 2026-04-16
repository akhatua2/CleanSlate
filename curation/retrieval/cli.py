"""Unified CLI for retrieval methods."""

import argparse

from curation.retrieval.backend_bm25 import build_bm25_index
from curation.retrieval.backend_infini_gram import build_fm_index
from curation.retrieval.schema import Cfg
from curation.retrieval.stages import export, run_all, search


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default="infini_gram_mini", choices=["infini_gram_mini", "bm25"])
    parser.add_argument("--corpus", default="10B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--index-workers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-span-chars", type=int, default=20)
    parser.add_argument("--segment-chars", type=int, default=2000)
    parser.add_argument("--segment-overlap", "--segment-overlap-chars", dest="segment_overlap", type=int, default=500)
    parser.add_argument("--min-doc-chars", type=int, default=100)
    parser.add_argument("--min-segment-chars", type=int, default=200)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--index-mem-gb", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--num-index-shards", type=int, default=1, help="Split FM-index into N shards to reduce peak disk usage during build")
    parser.add_argument("--dolma-dir", default=None)
    parser.add_argument("--curation-root", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--content-id", dest="content_ids", action="append", default=[])


def _cfg(args: argparse.Namespace) -> Cfg:
    return Cfg(
        variant=args.corpus,
        method=args.method,
        top_k=args.top_k,
        min_span_chars=args.min_span_chars,
        segment_chars=args.segment_chars,
        segment_overlap=args.segment_overlap,
        min_doc_chars=args.min_doc_chars,
        min_segment_chars=args.min_segment_chars,
        categories=args.categories,
        workers=args.workers,
        index_workers=args.index_workers,
        index_mem_gb=args.index_mem_gb,
        num_shards=args.num_shards,
        num_index_shards=args.num_index_shards,
        dolma_root=args.dolma_dir,
        curation_root=args.curation_root,
        dataset_path=args.dataset_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CleanSlate retrieval pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for name in ("build-index", "search", "export", "run"):
        subparser = subparsers.add_parser(name)
        _add_args(subparser)
        if name in ("export", "run"):
            subparser.add_argument("--push-to-hub", action="store_true")
            subparser.add_argument("--hf-repo", default=None)

    args = parser.parse_args()
    cfg = _cfg(args)
    content_ids = args.content_ids or None

    if args.cmd == "build-index":
        if cfg.method == "infini_gram_mini":
            build_fm_index(cfg)
        elif cfg.method == "bm25":
            build_bm25_index(cfg)
        else:
            raise ValueError(f"Unknown method: {cfg.method}")
        return

    if args.cmd == "search":
        search(cfg, limit=args.limit, content_ids=content_ids)
        return

    if args.cmd == "export":
        export(cfg, limit=args.limit, content_ids=content_ids, push_to_hub=args.push_to_hub, hf_repo=args.hf_repo)
        return

    if args.cmd == "run":
        run_all(cfg, limit=args.limit, content_ids=content_ids, push_to_hub=args.push_to_hub, hf_repo=args.hf_repo)
        return

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
