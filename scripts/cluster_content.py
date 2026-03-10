#!/usr/bin/env python3
"""Fast GPU-based content clustering for cleanslate dataset."""

import argparse
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=50, help="Number of clusters")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model")
    parser.add_argument("--output", default="cluster_assignments.json")
    args = parser.parse_args()

    print(f"Loading dataset...")
    ds = load_dataset("unlearning-cleanslate/cleanslate_dataset", "default")["train"]
    texts = ds["reference_target"]
    ids = ds["content_id"]
    titles = ds["content_title"]
    print(f"  {len(texts)} items")

    print(f"\nEmbedding with {args.model} (GPU)...")
    model = SentenceTransformer(args.model, device="cuda")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"  Embedding shape: {embeddings.shape}")

    print(f"\nClustering into {args.k} clusters...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Build cluster info
    clusters = {}
    for i, (cid, title, label) in enumerate(zip(ids, titles, labels)):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({"id": cid, "title": title})

    # Summary
    print(f"\nCluster sizes:")
    sizes = sorted([(k, len(v)) for k, v in clusters.items()], key=lambda x: -x[1])
    for cluster_id, size in sizes[:10]:
        sample_titles = [x["title"] for x in clusters[cluster_id][:3]]
        print(f"  Cluster {cluster_id}: {size} items - {sample_titles}")
    print(f"  ... ({len(sizes)} total clusters)")

    # Save
    output = {
        "k": args.k,
        "model": args.model,
        "num_items": len(texts),
        "clusters": clusters,
        "labels": [int(l) for l in labels],
        "content_ids": list(ids),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
