#!/usr/bin/env python3
"""
Generate embeddings for a BEIR dataset using Azure OpenAI and save in HybridSearch binary cache format.

Produces a .bin file compatible with HybridSearch.Benchmarks EmbeddingCache.

Usage:
    pip install openai azure-identity
    python tools/generate_embeddings.py --data-dir benchmarks/data/nfcorpus --output nfcorpus-embeddings.bin
    python tools/generate_embeddings.py --data-dir benchmarks/data/scifact --output scifact-embeddings.bin

Environment variables:
    HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT    Azure OpenAI endpoint URL (e.g. https://myresource.openai.azure.com/)
    HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT  Deployment name for the embedding model (default: text-embedding-3-small)

Authentication uses DefaultAzureCredential (Azure CLI, managed identity, etc.).

The binary format matches C# BinaryWriter conventions:
    [count:int32]
    For each entry:
        [id: 7-bit-encoded-length-prefixed UTF-8 string]
        [dim:int32]
        [floats:float32_le * dim]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path


def write_7bit_encoded_int(f, value: int) -> None:
    """Write a non-negative integer in .NET 7-bit encoded format."""
    while value >= 0x80:
        f.write(struct.pack("B", (value & 0x7F) | 0x80))
        value >>= 7
    f.write(struct.pack("B", value & 0x7F))


def write_dotnet_string(f, s: str) -> None:
    """Write a string in C# BinaryWriter.Write(string) format: 7-bit length prefix + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    write_7bit_encoded_int(f, len(encoded))
    f.write(encoded)


def load_corpus(path: Path) -> dict[str, tuple[str, str]]:
    """Load BEIR corpus.jsonl -> {id: (title, text)}."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            corpus[obj["_id"]] = (obj.get("title", ""), obj.get("text", ""))
    print(f"Loaded {len(corpus)} corpus documents", file=sys.stderr)
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Load BEIR queries.jsonl -> {id: text}."""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[obj["_id"]] = obj.get("text", "")
    print(f"Loaded {len(queries)} queries", file=sys.stderr)
    return queries


def save_embedding_cache(
    path: Path, embeddings: dict[str, list[float]]
) -> None:
    """Save embeddings in HybridSearch EmbeddingCache binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(embeddings)))
        for doc_id, vector in embeddings.items():
            write_dotnet_string(f, doc_id)
            f.write(struct.pack("<i", len(vector)))
            f.write(struct.pack(f"<{len(vector)}f", *vector))
    print(f"Saved {len(embeddings)} embeddings to {path}", file=sys.stderr)


def embed_batch(client, deployment: str, texts: list[str]) -> list[list[float]]:
    """Call Azure OpenAI embeddings API for a batch of texts."""
    response = client.embeddings.create(model=deployment, input=texts)
    # Results may not be in input order — sort by index
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def embed_all(
    client,
    deployment: str,
    ids: list[str],
    texts: list[str],
    batch_size: int,
    label: str,
) -> dict[str, list[float]]:
    """Embed a list of texts in batches with progress reporting."""
    embeddings: dict[str, list[float]] = {}
    total = len(texts)

    print(f"Encoding {total} {label}...", file=sys.stderr)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        vectors = embed_batch(client, deployment, batch_texts)

        for doc_id, vec in zip(batch_ids, vectors):
            embeddings[doc_id] = vec

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"  [{done}/{total}] {rate:.0f} items/s",
            file=sys.stderr,
        )

    t1 = time.time()
    print(f"{label.capitalize()} encoded in {t1 - t0:.1f}s", file=sys.stderr)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a BEIR dataset using Azure OpenAI (AAD auth)."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to extracted BEIR dataset directory (containing corpus.jsonl, queries.jsonl).",
    )
    parser.add_argument(
        "--output",
        default="embeddings.bin",
        help="Output binary cache path (default: embeddings.bin).",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Azure OpenAI deployment name (default: env HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT or 'text-embedding-3-small').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Texts per API call (default: 256, max 2048).",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 2048:
        print("Error: --batch-size must be between 1 and 2048.", file=sys.stderr)
        sys.exit(1)

    # Resolve Azure OpenAI config
    endpoint = os.environ.get("HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        print(
            "Error: set HYBRIDSEARCH_AZURE_OPENAI_ENDPOINT environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    deployment = args.deployment or os.environ.get(
        "HYBRIDSEARCH_AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small"
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        print(
            f"Error: expected corpus.jsonl and queries.jsonl in {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here so --help works without packages installed
    try:
        from openai import AzureOpenAI
    except ImportError:
        print(
            "Error: openai package not installed.\n"
            "  pip install openai azure-identity",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError:
        print(
            "Error: azure-identity package not installed.\n"
            "  pip install azure-identity",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create client with AAD auth
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
    )

    print(f"Azure OpenAI endpoint: {endpoint}", file=sys.stderr)
    print(f"Deployment: {deployment}", file=sys.stderr)

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)

    # Embed corpus: title + " " + text
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        f"{title} {text}".strip() for title, text in corpus.values()
    ]

    all_embeddings = embed_all(
        client, deployment, corpus_ids, corpus_texts, args.batch_size, "corpus documents"
    )

    # Embed queries
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    query_embeddings = embed_all(
        client, deployment, query_ids, query_texts, args.batch_size, "queries"
    )

    all_embeddings.update(query_embeddings)

    dim = len(next(iter(all_embeddings.values())))
    print(
        f"Total: {len(all_embeddings)} embeddings "
        f"({len(corpus_ids)} docs + {len(query_ids)} queries, {dim} dims)",
        file=sys.stderr,
    )

    save_embedding_cache(Path(args.output), all_embeddings)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
