# BEIR Benchmarks

Evaluation of Retrievo retrieval quality against [BEIR (Benchmarking Information Retrieval)](https://github.com/beir-cellar/beir) datasets.

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| `nfcorpus` (default) | Biomedical IR — 3,633 PubMed articles, 323 test queries, graded relevance (0/1/2) |
| `scifact` | Scientific claim verification — 5,183 abstracts, 300 test queries, binary relevance |
| `fiqa` | Financial opinion QA — 57,638 documents, 648 test queries |
| `trec-covid` | COVID-19 biomedical literature — 171,332 documents, 50 test queries |
| `hotpotqa` | Multi-hop QA — 5,233,329 documents, 7,405 test queries |
| `arguana` | Argument retrieval — 8,674 documents, 1,406 test queries |
| `quora` | Duplicate question retrieval — 522,931 documents, 10,000 test queries |

Any BEIR dataset ID can be used — the tool downloads from the BEIR CDN automatically.

## Running Benchmarks

```bash
# Run NFCorpus benchmark (default, auto-downloads on first run)
dotnet run --project benchmarks/Retrievo.Benchmarks

# Run SciFact benchmark
dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset scifact

# Run with precomputed embeddings (enables vector-only and hybrid modes)
dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset scifact --embeddings scifact-embeddings.bin

# List all available datasets
dotnet run --project benchmarks/Retrievo.Benchmarks -- --list-datasets
```

### CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--dataset` | `-d` | `nfcorpus` | BEIR dataset to evaluate |
| `--data-dir` | | `./benchmarks/data` | Base directory for dataset storage |
| `--embeddings` | | (none) | Path to pre-computed embeddings binary cache |
| `--list-datasets` | | | Show available datasets and exit |
| `--sweep` | | | Run parameter sweep over fusion weights, RRF k, and title boost |
| `--help` | `-h` | | Show usage and exit |

## Generating Embeddings

Without embeddings, the benchmark runs **lexical-only** (BM25). To enable vector-only and hybrid modes, pre-compute embeddings with the included Python script:

```bash
pip install openai azure-identity

# Generate embeddings for NFCorpus
python tools/generate_embeddings.py --data-dir benchmarks/data/nfcorpus --output nfcorpus-embeddings.bin

# Generate embeddings for SciFact
python tools/generate_embeddings.py --data-dir benchmarks/data/scifact --output scifact-embeddings.bin
```

Set `RETRIEVO_AZURE_OPENAI_ENDPOINT` to your Azure OpenAI endpoint. Authentication uses `DefaultAzureCredential` (Azure CLI, managed identity, etc.). Optionally set `RETRIEVO_AZURE_OPENAI_DEPLOYMENT` (defaults to `text-embedding-3-small`).

The script encodes all documents and queries using the specified model and saves them in the binary cache format. Then re-run the benchmark with `--embeddings <path>`.

## Results

All embeddings use `text-embedding-3-small` (1536 dims) via Azure OpenAI.

### NFCorpus

Biomedical IR — 3,633 PubMed articles, 323 test queries, graded relevance (0/1/2).

| Configuration | nDCG@10 | MAP@10 | Recall@100 | Avg Query |
|---------------|---------|--------|------------|-----------|
| Lexical-only (BM25) | 0.330 | 0.242 | 0.247 | 2.3ms |
| Vector-only | 0.384 | 0.291 | 0.360 | 1.8ms |
| Hybrid (default weights) | 0.392 | 0.293 | 0.369 | 2.2ms |
| **Hybrid (tuned)** | **0.392** | 0.295 | 0.362 | 2.2ms |
| BEIR BM25 baseline (Anserini) | 0.325 | — | — | — |

Best NFCorpus config: `LexicalWeight=0.3, VectorWeight=0.5, RrfK=1, TitleBoost=0.5`

### SciFact

Scientific claim verification — 5,183 abstracts, 300 test queries, binary relevance.

| Configuration | nDCG@10 | MAP@10 | Recall@100 | Avg Query |
|---------------|---------|--------|------------|-----------|
| Lexical-only (BM25) | 0.685 | 0.639 | 0.922 | 3.2ms |
| Lexical-only (BM25, TitleBoost=0.5) | 0.685 | 0.639 | 0.922 | 3.2ms |
| Vector-only | 0.731 | 0.687 | 0.973 | 2.7ms |
| Hybrid (default weights) | 0.756 | 0.710 | 0.987 | 3.3ms |
| **Hybrid (tuned)** | **0.757** | 0.709 | 0.983 | 3.3ms |
| BEIR BM25 baseline (Anserini MF) | 0.679 | — | — | — |

Best SciFact config: `LexicalWeight=1.0, VectorWeight=1.5, RrfK=20, TitleBoost=0.5`

### Parameter Sweep

The `--sweep` flag runs a grid search over 245 configurations per dataset:

| Parameter | Sweep Values |
|-----------|-------------|
| LexicalWeight | 0, 0.1, 0.3, 0.5, 1.0, 1.5 |
| VectorWeight | 0, 0.1, 0.3, 0.5, 1.0, 1.5 |
| RrfK | 1, 20, 60 |
| TitleBoost | 0.5, 1.0, 2.0 |

```bash
# Hybrid sweep (requires embeddings)
dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset nfcorpus --embeddings embeddings.bin --sweep

# Lexical-only sweep (no embeddings needed)
dotnet run --project benchmarks/Retrievo.Benchmarks -- --dataset scifact --sweep
```

### Analysis

#### Cross-Dataset Findings

1. **TitleBoost=0.5 is universally better** — Top configs on both datasets use TitleBoost=0.5. On SciFact lexical-only, lowering TitleBoost from 1.0 to 0.5 improves nDCG@10 by +0.020 (0.665 → 0.685).

2. **RrfK=60 (paper default) is suboptimal** — NFCorpus prefers k=1 (sharp top-rank emphasis), SciFact prefers k=20. The original RRF paper's k=60 is consistently outperformed.

3. **Vector retrieval dominates** — Both datasets benefit from higher vector weight relative to lexical weight. Pure equal weights (L=1, V=1) underperform tuned ratios.

4. **Top configs are tightly clustered** — The top ~20 configs on each dataset are within 0.002-0.003 nDCG@10 of the best, indicating robustness to exact parameter choices within the optimal region.

#### Balanced Default Config

Cross-dataset harmonic mean optimization across both sweep datasets identified `LexicalWeight=0.5, VectorWeight=1.0, RrfK=20, TitleBoost=0.5` as the best universal default:

| Config | NFCorpus nDCG@10 | SciFact nDCG@10 | Harmonic Mean |
|--------|-----------------|-----------------|---------------|
| **L=0.5 V=1.0 k=20 tb=0.5** | **0.392** | **0.756** | **0.516** |
| L=0.3 V=0.5 k=1 tb=0.5 | 0.392 | 0.754 | 0.516 |
| L=0.3 V=1.0 k=20 tb=0.5 | 0.391 | 0.757 | 0.516 |
| L=0.5 V=1.5 k=20 tb=0.5 | 0.391 | 0.756 | 0.515 |
| L=0.1 V=0.3 k=20 tb=0.5 | 0.391 | 0.756 | 0.515 |

The top 5 candidates are within 0.001 harmonic mean of each other. The chosen default uses clean, memorable values with a 2:1 vector-to-lexical weight ratio.

#### BM25 Implementation

Our BM25 implementation matches the reference Anserini/Lucene baseline (0.325 nDCG@10 on NFCorpus), achieved by:

1. **Tuning BM25 parameters** — k1=0.9, b=0.4 (matching Anserini defaults for BEIR)
2. **English stemming** — Porter stemmer with English possessive filter and stop words
3. **Bag-of-words query construction** — analyzer-based term extraction instead of query parser escaping

## How It Works

The benchmark parses standard BEIR files (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), builds a Retrievo index from the corpus, runs all test queries, and computes:

- **nDCG@10** — normalized discounted cumulative gain at rank 10
- **MAP@10** — mean average precision at rank 10
- **Recall@100** — fraction of relevant documents retrieved in top 100

Each dataset is evaluated in up to three modes: lexical-only (BM25), vector-only (cosine similarity), and hybrid (RRF fusion). Vector and hybrid modes require pre-computed embeddings.
