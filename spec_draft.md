# HybridSearch.NET — Phased Spec

## 0. Purpose
HybridSearch.NET is an open-source, in-process, **in-memory hybrid retrieval** library for **small, local corpora** where running a full search backend is too heavy.

Typical usage:
- **Local agent memory** (file-based notes, digests, conversation summaries)
- **Small RAG** over private docs (team SOPs, specs, runbooks)
- Developer tools indexing local **Markdown/text**
- Offline / edge scenarios with strict locality requirements

Core retrieval model:
- **Lexical**: BM25 / keyword search
- **Vector**: cosine/dot similarity over embeddings
- **Fusion**: Reciprocal Rank Fusion (RRF)

Non-goals: large-scale search (>10k–50k docs), distributed indexing, semantic re-ranking (cross-encoders).

---

## 1. Functional requirements

### 1.1 Library API (baseline)
- Provide a .NET library usable in-process.
- Support search over a corpus of documents:
  - `Text` query only (lexical)
  - `Vector` query only (vector)
  - `Text + Vector` (hybrid with fusion)
- Return top-K results.
- Provide **explainability** option:
  - lexical rank (if present)
  - vector rank (if present)
  - fused score breakdown (RRF contributions)
- Provide basic operational metadata:
  - corpus size, embedding dimension, index build time, query timing breakdown.

### 1.2 Document model
- Each document MUST have:
  - stable `Id` (string)
  - one or more text fields (MVP can default to a single `body` field)
- Each document MAY have:
  - `title`
  - `metadata` (key-value)
  - a single embedding vector (MVP)

### 1.3 Corpus ingestion
- MVP MUST support building an index from:
  - `IEnumerable<Document>`
  - a folder of local files (at least `*.md`, `*.txt`)
- Folder ingestion defaults:
  - filename → `title`
  - file contents → `body`
  - metadata includes at least `sourcePath` and `lastModified`

### 1.4 Index update model
HybridSearch.NET MUST support at least one of these usage models:

**Model A — Batch build (MVP):**
- Build index from a set of docs.
- Query is read-only.
- Updates are applied via full rebuild.

**Model B — Incremental (Phase 2+):**
- Support `Upsert(id)` and `Delete(id)`.
- Define a clear **visibility boundary**:
  - updates become visible after `Commit()` (preferred for determinism), OR
  - auto-refresh with `Flush()` for tests.

### 1.5 CLI tool
- Provide a CLI for common local-doc workflows.
- MVP CLI commands:
  - `index <folder>` (build in memory for interactive querying)
  - `query <folder> --text "..."` (build then query)
  - `query <folder> --text "..." --explain`
- Optional MVP CLI mode:
  - `--watch` to rebuild on file changes (debounced).

### 1.6 Fielded search (optional for MVP)
- Phase 2 MUST support basic fielded search:
  - at least `title` and `body`
  - field boosts (e.g., `titleBoost`, `bodyBoost`)

### 1.7 Language support policy
- Default experience targets English (reasonable tokenizer, lowercase, etc.).
- MUST NOT hardcode “English-only.”
- Text analysis MUST be pluggable so future multi-language analyzers can be added.

---

## 2. Phased roadmap (functional)

### Phase 0 — Foundation
- Core models, interfaces, orchestration.
- Minimal sample app wiring.

**Acceptance**
- Build passes; unit tests compile and run.

### Phase 1 — MVP Hybrid Retrieval (1k–5k docs)
- Batch build + query-only index.
- Folder ingestion for Markdown/text.
- Lexical retrieval (BM25).
- Vector retrieval (brute-force cosine/dot).
- RRF fusion + weights.
- Explain mode.
- CLI demo.

**Acceptance**
- Correctness tests for lexical-only, vector-only, hybrid.
- Deterministic fused ranking for fixed inputs.
- Interactive latency on typical dev machine for ~3k docs.

### Phase 2 — Incremental Updates + Fielded Search (up to ~10k)
- Optional mutable index with `Upsert/Delete`.
- Explicit `Commit()` visibility boundary.
- `title/body` fields with boosts.
- Basic metadata filters (exact match).
- Diagnostics: query time breakdown.

**Acceptance**
- Consistent read snapshots during updates.
- Integration tests proving visibility boundary.
- Usable performance at ~10k docs.

### Phase 3 — Optional Snapshot + Chunking
- Optional snapshot export/import (avoid re-embedding).
- Optional chunking strategy for long Markdown.
- Parent-child mapping (chunk → document).

**Acceptance**
- Reload works without embedding recomputation.
- Chunking improves retrieval on long docs.

### Phase 4 — Optional ANN
- Pluggable vector index interface.
- Brute-force remains default.
- Optional ANN implementation behind a flag.

---

## 3. Implementation details (design + engineering)

### 3.1 Proposed architecture
- Separate concerns into components:
  - `ILexicalRetriever`
  - `IVectorRetriever`
  - `IFuser`
  - `ITextAnalyzer`
  - optional `IEmbeddingProvider`

### 3.2 Lexical retrieval implementation
- Recommended: **Lucene.NET** with in-memory storage.
- Use BM25 scoring.
- Analyzer selection behind `ITextAnalyzer`.

### 3.3 Vector retrieval implementation (MVP)
- Store embeddings in memory as `float[]`.
- Pre-normalize vectors for cosine similarity.
- Brute-force scan for top-K:
  - SIMD via `System.Numerics.Vector<float>`
  - optional parallelization (toggle; avoid default thread overhead for small N).

### 3.4 Fusion implementation
- RRF (Reciprocal Rank Fusion):
  - for each list rank `r`, add `weight * 1/(rrfK + r)`
  - sum contributions across lists
  - sort by fused score
- Deterministic tie-breakers:
  - stable ordering by `Id` when scores match.

### 3.5 Incremental update strategy (Phase 2)
**Goal:** avoid partial visibility and keep tests deterministic.

Preferred design: **read snapshots + commit boundary**.

- Maintain an immutable `IndexSnapshot` for readers:
  - lexical searcher
  - embedding store snapshot
  - metadata snapshot
  - version
- Queries read the current snapshot atomically.
- Writers apply updates and then `Commit()` swaps in a new snapshot.

Why this matters:
- Readers never lock.
- No mixed-state results.
- Tests can assert “not visible until commit.”

### 3.6 CLI implementation
- Use a folder reader with basic frontmatter-aware parsing later (optional).
- Implement `--watch` via debounced file system events.
- For CLI testability:
  - prefer a file system abstraction (e.g., `System.IO.Abstractions`).

### 3.7 Testing strategy
- Unit tests:
  - RRF correctness
  - vector similarity correctness
  - analyzer/tokenizer behavior via injectable analyzer
- Integration tests:
  - folder ingestion → build → query
  - incremental update visibility boundary (Phase 2)
- Perf sanity (non-gating):
  - synthetic corpora at 1k/5k/10k
  - record timing breakdown

---

## 4. Public API sketch (non-final)

```csharp
public interface IHybridSearchIndex : IDisposable
{
    SearchResponse Search(HybridQuery query);
    IndexStats GetStats();
}

public interface IMutableHybridSearchIndex : IHybridSearchIndex
{
    void Upsert(Document doc);
    bool Delete(string id);

    // Defines update visibility boundary
    void Commit();
}

public sealed record HybridQuery(
    string? Text,
    float[]? Vector,
    int TopK = 10,
    int LexicalK = 50,
    int VectorK = 50,
    float LexicalWeight = 1f,
    float VectorWeight = 1f,
    int RrfK = 60,
    bool Explain = false);
```

---

## 5. Acceptance test checklist (by phase)

> The goal of this section is to make it easy to validate correctness and avoid regressions. These are phrased as *tests you can actually implement* (unit/integration), with clear pass/fail conditions.

### 5.1 Phase 0 — Foundation
**Unit tests**
- [ ] **Model immutability / equality:** `Document`, `HybridQuery`, and result models behave as expected (value equality where intended).
- [ ] **Deterministic ordering contract:** when fused scores tie, results are ordered by `Id` (or another explicitly documented stable tie-breaker).

**Integration tests**
- [ ] **Engine wiring smoke test:** create index from 3 documents and execute a query without exceptions.

---

### 5.2 Phase 1 — MVP Hybrid Retrieval
**Unit tests**
- [ ] **RRF correctness (single list):** given a ranked list `[A,B,C]`, fused scores follow `1/(k+r)` with correct ranks.
- [ ] **RRF correctness (two lists):** given lexical `[A,B,C]` and vector `[B,A,D]`, fused ranking matches expected ordering for a fixed `RrfK`, including weight handling.
- [ ] **Weighting behavior:** increasing `VectorWeight` shifts fused ordering in expected direction (e.g., doc that appears only in vector list rises).
- [ ] **Vector similarity:** cosine similarity returns 1.0 for identical vectors, ~0 for orthogonal vectors (within tolerance).
- [ ] **TopK truncation:** results length equals `TopK` (or `<=TopK` if corpus smaller).
- [ ] **Explain payload:** when `Explain=true`, each result includes lexical rank or null, vector rank or null, and fused contribution totals.

**Integration tests (folder ingestion)**
- [ ] **Index from folder:** building from a temp folder with `a.md`, `b.txt` produces 2 docs with expected `title/body` mapping.
- [ ] **Markdown ingestion sanity:** markdown file content is searchable (at least plain text extraction; no requirement for AST parsing).
- [ ] **Lexical-only query:** a keyword query retrieves the document containing that keyword in body.
- [ ] **Vector-only query:** a vector query retrieves the nearest document by cosine similarity using fixed test vectors.
- [ ] **Hybrid query:** with both text+vector, fused list contains union of candidates and returns expected top result.

**CLI tests**
- [ ] **CLI query smoke:** `hybridsearch query <folder> --text "..."` returns non-empty output and exit code 0.
- [ ] **CLI explain:** `--explain` prints lexical/vector ranks for returned docs.

**Perf sanity (non-gating, but tracked)**
- [ ] **3k docs scan:** vector brute-force for 3k docs (e.g., 768 dims) completes within a reasonable interactive budget on CI machine; record timing (do not hardcode strict SLA).

---

### 5.3 Phase 2 — Incremental Updates + Fielded Search
**Unit tests**
- [ ] **Visibility boundary (commit model):** after `Upsert()` but before `Commit()`, searches do **not** reflect the change.
- [ ] **Commit applies:** after `Commit()`, searches reflect the upserted/deleted doc.
- [ ] **Delete semantics:** after delete+commit, doc never appears even if it would match text/vector.
- [ ] **Field boosts:** when the query term appears in `title` only, increasing `titleBoost` increases rank relative to body-only matches.
- [ ] **Metadata filter include/exclude:** exact-match filter returns only matching docs.

**Integration tests**
- [ ] **Snapshot consistency:** during a writer update cycle, concurrent queries never throw and never return mixed-state artifacts (e.g., doc present in lexical results but missing from metadata store).
- [ ] **Incremental folder watcher (optional):** modifying a file triggers an upsert + commit and changes appear after the debounced cycle.

**Perf sanity**
- [ ] **10k docs scan:** vector brute-force query over ~10k docs remains usable; record timing and memory.

---

### 5.4 Phase 3 — Snapshot + Chunking (optional)
**Integration tests**
- [ ] **Snapshot round-trip:** build → export snapshot → import → query results match (within deterministic tie-breaker) for a fixed corpus.
- [ ] **No re-embed on import:** import does not call `IEmbeddingProvider` (assert via mock).
- [ ] **Chunk retrieval:** when chunking enabled, a match in the middle of a long doc returns the correct chunk and maps back to parent doc.

---

### 5.5 Phase 4 — ANN (optional)
**Correctness tests**
- [ ] **ANN parity check:** for a fixed small corpus, ANN top-K overlaps brute-force top-K above a configured threshold (e.g., recall@K).
- [ ] **Fallback:** brute-force remains functional and selectable.

---

## 6. Open questions
- Default analyzer selection and configuration.
- Snapshot format for Phase 3 (if implemented).
- ANN library selection (Phase 4).
- Chunking strategy (Phase 3): headings vs fixed-size tokens.

