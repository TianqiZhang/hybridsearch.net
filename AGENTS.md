# AGENTS.md — Retrievo

> Knowledge base for AI agents. Read this before making changes. See README.md for project overview.

---

## Build & Test Commands

Requires [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).

```shell
dotnet build                                          # Build entire solution
dotnet test                                           # Run all 238 tests (unit + integration)
dotnet test tests/Retrievo.Tests                      # Unit tests only
dotnet test tests/Retrievo.IntegrationTests           # Integration tests only (CLI subprocess tests)
dotnet test --filter "FullyQualifiedName~ClassName.MethodName"  # Run a single test
```

**CI settings** (`Directory.Build.props`): `TreatWarningsAsErrors=true`, `Nullable=enable`, `ImplicitUsings=enable`, `LangVersion=latest`. Zero warnings required.

---

## Project Structure

```
Retrievo.slnx
├── src/
│   ├── Retrievo/                    # Core library (NuGet: Retrievo)
│   ├── Retrievo.AzureOpenAI/       # Azure OpenAI embedding provider (NuGet: Retrievo.AzureOpenAI)
│   └── Retrievo.Cli/               # CLI tool (System.CommandLine)
├── tests/
│   ├── Retrievo.Tests/             # Unit tests (xUnit + NSubstitute)
│   └── Retrievo.IntegrationTests/  # CLI subprocess integration tests
├── benchmarks/Retrievo.Benchmarks/ # Performance benchmarks
└── Directory.Build.props           # Shared build settings, version, NuGet metadata
```

Namespaces map to folders.

---

## Workflow Rules

1. **Update README on every change** — features, tests, structure, or roadmap changes must be reflected in `README.md`.
2. **Code review before every commit** — ask a specialized code review expert agent to review all pending changes.

---

## Coding Conventions

### Types & Properties

| Pattern | When | Example |
|---------|------|---------|
| `public sealed record` | DTOs, value objects | `HybridQuery`, `SearchResult`, `SearchResponse` |
| `public sealed class` | Orchestrators, mutable state | `HybridSearchIndex`, `Document` |
| `internal` constructors | Force builder usage | `internal HybridSearchIndex(...)` |
| `public` builders | Fluent API | `HybridSearchIndexBuilder` |

- `init`-only properties: `public string Name { get; init; }`
- `required` for mandatory: `public required string Id { get; init; }`
- `T?` for optional: `public string? Title { get; init; }`
- Collections: `IReadOnlyList<T>`, `IReadOnlyDictionary<K,V>`
- Embeddings: `float[]?`

### Methods

- Provide sync + async: `Search()` + `SearchAsync()`
- Async-first for I/O (embedding providers)
- Propagate `CancellationToken` in async methods
- Helpers are `private` or `internal`

### Formatting & Imports

- File-scoped namespaces: `namespace Retrievo;`
- File-scoped `using` statements
- Alias long types: `using LuceneDocument = Lucene.Net.Documents.Document;`
- No `.editorconfig` — follow existing code style

### XML Documentation

- **Required** on all public types, interfaces, and public methods
- `/// <inheritdoc/>` on implementations
- Include `<summary>`, `<param>`, `<returns>` where useful

### Test Conventions

- **Framework**: xUnit 2.9 + NSubstitute 5.3
- **Global using**: `<Using Include="Xunit" />` in csproj (no explicit `using Xunit;` needed)
- **Naming**: `MethodOrFeature_Scenario_ExpectedResult` — e.g., `ContainsFilter_NoMatch_ReturnsEmpty`
- **No traits** — tests are organized by folder/class, not `[Trait]`
- **Test data**: `SyntheticCorpusGenerator` in `TestData/` for generating documents
- **`[Fact]`** for single cases, **`[Theory]`** for parameterized
- **Internals access**: `InternalsVisibleTo("Retrievo.Tests")` on core project

---

## Validation & Error Handling

### Guard Clauses

```csharp
ArgumentNullException.ThrowIfNull(query);
ObjectDisposedException.ThrowIf(_disposed, this);
if (float.IsNaN(value) || float.IsInfinity(value) || value < 0)
    throw new ArgumentOutOfRangeException(nameof(value), "...");
```

### Rules

- **Narrow catches only**: `catch (ParseException)` — never `catch (Exception) {}`
- **No logging in library** — exceptions and return values are the API
- CLI uses `Console.Error.WriteLine` / `Console.WriteLine` for user output

---

## Resource Management

- `_disposed` flag guarded by `ObjectDisposedException.ThrowIf`
- Dispose all owned resources (Lucene readers, writers, directories, analyzers)
- Mutable index: `AcquireSearcherSnapshot()` → `reader.IncRef()`, caller must `DecRef()`
- Snapshot swap on `Commit()` — old released, new acquired atomically

---

## Key Patterns

- **Builder pattern** — fluent API → `internal` constructor
- **Snapshot isolation** — mutable index swaps immutable snapshots atomically
- **Deterministic tie-breaking**: `OrderByDescending(score).ThenBy(id, StringComparer.Ordinal)`
- **Optional deps** via nullable: `IEmbeddingProvider?` with `if (provider is not null)`
- **SIMD-first** vector math with scalar fallback in `VectorMath`

## Anti-Patterns to Avoid

| Don't | Why |
|-------|-----|
| `.GetAwaiter().GetResult()` | Throws `InvalidOperationException` directing to async API instead |
| `ex.Message.Contains("...")` | Brittle — catch specific exception types |
| `#pragma warning disable` / type suppression | `TreatWarningsAsErrors` is on; keep it clean |
| Empty catch blocks | Never swallow exceptions |
| `ILogger` in core library | Not used — exceptions are the API |
