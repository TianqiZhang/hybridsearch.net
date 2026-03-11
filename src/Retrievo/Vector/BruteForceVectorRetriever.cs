using Retrievo.Abstractions;
using Retrievo.Models;

namespace Retrievo.Vector;

/// <summary>
/// Brute-force vector retrieval using cosine similarity.
/// Stores all embeddings in memory as pre-normalized float arrays.
/// Uses SIMD-accelerated dot product for the similarity scan.
/// </summary>
public sealed class BruteForceVectorRetriever : IVectorRetriever
{
    private readonly List<(string Id, float[] NormalizedEmbedding)> _entries = new();

    /// <summary>
    /// Get a snapshot copy of all entries for use in snapshot-based search.
    /// </summary>
    internal IReadOnlyList<(string Id, float[] NormalizedEmbedding)> GetSnapshotEntries()
    {
        return _entries.ToList();
    }

    /// <summary>
    /// The expected embedding dimension, or 0 if no embeddings have been added yet.
    /// </summary>
    public int Dimensions { get; private set; }

    /// <summary>
    /// Number of indexed embeddings.
    /// </summary>
    public int Count => _entries.Count;

    /// <summary>
    /// Add a document embedding to the index. The embedding is normalized on insert.
    /// </summary>
    public void Add(string id, float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(embedding);

        if (embedding.Length == 0)
            throw new ArgumentException("Embedding must not be empty.", nameof(embedding));

        VectorEntrySearcher.ValidateFiniteValues(embedding, nameof(embedding));

        if (Dimensions == 0)
            Dimensions = embedding.Length;
        else if (embedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {embedding.Length}.", nameof(embedding));

        var normalized = VectorMath.Normalize(embedding);
        _entries.Add((id, normalized));
    }

    /// <summary>
    /// Add an already-normalized embedding to the index without re-normalizing it.
    /// Used when rebuilding an exact snapshot of the live vector state.
    /// </summary>
    internal void AddNormalized(string id, float[] normalizedEmbedding)
    {
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(normalizedEmbedding);

        if (normalizedEmbedding.Length == 0)
            throw new ArgumentException("Embedding must not be empty.", nameof(normalizedEmbedding));

        VectorEntrySearcher.ValidateFiniteValues(normalizedEmbedding, nameof(normalizedEmbedding));

        if (Dimensions == 0)
            Dimensions = normalizedEmbedding.Length;
        else if (normalizedEmbedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {normalizedEmbedding.Length}.",
                nameof(normalizedEmbedding));

        _entries.Add((id, normalizedEmbedding.ToArray()));
    }

    /// <summary>
    /// Update an existing embedding or add it if not found.
    /// The embedding is normalized on insert.
    /// </summary>
    public void Update(string id, float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(embedding);

        if (embedding.Length == 0)
            throw new ArgumentException("Embedding must not be empty.", nameof(embedding));

        VectorEntrySearcher.ValidateFiniteValues(embedding, nameof(embedding));

        if (Dimensions == 0)
            Dimensions = embedding.Length;
        else if (embedding.Length != Dimensions)
            throw new ArgumentException(
                $"Embedding dimension mismatch: expected {Dimensions}, got {embedding.Length}.", nameof(embedding));

        var normalized = VectorMath.Normalize(embedding);

        for (int i = 0; i < _entries.Count; i++)
        {
            if (string.Equals(_entries[i].Id, id, StringComparison.Ordinal))
            {
                _entries[i] = (id, normalized);
                return;
            }
        }

        _entries.Add((id, normalized));
    }

    /// <summary>
    /// Remove a document embedding by its ID.
    /// </summary>
    /// <returns>True if the document was found and removed; false otherwise.</returns>
    public bool Remove(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        for (int i = 0; i < _entries.Count; i++)
        {
            if (string.Equals(_entries[i].Id, id, StringComparison.Ordinal))
            {
                _entries.RemoveAt(i);
                if (_entries.Count == 0)
                    Dimensions = 0;
                return true;
            }
        }

        return false;
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(float[] vector, int topK)
    {
        return Search(vector, topK, CancellationToken.None);
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(float[] vector, int topK, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(vector);

        return VectorEntrySearcher.Search(vector, topK, _entries, ct);
    }
}
