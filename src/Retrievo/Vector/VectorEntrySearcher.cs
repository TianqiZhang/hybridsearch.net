using System.Numerics.Tensors;
using Retrievo.Models;

namespace Retrievo.Vector;

/// <summary>
/// Shared brute-force search implementation over pre-normalized vector entries.
/// Used by both immutable and snapshot-based mutable vector retrieval paths.
/// </summary>
internal static class VectorEntrySearcher
{
    private static readonly Comparer<(string Id, float Similarity)> DescendingComparer =
        Comparer<(string Id, float Similarity)>.Create(CompareDescending);

    internal static IReadOnlyList<RankedItem> Search(
        float[] queryVector,
        int topK,
        IReadOnlyList<(string Id, float[] NormalizedEmbedding)> entries,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(queryVector);
        ArgumentNullException.ThrowIfNull(entries);

        ValidateFiniteValues(queryVector, nameof(queryVector));

        if (topK <= 0 || entries.Count == 0)
            return Array.Empty<RankedItem>();

        int dimensions = entries[0].NormalizedEmbedding.Length;
        if (queryVector.Length != dimensions)
        {
            throw new ArgumentException(
                $"Query vector dimension mismatch: expected {dimensions}, got {queryVector.Length}.",
                nameof(queryVector));
        }

        var normalizedQuery = VectorMath.Normalize(queryVector);

        ct.ThrowIfCancellationRequested();

        int k = Math.Min(topK, entries.Count);
        var heap = new (string Id, float Similarity)[k];
        int heapSize = 0;

        for (int i = 0; i < entries.Count; i++)
        {
            if ((i & 0xFF) == 0)
                ct.ThrowIfCancellationRequested();

            var (id, embedding) = entries[i];
            float similarity = VectorMath.DotProduct(normalizedQuery, embedding);
            var item = (id, similarity);

            if (heapSize < k)
            {
                heap[heapSize] = item;
                heapSize++;
                if (heapSize == k)
                    BuildMinHeap(heap, k);
            }
            else if (CompareDescending(item, heap[0]) < 0)
            {
                heap[0] = item;
                SiftDown(heap, 0, k);
            }
        }

        ct.ThrowIfCancellationRequested();

        Array.Sort(heap, 0, heapSize, DescendingComparer);

        var results = new RankedItem[heapSize];
        for (int i = 0; i < heapSize; i++)
        {
            results[i] = new RankedItem
            {
                Id = heap[i].Id,
                Score = heap[i].Similarity,
                Rank = i + 1
            };
        }

        return results;
    }

    /// <summary>
    /// Validates that a vector contains only finite values (no NaN or Infinity).
    /// Uses SIMD-accelerated dot-self to propagate NaN/Infinity in O(1) checks.
    /// </summary>
    internal static void ValidateFiniteValues(float[] vector, string paramName)
    {
        float dotSelf = TensorPrimitives.Dot(vector, vector);
        if (!float.IsFinite(dotSelf))
            throw new ArgumentException("Vector contains non-finite values (NaN or Infinity).", paramName);
    }

    private static int CompareDescending((string Id, float Similarity) a, (string Id, float Similarity) b)
    {
        int cmp = b.Similarity.CompareTo(a.Similarity);
        return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
    }

    private static int CompareAscending((string Id, float Similarity) a, (string Id, float Similarity) b)
    {
        int cmp = a.Similarity.CompareTo(b.Similarity);
        return cmp != 0 ? cmp : string.Compare(b.Id, a.Id, StringComparison.Ordinal);
    }

    private static void BuildMinHeap((string Id, float Similarity)[] heap, int size)
    {
        for (int i = size / 2 - 1; i >= 0; i--)
            SiftDown(heap, i, size);
    }

    private static void SiftDown((string Id, float Similarity)[] heap, int i, int size)
    {
        while (true)
        {
            int left = (2 * i) + 1;
            int right = (2 * i) + 2;
            int smallest = i;

            if (left < size && CompareAscending(heap[left], heap[smallest]) < 0)
                smallest = left;
            if (right < size && CompareAscending(heap[right], heap[smallest]) < 0)
                smallest = right;

            if (smallest == i)
                break;

            (heap[i], heap[smallest]) = (heap[smallest], heap[i]);
            i = smallest;
        }
    }
}
