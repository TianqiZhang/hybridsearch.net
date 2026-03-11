using BenchmarkDotNet.Attributes;
using Retrievo.Models;
using Retrievo.Vector;

namespace Retrievo.PerfBenchmarks;

/// <summary>
/// Benchmarks the mutable snapshot vector search path after sharing the
/// bounded top-K selector with the immutable retriever.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class MutableVectorSearchBenchmarks
{
    private List<(string Id, float[] NormalizedEmbedding)> _entries = null!;
    private float[] _queryVector = null!;

    [Params(1000, 5000)]
    public int DocumentCount { get; set; }

    [Params(384, 1536)]
    public int Dimensions { get; set; }

    [Params(10, 50)]
    public int TopK { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _entries = new List<(string Id, float[] NormalizedEmbedding)>(DocumentCount);

        for (int i = 0; i < DocumentCount; i++)
        {
            var embedding = new float[Dimensions];
            for (int j = 0; j < embedding.Length; j++)
                embedding[j] = (float)(rng.NextDouble() * 2.0 - 1.0);

            _entries.Add(($"doc-{i:D6}", VectorMath.Normalize(embedding)));
        }

        _queryVector = new float[Dimensions];
        for (int i = 0; i < _queryVector.Length; i++)
            _queryVector[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    /// <summary>
    /// Legacy mutable path: normalize, score every entry, sort the whole corpus, then truncate.
    /// </summary>
    [Benchmark(Baseline = true)]
    public IReadOnlyList<RankedItem> LegacyFullSort()
    {
        var normalizedQuery = VectorMath.Normalize(_queryVector);
        var scored = new (string Id, float Similarity)[_entries.Count];

        for (int i = 0; i < _entries.Count; i++)
        {
            var (id, embedding) = _entries[i];
            scored[i] = (id, VectorMath.DotProduct(normalizedQuery, embedding));
        }

        Array.Sort(scored, CompareDescending);

        int resultCount = Math.Min(TopK, scored.Length);
        var results = new RankedItem[resultCount];
        for (int i = 0; i < resultCount; i++)
        {
            results[i] = new RankedItem
            {
                Id = scored[i].Id,
                Score = scored[i].Similarity,
                Rank = i + 1
            };
        }

        return results;
    }

    /// <summary>
    /// Current shared path: bounded min-heap selection with deterministic final ordering.
    /// </summary>
    [Benchmark]
    public IReadOnlyList<RankedItem> SharedTopKHeap()
    {
        return VectorEntrySearcher.Search(_queryVector, TopK, _entries, CancellationToken.None);
    }

    private static int CompareDescending((string Id, float Similarity) a, (string Id, float Similarity) b)
    {
        int cmp = b.Similarity.CompareTo(a.Similarity);
        return cmp != 0 ? cmp : string.Compare(a.Id, b.Id, StringComparison.Ordinal);
    }
}
