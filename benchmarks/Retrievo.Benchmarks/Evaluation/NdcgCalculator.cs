namespace Retrievo.Benchmarks.Evaluation;

/// <summary>
/// Computes standard information retrieval metrics: nDCG@K, MAP@K, Recall@K.
/// Follows the same methodology as BEIR (pytrec_eval).
/// </summary>
public static class NdcgCalculator
{
    /// <summary>
    /// Compute nDCG@K for a single query.
    /// Uses raw relevance as gain (not 2^rel-1), matching pytrec_eval ndcg_cut behavior.
    /// </summary>
    /// <param name="rankedDocIds">Ranked document IDs in descending retrieval order.</param>
    /// <param name="qrels">Query relevance judgments by document ID.</param>
    /// <param name="k">Rank cutoff.</param>
    /// <returns>nDCG at k in range [0, 1].</returns>
    public static double ComputeNdcg(IReadOnlyList<string> rankedDocIds, IReadOnlyDictionary<string, int> qrels, int k)
    {
        ArgumentNullException.ThrowIfNull(rankedDocIds);
        ArgumentNullException.ThrowIfNull(qrels);
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be greater than zero.");

        var dcg = ComputeDcg(rankedDocIds, qrels, k);
        var idcg = ComputeIdealDcg(qrels, k);

        return idcg == 0d ? 0d : dcg / idcg;
    }

    /// <summary>
    /// Compute Average Precision at K for a single query (binary: rel &gt; 0 is relevant).
    /// </summary>
    /// <param name="rankedDocIds">Ranked document IDs in descending retrieval order.</param>
    /// <param name="qrels">Query relevance judgments by document ID.</param>
    /// <param name="k">Rank cutoff.</param>
    /// <returns>Average precision at k.</returns>
    public static double ComputeAveragePrecision(IReadOnlyList<string> rankedDocIds, IReadOnlyDictionary<string, int> qrels, int k)
    {
        ArgumentNullException.ThrowIfNull(rankedDocIds);
        ArgumentNullException.ThrowIfNull(qrels);
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be greater than zero.");

        var relevantTotal = qrels.Values.Count(score => score > 0);
        if (relevantTotal == 0)
            return 0d;

        var limit = Math.Min(k, rankedDocIds.Count);
        var relevantRetrieved = 0;
        var precisionSum = 0d;

        for (var i = 0; i < limit; i++)
        {
            var docId = rankedDocIds[i];
            var isRelevant = qrels.TryGetValue(docId, out var rel) && rel > 0;
            if (!isRelevant)
                continue;

            relevantRetrieved++;
            var precisionAtRank = relevantRetrieved / (double)(i + 1);
            precisionSum += precisionAtRank;
        }

        var denominator = Math.Min(relevantTotal, k);
        return denominator == 0 ? 0d : precisionSum / denominator;
    }

    /// <summary>
    /// Compute Recall@K for a single query (binary: rel &gt; 0 is relevant).
    /// </summary>
    /// <param name="rankedDocIds">Ranked document IDs in descending retrieval order.</param>
    /// <param name="qrels">Query relevance judgments by document ID.</param>
    /// <param name="k">Rank cutoff.</param>
    /// <returns>Recall at k.</returns>
    public static double ComputeRecall(IReadOnlyList<string> rankedDocIds, IReadOnlyDictionary<string, int> qrels, int k)
    {
        ArgumentNullException.ThrowIfNull(rankedDocIds);
        ArgumentNullException.ThrowIfNull(qrels);
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "k must be greater than zero.");

        var relevant = qrels
            .Where(kvp => kvp.Value > 0)
            .Select(kvp => kvp.Key)
            .ToHashSet(StringComparer.Ordinal);

        if (relevant.Count == 0)
            return 0d;

        var hits = 0;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var limit = Math.Min(k, rankedDocIds.Count);
        for (var i = 0; i < limit; i++)
        {
            var docId = rankedDocIds[i];
            if (!seen.Add(docId))
                continue;

            if (relevant.Contains(docId))
                hits++;
        }

        return hits / (double)relevant.Count;
    }

    private static double ComputeDcg(IReadOnlyList<string> rankedDocIds, IReadOnlyDictionary<string, int> qrels, int k)
    {
        var dcg = 0d;
        var limit = Math.Min(k, rankedDocIds.Count);
        for (var i = 0; i < limit; i++)
        {
            var rel = qrels.TryGetValue(rankedDocIds[i], out var score) ? score : 0;
            var rank = i + 1;
            dcg += rel / Math.Log2(rank + 1);
        }

        return dcg;
    }

    private static double ComputeIdealDcg(IReadOnlyDictionary<string, int> qrels, int k)
    {
        var idealRelevances = qrels.Values.OrderByDescending(score => score).Take(k).ToList();

        var idcg = 0d;
        for (var i = 0; i < idealRelevances.Count; i++)
        {
            var rank = i + 1;
            idcg += idealRelevances[i] / Math.Log2(rank + 1);
        }

        return idcg;
    }
}
