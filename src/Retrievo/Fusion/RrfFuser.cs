using System.Runtime.InteropServices;
using Retrievo.Abstractions;
using Retrievo.Models;

namespace Retrievo.Fusion;

/// <summary>
/// Implements Reciprocal Rank Fusion (RRF) as described by Cormack, Clarke &amp; Butt (2009).
/// For each document across all ranked lists, the fused score is:
///   sum over lists of: weight * 1/(rrfK + rank)
/// where rank is 1-based.
/// Ties are broken by ordinal (case-sensitive) comparison of document IDs.
/// </summary>
public sealed class RrfFuser : IFuser
{
    private static readonly Comparer<KeyValuePair<string, double>> DescendingComparer =
        Comparer<KeyValuePair<string, double>>.Create(CompareDescending);

    /// <inheritdoc/>
    public IReadOnlyList<SearchResult> Fuse(
        IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)> rankedLists,
        int rrfK,
        int topK,
        bool explain)
    {
        if (rankedLists.Count == 0 || topK <= 0)
            return Array.Empty<SearchResult>();

        int scoreCapacity = EstimateScoreCapacity(rankedLists);
        var scores = new Dictionary<string, double>(scoreCapacity, StringComparer.Ordinal);
        var explainData = explain
            ? new Dictionary<string, Dictionary<string, (int Rank, double Contribution)>>(scoreCapacity, StringComparer.Ordinal)
            : null;

        for (int listIdx = 0; listIdx < rankedLists.Count; listIdx++)
        {
            var (items, weight, listName) = rankedLists[listIdx];

            for (int i = 0; i < items.Count; i++)
            {
                var item = items[i];
                int rank = item.Rank; // 1-based
                double contribution = weight * (1.0 / (rrfK + rank));

                ref double scoreRef = ref CollectionsMarshal.GetValueRefOrAddDefault(scores, item.Id, out _);
                scoreRef += contribution;

                if (explainData is not null)
                {
                    if (!explainData.TryGetValue(item.Id, out var listContributions))
                    {
                        listContributions = new Dictionary<string, (int, double)>(rankedLists.Count, StringComparer.Ordinal);
                        explainData[item.Id] = listContributions;
                    }

                    listContributions[listName] = (rank, contribution);
                }
            }
        }

        return SelectTopResults(scores, explainData, topK);
    }

    private static IReadOnlyList<SearchResult> SelectTopResults(
        Dictionary<string, double> scores,
        Dictionary<string, Dictionary<string, (int Rank, double Contribution)>>? explainData,
        int topK)
    {
        int k = Math.Min(topK, scores.Count);
        if (k == 0)
            return Array.Empty<SearchResult>();

        var heap = new KeyValuePair<string, double>[k];
        int heapSize = 0;

        foreach (var score in scores)
        {
            if (heapSize < k)
            {
                heap[heapSize] = score;
                heapSize++;
                if (heapSize == k)
                    BuildMinHeap(heap, k);
            }
            else if (CompareDescending(score, heap[0]) < 0)
            {
                heap[0] = score;
                SiftDown(heap, 0, k);
            }
        }

        Array.Sort(heap, 0, heapSize, DescendingComparer);

        var results = new List<SearchResult>(heapSize);
        for (int i = 0; i < heapSize; i++)
            results.Add(CreateResult(heap[i], explainData));

        return results;
    }

    private static SearchResult CreateResult(
        KeyValuePair<string, double> score,
        Dictionary<string, Dictionary<string, (int Rank, double Contribution)>>? explainData)
    {
        ExplainDetails? explainDetails = null;

        if (explainData is not null)
        {
            int? lexicalRank = null;
            int? vectorRank = null;
            double lexicalContribution = 0.0;
            double vectorContribution = 0.0;

            if (explainData.TryGetValue(score.Key, out var contributions))
            {
                if (contributions.TryGetValue("lexical", out var lexInfo))
                {
                    lexicalRank = lexInfo.Rank;
                    lexicalContribution = lexInfo.Contribution;
                }

                if (contributions.TryGetValue("vector", out var vecInfo))
                {
                    vectorRank = vecInfo.Rank;
                    vectorContribution = vecInfo.Contribution;
                }
            }

            explainDetails = new ExplainDetails
            {
                LexicalRank = lexicalRank,
                VectorRank = vectorRank,
                LexicalContribution = lexicalContribution,
                VectorContribution = vectorContribution,
                FusedScore = score.Value
            };
        }

        return new SearchResult
        {
            Id = score.Key,
            Score = score.Value,
            Explain = explainDetails
        };
    }

    private static int EstimateScoreCapacity(
        IReadOnlyList<(IReadOnlyList<RankedItem> Items, float Weight, string ListName)> rankedLists)
    {
        int totalItems = 0;
        for (int i = 0; i < rankedLists.Count; i++)
            totalItems += rankedLists[i].Items.Count;

        return totalItems;
    }

    private static int CompareDescending(KeyValuePair<string, double> a, KeyValuePair<string, double> b)
    {
        int cmp = b.Value.CompareTo(a.Value);
        return cmp != 0 ? cmp : string.Compare(a.Key, b.Key, StringComparison.Ordinal);
    }

    private static int CompareAscending(KeyValuePair<string, double> a, KeyValuePair<string, double> b)
    {
        int cmp = a.Value.CompareTo(b.Value);
        return cmp != 0 ? cmp : string.Compare(b.Key, a.Key, StringComparison.Ordinal);
    }

    private static void BuildMinHeap(KeyValuePair<string, double>[] heap, int size)
    {
        for (int i = size / 2 - 1; i >= 0; i--)
            SiftDown(heap, i, size);
    }

    private static void SiftDown(KeyValuePair<string, double>[] heap, int index, int size)
    {
        while (true)
        {
            int left = (2 * index) + 1;
            int right = (2 * index) + 2;
            int smallest = index;

            if (left < size && CompareAscending(heap[left], heap[smallest]) < 0)
                smallest = left;
            if (right < size && CompareAscending(heap[right], heap[smallest]) < 0)
                smallest = right;

            if (smallest == index)
                break;

            (heap[index], heap[smallest]) = (heap[smallest], heap[index]);
            index = smallest;
        }
    }
}
