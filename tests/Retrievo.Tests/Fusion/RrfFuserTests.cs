using Retrievo.Fusion;
using Retrievo.Models;

namespace Retrievo.Tests.Fusion;

public class RrfFuserTests
{
    private readonly RrfFuser _fuser = new();

    [Fact]
    public void SingleList_ScoresFollowRrfFormula()
    {
        // Given a single ranked list [A, B, C] with weight 1.0 and rrfK=60
        var items = new[]
        {
            new RankedItem { Id = "A", Score = 10.0, Rank = 1 },
            new RankedItem { Id = "B", Score = 8.0, Rank = 2 },
            new RankedItem { Id = "C", Score = 5.0, Rank = 3 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (items, 1.0f, "lexical")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        Assert.Equal(3, results.Count);

        // Expected scores: 1/(60+1), 1/(60+2), 1/(60+3)
        Assert.Equal("A", results[0].Id);
        Assert.Equal(1.0 / 61, results[0].Score, precision: 10);

        Assert.Equal("B", results[1].Id);
        Assert.Equal(1.0 / 62, results[1].Score, precision: 10);

        Assert.Equal("C", results[2].Id);
        Assert.Equal(1.0 / 63, results[2].Score, precision: 10);
    }

    [Fact]
    public void TwoLists_FusedRankingMatchesExpected()
    {
        // Lexical: [A, B, C], Vector: [B, A, D]
        // With rrfK=60 and equal weights:
        // A: lexical=1/(60+1) + vector=1/(60+2) = 1/61 + 1/62
        // B: lexical=1/(60+2) + vector=1/(60+1) = 1/62 + 1/61
        // C: lexical=1/(60+3) = 1/63
        // D: vector=1/(60+3) = 1/63
        // A and B have the same score; tie-break by ordinal: A < B
        var lexical = new[]
        {
            new RankedItem { Id = "A", Score = 10, Rank = 1 },
            new RankedItem { Id = "B", Score = 8, Rank = 2 },
            new RankedItem { Id = "C", Score = 5, Rank = 3 },
        };
        var vector = new[]
        {
            new RankedItem { Id = "B", Score = 0.95, Rank = 1 },
            new RankedItem { Id = "A", Score = 0.90, Rank = 2 },
            new RankedItem { Id = "D", Score = 0.80, Rank = 3 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (lexical, 1.0f, "lexical"),
            (vector, 1.0f, "vector")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        Assert.Equal(4, results.Count);

        double expectedAB = (1.0 / 61) + (1.0 / 62);
        double expectedCD = 1.0 / 63;

        // A and B tied, A comes first by ordinal tie-break
        Assert.Equal("A", results[0].Id);
        Assert.Equal(expectedAB, results[0].Score, precision: 10);

        Assert.Equal("B", results[1].Id);
        Assert.Equal(expectedAB, results[1].Score, precision: 10);

        // C and D tied, C < D by ordinal
        Assert.Equal("C", results[2].Id);
        Assert.Equal(expectedCD, results[2].Score, precision: 10);

        Assert.Equal("D", results[3].Id);
        Assert.Equal(expectedCD, results[3].Score, precision: 10);
    }

    [Fact]
    public void WeightShifting_IncreasingVectorWeight_PromotesVectorOnlyDoc()
    {
        // D only appears in vector list, C only in lexical list
        // With equal weights: C score = D score (both rank 3)
        // With vectorWeight=2.0: D should score higher than C
        var lexical = new[]
        {
            new RankedItem { Id = "A", Score = 10, Rank = 1 },
            new RankedItem { Id = "B", Score = 8, Rank = 2 },
            new RankedItem { Id = "C", Score = 5, Rank = 3 },
        };
        var vector = new[]
        {
            new RankedItem { Id = "B", Score = 0.95, Rank = 1 },
            new RankedItem { Id = "A", Score = 0.90, Rank = 2 },
            new RankedItem { Id = "D", Score = 0.80, Rank = 3 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (lexical, 1.0f, "lexical"),
            (vector, 2.0f, "vector")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        // D: vector contribution = 2.0 * 1/(60+3) = 2/63
        // C: lexical contribution = 1.0 * 1/(60+3) = 1/63
        var dResult = results.First(r => r.Id == "D");
        var cResult = results.First(r => r.Id == "C");

        Assert.True(dResult.Score > cResult.Score,
            $"D ({dResult.Score}) should rank higher than C ({cResult.Score}) when vector weight is doubled");
    }

    [Fact]
    public void TieBreak_OrdinalById()
    {
        // Two docs with identical fused scores, should be ordered by ordinal Id
        var list = new[]
        {
            new RankedItem { Id = "zebra", Score = 1.0, Rank = 1 },
            new RankedItem { Id = "apple", Score = 0.5, Rank = 2 },
        };
        var list2 = new[]
        {
            new RankedItem { Id = "apple", Score = 1.0, Rank = 1 },
            new RankedItem { Id = "zebra", Score = 0.5, Rank = 2 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (list, 1.0f, "lexical"),
            (list2, 1.0f, "vector")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        // Both have score 1/(60+1) + 1/(60+2) — tied
        Assert.Equal("apple", results[0].Id); // 'a' < 'z' in ordinal
        Assert.Equal("zebra", results[1].Id);
    }

    [Fact]
    public void TopK_Truncation()
    {
        var items = Enumerable.Range(1, 10)
            .Select(i => new RankedItem { Id = $"doc-{i:D2}", Score = 10.0 - i, Rank = i })
            .ToArray();

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (items, 1.0f, "lexical")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 3, explain: false);

        Assert.Equal(3, results.Count);
        Assert.Equal("doc-01", results[0].Id);
        Assert.Equal("doc-02", results[1].Id);
        Assert.Equal("doc-03", results[2].Id);
    }

    [Fact]
    public void TopK_LargerThanCorpus_ReturnsAll()
    {
        var items = new[]
        {
            new RankedItem { Id = "A", Score = 1.0, Rank = 1 },
            new RankedItem { Id = "B", Score = 0.5, Rank = 2 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (items, 1.0f, "lexical")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 100, explain: false);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void EmptyLists_ReturnsEmpty()
    {
        var rankedLists = Array.Empty<(IReadOnlyList<RankedItem>, float, string)>();

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        Assert.Empty(results);
    }

    [Fact]
    public void Explain_PopulatesContributions()
    {
        var lexical = new[]
        {
            new RankedItem { Id = "A", Score = 10, Rank = 1 },
            new RankedItem { Id = "B", Score = 8, Rank = 2 },
        };
        var vector = new[]
        {
            new RankedItem { Id = "B", Score = 0.95, Rank = 1 },
            new RankedItem { Id = "C", Score = 0.80, Rank = 2 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (lexical, 1.0f, "lexical"),
            (vector, 1.0f, "vector")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: true);

        // A: only in lexical at rank 1
        var aResult = results.First(r => r.Id == "A");
        Assert.NotNull(aResult.Explain);
        Assert.Equal(1, aResult.Explain.LexicalRank);
        Assert.Null(aResult.Explain.VectorRank);
        Assert.Equal(1.0 / 61, aResult.Explain.LexicalContribution, precision: 10);
        Assert.Equal(0.0, aResult.Explain.VectorContribution);

        // B: lexical rank 2, vector rank 1
        var bResult = results.First(r => r.Id == "B");
        Assert.NotNull(bResult.Explain);
        Assert.Equal(2, bResult.Explain.LexicalRank);
        Assert.Equal(1, bResult.Explain.VectorRank);
        Assert.Equal(1.0 / 62, bResult.Explain.LexicalContribution, precision: 10);
        Assert.Equal(1.0 / 61, bResult.Explain.VectorContribution, precision: 10);

        // C: only in vector at rank 2
        var cResult = results.First(r => r.Id == "C");
        Assert.NotNull(cResult.Explain);
        Assert.Null(cResult.Explain.LexicalRank);
        Assert.Equal(2, cResult.Explain.VectorRank);
        Assert.Equal(0.0, cResult.Explain.LexicalContribution);
        Assert.Equal(1.0 / 62, cResult.Explain.VectorContribution, precision: 10);
    }

    [Fact]
    public void Explain_ContributionsSumToFusedScore()
    {
        var lexical = new[]
        {
            new RankedItem { Id = "A", Score = 10, Rank = 1 },
        };
        var vector = new[]
        {
            new RankedItem { Id = "A", Score = 0.95, Rank = 1 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (lexical, 1.5f, "lexical"),
            (vector, 0.8f, "vector")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: true);

        var a = results[0];
        Assert.NotNull(a.Explain);

        double expectedLexical = 1.5 * (1.0 / 61);
        double expectedVector = 0.8 * (1.0 / 61);

        Assert.Equal(expectedLexical, a.Explain.LexicalContribution, precision: 8);
        Assert.Equal(expectedVector, a.Explain.VectorContribution, precision: 8);
        Assert.Equal(expectedLexical + expectedVector, a.Explain.FusedScore, precision: 8);
        Assert.Equal(a.Score, a.Explain.FusedScore, precision: 8);
    }

    [Fact]
    public void Explain_False_NoExplainDetails()
    {
        var items = new[]
        {
            new RankedItem { Id = "A", Score = 10, Rank = 1 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (items, 1.0f, "lexical")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        Assert.Null(results[0].Explain);
    }

    [Fact]
    public void SingleItemLists_Work()
    {
        var lexical = new[]
        {
            new RankedItem { Id = "only", Score = 5.0, Rank = 1 },
        };

        var rankedLists = new (IReadOnlyList<RankedItem>, float, string)[]
        {
            (lexical, 1.0f, "lexical")
        };

        var results = _fuser.Fuse(rankedLists, rrfK: 60, topK: 10, explain: false);

        Assert.Single(results);
        Assert.Equal("only", results[0].Id);
        Assert.Equal(1.0 / 61, results[0].Score, precision: 10);
    }
}
