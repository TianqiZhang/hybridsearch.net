using HybridSearch.Lexical;

namespace HybridSearch.Tests.Lexical;

public class LuceneLexicalRetrieverTests
{
    [Fact]
    public void Search_ExactMatch_ReturnsDocument()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "machine learning is transforming software engineering");

        var results = retriever.Search("machine learning", topK: 10);

        Assert.Single(results);
        Assert.Equal("doc-1", results[0].Id);
        Assert.True(results[0].Score > 0);
        Assert.Equal(1, results[0].Rank);
    }

    [Fact]
    public void Search_NoMatch_ReturnsEmpty()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "machine learning is transforming software engineering");

        var results = retriever.Search("quantum physics", topK: 10);

        Assert.Empty(results);
    }

    [Fact]
    public void Search_TopK_LimitsResults()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "neural network deep learning");
        retriever.Add("doc-2", "deep learning frameworks");
        retriever.Add("doc-3", "learning algorithms for deep data");

        var results = retriever.Search("deep learning", topK: 2);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void Search_ResultsOrderedByRelevance()
    {
        using var retriever = new LuceneLexicalRetriever();
        // doc-1 has "kubernetes" twice, doc-2 has it once
        retriever.Add("doc-1", "kubernetes orchestration with kubernetes deployment");
        retriever.Add("doc-2", "kubernetes container management");
        retriever.Add("doc-3", "docker container images");

        var results = retriever.Search("kubernetes", topK: 10);

        // doc-3 shouldn't match at all
        Assert.All(results, r => Assert.NotEqual("doc-3", r.Id));

        // Results should be ordered by descending score
        for (int i = 1; i < results.Count; i++)
            Assert.True(results[i - 1].Score >= results[i].Score);

        // Ranks should be 1-based sequential
        for (int i = 0; i < results.Count; i++)
            Assert.Equal(i + 1, results[i].Rank);
    }

    [Fact]
    public void Search_EmptyQuery_ReturnsEmpty()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "hello world");

        var results = retriever.Search("", topK: 10);
        Assert.Empty(results);
    }

    [Fact]
    public void Search_WhitespaceQuery_ReturnsEmpty()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "hello world");

        var results = retriever.Search("   ", topK: 10);
        Assert.Empty(results);
    }

    [Fact]
    public void Search_EmptyIndex_ReturnsEmpty()
    {
        using var retriever = new LuceneLexicalRetriever();
        var results = retriever.Search("hello", topK: 10);
        Assert.Empty(results);
    }

    [Fact]
    public void Count_ReflectsAddedDocuments()
    {
        using var retriever = new LuceneLexicalRetriever();
        Assert.Equal(0, retriever.Count);

        retriever.Add("doc-1", "hello world");
        Assert.Equal(1, retriever.Count);

        retriever.Add("doc-2", "goodbye world");
        Assert.Equal(2, retriever.Count);
    }

    [Fact]
    public void Search_CaseInsensitive()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "Machine Learning Algorithms");

        var results = retriever.Search("machine learning", topK: 10);

        Assert.Single(results);
        Assert.Equal("doc-1", results[0].Id);
    }

    [Fact]
    public void Search_PartialTermMatch()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "distributed computing systems handle parallel workloads");
        retriever.Add("doc-2", "parallel processing in modern CPUs");

        var results = retriever.Search("parallel", topK: 10);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void Search_MultipleTermQuery_MatchesDocumentsWithAnyTerm()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "rust programming language");
        retriever.Add("doc-2", "python programming language");
        retriever.Add("doc-3", "database management system");

        var results = retriever.Search("rust python", topK: 10);

        // Both doc-1 and doc-2 contain at least one query term
        Assert.Equal(2, results.Count);
        var ids = results.Select(r => r.Id).ToHashSet();
        Assert.Contains("doc-1", ids);
        Assert.Contains("doc-2", ids);
    }

    [Fact]
    public void Search_SpecialCharacters_DoesNotThrow()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "hello world");

        // QueryParser.Escape should handle special characters
        var results = retriever.Search("hello + world && (test)", topK: 10);

        // Should not throw — escaped query should still work
        Assert.NotNull(results);
    }

    [Fact]
    public void Search_TopK_LargerThanCorpus_ReturnsAll()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "test document alpha");
        retriever.Add("doc-2", "test document beta");

        var results = retriever.Search("test", topK: 100);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void Search_AfterMultipleAdds_ReflectsAllDocuments()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "first document about retrieval");
        retriever.Add("doc-2", "second document about indexing");

        // Add more after a search
        var results1 = retriever.Search("retrieval", topK: 10);
        Assert.Single(results1);

        retriever.Add("doc-3", "third document about retrieval systems");

        var results2 = retriever.Search("retrieval", topK: 10);
        Assert.Equal(2, results2.Count);
    }

    [Fact]
    public void Dispose_ThenSearch_Throws()
    {
        var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "hello world");
        retriever.Dispose();

        Assert.Throws<ObjectDisposedException>(() => retriever.Search("hello", topK: 10));
    }

    [Fact]
    public void Dispose_ThenAdd_Throws()
    {
        var retriever = new LuceneLexicalRetriever();
        retriever.Dispose();

        Assert.Throws<ObjectDisposedException>(() => retriever.Add("doc-1", "hello"));
    }

    [Fact]
    public void Search_AllStopWords_ReturnsEmpty()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "the quick brown fox jumps over the lazy dog");

        // "the", "a", "is", "are" are all stop words — analyzer strips them, leaving no query terms
        var results = retriever.Search("the a is are", topK: 10);

        Assert.Empty(results);
    }

    [Fact]
    public void Search_RepeatedTerms_BoostedByFrequency()
    {
        using var retriever = new LuceneLexicalRetriever();
        // doc-1 has both "neural" and "network" — matches both unique query terms
        retriever.Add("doc-1", "neural network architectures for image recognition");
        // doc-2 has "network" but not "neural"
        retriever.Add("doc-2", "network security and firewall configuration");

        // Repeating "neural" should boost documents matching "neural" more
        var resultsNormal = retriever.Search("neural network", topK: 10);
        var resultsBoosted = retriever.Search("neural neural neural network", topK: 10);

        // doc-1 should be first in both cases (matches both terms)
        Assert.Equal("doc-1", resultsNormal[0].Id);
        Assert.Equal("doc-1", resultsBoosted[0].Id);

        // With repeated "neural", doc-1's score should increase relative to doc-2
        // because the frequency-weighted boost on "neural" TermQuery is higher
        double normalRatio = resultsNormal[0].Score / resultsNormal[1].Score;
        double boostedRatio = resultsBoosted[0].Score / resultsBoosted[1].Score;
        Assert.True(boostedRatio > normalRatio,
            $"Expected boosted ratio ({boostedRatio:F4}) > normal ratio ({normalRatio:F4})");
    }

    [Fact]
    public void Search_WithBoosts_TitleBoostAffectsRanking()
    {
        using var retriever = new LuceneLexicalRetriever();
        // doc-1: "search" only in title
        retriever.Add("doc-1", "unrelated body content about databases", title: "search algorithms");
        // doc-2: "search" only in body
        retriever.Add("doc-2", "information search and retrieval systems", title: "unrelated title");

        // With high title boost, doc-1 (title match) should rank higher
        var highTitleResults = retriever.Search("search", topK: 10, titleBoost: 10f, bodyBoost: 1f);
        Assert.True(highTitleResults.Count >= 2);
        Assert.Equal("doc-1", highTitleResults[0].Id);

        // With high body boost, doc-2 (body match) should rank higher
        var highBodyResults = retriever.Search("search", topK: 10, titleBoost: 1f, bodyBoost: 10f);
        Assert.True(highBodyResults.Count >= 2);
        Assert.Equal("doc-2", highBodyResults[0].Id);
    }

    [Fact]
    public void Search_NullText_Throws()
    {
        using var retriever = new LuceneLexicalRetriever();
        retriever.Add("doc-1", "hello world");

        Assert.Throws<ArgumentNullException>(() => retriever.Search(null!, topK: 10));
    }
}
