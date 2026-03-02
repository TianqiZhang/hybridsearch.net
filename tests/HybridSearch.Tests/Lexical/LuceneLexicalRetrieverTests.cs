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
}
