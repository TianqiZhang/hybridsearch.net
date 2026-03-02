using HybridSearch.Abstractions;
using Lucene.Net.Analysis;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Analysis.TokenAttributes;
using Lucene.Net.Util;

namespace HybridSearch.Lexical;

/// <summary>
/// ITextAnalyzer implementation backed by Lucene.NET's StandardAnalyzer.
/// Provides tokenization and normalization (lowercasing, stop-word removal).
/// </summary>
public sealed class LuceneTextAnalyzer : ITextAnalyzer
{
    private readonly Analyzer _analyzer;
    private bool _disposed;

    /// <summary>
    /// Creates a new LuceneTextAnalyzer with a StandardAnalyzer using the specified Lucene version.
    /// </summary>
    public LuceneTextAnalyzer()
    {
        _analyzer = new StandardAnalyzer(LuceneVersion.LUCENE_48);
    }

    /// <summary>
    /// Creates a new LuceneTextAnalyzer wrapping the provided Lucene analyzer.
    /// </summary>
    internal LuceneTextAnalyzer(Analyzer analyzer)
    {
        _analyzer = analyzer ?? throw new ArgumentNullException(nameof(analyzer));
    }

    /// <summary>
    /// The underlying Lucene.NET Analyzer instance.
    /// Exposed internally for use by LuceneLexicalRetriever.
    /// </summary>
    internal Analyzer Analyzer => _analyzer;

    /// <inheritdoc/>
    public IEnumerable<string> Analyze(string fieldName, string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(fieldName);
        ArgumentNullException.ThrowIfNull(text);

        return AnalyzeCore(fieldName, text);
    }

    private IEnumerable<string> AnalyzeCore(string fieldName, string text)
    {
        using var tokenStream = _analyzer.GetTokenStream(fieldName, text);
        var termAttr = tokenStream.AddAttribute<ICharTermAttribute>();
        tokenStream.Reset();

        while (tokenStream.IncrementToken())
        {
            yield return termAttr.ToString();
        }

        tokenStream.End();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _analyzer.Dispose();
            _disposed = true;
        }
    }
}
