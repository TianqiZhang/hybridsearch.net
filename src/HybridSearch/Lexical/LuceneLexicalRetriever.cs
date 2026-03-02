using HybridSearch.Abstractions;
using HybridSearch.Models;
using Lucene.Net.Index;
using Lucene.Net.QueryParsers.Classic;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;
using LuceneDocument = Lucene.Net.Documents.Document;
using LuceneField = Lucene.Net.Documents.Field;
using LuceneStringField = Lucene.Net.Documents.StringField;
using LuceneTextField = Lucene.Net.Documents.TextField;

namespace HybridSearch.Lexical;

/// <summary>
/// Lexical retriever backed by Lucene.NET's in-memory RAMDirectory with BM25 scoring.
/// Documents are indexed by their body text and searched via parsed text queries.
/// </summary>
public sealed class LuceneLexicalRetriever : ILexicalRetriever
{
    private const string FieldId = "id";
    private const string FieldBody = "body";
    private const LuceneVersion Version = LuceneVersion.LUCENE_48;

    private readonly RAMDirectory _directory;
    private readonly LuceneTextAnalyzer _textAnalyzer;
    private readonly IndexWriter _writer;
    private IndexSearcher? _searcher;
    private DirectoryReader? _reader;
    private bool _dirty = true;
    private bool _disposed;

    /// <summary>
    /// Number of documents in the index.
    /// </summary>
    public int Count => _writer.NumDocs;

    /// <summary>
    /// Creates a new LuceneLexicalRetriever with a default StandardAnalyzer.
    /// </summary>
    public LuceneLexicalRetriever()
        : this(new LuceneTextAnalyzer())
    {
    }

    /// <summary>
    /// Creates a new LuceneLexicalRetriever with the specified text analyzer.
    /// </summary>
    public LuceneLexicalRetriever(LuceneTextAnalyzer textAnalyzer)
    {
        _textAnalyzer = textAnalyzer ?? throw new ArgumentNullException(nameof(textAnalyzer));
        _directory = new RAMDirectory();
        var config = new IndexWriterConfig(Version, _textAnalyzer.Analyzer)
        {
            OpenMode = OpenMode.CREATE
        };
        _writer = new IndexWriter(_directory, config);
    }

    /// <summary>
    /// Add a document to the lexical index.
    /// </summary>
    /// <param name="id">The unique document ID.</param>
    /// <param name="body">The document body text to index.</param>
    public void Add(string id, string body)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(id);
        ArgumentNullException.ThrowIfNull(body);

        var doc = new LuceneDocument
        {
            new LuceneStringField(FieldId, id, LuceneField.Store.YES),
            new LuceneTextField(FieldBody, body, LuceneField.Store.NO)
        };

        _writer.AddDocument(doc);
        _dirty = true;
    }

    /// <summary>
    /// Ensure the searcher is refreshed to reflect any pending writes.
    /// </summary>
    private void EnsureSearcher()
    {
        if (_dirty || _searcher is null)
        {
            _writer.Commit();

            var newReader = _reader is null
                ? DirectoryReader.Open(_directory)
                : DirectoryReader.OpenIfChanged(_reader);

            if (newReader is not null)
            {
                _reader?.Dispose();
                _reader = newReader;
                _searcher = new IndexSearcher(_reader);
            }

            _dirty = false;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<RankedItem> Search(string text, int topK)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<RankedItem>();

        EnsureSearcher();

        if (_searcher is null || _reader!.NumDocs == 0)
            return Array.Empty<RankedItem>();

        // Parse the query against the body field
        var parser = new QueryParser(Version, FieldBody, _textAnalyzer.Analyzer);
        Query query;
        try
        {
            query = parser.Parse(QueryParser.Escape(text));
        }
        catch (ParseException)
        {
            return Array.Empty<RankedItem>();
        }

        var topDocs = _searcher.Search(query, topK);
        var results = new RankedItem[topDocs.ScoreDocs.Length];

        for (int i = 0; i < topDocs.ScoreDocs.Length; i++)
        {
            var scoreDoc = topDocs.ScoreDocs[i];
            var storedDoc = _searcher.Doc(scoreDoc.Doc);
            var docId = storedDoc.Get(FieldId);

            results[i] = new RankedItem
            {
                Id = docId,
                Score = scoreDoc.Score,
                Rank = i + 1 // 1-based
            };
        }

        return results;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _reader?.Dispose();
            _writer.Dispose();
            _directory.Dispose();
            _textAnalyzer.Dispose();
            _disposed = true;
        }
    }
}
