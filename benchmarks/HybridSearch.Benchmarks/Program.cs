using HybridSearch.Benchmarks.Data;
using HybridSearch.Benchmarks.Embeddings;
using HybridSearch.Benchmarks.Evaluation;
using HybridSearch.Models;

var exitCode = await RunAsync(args).ConfigureAwait(false);
return exitCode;

static async Task<int> RunAsync(string[] args)
{
    try
    {
        var options = ParseArguments(args);

        var dataDir = Path.GetFullPath(options.DataDir);
        var datasetDir = await NfCorpusDownloader.EnsureDataAsync(dataDir).ConfigureAwait(false);

        var corpusPath = Path.Combine(datasetDir, "corpus.jsonl");
        var queriesPath = Path.Combine(datasetDir, "queries.jsonl");
        var qrelsPath = Path.Combine(datasetDir, "qrels", "test.tsv");

        Console.Error.WriteLine("Loading BEIR files...");
        var corpus = await BeirDataLoader.LoadCorpusAsync(corpusPath).ConfigureAwait(false);
        var queries = await BeirDataLoader.LoadQueriesAsync(queriesPath).ConfigureAwait(false);
        var qrels = await BeirDataLoader.LoadQrelsAsync(qrelsPath).ConfigureAwait(false);

        IReadOnlyDictionary<string, float[]>? embeddings = null;
        if (!string.IsNullOrWhiteSpace(options.EmbeddingsPath))
        {
            var embeddingsPath = Path.GetFullPath(options.EmbeddingsPath);
            if (!EmbeddingCache.Exists(embeddingsPath))
            {
                Console.Error.WriteLine($"Embedding cache not found: {embeddingsPath}");
                Console.Error.WriteLine("Continuing with lexical-only evaluation.");
            }
            else
            {
                Console.Error.WriteLine($"Loading embeddings cache: {embeddingsPath}");
                embeddings = await EmbeddingCache.LoadAsync(embeddingsPath).ConfigureAwait(false);
            }
        }

        var documents = BuildDocuments(corpus, embeddings);

        using var runner = new EvaluationRunner(documents, queries, qrels, embeddings);

        var results = new List<EvaluationResult>
        {
            runner.Run(new EvaluationConfig
            {
                Name = "Lexical-only (BM25)",
                UseLexical = true,
                UseVector = false,
                TopK = 10,
                LexicalK = 100,
                VectorK = 100,
                LexicalWeight = 1f,
                VectorWeight = 0f,
                RecallAtK = 100
            })
        };

        if (embeddings is not null && HasQueryEmbeddings(qrels, embeddings))
        {
            results.Add(runner.Run(new EvaluationConfig
            {
                Name = "Vector-only",
                UseLexical = false,
                UseVector = true,
                TopK = 10,
                LexicalK = 100,
                VectorK = 100,
                LexicalWeight = 0f,
                VectorWeight = 1f,
                RecallAtK = 100
            }));

            results.Add(runner.Run(new EvaluationConfig
            {
                Name = "Hybrid (BM25 + Vector)",
                UseLexical = true,
                UseVector = true,
                TopK = 10,
                LexicalK = 100,
                VectorK = 100,
                LexicalWeight = 1f,
                VectorWeight = 1f,
                RecallAtK = 100
            }));
        }
        else if (embeddings is not null)
        {
            Console.Error.WriteLine("No query embeddings found in cache. Skipping vector-only and hybrid modes.");
        }

        PrintResults(corpus.Count, qrels.Count, results);
        return 0;
    }
    catch (ArgumentException ex)
    {
        Console.Error.WriteLine($"Invalid argument: {ex.Message}");
        return 1;
    }
    catch (FileNotFoundException ex)
    {
        Console.Error.WriteLine($"File not found: {ex.Message}");
        return 1;
    }
    catch (InvalidDataException ex)
    {
        Console.Error.WriteLine($"Data error: {ex.Message}");
        return 1;
    }
    catch (DirectoryNotFoundException ex)
    {
        Console.Error.WriteLine($"Directory not found: {ex.Message}");
        return 1;
    }
    catch (HttpRequestException ex)
    {
        Console.Error.WriteLine($"Download failed: {ex.Message}");
        return 1;
    }
}

static IReadOnlyList<Document> BuildDocuments(
    IReadOnlyDictionary<string, (string Title, string Text)> corpus,
    IReadOnlyDictionary<string, float[]>? embeddings)
{
    ArgumentNullException.ThrowIfNull(corpus);

    var docs = new List<Document>(corpus.Count);
    foreach (var (id, payload) in corpus)
    {
        float[]? vector = null;
        if (embeddings is not null)
            embeddings.TryGetValue(id, out vector);

        docs.Add(new Document
        {
            Id = id,
            Title = payload.Title,
            Body = payload.Text,
            Embedding = vector
        });
    }

    return docs;
}

static bool HasQueryEmbeddings(
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels,
    IReadOnlyDictionary<string, float[]> embeddings)
{
    ArgumentNullException.ThrowIfNull(qrels);
    ArgumentNullException.ThrowIfNull(embeddings);

    foreach (var queryId in qrels.Keys)
    {
        if (!embeddings.ContainsKey(queryId))
            return false;
    }

    return true;
}

static ParsedOptions ParseArguments(string[] args)
{
    ArgumentNullException.ThrowIfNull(args);

    var dataDir = "./benchmarks/data";
    string? embeddingsPath = null;

    for (var i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--data-dir":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --data-dir.");
                dataDir = args[++i];
                break;

            case "--embeddings":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --embeddings.");
                embeddingsPath = args[++i];
                break;

            case "--help":
            case "-h":
                PrintUsage();
                Environment.Exit(0);
                break;

            default:
                throw new ArgumentException($"Unknown argument: {args[i]}");
        }
    }

    return new ParsedOptions
    {
        DataDir = dataDir,
        EmbeddingsPath = embeddingsPath
    };
}

static void PrintUsage()
{
    Console.WriteLine("HybridSearch.Benchmarks");
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project benchmarks/HybridSearch.Benchmarks -- [--data-dir <dir>] [--embeddings <path>]");
}

static void PrintResults(int corpusCount, int queryCount, IReadOnlyList<EvaluationResult> results)
{
    Console.WriteLine("NFCorpus Benchmark (BEIR)");
    Console.WriteLine("=========================");
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries | Graded relevance (0/1/2)");
    Console.WriteLine();
    Console.WriteLine("Configuration            nDCG@10   MAP@10   Recall@100  Avg Query");
    Console.WriteLine("-------------------------------------------------------------------");

    foreach (var result in results)
    {
        Console.WriteLine(
            $"{result.Name,-24} {result.NdcgAt10,8:F5} {result.MapAt10,8:F5} {result.RecallAt100,10:F5}  {result.AvgQueryTimeMs,8:F1}ms");
    }

    Console.WriteLine($"{"BEIR BM25 baseline",-24} {0.32500,8:F5} {"-",8} {"-",10}  {"-",8}");
}

file sealed record ParsedOptions
{
    public required string DataDir { get; init; }

    public string? EmbeddingsPath { get; init; }
}
