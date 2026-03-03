using System.Diagnostics;
using Retrievo.Benchmarks.Data;
using Retrievo.Benchmarks.Embeddings;
using Retrievo.Benchmarks.Evaluation;
using Retrievo.Models;

var exitCode = await RunAsync(args).ConfigureAwait(false);
return exitCode;

static async Task<int> RunAsync(string[] args)
{
    try
    {
        var options = ParseArguments(args);

        var dataDir = Path.GetFullPath(options.DataDir);
        var datasetDir = await BeirDatasetDownloader.EnsureDataAsync(options.Dataset, dataDir).ConfigureAwait(false);

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

        if (options.Sweep)
        {
            RunSweep(runner, options.Dataset, corpus.Count, qrels.Count, embeddings, qrels);
        }
        else
        {
            RunStandard(runner, options.Dataset, corpus.Count, qrels.Count, embeddings, qrels);
        }

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

static void RunStandard(
    EvaluationRunner runner,
    string datasetId,
    int corpusCount,
    int queryCount,
    IReadOnlyDictionary<string, float[]>? embeddings,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels)
{
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
            LexicalWeight = 0.5f,
            VectorWeight = 1f,
            RecallAtK = 100
        }));
    }
    else if (embeddings is not null)
    {
        Console.Error.WriteLine("No query embeddings found in cache. Skipping vector-only and hybrid modes.");
    }

    PrintResults(datasetId, corpusCount, queryCount, results);
}

static void RunSweep(
    EvaluationRunner runner,
    string datasetId,
    int corpusCount,
    int queryCount,
    IReadOnlyDictionary<string, float[]>? embeddings,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> qrels)
{
    var hasEmbeddings = embeddings is not null && HasQueryEmbeddings(qrels, embeddings);

    var displayName = BeirDatasetDownloader.KnownDatasets.TryGetValue(datasetId, out var info)
        ? info.DisplayName
        : datasetId;

    Console.Error.WriteLine($"Starting parameter sweep for {displayName}...");
    Console.Error.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries");
    Console.Error.WriteLine($"Embeddings: {(hasEmbeddings ? "available (hybrid sweep)" : "not available (lexical-only sweep)")}");

    // Sweep grid: ~200 configs for hybrid (~10 min), ~12 for lexical-only
    float[] lexicalWeights = hasEmbeddings
        ? [0f, 0.1f, 0.3f, 0.5f, 1f, 1.5f]
        : [1f];
    float[] vectorWeights = hasEmbeddings
        ? [0f, 0.1f, 0.3f, 0.5f, 1f, 1.5f]
        : [0f];
    int[] rrfKValues = hasEmbeddings
        ? [1, 20, 60]
        : [60];
    float[] titleBoosts = [0.5f, 1f, 2f];

    const int candidateK = 100;

    var configs = new List<EvaluationConfig>();

    foreach (var lw in lexicalWeights)
    foreach (var vw in vectorWeights)
    foreach (var rrfK in rrfKValues)
    foreach (var tb in titleBoosts)
    {
        // Skip invalid combos: both weights zero means no retrieval
        if (lw == 0f && vw == 0f)
            continue;

        // Skip combos requiring embeddings we don't have
        if (vw > 0f && !hasEmbeddings)
            continue;

        var useLexical = lw > 0f;
        var useVector = vw > 0f;

        // For single-list configs, RRF k doesn't matter — only run with default k=60
        if ((!useLexical || !useVector) && rrfK != 60)
            continue;

        // Title boost only matters when lexical is active
        if (!useLexical && tb != 1f)
            continue;

        var name = $"L={lw:F1} V={vw:F1} k={rrfK} tb={tb:F1}";

        configs.Add(new EvaluationConfig
        {
            Name = name,
            UseLexical = useLexical,
            UseVector = useVector,
            TopK = 10,
            LexicalK = candidateK,
            VectorK = candidateK,
            LexicalWeight = lw,
            VectorWeight = vw,
            RrfK = rrfK,
            RecallAtK = 100,
            TitleBoost = tb
        });
    }

    Console.Error.WriteLine($"Running {configs.Count} configurations...");
    Console.Error.WriteLine();

    var results = new List<EvaluationResult>(configs.Count);
    var timer = Stopwatch.StartNew();

    for (var i = 0; i < configs.Count; i++)
    {
        var config = configs[i];
        results.Add(runner.Run(config));

        if ((i + 1) % 50 == 0 || i + 1 == configs.Count)
            Console.Error.WriteLine($"  [{i + 1}/{configs.Count}] {timer.Elapsed.TotalSeconds:F1}s");
    }

    timer.Stop();

    // Sort by nDCG@10 descending
    results.Sort((a, b) => b.NdcgAt10.CompareTo(a.NdcgAt10));

    Console.WriteLine($"{displayName} Parameter Sweep (BEIR)");
    Console.WriteLine(new string('=', displayName.Length + 22));
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries | {configs.Count} configs | {timer.Elapsed.TotalSeconds:F1}s total");
    Console.WriteLine();
    Console.WriteLine($"{"Config",-38} {"nDCG@10",8} {"MAP@10",8} {"R@100",10} {"Avg ms",8}");
    Console.WriteLine(new string('-', 78));

    foreach (var result in results)
    {
        Console.WriteLine(
            $"{result.Name,-38} {result.NdcgAt10,8:F5} {result.MapAt10,8:F5} {result.RecallAt100,10:F5} {result.AvgQueryTimeMs,8:F1}");
    }

    Console.WriteLine();
    Console.WriteLine($"Best: {results[0].Name} -> nDCG@10 = {results[0].NdcgAt10:F5}");
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

    var dataset = "nfcorpus";
    var dataDir = "./benchmarks/data";
    string? embeddingsPath = null;
    var sweep = false;

    for (var i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--dataset":
            case "-d":
                if (i + 1 >= args.Length)
                    throw new ArgumentException("Missing value for --dataset.");
                dataset = args[++i];
                break;

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

            case "--sweep":
                sweep = true;
                break;

            case "--list-datasets":
                PrintAvailableDatasets();
                Environment.Exit(0);
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
        Dataset = dataset,
        DataDir = dataDir,
        EmbeddingsPath = embeddingsPath,
        Sweep = sweep
    };
}

static void PrintUsage()
{
    Console.WriteLine("HybridSearch.Benchmarks — BEIR dataset evaluation");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project benchmarks/HybridSearch.Benchmarks -- [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --dataset, -d <name>    BEIR dataset to evaluate (default: nfcorpus)");
    Console.WriteLine("  --data-dir <dir>        Base directory for dataset storage (default: ./benchmarks/data)");
    Console.WriteLine("  --embeddings <path>     Path to pre-computed embeddings binary cache");
    Console.WriteLine("  --sweep                 Run parameter sweep (grid search over fusion params)");
    Console.WriteLine("  --list-datasets         Show available BEIR datasets and exit");
    Console.WriteLine("  --help, -h              Show this help and exit");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine("  dotnet run --project benchmarks/HybridSearch.Benchmarks");
    Console.WriteLine("  dotnet run --project benchmarks/HybridSearch.Benchmarks -- --dataset scifact");
    Console.WriteLine("  dotnet run --project benchmarks/HybridSearch.Benchmarks -- --dataset nfcorpus --embeddings embeddings.bin --sweep");
}

static void PrintAvailableDatasets()
{
    Console.WriteLine("Available BEIR datasets:");
    Console.WriteLine();
    foreach (var (id, info) in BeirDatasetDownloader.KnownDatasets.OrderBy(kv => kv.Key, StringComparer.Ordinal))
    {
        Console.WriteLine($"  {id,-16} {info.Description}");
    }
    Console.WriteLine();
    Console.WriteLine("Any BEIR dataset ID can be used, even if not listed above.");
    Console.WriteLine("The tool will attempt to download it from the BEIR CDN.");
}

static void PrintResults(string datasetId, int corpusCount, int queryCount, IReadOnlyList<EvaluationResult> results)
{
    var displayName = BeirDatasetDownloader.KnownDatasets.TryGetValue(datasetId, out var info)
        ? info.DisplayName
        : datasetId;

    Console.WriteLine($"{displayName} Benchmark (BEIR)");
    Console.WriteLine(new string('=', displayName.Length + 17));
    Console.WriteLine($"Dataset: {corpusCount:N0} docs | {queryCount:N0} queries");
    Console.WriteLine();
    Console.WriteLine("Configuration            nDCG@10   MAP@10   Recall@100  Avg Query");
    Console.WriteLine("-------------------------------------------------------------------");

    foreach (var result in results)
    {
        Console.WriteLine(
            $"{result.Name,-24} {result.NdcgAt10,8:F5} {result.MapAt10,8:F5} {result.RecallAt100,10:F5}  {result.AvgQueryTimeMs,8:F1}ms");
    }
}

file sealed record ParsedOptions
{
    public required string Dataset { get; init; }

    public required string DataDir { get; init; }

    public string? EmbeddingsPath { get; init; }

    public bool Sweep { get; init; }
}
