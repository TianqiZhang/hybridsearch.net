using System.IO.Compression;

namespace HybridSearch.Benchmarks.Data;

/// <summary>
/// Downloads and extracts BEIR datasets from the official repository.
/// Supports any dataset available at the BEIR CDN (NFCorpus, SciFact, FiQA, TREC-COVID, etc.).
/// </summary>
public static class BeirDatasetDownloader
{
    private const string BeirCdnBaseUrl = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets";

    /// <summary>
    /// Known BEIR datasets with display names and corpus descriptions.
    /// </summary>
    public static IReadOnlyDictionary<string, BeirDatasetInfo> KnownDatasets { get; } =
        new Dictionary<string, BeirDatasetInfo>(StringComparer.OrdinalIgnoreCase)
        {
            ["nfcorpus"] = new()
            {
                Id = "nfcorpus",
                DisplayName = "NFCorpus",
                Description = "Biomedical information retrieval — 3,633 PubMed articles, 323 test queries, graded relevance (0/1/2)"
            },
            ["scifact"] = new()
            {
                Id = "scifact",
                DisplayName = "SciFact",
                Description = "Scientific claim verification — 5,183 abstracts, 300 test queries, binary relevance"
            },
            ["fiqa"] = new()
            {
                Id = "fiqa",
                DisplayName = "FiQA",
                Description = "Financial opinion QA — 57,638 documents, 648 test queries, graded relevance"
            },
            ["trec-covid"] = new()
            {
                Id = "trec-covid",
                DisplayName = "TREC-COVID",
                Description = "COVID-19 biomedical literature — 171,332 documents, 50 test queries, graded relevance"
            },
            ["hotpotqa"] = new()
            {
                Id = "hotpotqa",
                DisplayName = "HotpotQA",
                Description = "Multi-hop question answering — 5,233,329 documents, 7,405 test queries"
            },
            ["arguana"] = new()
            {
                Id = "arguana",
                DisplayName = "ArguAna",
                Description = "Argument retrieval — 8,674 documents, 1,406 test queries"
            },
            ["quora"] = new()
            {
                Id = "quora",
                DisplayName = "Quora",
                Description = "Duplicate question retrieval — 522,931 documents, 10,000 test queries"
            }
        };

    /// <summary>
    /// Lists all known dataset IDs.
    /// </summary>
    public static IReadOnlyList<string> ListDatasetIds() =>
        KnownDatasets.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();

    /// <summary>
    /// Ensures a BEIR dataset is available at the specified directory.
    /// Downloads and extracts if not already present.
    /// </summary>
    /// <param name="datasetId">BEIR dataset identifier (e.g. "nfcorpus", "scifact").</param>
    /// <param name="dataDir">Base directory where the dataset subdirectory should exist.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Path to extracted dataset directory.</returns>
    public static async Task<string> EnsureDataAsync(string datasetId, string dataDir, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(datasetId);
        ArgumentNullException.ThrowIfNull(dataDir);

        if (string.IsNullOrWhiteSpace(datasetId))
            throw new ArgumentException("Dataset ID must not be empty.", nameof(datasetId));

        var fullDataDir = Path.GetFullPath(dataDir);
        Directory.CreateDirectory(fullDataDir);

        var datasetDir = Path.Combine(fullDataDir, datasetId);
        var corpusPath = Path.Combine(datasetDir, "corpus.jsonl");
        if (File.Exists(corpusPath))
        {
            var displayName = GetDisplayName(datasetId);
            Console.Error.WriteLine($"{displayName} already present: {datasetDir}");
            return datasetDir;
        }

        var downloadUrl = $"{BeirCdnBaseUrl}/{datasetId}.zip";
        var zipPath = Path.Combine(fullDataDir, $"{datasetId}.zip");

        Console.Error.WriteLine($"Downloading {GetDisplayName(datasetId)} from {downloadUrl}");

        using var httpClient = new HttpClient();
        using var response = await httpClient.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength;
        await using (var source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (var destination = new FileStream(zipPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true))
        {
            var buffer = new byte[81920];
            long downloaded = 0;
            DateTime nextProgressUpdateUtc = DateTime.UtcNow;

            while (true)
            {
                var read = await source.ReadAsync(buffer.AsMemory(0, buffer.Length), ct).ConfigureAwait(false);
                if (read == 0)
                    break;

                await destination.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                downloaded += read;

                if (DateTime.UtcNow >= nextProgressUpdateUtc)
                {
                    if (totalBytes is > 0)
                    {
                        var percent = downloaded * 100d / totalBytes.Value;
                        Console.Error.WriteLine($"Download progress: {percent:F1}% ({downloaded:N0}/{totalBytes.Value:N0} bytes)");
                    }
                    else
                    {
                        Console.Error.WriteLine($"Download progress: {downloaded:N0} bytes");
                    }

                    nextProgressUpdateUtc = DateTime.UtcNow.AddMilliseconds(500);
                }
            }
        }

        Console.Error.WriteLine("Download completed. Extracting archive...");

        if (Directory.Exists(datasetDir))
            Directory.Delete(datasetDir, recursive: true);

        ZipFile.ExtractToDirectory(zipPath, fullDataDir, overwriteFiles: true);
        File.Delete(zipPath);

        if (!File.Exists(corpusPath))
            throw new DirectoryNotFoundException($"Dataset extraction failed. Missing file: {corpusPath}");

        Console.Error.WriteLine($"{GetDisplayName(datasetId)} ready: {datasetDir}");
        return datasetDir;
    }

    private static string GetDisplayName(string datasetId)
    {
        if (KnownDatasets.TryGetValue(datasetId, out var info))
            return info.DisplayName;
        return datasetId;
    }
}

/// <summary>
/// Metadata for a known BEIR dataset.
/// </summary>
public sealed record BeirDatasetInfo
{
    /// <summary>
    /// Lowercase identifier used in URLs and directory names.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Human-readable display name.
    /// </summary>
    public required string DisplayName { get; init; }

    /// <summary>
    /// Short description of the dataset.
    /// </summary>
    public required string Description { get; init; }
}
