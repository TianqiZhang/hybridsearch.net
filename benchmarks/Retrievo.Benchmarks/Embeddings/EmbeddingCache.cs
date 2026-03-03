using System.Text;

namespace Retrievo.Benchmarks.Embeddings;

/// <summary>
/// Caches pre-computed embeddings in a compact binary format.
/// Format: [count:int32] then per entry [id:7bit-length-prefixed-utf8][dim:int32][floats:float32_le[dim]].
/// </summary>
public static class EmbeddingCache
{
    /// <summary>
    /// Save embeddings to a binary cache file.
    /// </summary>
    /// <param name="path">Destination cache path.</param>
    /// <param name="embeddings">Embeddings keyed by ID.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task SaveAsync(string path, IReadOnlyDictionary<string, float[]> embeddings, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(embeddings);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        await using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(embeddings.Count);
        foreach (var (id, vector) in embeddings)
        {
            ct.ThrowIfCancellationRequested();

            ArgumentNullException.ThrowIfNull(id);
            ArgumentNullException.ThrowIfNull(vector);

            writer.Write(id);
            writer.Write(vector.Length);
            foreach (var value in vector)
                writer.Write(value);
        }

        await stream.FlushAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Load embeddings from a binary cache file.
    /// </summary>
    /// <param name="path">Source cache path.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Embeddings keyed by ID.</returns>
    public static Task<IReadOnlyDictionary<string, float[]>> LoadAsync(string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(path);

        if (!File.Exists(path))
            throw new FileNotFoundException($"Embedding cache file not found: {path}", path);

        return Task.Run<IReadOnlyDictionary<string, float[]>>(() =>
        {
            var embeddings = new Dictionary<string, float[]>(StringComparer.Ordinal);

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

            var count = reader.ReadInt32();
            if (count < 0)
                throw new InvalidDataException("Invalid embedding cache: negative record count.");

            for (var i = 0; i < count; i++)
            {
                ct.ThrowIfCancellationRequested();

                var id = reader.ReadString();
                var dimension = reader.ReadInt32();

                if (string.IsNullOrWhiteSpace(id))
                    throw new InvalidDataException("Invalid embedding cache: empty ID encountered.");
                if (dimension <= 0)
                    throw new InvalidDataException($"Invalid embedding cache: non-positive dimension for '{id}'.");

                var vector = new float[dimension];
                for (var d = 0; d < dimension; d++)
                    vector[d] = reader.ReadSingle();

                embeddings[id] = vector;
            }

            return embeddings;
        }, ct);
    }

    /// <summary>
    /// Check whether an embedding cache file exists.
    /// </summary>
    /// <param name="path">Cache file path.</param>
    /// <returns>True when the file exists; otherwise false.</returns>
    public static bool Exists(string path) => File.Exists(path);
}
