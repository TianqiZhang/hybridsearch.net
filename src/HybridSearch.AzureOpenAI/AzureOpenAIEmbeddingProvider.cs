using System.ClientModel;
using Azure.AI.OpenAI;
using HybridSearch.Abstractions;
using OpenAI.Embeddings;

namespace HybridSearch.AzureOpenAI;

/// <summary>
/// IEmbeddingProvider implementation using Azure OpenAI embeddings.
/// Uses the Azure.AI.OpenAI SDK v2.x client.
/// Default model is text-embedding-3-small (1536 dimensions).
/// </summary>
public sealed class AzureOpenAIEmbeddingProvider : IEmbeddingProvider
{
    private readonly EmbeddingClient _client;
    private readonly EmbeddingGenerationOptions _options;

    /// <inheritdoc/>
    public int Dimensions { get; }

    /// <summary>
    /// Creates a new AzureOpenAIEmbeddingProvider.
    /// </summary>
    /// <param name="endpoint">The Azure OpenAI endpoint URL (e.g., https://myresource.openai.azure.com/).</param>
    /// <param name="apiKey">The API key for authentication.</param>
    /// <param name="deploymentName">The deployment name (model deployment) for the embedding model.</param>
    /// <param name="dimensions">The desired embedding dimensions (default: 1536 for text-embedding-3-small).</param>
    public AzureOpenAIEmbeddingProvider(
        Uri endpoint,
        string apiKey,
        string deploymentName,
        int dimensions = 1536)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        ArgumentNullException.ThrowIfNull(apiKey);
        ArgumentNullException.ThrowIfNull(deploymentName);

        if (dimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Dimensions must be positive.");

        Dimensions = dimensions;

        var azureClient = new AzureOpenAIClient(endpoint, new ApiKeyCredential(apiKey));
        _client = azureClient.GetEmbeddingClient(deploymentName);
        _options = new EmbeddingGenerationOptions
        {
            Dimensions = dimensions
        };
    }

    /// <summary>
    /// Creates a new AzureOpenAIEmbeddingProvider with a pre-configured EmbeddingClient.
    /// Useful for testing or custom client configurations.
    /// </summary>
    internal AzureOpenAIEmbeddingProvider(EmbeddingClient client, int dimensions)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));

        if (dimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Dimensions must be positive.");

        Dimensions = dimensions;
        _options = new EmbeddingGenerationOptions
        {
            Dimensions = dimensions
        };
    }

    /// <inheritdoc/>
    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);

        var result = await _client.GenerateEmbeddingAsync(text, _options, ct).ConfigureAwait(false);
        return ToFloatArray(result.Value);
    }

    /// <inheritdoc/>
    public async Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(texts);

        if (texts.Count == 0)
            return Array.Empty<float[]>();

        var result = await _client.GenerateEmbeddingsAsync(texts, _options, ct).ConfigureAwait(false);

        // Results may not be in input order — sort by index
        var embeddings = new float[texts.Count][];
        foreach (var embedding in result.Value)
        {
            embeddings[embedding.Index] = ToFloatArray(embedding);
        }

        return embeddings;
    }

    private static float[] ToFloatArray(OpenAIEmbedding embedding)
    {
        var vector = embedding.ToFloats();
        return vector.ToArray();
    }
}
