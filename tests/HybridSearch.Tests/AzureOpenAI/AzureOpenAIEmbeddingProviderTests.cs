using HybridSearch.AzureOpenAI;

namespace HybridSearch.Tests.AzureOpenAI;

public class AzureOpenAIEmbeddingProviderTests
{
    [Fact]
    public void Constructor_NullEndpoint_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AzureOpenAIEmbeddingProvider(null!, "key", "deployment"));
    }

    [Fact]
    public void Constructor_NullApiKey_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AzureOpenAIEmbeddingProvider(new Uri("https://test.openai.azure.com"), null!, "deployment"));
    }

    [Fact]
    public void Constructor_NullDeploymentName_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AzureOpenAIEmbeddingProvider(new Uri("https://test.openai.azure.com"), "key", null!));
    }

    [Fact]
    public void Constructor_ZeroDimensions_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AzureOpenAIEmbeddingProvider(
                new Uri("https://test.openai.azure.com"), "key", "deployment", dimensions: 0));
    }

    [Fact]
    public void Constructor_NegativeDimensions_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AzureOpenAIEmbeddingProvider(
                new Uri("https://test.openai.azure.com"), "key", "deployment", dimensions: -1));
    }

    [Fact]
    public void Dimensions_ReturnsConfiguredValue()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment", dimensions: 768);

        Assert.Equal(768, provider.Dimensions);
    }

    [Fact]
    public void Dimensions_DefaultIs1536()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment");

        Assert.Equal(1536, provider.Dimensions);
    }

    [Fact]
    public async Task EmbedAsync_NullText_Throws()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment");

        await Assert.ThrowsAsync<ArgumentNullException>(() => provider.EmbedAsync(null!));
    }

    [Fact]
    public async Task EmbedBatchAsync_NullTexts_Throws()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment");

        await Assert.ThrowsAsync<ArgumentNullException>(() => provider.EmbedBatchAsync(null!));
    }

    [Fact]
    public async Task EmbedBatchAsync_EmptyList_ReturnsEmpty()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment");

        var result = await provider.EmbedBatchAsync(Array.Empty<string>());

        Assert.Empty(result);
    }

    [Fact]
    public void ImplementsIEmbeddingProvider()
    {
        var provider = new AzureOpenAIEmbeddingProvider(
            new Uri("https://test.openai.azure.com"), "key", "deployment");

        Assert.IsAssignableFrom<HybridSearch.Abstractions.IEmbeddingProvider>(provider);
    }
}
