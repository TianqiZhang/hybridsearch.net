using Retrievo.Models;

namespace Retrievo.Tests;

public class FieldDefinitionTests
{
    [Fact]
    public void FieldDefinition_DefaultValues()
    {
        var def = new FieldDefinition { Name = "test" };

        Assert.Equal("test", def.Name);
        Assert.Equal(FieldType.String, def.Type);
        Assert.Equal('|', def.Delimiter);
    }

    [Fact]
    public void FieldDefinition_StringArrayWithCustomDelimiter()
    {
        var def = new FieldDefinition { Name = "tags", Type = FieldType.StringArray, Delimiter = ',' };

        Assert.Equal(FieldType.StringArray, def.Type);
        Assert.Equal(',', def.Delimiter);
    }

    [Fact]
    public void FieldDefinition_Validate_NullName_Throws()
    {
        var def = new FieldDefinition { Name = null! };

        Assert.Throws<ArgumentException>(() => def.Validate());
    }

    [Fact]
    public void FieldDefinition_RecordEquality()
    {
        var a = new FieldDefinition { Name = "x", Type = FieldType.StringArray, Delimiter = '|' };
        var b = new FieldDefinition { Name = "x", Type = FieldType.StringArray, Delimiter = '|' };
        var c = new FieldDefinition { Name = "x", Type = FieldType.String, Delimiter = '|' };

        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }

    [Fact]
    public void DefineField_OnMutableBuilder_PassedToIndex()
    {
        // When a field is declared StringArray on the builder, MetadataFilters does contains-match
        using var index = new MutableHybridSearchIndexBuilder()
            .DefineField("tags", FieldType.StringArray)
            .Build();

        index.Upsert(new Document
        {
            Id = "d1",
            Body = "Document with multiple tags.",
            Metadata = new Dictionary<string, string> { ["tags"] = "a|b|c" }
        });
        index.Commit();

        var response = index.Search(new HybridQuery
        {
            Text = "document tags",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["tags"] = "b" }
        });

        Assert.Single(response.Results);
        Assert.Equal("d1", response.Results[0].Id);
    }

    [Fact]
    public void DefineField_OnImmutableBuilder_PassedToIndex()
    {
        using var index = new HybridSearchIndexBuilder()
            .DefineField("tags", FieldType.StringArray)
            .AddDocument(new Document
            {
                Id = "d1",
                Body = "Document with multiple tags.",
                Metadata = new Dictionary<string, string> { ["tags"] = "a|b|c" }
            })
            .Build();

        var response = index.Search(new HybridQuery
        {
            Text = "document tags",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["tags"] = "b" }
        });

        Assert.Single(response.Results);
        Assert.Equal("d1", response.Results[0].Id);
    }

    [Fact]
    public void UndeclaredField_UsesExactMatch_BackwardCompatible()
    {
        // When no field definition is declared, MetadataFilters does exact-match (backward compat)
        using var index = new MutableHybridSearchIndexBuilder().Build();

        index.Upsert(new Document
        {
            Id = "d1",
            Body = "Document with delimited value.",
            Metadata = new Dictionary<string, string> { ["tags"] = "a|b|c" }
        });
        index.Commit();

        // Exact match: "b" != "a|b|c" → no results
        var response = index.Search(new HybridQuery
        {
            Text = "document tags",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["tags"] = "b" }
        });

        Assert.Empty(response.Results);

        // Exact match: "a|b|c" == "a|b|c" → matches
        var exactResponse = index.Search(new HybridQuery
        {
            Text = "document tags",
            TopK = 10,
            MetadataFilters = new Dictionary<string, string> { ["tags"] = "a|b|c" }
        });

        Assert.Single(exactResponse.Results);
    }

    [Fact]
    public void DefineField_NullName_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MutableHybridSearchIndexBuilder().DefineField(null!, FieldType.String));
    }

    [Fact]
    public void DefineField_WhitespaceName_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new MutableHybridSearchIndexBuilder().DefineField("  ", FieldType.String));
    }

    [Fact]
    public void DefineField_EmptyName_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new MutableHybridSearchIndexBuilder().DefineField("", FieldType.String));
    }
}
