namespace Retrievo.Models;

/// <summary>
/// Metadata field types supported by the index.
/// </summary>
public enum FieldType
{
    /// <summary>
    /// A single string value. Filtered by exact ordinal match (default).
    /// </summary>
    String,

    /// <summary>
    /// A delimited string representing multiple values (e.g., "a|b|c").
    /// Filtered by checking whether any delimited element matches the filter value.
    /// The delimiter is specified per-field via <see cref="FieldDefinition.Delimiter"/>.
    /// </summary>
    StringArray
}

/// <summary>
/// Declares the type and storage conventions of a metadata field in the index.
/// Fields not explicitly defined default to <see cref="FieldType.String"/>.
/// </summary>
public sealed record FieldDefinition
{
    /// <summary>
    /// The metadata key this definition applies to.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// The field type. Determines how metadata filters are evaluated.
    /// </summary>
    public FieldType Type { get; init; } = FieldType.String;

    /// <summary>
    /// Delimiter used to split <see cref="FieldType.StringArray"/> values.
    /// Only meaningful when <see cref="Type"/> is <see cref="FieldType.StringArray"/>.
    /// Default is '|'.
    /// </summary>
    public char Delimiter { get; init; } = '|';

    /// <summary>
    /// Validates that the field definition has a non-null, non-whitespace name.
    /// </summary>
    internal void Validate()
    {
        if (string.IsNullOrWhiteSpace(Name))
            throw new ArgumentException("Field name must not be null or whitespace.", nameof(Name));
    }
}
