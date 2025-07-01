package org.rogmann.llmva4j.mcp;

/**
 * Description of a property.
 * @param type type of the property (e.g. "string" or "array")
 * @param description description of the property
 * @param itemsType optional description of the element type in the case of an array.
 */
public record McpToolPropertyDescription(String type, String description, String itemsType) {
    public McpToolPropertyDescription(String type, String description) {
        this(type, description, null);
    }
}