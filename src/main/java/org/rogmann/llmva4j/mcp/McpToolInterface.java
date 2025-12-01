package org.rogmann.llmva4j.mcp;

/**
 * Description of the interface of a tool.
 *
 * @param name name of the tool
 * @param title title of the tool
 * @param description description of the tool
 * @param inputSchema properties of the tool
 */
public record McpToolInterface(String name, String title, String description, McpToolInputSchema inputSchema) { }
