package org.rogmann.llmva4j.mcp;

/**
 * Description of the interface of a tool.
 */
public record McpToolInterface(String name, String title, String description, McpToolInputSchema inputSchema) { }