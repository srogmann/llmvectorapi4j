package org.rogmann.llmva4j.mcp;

import java.util.List;
import java.util.Map;

public record McpToolInputSchema(String type, Map<String, McpToolPropertyDescription> properties, List<String> required) { }