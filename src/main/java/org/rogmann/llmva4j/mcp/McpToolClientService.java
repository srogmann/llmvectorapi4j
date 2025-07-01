package org.rogmann.llmva4j.mcp;

import org.rogmann.llmva4j.mcp.McpHttpClient.McpToolWithUri;

/**
 * Interface to provide a tool description together with a URI (client-side).
 */
public interface McpToolClientService {
	/**
	 * Provides a tool description with URI.
	 * @return tool description with URI
	 */
	McpToolWithUri provide();
}