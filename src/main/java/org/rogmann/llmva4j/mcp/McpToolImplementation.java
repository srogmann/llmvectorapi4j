package org.rogmann.llmva4j.mcp;

import java.util.List;
import java.util.Map;

/**
 * Interface of an implementation of a MCP tool (server-side).
 */
public interface McpToolImplementation {

	/**
	 * Gets the name of the tool.
	 * @return tool-name
	 */
	String getName();
	
	/**
	 * Gets the tool definition.
	 * @return tool interface
	 */
	McpToolInterface getTool();

	/**
	 * Executes a tool.
	 * @param params params-map with keys "name" and "arguments"
	 * @return value of content attribute of the result (e.g. array of type-text-maps)
	 */
	List<Map<String, Object>> call(Map<String, Object> params);

}
