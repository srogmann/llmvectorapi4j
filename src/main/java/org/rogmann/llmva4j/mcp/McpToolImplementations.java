package org.rogmann.llmva4j.mcp;

import java.util.List;

/**
 * Interface to get a list of tool-implementations.
 */
public interface McpToolImplementations {

    /**
     * Gets the list of tool-implmentations.
     * @return tools
     */
    List<McpToolImplementation> get();
}
