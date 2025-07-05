package org.rogmann.llmva4j.server;

import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Interface used to forward a request to a LLM server.
 */
public interface RequestForwarder {

    /**
     * Forwards a request to a language model (LLM) server at the specified URL, handling tool call information
     * and preparing the request with the provided configuration parameters. This method constructs the request
     * payload from the given message history and tools, sends it using HTTP POST, and returns the raw response string.
     *
     * @param requestMap           The request configuration map containing parameters such as temperature,
     *                             max_tokens, top_p, and stream. These values are used to configure the LLM request.
     * @param messagesWithTools    A list of message dictionaries to include in the LLM request. This typically
     *                             includes user and assistant messages, as well as any tool call responses.
     * @param listOpenAITools      A list of tool definitions in the OpenAI format. These are added to the request
     *                             if the list is not empty.
     * @param llmUrl               The URL of the LLM server to which the request will be sent.
     *
     * @return                     The raw JSON response string from the LLM server. Returns null if the HTTP
     *                             request fails or an exception is thrown.
     *
     * @throws MalformedURLException If the provided llmUrl is invalid and cannot be converted to a valid URL.
     * @throws URISyntaxException    If the provided llmUrl is invalid and cannot be converted to a valid URI.
     */
    String forwardRequest(Map<String, Object> requestMap, ArrayList<Map<String, Object>> messagesWithTools,
            final List<Map<String, Object>> listOpenAITools, String llmUrl)
            throws MalformedURLException, URISyntaxException;
}
