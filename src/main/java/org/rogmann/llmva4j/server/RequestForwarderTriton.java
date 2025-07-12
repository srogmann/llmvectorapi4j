package org.rogmann.llmva4j.server;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.rogmann.llmva4j.LightweightJsonHandler;

/**
 * Forward request to a triton backend.
 * 
 * <p>This implementation uses a Mistral chat-template.</p>
 */
public class RequestForwarderTriton implements RequestForwarder {
    /** logger */
    private static final Logger LOG = Logger.getLogger(RequestForwarderTriton.class.getName());

    /** optional system-prompt */
    private static final String SYSTEM_PROMPT = System.getProperty("uiserver.systemPrompt");
    
    /** Mistral-response without tool-call, group(1) contains the answer text. */
    private static final Pattern PATTERN_MISTRAL_PLAIN_RESPONSE = Pattern.compile("(.*?)</s>.*", Pattern.DOTALL);
    /** Mistral-response with tool-call, group(1) contains an answer text, group(2) the tool-name and group(3) the tool-arguments. */
    private static final Pattern PATTERN_MISTRAL_TOOL_RESPONSE = Pattern.compile("(.*?)\\[TOOL_CALLS\\](.*?)(?:\\[CALL_ID\\].*?)?\\[ARGS\\](.*?)</s>.*", Pattern.DOTALL);
    
    public RequestForwarderTriton() {
        LOG.info("initialized triton-forwarder (Mistral)");
    }

    @Override
    public String forwardRequest(Map<String, Object> requestMap, List<Map<String, Object>> messagesWithTools,
            List<Map<String, Object>> listOpenAITools, String llmUrl) throws MalformedURLException, URISyntaxException {
        // Build a text-input using the chat-format of Mistral-3.2
        var sb = new StringBuilder(200);
        // <s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]My question ...[/INST]The answer.</s>[INST]2nd prompt[/INST]
        sb.append("[SYSTEM_PROMPT]");
        if (SYSTEM_PROMPT == null) {
            // Default is a Mistral system-prompt.
            sb.append("You are Mistral-Small, a Large Language Model (LLM) created by Mistral AI.\n");
            sb.append("Your knowledge base was last updated on 2023-10-01.\n");
            var todayDate = LocalDateTime.now().format(DateTimeFormatter.ISO_DATE);
            sb.append("The current date is " + todayDate + ".\n");
            sb.append("When you\'re not sure about some information or when the user\'s request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don\'t have the information and avoid making up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request.\n");
            sb.append("\n");
            sb.append("# TOOL CALLING INSTRUCTIONS\n\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\n\n1. When the request requires up-to-date information.\n2. When the request requires specific data that you do not have in your knowledge base.\n3. When the request involves actions that you cannot perform without tools.\n\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.");
        } else {
            sb.append(SYSTEM_PROMPT);
        }
        sb.append("[/SYSTEM_PROMPT]");
        for (Map<String, Object> roleMsg : messagesWithTools) {
            var role = LightweightJsonHandler.getJsonValue(roleMsg, "role", String.class);
            if ("user".equals(role)) {
                var content = LightweightJsonHandler.getJsonValue(roleMsg, "content", String.class);
                sb.append("[INST]").append(content).append("[/INST]");
            } else if ("assistant".equals(role)) {
                var content = LightweightJsonHandler.getJsonValue(roleMsg, "content", String.class);
                List<Map<String, Object>> toolCalls = LightweightJsonHandler.getJsonArrayDicts(roleMsg, "tool_calls");
                if (toolCalls != null) {
                    @SuppressWarnings("unchecked")
                    List<Object> toolCallsO = (List<Object>) (List<?>) toolCalls;
                    LightweightJsonHandler.dumpJson(sb, toolCallsO);
                    sb.append("</s>");
                } else {
                    sb.append(content).append("</s>");
                }
            } else {
                throw new RuntimeException(String.format("Unexpected role (%s) in role-msg: %s", role, roleMsg));
            }
        }
        if (!listOpenAITools.isEmpty()) {
            sb.append("[AVAILABLE_TOOLS]");
            @SuppressWarnings("unchecked")
            List<Object> listTools = (List<Object>) (List<?>) listOpenAITools;
            LightweightJsonHandler.dumpJson(sb, listTools);
            sb.append("[/AVAILABLE_TOOLS]");
        }
        var textInput = sb.toString();
        LOG.fine(String.format("llm.text_input: %s", textInput));

        // Build request for LLM
        var llmRequest = new HashMap<String, Object>();
        
        llmRequest.put("temperature", LightweightJsonHandler.readFloat(requestMap, "temperature", 0.5f));
        llmRequest.put("top_p", LightweightJsonHandler.readFloat(requestMap, "top_p", 0.95f));
        llmRequest.put("stream", true);
        llmRequest.put("text_input", textInput);
        llmRequest.put("max_tokens", Integer.parseInt(System.getProperty("forwarder.triton.maxTokens", "200")));

        var sbRequest = new StringBuilder(200);
        LightweightJsonHandler.dumpJson(sbRequest, llmRequest);
        var requestOut = sbRequest.toString();
        LOG.fine(String.format("Request out: " + requestOut));

        // Send request to LLM
        URL url = new URI(llmUrl).toURL();
        String uiResponse;
        AtomicReference<String> modelName = new AtomicReference<>();
        try {
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);

            try (OutputStream os = connection.getOutputStream();
                    OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8)) {
                osw.write(requestOut);
            }

            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                System.err.format("%s HTTP-error accessing %s: %s - %s%n", LocalDateTime.now(),
                        url, responseCode, connection.getResponseMessage());
                return null;
            }
            String contentType = connection.getContentType();
            LOG.fine("Content-Type: " + contentType);
            LOG.fine(String.format("Status: %d - %s", connection.getResponseCode(), connection.getResponseMessage()));
            boolean isEventStream = "text/event-stream".equals(contentType) || "text/event-stream; charset=utf-8".equals(contentType);

            // Read the response.
            String textOutput;
            if (isEventStream) {
                StringBuilder sbResponseContent = new StringBuilder(500);
                Consumer<String> dataConsumer = (data -> {
                    try {
                        Map<String, Object> mapChunk = LightweightJsonHandler.parseJsonDict(data);
                        String lModelName = LightweightJsonHandler.getJsonValue(mapChunk, "model_name", String.class);
                        if (modelName.get() == null && lModelName != null) {
                            modelName.set(lModelName);
                        }
                        String textChunk = LightweightJsonHandler.getJsonValue(mapChunk, "text_output", String.class);
                        if (LOG.isLoggable(Level.FINE)) {
                            System.out.print(textChunk);
                        }
                        sbResponseContent.append(textChunk);
                    } catch (IOException e) {
                        throw new RuntimeException("Unexpected exception while parsing chunk: " + data, e);
                    }
                });
                LOG.fine("Response (merged):");
                if (LOG.isLoggable(Level.FINE)) {
                    System.out.print("<RESPONSE>");
                }
                UiServer.readEventStream(connection, dataConsumer);
                if (LOG.isLoggable(Level.FINE)) {
                    System.out.println("</RESPONSE>");
                }
                textOutput = sbResponseContent.toString();
            } else {
                var sResponse = UiServer.readResponse(connection);
                Map<String, Object> mapResponse = LightweightJsonHandler.parseJsonDict(sResponse);
                modelName.set(LightweightJsonHandler.getJsonValue(mapResponse, "model_name", String.class));
                textOutput = LightweightJsonHandler.getJsonValue(mapResponse, "text_output", String.class);
            }
            LOG.fine("Response: " + textOutput);
            
            // Map the response to llama.cpp-format.
            Map<String, Object> responseMapped = new LinkedHashMap<>();
            responseMapped.put("model_name", modelName);
            List<Map<String, Object>> listChoices = new ArrayList<>();
            Map<String, Object> choice0 = new LinkedHashMap<>();
            choice0.put("index", 0);
            Map<String, Object> respMsg = new LinkedHashMap<>();
            respMsg.put("role", "assistant");
            Matcher mRespTool = PATTERN_MISTRAL_TOOL_RESPONSE.matcher(textOutput);
            Matcher mRespPlain = PATTERN_MISTRAL_PLAIN_RESPONSE.matcher(textOutput);
            if (textOutput == null) {
                LOG.severe("Missing text_output");
            } else if (mRespTool.matches()) {
                // "[TOOL_CALLS]" + tool['function']['name'] + "[CALL_ID]" + tool_call_id + "[ARGS]" + arguments
                textOutput = mRespTool.group(1);
                var toolName = mRespTool.group(2);
                var toolArgs = mRespTool.group(3);
                LOG.info(String.format("Tool-Call of '%s' with arguments: %s", toolName, toolArgs));
                Map<String, Object> mapToolCall = new LinkedHashMap<>();
                mapToolCall.put("name", toolName);
                mapToolCall.put("arguments", toolArgs);
                List<Map<String, Object>> listTools = new ArrayList<>();
                listTools.add(mapToolCall);
                respMsg.put("tool_calls", listTools);
            } else if (mRespPlain.matches()) {
                textOutput = mRespTool.group(1);
            }
            respMsg.put("content", textOutput);
            choice0.put("message", respMsg);
            listChoices.add(choice0);
            responseMapped.put("choices", listChoices);
            
            sb.setLength(0);
            LightweightJsonHandler.dumpJson(sb, responseMapped);
            uiResponse = sb.toString();
            LOG.fine("Response mapped: " + uiResponse);
        } catch (IOException e) {
            System.err.format("%s IO-error", LocalDateTime.now(), e.getMessage());
            e.printStackTrace();
            return null;
        }
        return uiResponse;
    }

}
