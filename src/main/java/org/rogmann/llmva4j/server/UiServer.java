package org.rogmann.llmva4j.server;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.ServiceLoader;
import java.util.function.Consumer;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import org.rogmann.llmva4j.LightweightJsonHandler;
import org.rogmann.llmva4j.mcp.McpHttpClient;
import org.rogmann.llmva4j.mcp.McpHttpClient.McpToolWithUri;
import org.rogmann.llmva4j.mcp.McpToolInterface;
import org.rogmann.llmva4j.mcp.McpToolPropertyDescription;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Simple web-server for providing a web interface for an OpenAI-chat/completions-compatible LLM-API endpoint.
 *
 * <p>A different LLM-API can be used by providing an implementation of {@link RequestForwarder}./p>
 */
public class UiServer {
    /** logger */
    private static final Logger LOG = Logger.getLogger(UiServer.class.getName());

    /** property to set a maximum number of tool-calls (default is 3) in a request */
    private static final String PROP_MAX_TOOL_CALLS = "uiserver.max.toolCalls";

    /**
     * Entry method, starts a web-server.
     * @param args UiServer &lt;server-ip&gt; &lt;server-port&gt; &lt;llm-url&gt; &lt;public-path&gt;
     */
    public static void main(String[] args) {
        if (args.length < 4) {
            System.out.println("Usage: java UiServer <server-ip> <server-port> <llm-url> <public-path> [<mcp-server-url>]*");
            System.exit(1);
        }

        String host = args[0];
        int port = Integer.parseInt(args[1]);
        String llmUrl = args[2];
        String publicPath = args[3];
        if (!new File(publicPath).isDirectory()) {
            throw new IllegalArgumentException("Missing path directory (web-content): " + publicPath);
        }
        List<String> mcpServerUrls = IntStream.range(4, args.length).mapToObj(i -> args[i]).toList();

        McpHttpClient mcpClient = new McpHttpClient();
        for (String mcpServerUrl : mcpServerUrls) {
            URL url;
            try {
                url = URI.create(mcpServerUrl).toURL();
            } catch (MalformedURLException e) {
                throw new RuntimeException("Invalid mcp-server-URL " + mcpServerUrl, e);
            }
            List<McpToolWithUri> tools = mcpClient.listTools(url, null);
            for (McpToolWithUri toolWithUri : tools) {
                McpToolInterface tool = toolWithUri.tool();
                LOG.info(String.format("Register tool: %s at %s", tool.name(), toolWithUri.url()));
                mcpClient.registerTool(toolWithUri);
            }
        }

        startServer(host, port, llmUrl, publicPath, mcpClient);
    }

    private static void startServer(String host, int port, String llmUrl, String publicPath, McpHttpClient mcpClient) {
        ServiceLoader<RequestForwarder> slRequestForwarder = ServiceLoader.load(RequestForwarder.class);
        RequestForwarder requestForwarder = slRequestForwarder.findFirst()
                .orElse((requestMap, messagesWithTools, listOpenAITools, url) -> forwardRequest(requestMap, messagesWithTools, listOpenAITools, url));
        LOG.info("RequestForwarder: " + requestForwarder);

        try {
            var addr = new InetSocketAddress(host, port);
            var server = HttpServer.create(addr, 0);
            server.createContext("/", exchange -> handleRequest(new HttpExchangeDecorator(exchange),
                    llmUrl, publicPath, mcpClient, requestForwarder));
            server.setExecutor(null); // use default executor
            server.start();
            LOG.info("Server started on " + host + ":" + port);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void handleRequest(IHttpExchange exchange, String llmUrl, String publicPath, McpHttpClient mcpClient,
            RequestForwarder requestForwarder) {
        LOG.info(String.format("%s %s request %s%n", LocalDateTime.now(), exchange.getRequestMethod(), exchange.getRequestURI()));
        if ("GET".equals(exchange.getRequestMethod())) {
            processGetRequest(exchange, publicPath);
            return;
        }

        if (!"POST".equals(exchange.getRequestMethod())) {
            sendError(exchange, 405, "Method not allowed");
            return;
        }
        
        String cookie = exchange.getRequestHeaders().getFirst("Cookie");

        try {
            String httpRequestBody;
            try (var is = exchange.getRequestBody();
                    var isr = new InputStreamReader(is);
                    var br = new BufferedReader(isr)) {

                var jsonBody = new StringBuilder();
                while (true) {
                    String line = br.readLine();
                    if (line == null) {
                        break;
                    }
                    LOG.fine("< " + line);
                    jsonBody.append(line);
                }
                httpRequestBody = jsonBody.toString();
            }

            // Parse incoming JSON
            var requestMap = LightweightJsonHandler.parseJsonDict(httpRequestBody);
            var messages = LightweightJsonHandler.getJsonArrayDicts(requestMap, "messages");
            var messagesWithTools = new ArrayList<>(messages);

            boolean hasToolResponse = false;
            String uiResponse = null;
            int numToolCalls = 0;
            do {
                final List<Map<String, Object>> listOpenAITools = convertMcp2OpenAI(mcpClient.getTools());
                uiResponse = requestForwarder.forwardRequest(requestMap, messagesWithTools, listOpenAITools, llmUrl);
                hasToolResponse = false;
                if (!listOpenAITools.isEmpty() && uiResponse != null) {
                    // Check for a tool-call.
                    hasToolResponse = checkForToolCall(mcpClient, messagesWithTools, uiResponse, cookie);
                    numToolCalls++;
                    int maxNumToolCalls = Integer.getInteger(PROP_MAX_TOOL_CALLS, 3);
                    if (hasToolResponse && numToolCalls > maxNumToolCalls) {
                        // We don't want an infinite loop.
                        LOG.severe(String.format("Reached maximum number of tool-calls (%d) in a request", maxNumToolCalls));
                        break;
                    };
                }
            } while (hasToolResponse);

            if (uiResponse != null) {
                Map<String, Object> mapResponse = LightweightJsonHandler.parseJsonDict(uiResponse);
                List<Map<String, Object>> listChoices = LightweightJsonHandler.getJsonArrayDicts(mapResponse, "choices");
                if (listChoices != null && !listChoices.isEmpty()) {
                    Map<String, Object> choice = listChoices.get(0);
                    @SuppressWarnings("unchecked")
                    Map<String, Object> mapMessage = LightweightJsonHandler.getJsonValue(choice, "message", Map.class);
                    if (mapMessage != null) {
                        choice.put("delta", mapMessage);
                        var sb = new StringBuilder(500);
                        LightweightJsonHandler.dumpJson(sb, mapResponse);
                        uiResponse = sb.toString();
                    }
                }
                LOG.fine(String.format("UI-response: %s", uiResponse));
                exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream os = exchange.getResponseBody();
                        OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8)) {
                    String sseResponse = String.format("data: %s\n\ndata: [DONE]\n\n", uiResponse);
                    osw.write(sseResponse);
                }
            } else {
                sendError(exchange, 500, "No UI-response");
            }

        } catch (Exception e) {
            sendError(exchange, 500, "Internal server error: " + e.getMessage());
            e.printStackTrace();
        }
    }

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
     *
     * @note                       The request is modified to always set the "stream" parameter to false before
     *                             being sent, even if it was originally true.
     */
    public static String forwardRequest(Map<String, Object> requestMap, List<Map<String, Object>> messagesWithTools,
            final List<Map<String, Object>> listOpenAITools, String llmUrl)
            throws MalformedURLException, URISyntaxException {
        // Build request for LLM
        var llmRequest = new HashMap<String, Object>();
        if (!listOpenAITools.isEmpty()) {
            llmRequest.put("tools", listOpenAITools);
        }
        LOG.fine(String.format("llm.messages-out: %s", messagesWithTools));
        llmRequest.put("messages", messagesWithTools);
        llmRequest.put("temperature", LightweightJsonHandler.readFloat(requestMap, "temperature", 0.7f));
        llmRequest.put("max_tokens", LightweightJsonHandler.readInt(requestMap, "max_tokens", 100));
        llmRequest.put("top_p", LightweightJsonHandler.readFloat(requestMap, "top_p", 1.0f));
        llmRequest.put("stream", LightweightJsonHandler.readBoolean(requestMap, "stream", false));

        var sbRequest = new StringBuilder(200);
        LightweightJsonHandler.dumpJson(sbRequest, llmRequest);
        var requestOut = sbRequest.toString().replace("\"stream\":true", "\"stream\":false");
        LOG.fine(String.format("Request out: " + requestOut));

        // Send request to LLM
        URL url = new URI(llmUrl).toURL();
        String uiResponse;
        try {
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            connection.addRequestProperty("Accept", "text/event-stream");
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
            LOG.fine("Content-Type: " + connection.getContentType());

            // Read the response in chunks.
            String sResponse = readResponse(connection);
            LOG.fine("Response: " + sResponse);
            uiResponse = sResponse;
        } catch (IOException e) {
            LOG.severe("IO-error while calling " + url);
            System.err.format("%s IO-error", LocalDateTime.now(), e.getMessage());
            e.printStackTrace();
            return null;
        }
        return uiResponse;
    }

    /**
     * Reads the content (body, UTF-8) of the HTTP-response.
     * @param connection connection
     * @return content
     * @throws IOException
     */
    public static String readResponse(HttpURLConnection connection) throws IOException {
        String sResponse;
        try (InputStream inputStream = connection.getInputStream();
                InputStreamReader isr = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
            final StringBuilder sb = new StringBuilder(500);
            char[] cBuf = new char[4096];
            while (true) {
                int len = isr.read(cBuf);
                if (len == -1) {
                    break;
                }
                sb.append(cBuf, 0, len);
            }
            sResponse = sb.toString();
        }
        return sResponse;
    }

    /**
     * Reads a SSE-body (event stream).
     * @param connection connection
     * @param dataConsumer consumer to read data-chunks 
     * @throws IOException
     */
    public static void readEventStream(HttpURLConnection connection, Consumer<String> dataConsumer) throws IOException {
        try (InputStream inputStream = connection.getInputStream();
                InputStreamReader isr = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
            StringBuilder sb = new StringBuilder(1000);
            char[] cBuf = new char[4096];
            int dataCount = 0;
            while (true) {
                int len = isr.read(cBuf);
                if (len == -1) {
                    break;
                }
                int curOffset = sb.length();
                sb.append(cBuf, 0, len);
                while (true) {
                    int idxCR = sb.indexOf("\r", curOffset);
                    if (idxCR < curOffset) {
                        break;
                    }
                    // Remove carriage-return.
                    sb.replace(idxCR, idxCR + 1, "");
                }
                if (sb.length() == 0) {
                    LOG.fine(String.format("missing data-line in response (#data-lines=%d)", dataCount));
                    break;
                }
                while (sb.length() >= 6 && "data: ".equals(sb.substring(0, 6))) {
                    dataCount++;
                    int idxLF = sb.indexOf("\n\n");
                    if (idxLF < 0) {
                        break;
                    }
                    String data = sb.substring(6, idxLF);
                    if (data.length() == 0 || "[DONE]".equalsIgnoreCase(data)) {
                        break;
                    }
                    dataConsumer.accept(data);
                    sb.delete(0, idxLF + 2);
                }
            }
            if (sb.length() > 0) {
                LOG.warning(String.format("Unexpected end of event-stream (#data-lines=%d): %s", dataCount, sb));
            }
        }
    }

    /**
     * Processes the JSON response to detect and handle tool calls within the message content.
     * If tool calls are found, executes the corresponding tools using the provided client
     * and appends both the original message and the tool responses to the messages list.
     *
     * @param mcpClient      The HTTP client used to execute tool calls.
     * @param messagesWithTools  A list of messages to which the tool call message and its response will be appended.
     * @param sResponse      The raw JSON response string to parse for tool calls.
     * @param cookie         optional HTTP cookie
     * @return               True if any tool calls were processed and added to the messages list, false otherwise.
     * @throws IOException   If an error occurs during the tool execution via the McpHttpClient.
     * @note                 This method assumes the response follows a specific JSON structure containing
     *                       "choices" array with message objects that might include "tool_calls".
     */
    private static boolean checkForToolCall(McpHttpClient mcpClient, ArrayList<Map<String, Object>> messagesWithTools,
            String sResponse, String cookie) throws IOException {
        boolean hasToolResponse = false;
        Map<String, Object> mapResponse = LightweightJsonHandler.parseJsonDict(sResponse);
        List<Map<String, Object>> listChoices = LightweightJsonHandler.getJsonArrayDicts(mapResponse, "choices");
        if (listChoices != null && !listChoices.isEmpty()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> mapMessage = LightweightJsonHandler.getJsonValue(listChoices.get(0), "message", Map.class);
            if (mapMessage != null) {
                List<Map<String, Object>> listToolCalls = LightweightJsonHandler.getJsonArrayDicts(mapMessage, "tool_calls");
                if (listToolCalls != null && !listToolCalls.isEmpty()) {
                    for (Map<String, Object> mapToolCall : listToolCalls) {
                        String type = LightweightJsonHandler.getJsonValue(mapToolCall, "type", String.class);
                        if (!"function".equals(type)) {
                            throw new RuntimeException("Unexpected tool_call: " + mapToolCall);
                        }
                        @SuppressWarnings("unchecked")
                        Map<String, Object> mapFunction = LightweightJsonHandler.getJsonValue(mapToolCall, "function", Map.class);
                        String functionName = LightweightJsonHandler.getJsonValue(mapFunction, "name", String.class);
                        String arguments = LightweightJsonHandler.getJsonValue(mapFunction, "arguments", String.class);
                        String id = LightweightJsonHandler.getJsonValue(mapToolCall, "id", String.class);
                        Map<String, Object> mapArgs = LightweightJsonHandler.parseJsonDict(arguments);
                        Map<String, Object> toolResult = mcpClient.callTool(functionName, mapArgs, id, cookie);
                        LOG.info("Tool-Result: " + toolResult);
                        @SuppressWarnings("unchecked")
                        List<Map<String, Object>> aToolContent = LightweightJsonHandler.getJsonValue(toolResult, "content", List.class);
                        String text = null;
                        if (!aToolContent.isEmpty()) {
                            Map<String, Object> mapFirstContent = aToolContent.get(0);
                            text = LightweightJsonHandler.getJsonValue(mapFirstContent, "text", String.class);
                        }
                        hasToolResponse = true;
                        messagesWithTools.add(mapMessage);
                        Map<String, Object> mapToolResponse = new LinkedHashMap<>();
                        mapToolResponse.put("role", "tool");
                        mapToolResponse.put("content", text);
                        mapToolResponse.put("tool_call_id", id);
                        messagesWithTools.add(mapToolResponse);
                        LOG.fine(String.format("Next messages: %s", messagesWithTools));
                    }
                }
            }
        }
        return hasToolResponse;
    }

    /**
     * Reads a response of the HTTP-connection and streams it to the HTTP-exchange.
     * @param exchange HTTP-exchange
     * @param connection HTTP-connection
     * @throws IOException in case of a network error
     */
    static void streamResponse(HttpExchange exchange, HttpURLConnection connection) throws IOException {
        try (InputStream inputStream = connection.getInputStream();
                 BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            exchange.sendResponseHeaders(200, 0);
            try (OutputStream os = exchange.getResponseBody();
                    OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8);
                    BufferedWriter bw = new BufferedWriter(osw)) {
                while (true) {
                    String line = reader.readLine();
                    if (line == null) {
                        break;
                    }
                    LOG.fine("> " + line);
                    bw.write(line); bw.write((char) '\n');
                    bw.flush();
                }
            }
        }
    }

    private static List<Map<String, Object>> convertMcp2OpenAI(List<McpToolInterface> listMcpTools) {
        List<Map<String, Object>> listOpenAITools = new ArrayList<>();
        for (McpToolInterface mcpTool : listMcpTools) {
            Map<String, Object> tool = new LinkedHashMap<>();
            tool.put("type", "function");
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", mcpTool.name());
            function.put("description", mcpTool.description());
            Map<String, Object> parameters = new LinkedHashMap<>();
            parameters.put("type", "object");
            Map<String, Object> properties = new LinkedHashMap<>();
            for (Entry<String, McpToolPropertyDescription> entry : mcpTool.inputSchema().properties().entrySet()) {
                Map<String, Object> mapProp = new LinkedHashMap<>();
                mapProp.put("type", entry.getValue().type());
                mapProp.put("description", entry.getValue().description());
                if (entry.getValue().itemsType() != null) {
                    var mapItems = new LinkedHashMap<>();
                    mapItems.put("type", entry.getValue().itemsType());
                    mapProp.put("items", mapItems);
                }
                properties.put(entry.getKey(), mapProp);
            }
            parameters.put("properties", properties);
            function.put("parameters", parameters);
            function.put("required", mcpTool.inputSchema().required());
            tool.put("function", function);
            listOpenAITools.add(tool);
        }
        return listOpenAITools;
    }

    private static void processGetRequest(IHttpExchange exchange, String publicPath) {
        String path = exchange.getRequestURI().getPath();
        if (path.equals("/")) {
            path = "/index.html";
        }

        // Build the file paths
        File requestedFile = new File(publicPath + path);
        File gzFile = new File(publicPath + path + ".gz");

        // Security check for path traversal
        try {
            File canonicalFile = requestedFile.getCanonicalFile();
            if (!canonicalFile.getAbsolutePath().startsWith(publicPath)) {
                System.err.format("%s Forbidden path (%s) -> (%s)%n", LocalDateTime.now(), path, canonicalFile);
                sendError(exchange, 403, "Forbidden path");
                return;
            }
        } catch (IOException e) {
            System.err.format("%s Error accessing file: %s%n", LocalDateTime.now(), requestedFile);
            e.printStackTrace();
            sendError(exchange, 500, "Error accessing file");
            return;
        }

        // Determine which file to serve
        File serveFile = null;
        if (gzFile.exists()) {
            serveFile = gzFile;
        } else if (requestedFile.exists()) {
            serveFile = requestedFile;
        } else {
            sendError(exchange, 404, "File not found");
            return;
        }

        // Determine content type based on extension
        String ext = path.substring(path.lastIndexOf('.') + 1);
        String contentType = "application/octet-stream"; // default
        switch (ext.toLowerCase()) {
            case "html" -> contentType = "text/html";
            case "css" -> contentType = "text/css";
            case "js", "mjs" -> contentType = "text/javascript";
            case "ico" -> contentType = "image/x-icon";
            case "png" -> contentType = "image/png";
            case "jpg", "jpeg" -> contentType = "image/jpeg";
            case "svg" -> contentType = "image/svg+xml";
            case "json" -> contentType = "application/json";
            default -> { /* keep default */ }
        }

        try {
            // Read the file content
            byte[] content;
            if (serveFile == gzFile) {
                // Decompress gz file
                try (InputStream fis = new FileInputStream(gzFile);
                     GZIPInputStream gis = new GZIPInputStream(fis)) {

                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = gis.read(buffer)) != -1) {
                        baos.write(buffer, 0, read);
                    }
                    content = baos.toByteArray();
                }
            } else {
                content = Files.readAllBytes(serveFile.toPath());
            }

            // Send the response
            exchange.getResponseHeaders().set("Content-Type", contentType);
            exchange.sendResponseHeaders(200, content.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(content);
            }
        } catch (IOException e) {
            sendError(exchange, 500, "Error reading file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void sendError(IHttpExchange exchange, int code, String message) {
        try {
            var errorResponse = String.format("{\"error\": \"%s\"}", message);
            exchange.sendResponseHeaders(code, errorResponse.length());
            var os = exchange.getResponseBody();
            os.write(errorResponse.getBytes());
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
