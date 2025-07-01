package org.rogmann.llmva4j.mcp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

import org.rogmann.llmva4j.LightweightJsonHandler;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcError;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcRequest;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcResponse;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Implementation of a simple MCP-server, see https://modelcontextprotocol.io/.
 */
public class McpHttpServer {
    /** logger */
    private static final Logger LOG = Logger.getLogger(McpHttpServer.class.getName());

    private static final String SERVER_VERSION = "0.1.0";
    
    private ConcurrentMap<String, McpToolImplementation> mapTools = new ConcurrentHashMap<>();

    private void startServer(String host, int port) {
        try {
            var addr = new java.net.InetSocketAddress(host, port);
            var server = HttpServer.create(addr, 0);
            server.createContext("/", exchange -> handleRequest(exchange));
            server.setExecutor(null); // use default executor
            server.start();
            System.out.println("Server started on " + host + ":" + port);
        } catch (IOException e) {
            throw new RuntimeException("IO-error while starting server", e);
        }
    }

    private void handleRequest(HttpExchange exchange) {
        LOG.info(String.format("%s %s request %s", LocalDateTime.now(), exchange.getRequestMethod(), exchange.getRequestURI()));
        if ("GET".equals(exchange.getRequestMethod())) {
            processGetRequest(exchange);
            return;
        }

        if (!"POST".equals(exchange.getRequestMethod())) {
            sendError(exchange, 405, -32601, "Method not allowed", null);
            return;
        }

        try (var isr = new InputStreamReader(exchange.getRequestBody());
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

            // Parse incoming JSON
            JsonRpcRequest jsonRpcRequest = JsonRpc.parseJsonRequest(jsonBody.toString());
            LOG.info(String.format("JSON-Request: %s", jsonRpcRequest));
            
            JsonRpcResponse response = switch (jsonRpcRequest.method()) {
                case "initialize" -> initializeConnection(jsonRpcRequest.params(), jsonRpcRequest.id());
                case "notifications/initialized" -> new JsonRpcResponse(null, null, jsonRpcRequest.id());
                case "tools/list" -> createToolsList(jsonRpcRequest.params(), jsonRpcRequest.id());
                case "tools/call" -> callTool(jsonRpcRequest.params(), jsonRpcRequest.id());
                default -> new JsonRpcResponse(null, new JsonRpcError(32000, "Invalid method", null), jsonRpcRequest.id());
            };

            // Build request for LLM
            var llmRequest = new HashMap<String, Object>();
            
            var sbRequest = new StringBuilder(200);
            LightweightJsonHandler.dumpJson(sbRequest, llmRequest);

            // Send request to LLM
            try {
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, 0);
                try (OutputStream os = exchange.getResponseBody();
                        OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8);
                        BufferedWriter bw = new BufferedWriter(osw)) {
                    String sResponse = JsonRpc.serializeToJson(response);
                    LOG.info(String.format("Response: %s", sResponse));
                    bw.write(sResponse);
                }
            } catch (IOException e) {
                System.err.format("%s IO-error", LocalDateTime.now(), e.getMessage());
                e.printStackTrace();
            }

        } catch (Exception e) {
            e.printStackTrace();
            sendError(exchange, 500, -32603, "Internal server error: " + e.getMessage(), null);
        }
    }

    private JsonRpcResponse initializeConnection(Map<String, Object> params, Object id) {
        LOG.fine(String.format("params: %s, id: %s", params, id));
        @SuppressWarnings("unchecked")
        Map<String, Object> clientInfo = (Map<String, Object>) params.get("clientInfo");
        if (clientInfo != null) {
            LOG.info(String.format("initialize: Client '%s', Version '%s', id %s", clientInfo.get("name"), clientInfo.get("version"), id));
        }
        
        // See https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        result.put("protocolVersion", "2025-03-26");
        Map<String, Object> capabilities = new LinkedHashMap<String, Object>();
        result.put("capabilities", capabilities);
        capabilities.put("resources", new LinkedHashMap<String, Object>());
        Map<String, Object> tools = new LinkedHashMap<String, Object>();
        tools.put("listChanged", Boolean.FALSE);
        capabilities.put("tools", tools);
        Map<String, Object> serverInfo = new LinkedHashMap<>();
        serverInfo.put("name", McpHttpServer.class.getName());
        serverInfo.put("version", SERVER_VERSION);
        result.put("serverInfo", serverInfo);
        return new JsonRpcResponse(result, null, id);
    }

    private JsonRpcResponse createToolsList(Map<String, Object> params, Object id) {
        LOG.fine(String.format("listTools: params: %s, id: %s", params, id));
        
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        List<Map<String, Object>> listTools = new ArrayList<>();
        mapTools.forEach((name, toolImpl) -> {
        	McpToolInterface tool = toolImpl.getTool();
            Map<String, Object> mapTool = new LinkedHashMap<>();
            mapTool.put("name", name);
            mapTool.put("description", tool.description());
            Map<String, Object> mapInputSchema = new LinkedHashMap<>();
            mapInputSchema.put("type", tool.inputSchema().type());
            Map<String, Object> properties = new LinkedHashMap<>();
            tool.inputSchema().properties().forEach((key, prop) -> {
                Map<String, Object> mapProp = new LinkedHashMap<>();
                mapProp.put("type", prop.type());
                mapProp.put("description", prop.description());
                if (prop.itemsType() != null) {
                    var mapItems = new LinkedHashMap<>();
                    mapItems.put("type", prop.itemsType());
                    mapProp.put("items", mapItems);
                }
                properties.put(key, mapProp);
            });
            mapInputSchema.put("required", tool.inputSchema().required());
            mapTool.put("inputSchema", mapInputSchema);
            listTools.add(mapTool);
        });
        result.put("tools", listTools);
        return new JsonRpcResponse(result, null, id);
    }

    private JsonRpcResponse callTool(Map<String, Object> params, Object id) {
        // https://modelcontextprotocol.io/specification/2025-06-18/server/tools
        LOG.fine(String.format("callTool: params: %s, id: %s", params, id));

        String name = (String) params.get("name");
        if (name == null) {
            return new JsonRpcResponse(null, new JsonRpcError(32000, "name missing", null), id);
        }
        McpToolImplementation toolImpl = mapTools.get(name);
        if (toolImpl == null) {
        	LOG.severe(String.format("Unknown tool: %s", name));
        	return new JsonRpcResponse(null, new JsonRpcError(32000, "unknown tool", null), id);
        }
        List<Map<String, Object>> contentValue = toolImpl.call(params);
        
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        result.put("content", contentValue);
        result.put("isError", Boolean.FALSE);
        return new JsonRpcResponse(result, null, id);
    }

    private static void processGetRequest(HttpExchange exchange) {
        String path = exchange.getRequestURI().getPath();
        LOG.severe(String.format("GET-request: %s", path));

        try {
            var errorResponse = "<html><head><title>MCP server</title></head>"
                    + "<body><h1>MCP server</h1><p>Use a MCP client</p></body></html>";
            exchange.sendResponseHeaders(400, errorResponse.length());
            var os = exchange.getResponseBody();
            os.write(errorResponse.getBytes());
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void sendError(HttpExchange exchange, int code, int rpcCode, String message, Object id) {
        LOG.severe(String.format("Error: %d - %s", rpcCode, message));
        try {
            Map<String, Object> mapResponse = new LinkedHashMap<>(3);
            mapResponse.put("jsonrpc", "2.0");
            Map<String, Object> mapError = new LinkedHashMap<>(2);
            mapError.put("code", Integer.valueOf(rpcCode));
            mapError.put("message", message);
            mapResponse.put("error", mapError);
            mapResponse.put("id", id);
            var errorResponse = serializeToJson(mapResponse);
            exchange.setAttribute("Content-Type", "application/json");
            exchange.sendResponseHeaders(code, errorResponse.length());
            var os = exchange.getResponseBody();
            os.write(errorResponse.getBytes());
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String serializeToJson(Map<String, Object> map) {
        var sb = new StringBuilder(100);
        LightweightJsonHandler.dumpJson(sb, map);
        return sb.toString();
    }

    /**
     * Start MCP server.
     * @param args &lt;server-ip&gt; &lt;server-port&gt;
     */
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: <server-ip> <server-port>");
            System.exit(1);
        }

        String host = args[0];
        int port = Integer.parseInt(args[1]);

        McpHttpServer mcpServer = new McpHttpServer();

        ServiceLoader<McpToolImplementations> loaderTools = ServiceLoader.load(McpToolImplementations.class);
        for (McpToolImplementations listSupplier : loaderTools) {
            for (McpToolImplementation action : listSupplier.get()) {
                String name = action.getName();
                LOG.info(String.format("Register tool implementation: %s in %s", name, action));
                mcpServer.mapTools.put(name, action);
            }
        }

        mcpServer.startServer(host, port);
    }

}
