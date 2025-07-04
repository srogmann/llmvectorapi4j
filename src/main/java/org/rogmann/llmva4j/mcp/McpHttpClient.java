package org.rogmann.llmva4j.mcp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

import org.rogmann.llmva4j.LightweightJsonHandler;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcError;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcRequest;
import org.rogmann.llmva4j.mcp.JsonRpc.JsonRpcResponse;

/**
 * Implementation of a simple MCP-client, see https://modelcontextprotocol.io/.
 */
public class McpHttpClient {
    /** logger */
    private static final Logger LOG = Logger.getLogger(McpHttpClient.class.getName());

	/** maximum number of remote tools to be read */
	private static final int MAX_NUM_TOOLS = Integer.getInteger("mcp.client.maxNumTools", 1000);
	
	/** maximum number of requests to enumerate remote tools */
	private static final int MAX_NUM_REQS = Integer.getInteger("mcp.client.maxNumReqs", 10);

    private ConcurrentMap<String, McpToolWithUri> mapTools = new ConcurrentHashMap<>();
    
    /**
     * Interface of a MCP tool with an url of its server and an optional API key.
     * @param tool MCP tool
     * @param url url of the corresponding server
     */
    public record McpToolWithUri(McpToolInterface tool, URL url) { }
    
    /**
     * Create a tools map containing the interfaces of tools known to this client.
     * @return map with key "tools" and an array of tool-definitions as value
     */
    public Map<String, Object> createToolMap() {
    	Map<String, Object> map = new LinkedHashMap<>();
    	List<Map<String, Object>> listTools = new ArrayList<>();
    	mapTools.forEach((name, toolWithUri) -> {
    		McpToolInterface tool = toolWithUri.tool();
    		Map<String, Object> mapTool = new LinkedHashMap<>();
    		mapTool.put("type", "function");
    		Map<String, Object> mapFunction = new LinkedHashMap<>();
    		mapFunction.put("name", name);
    		mapFunction.put("description", tool.description());
    		Map<String, Object> mapParameters = new LinkedHashMap<>();
    		mapParameters.put("type", "object");
    		Map<String, Object> mapProps = new LinkedHashMap<>();
    		tool.inputSchema().properties().forEach((propName, propDesc) -> {
    			Map<String, Object> mapParam = new LinkedHashMap<>();
    			mapParam.put("type", propDesc.type());
    			mapParam.put("description", propDesc.description());
                if (propDesc.itemsType() != null) {
                    var mapItems = new LinkedHashMap<>();
                    mapItems.put("type", propDesc.itemsType());
                    mapParam.put("items", mapItems);
                }
    			mapProps.put(propName, mapParam);
    		});
    		mapParameters.put("properties", mapProps);
    		mapParameters.put("required", Arrays.asList(tool.inputSchema().required()));
    		mapTool.put("function", mapFunction);
    		listTools.add(mapTool);
    	});
    	map.put("tools", listTools);
		return map;
    }
    
    public List<McpToolInterface> getTools() {
        return mapTools.values().stream().map(toolWithIntf -> toolWithIntf.tool()).toList();
    }

    /**
     * Registers a tool.
     * @param toolWithUrl tool to be used in the client and its URL
     */
    public void registerTool(McpToolWithUri toolWithUrl) {
    	mapTools.put(toolWithUrl.tool().name(), toolWithUrl);
    }
    
    public List<McpToolWithUri> listTools(final URL url) {
    	List<McpToolWithUri> tools = new ArrayList<>();
    	int numRequests = 0;
    	String nextCursor = null;
    	do {
    	    numRequests++;
    	    if (numRequests > MAX_NUM_REQS) {
    	        throw new RuntimeException(String.format("Invalid number of requests (%d), current cursor '%s', sent by %s",
    	                numRequests, nextCursor, url));
    	    }
            // Build request
    		String id = UUID.randomUUID().toString();
    		Map<String, Object> params = new LinkedHashMap<>();
    		if (nextCursor != null) {
    			params.put("cursor", nextCursor);
    		}
			JsonRpcRequest request = new JsonRpcRequest("tools/list", params, id);
            JsonRpcResponse response = sendJsonRpcRequest(url, request);
            LOG.info(String.format("JSON-Response: %s", response));
    		
            JsonRpcError error = response.error();
            if (error != null) {
                throw new RuntimeException(String.format("JSON-RPC error %d - '%s' from %s", error.number(), error.message(), url));
            }
            Map<String, Object> result = response.result();
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> listTools = (List<Map<String, Object>>) LightweightJsonHandler.getJsonValue(result, "tools", List.class);
            for (Map<String, Object> mapTool : listTools) {
                String name = LightweightJsonHandler.getJsonValue(mapTool, "name", String.class);
                String title = LightweightJsonHandler.getJsonValue(mapTool, "title", String.class);
                String description = LightweightJsonHandler.getJsonValue(mapTool, "description", String.class);
                if (listTools.size() > MAX_NUM_TOOLS) {
                    throw new RuntimeException(String.format("Invalid number (%d) of tools, current tool '%s', sent by %s",
                            listTools.size(), name, url));
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> mapInputSchema = LightweightJsonHandler.getJsonValue(mapTool, "inputSchema", Map.class);
                String type = LightweightJsonHandler.getJsonValue(mapInputSchema, "type", String.class);
                @SuppressWarnings("unchecked")
                Map<String, Map<String, Object>> mapProperties = LightweightJsonHandler.getJsonValue(mapInputSchema, "properties", Map.class);
                @SuppressWarnings("unchecked")
                List<String> required = LightweightJsonHandler.getJsonValue(mapInputSchema, "required", List.class);
                Map<String, McpToolPropertyDescription> properties = new LinkedHashMap<>();
                if (mapProperties != null) {
                    mapProperties.forEach((k, v) -> {
                        String propType = LightweightJsonHandler.getJsonValue(v, "type", String.class);
                        String propDesc = LightweightJsonHandler.getJsonValue(v, "description", String.class);
                        properties.put(k, new McpToolPropertyDescription(propType, propDesc));
                    });
                }
                McpToolInputSchema inputSchema = new McpToolInputSchema(type, properties, required);
                McpToolInterface tool = new McpToolInterface(name, title, description, inputSchema);
                tools.add(new McpToolWithUri(tool, url));
            }
            nextCursor = LightweightJsonHandler.getJsonValue(result, "cursor", String.class);
    	} while (nextCursor != null);	
		return tools;
    }
    
    public Map<String, Object> callTool(String functionName, Map<String, Object> arguments, String id) {
        McpToolWithUri mcpToolWithUri = mapTools.get(functionName);
        if (mcpToolWithUri == null) {
            throw new RuntimeException("Unknown tool: " + functionName);
        }
        LOG.info(String.format("Call %s with: %s", functionName, arguments));
        Map<String, Object> mapParams = new LinkedHashMap<>(2);
        mapParams.put("name", functionName);
        mapParams.put("arguments", arguments);
        JsonRpcRequest rpcRequest = new JsonRpcRequest("tools/call", mapParams, id);
        LOG.fine(String.format("rpc-request: %s", rpcRequest));
        JsonRpcResponse rpcResponse = sendJsonRpcRequest(mcpToolWithUri.url(), rpcRequest);
        return rpcResponse.result();
    }

    public static JsonRpcResponse sendJsonRpcRequest(final URL url, JsonRpcRequest request) {
        String sRequest = JsonRpc.serializeToJson(request);

        HttpURLConnection conn;
        try {
        	conn = (HttpURLConnection) url.openConnection();
        	conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
        	conn.addRequestProperty("Accept", "application/json");
        	conn.addRequestProperty("Accept", "text/event-stream");
        	conn.setDoInput(true);
        	conn.setDoOutput(true);
        } catch (IOException e) {
        	throw new RuntimeException("IO-exception while trying connect to " + url, e);
        }
        
        try (OutputStream os = conn.getOutputStream();
                OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8)) {
            osw.write(sRequest);
        } catch (IOException e) {
        	throw new RuntimeException("IO-exception while sending request to " + url, e);
        }
        
        JsonRpcResponse response;
        try (var is = conn.getInputStream();
        		var isr = new InputStreamReader(is, StandardCharsets.UTF_8);
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
           response = JsonRpc.parseJsonResponse(jsonBody.toString(), request.id());
        } catch (IOException e) {
        	throw new RuntimeException("IO-exception while reading data from " + url, e);
        }
        return response;
    }
    
    /**
     * Sample entry used to list tools of a remote MCP server.
     * @param args MCP-server-URL
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException("Usage: <URL_MCP_SERVER>");
        }
        URL url;
        try {
            url = new URI(args[0]).toURL();
        } catch (MalformedURLException | URISyntaxException e) {
            throw new IllegalArgumentException("Invalid URL", e);
        }
        McpHttpClient mcpHttpClient = new McpHttpClient();
        List<McpToolWithUri> tools = mcpHttpClient.listTools(url);
        tools.forEach(tool -> LOG.info(String.format("Tool %s at %s%n", tool.tool().name(), tool.url())));
    }

}
