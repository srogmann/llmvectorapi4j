package org.rogmann.llmva4j.mcp;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import org.rogmann.llmva4j.LightweightJsonHandler;

/**
 * Internal utility class to parse or create <a href="https://www.jsonrpc.org/specification">JSON-RPC 2.0</a> messages.
 */
public class JsonRpc {
    private static final String RPC_VERSION = "2.0";

    /** JSON RPC 2.0 request */
    public record JsonRpcRequest(String method, Map<String, Object> params, Object id) { }
    
    public record JsonRpcResponse(Map<String, Object> result, JsonRpcError error, Object id) { }
    
    public record JsonRpcError(int number, String message, Map<String, Object> data) { }

    static JsonRpcRequest parseJsonRequest(String json) throws IOException {
        var requestMap = LightweightJsonHandler.parseJsonDict(json);
        String jsonrpc = LightweightJsonHandler.getJsonValue(requestMap, "jsonrpc", String.class);
        if (!RPC_VERSION.equals(jsonrpc)) {
        	throw new RuntimeException("Unexpected RPC-version: " + jsonrpc);
        }
        String method = LightweightJsonHandler.getJsonValue(requestMap, "method", String.class);
        @SuppressWarnings("unchecked")
        Map<String, Object> params = LightweightJsonHandler.getJsonValue(requestMap, "params", Map.class);
        Object id = requestMap.get("id");
        return new JsonRpcRequest(method, params, id);
    }

    static JsonRpcResponse parseJsonResponse(String json, Object idExcepted) throws IOException {
        var responseMap = LightweightJsonHandler.parseJsonDict(json);
        String jsonrpc = LightweightJsonHandler.getJsonValue(responseMap, "jsonrpc", String.class);
        if (!RPC_VERSION.equals(jsonrpc)) {
        	throw new RuntimeException("Unexpected RPC-version: " + jsonrpc);
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> result = LightweightJsonHandler.getJsonValue(responseMap, "result", Map.class);
        @SuppressWarnings("unchecked")
        Map<String, Object> mapError = LightweightJsonHandler.getJsonValue(responseMap, "error", Map.class);
        JsonRpcError error = null;
        if (mapError != null) {
            Integer code = LightweightJsonHandler.getJsonValue(mapError, "code", Integer.class);
            String message = LightweightJsonHandler.getJsonValue(mapError, "message", String.class);
            if (code == null) {
                throw new RuntimeException(String.format("Missing code in error object with message '%s'", message));
            }
            // TODO data may be a primitive.
            @SuppressWarnings("unchecked")
            Map<String, Object> data = LightweightJsonHandler.getJsonValue(mapError, "data", Map.class);
        	error = new JsonRpcError(code.intValue(), message, data);
        }
        Object id = responseMap.get("id");
        if (error == null && id == null) {
        	throw new RuntimeException(String.format("Missing, expected id '%s'",  idExcepted));
        }
        if (id != null && !idExcepted.equals(id)) {
        	throw new RuntimeException(String.format("Unexpected id, expected '%s', actual '%s'", id, idExcepted));
        }
        return new JsonRpcResponse(result, error, id);
    }

    static String serializeToJson(JsonRpcRequest request) {
        Map<String, Object> mapResponse = new LinkedHashMap<>(3);
        mapResponse.put("jsonrpc", RPC_VERSION);
        mapResponse.put("method", request.method());
        Map<String, Object> params = request.params();
        if (params != null) {
        	mapResponse.put("params", params);
        }
        mapResponse.put("id", request.id());
        return serializeToJson(mapResponse);
    }

    static String serializeToJson(JsonRpcResponse response) {
        Map<String, Object> mapResponse = new LinkedHashMap<>(3);
        mapResponse.put("jsonrpc", RPC_VERSION);
        mapResponse.put("result", response.result);
        if (response.error != null) {
            Map<String, Object> mapError = new LinkedHashMap<>(2);
            mapError.put("code", Integer.valueOf(response.error.number));
            mapError.put("message", response.error.message);
            mapResponse.put("error", mapError);
        }
        mapResponse.put("id", response.id);
        return serializeToJson(mapResponse);
    }

    private static String serializeToJson(Map<String, Object> map) {
        var sb = new StringBuilder(100);
        LightweightJsonHandler.dumpJson(sb, map);
        return sb.toString();
    }
}
