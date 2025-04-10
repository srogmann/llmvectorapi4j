package org.rogmann.llmva4j;

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
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Simple webserver for providing a web interface for an OpenAI-chat/completions-compatible LLM-API endpoint.
 */
public class UiServer {
    public static void main(String[] args) {
        if (args.length != kNumArgs) {
            System.out.println("Usage: java UiServer <server-ip> <server-port> <llm-url> <public-path>");
            return;
        }

        String host = args[0];
        int port = Integer.parseInt(args[1]);
        String llmUrl = args[2];
        String publicPath = args[3];

        startServer(host, port, llmUrl, publicPath);
    }

    private static final int kNumArgs = 4;

    private static void startServer(String host, int port, String llmUrl, String publicPath) {
        try {
            var addr = new java.net.InetSocketAddress(host, port);
            var server = HttpServer.create(addr, 0);
            server.createContext("/", exchange -> handleRequest(exchange, llmUrl, publicPath));
            server.setExecutor(null); // use default executor
            server.start();
            System.out.println("Server started on " + host + ":" + port);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void handleRequest(HttpExchange exchange, String llmUrl, String publicPath) {
        System.out.format("%s %s request %s%n", LocalDateTime.now(), exchange.getRequestMethod(), exchange.getRequestURI());
        if ("GET".equals(exchange.getRequestMethod())) {
            processGetRequest(exchange, publicPath);
            return;
        }

        if (!"POST".equals(exchange.getRequestMethod())) {
            sendError(exchange, 405, "Method not allowed");
            return;
        }

        try {
            var br = new java.io.BufferedReader(new java.io.InputStreamReader(exchange.getRequestBody()));
            var jsonBody = new StringBuilder();
            while (true) {
                String line = br.readLine();
                if (line == null) {
                    break;
                }
                System.out.println("< " + line);
                jsonBody.append(line);
            }

            // Parse incoming JSON
            var requestMap = LightweightJsonHandler.parseJsonDict(jsonBody.toString());
            var messages = getJsonArrayDicts(requestMap, "messages");

            // Build request for LLM
            var llmRequest = new HashMap<String, Object>();
            llmRequest.put("messages", messages);
            llmRequest.put("temperature", LightweightJsonHandler.readFloat(requestMap, "temperature", 0.7f));
            llmRequest.put("max_tokens", LightweightJsonHandler.readInt(requestMap, "max_tokens", 100));
            llmRequest.put("top_p", LightweightJsonHandler.readFloat(requestMap, "top_p", 1.0f));
            llmRequest.put("stream", LightweightJsonHandler.readBoolean(requestMap, "stream", false));
            
            var sbRequest = new StringBuilder(200);
            LightweightJsonHandler.dumpJson(sbRequest, llmRequest);

            // Send request to LLM
            URL url = new URI(llmUrl).toURL();
            try {
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json");
                connection.setRequestProperty("Accept", "application/json");
                connection.setDoOutput(true);

                try (OutputStream os = connection.getOutputStream();
                        OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8)) {
                    osw.write(sbRequest.toString());
                }

                int responseCode = connection.getResponseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    System.err.format("%s HTTP-error accessing %s: %s - %s%n", LocalDateTime.now(),
                            url, responseCode, connection.getResponseMessage());
                    return;
                }

                // Read the response in chunks
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
                            System.out.println("> " + line);
                            bw.write(line); bw.write((char) '\n');
                            bw.flush();
                        }
                    }
                }
            } catch (IOException e) {
                System.err.format("%s IO-error", LocalDateTime.now(), e.getMessage());
                e.printStackTrace();
            }

        } catch (Exception e) {
            sendError(exchange, 500, "Internal server error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void processGetRequest(HttpExchange exchange, String publicPath) {
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

    private static void sendError(HttpExchange exchange, int code, String message) {
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

    // Utility methods from original code
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> getJsonArrayDicts(Map<String, Object> map, String key) {
        var list = (List<?>) map.get(key);
        if (list == null) return null;
        return list.stream()
                .filter(o -> o instanceof Map)
                .map(o -> (Map<String, Object>) o)
                .collect(Collectors.toList());
    }

    // LightweightJsonHandler extension
    public static String serializeToJson(Map<String, Object> map) {
        var sb = new StringBuilder();
        LightweightJsonHandler.dumpJson(sb, map);
        return sb.toString();
    }
}
