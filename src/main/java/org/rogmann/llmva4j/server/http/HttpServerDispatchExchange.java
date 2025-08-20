package org.rogmann.llmva4j.server.http;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Represents a single HTTP request/response exchange.
 * Provides access to the underlying socket for WebSocket upgrades.
 */
public class HttpServerDispatchExchange {
    private static final Logger LOGGER = Logger.getLogger(HttpServerDispatchExchange.class.getName());
    
    private final Socket socket;
    private final BufferedInputStream inputStream;
    private final BufferedOutputStream outputStream;
    private final String method;
    private final String uri;
    private final String protocol;
    private final Map<String, String> requestHeaders;
    private final boolean keepAlive;
    
    private final Map<String, String> responseHeaders = new HashMap<>();
    private int responseStatusCode = 200;
    private boolean responseHeadersSent = false;
    private boolean upgradeRequested = false;

	private boolean isResponseChunked;
    
    /**
     * Creates a new HTTP exchange.
     * 
     * @param socket the underlying socket
     * @param inputStream the input stream
     * @param outputStream the output stream
     * @param method the HTTP method
     * @param uri the request URI
     * @param protocol the HTTP protocol version
     * @param requestHeaders the request headers
     * @param keepAlive whether connection should be kept alive
     */
    public HttpServerDispatchExchange(
            Socket socket,
            BufferedInputStream inputStream,
            BufferedOutputStream outputStream,
            String method,
            String uri,
            String protocol,
            Map<String, String> requestHeaders,
            boolean keepAlive) {
    	if (LOGGER.isLoggable(Level.FINE)) {
    		LOGGER.fine(String.format("method %s, uri %s, socket %s", method, uri, socket));
    		LOGGER.finer("HTTP-headers: " + requestHeaders);
    	}
        this.socket = socket;
        this.inputStream = inputStream;
        this.outputStream = outputStream;
        this.method = method;
        this.uri = uri;
        this.protocol = protocol;
        this.requestHeaders = Collections.unmodifiableMap(new HashMap<>(requestHeaders));
        this.keepAlive = keepAlive;
    }
    
    /**
     * Gets the HTTP method.
     * 
     * @return the method (GET, POST, etc.)
     */
    public String getRequestMethod() {
        return method;
    }
    
    /**
     * Gets the request URI.
     * 
     * @return the request URI
     */
    public String getRequestURI() {
        return uri;
    }
    
    /**
     * Gets the request headers.
     * 
     * @return the request headers
     */
    public Map<String, String> getRequestHeaders() {
        return requestHeaders;
    }
    
    /**
     * Gets the request body as an InputStream.
     * 
     * @return the request body input stream
     */
    public InputStream getRequestBody() {
        return new HttpInputStream(inputStream, requestHeaders);
    }
    
    /**
     * Gets the response headers.
     * 
     * @return the response headers
     */
    public Map<String, String> getResponseHeaders() {
        return responseHeaders;
    }
    
    /**
     * Gets the response body as an OutputStream.
     * 
     * @return the response body output stream
     * @throws IOException if response headers haven't been sent yet
     */
    public OutputStream getResponseBody() throws IOException {
        if (!responseHeadersSent) {
            throw new IllegalStateException("Response headers not sent yet");
        }
        return new HttpOutputStream(outputStream, isResponseChunked);
    }
    
    /**
     * Sends the response headers.
     * 
     * @param statusCode the HTTP status code
     * @param contentLength the content length, or -1 for chunked encoding
     * @throws IOException if an I/O error occurs
     */
    public void sendResponseHeaders(int statusCode, long contentLength) throws IOException {
        if (responseHeadersSent) {
            throw new IllegalStateException("Response headers already sent");
        }
        
        this.responseStatusCode = statusCode;
        
        StringBuilder responseLine = new StringBuilder();
        responseLine.append(protocol).append(" ").append(statusCode).append(" ");
        
        // Add status text
        switch (statusCode) {
            case 200: responseLine.append("OK"); break;
            case 400: responseLine.append("Bad Request"); break;
            case 404: responseLine.append("Not Found"); break;
            case 500: responseLine.append("Internal Server Error"); break;
            default: responseLine.append("Unknown"); break;
        }
        responseLine.append("\r\n");
        
        // Add standard headers
        if (!responseHeaders.containsKey("Connection")) {
            responseHeaders.put("Connection", keepAlive ? "keep-alive" : "close");
        }
        
        if (contentLength > 0) {
            responseHeaders.put("Content-Length", String.valueOf(contentLength));
        } else if (statusCode != 204 && statusCode != 304) {
            responseHeaders.put("Transfer-Encoding", "chunked");
            isResponseChunked = true;
        }
        
        // Write response line and headers
        outputStream.write(responseLine.toString().getBytes());
        for (Map.Entry<String, String> header : responseHeaders.entrySet()) {
            outputStream.write((header.getKey() + ": " + header.getValue() + "\r\n").getBytes());
        }
        outputStream.write("\r\n".getBytes());
        outputStream.flush();
        
        responseHeadersSent = true;
    }
    
    /**
     * Gets the underlying socket for WebSocket upgrades.
     * 
     * @return the socket
     */
    public Socket getSocket() {
        return socket;
    }
    
    /**
     * Checks if a WebSocket upgrade was requested.
     * 
     * @return true if upgrade requested
     */
    public boolean isUpgradeRequested() {
        return upgradeRequested;
    }
    
    /**
     * Requests a protocol upgrade (e.g., to WebSocket).
     * This should be called before sending response headers.
     */
    public void requestUpgrade() {
        upgradeRequested = true;
        responseHeaders.put("Connection", "Upgrade");
        responseHeaders.put("Upgrade", "websocket");
    }
}