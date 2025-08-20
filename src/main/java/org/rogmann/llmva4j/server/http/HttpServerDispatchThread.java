package org.rogmann.llmva4j.server.http;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Thread handling HTTP communication for a single client connection.
 * Supports persistent connections (HTTP/1.1 keep-alive).
 */
public class HttpServerDispatchThread implements Runnable {
    private static final Logger LOGGER = Logger.getLogger(HttpServerDispatchThread.class.getName());
    
    private final Socket socket;
    private final HttpHandler handler;
    private BufferedInputStream inputStream;
    private BufferedOutputStream outputStream;
    
    /**
     * Creates a new HTTP dispatch thread.
     * 
     * @param socket the client socket
     * @param handler the request handler
     */
    public HttpServerDispatchThread(Socket socket, HttpHandler handler) {
    	if (LOGGER.isLoggable(Level.FINE)) {
    		LOGGER.fine(String.format("thread %s, socket %s", Thread.currentThread(), socket));
    	}
        this.socket = socket;
        this.handler = handler;
    }
    
    @Override
    public void run() {
        try {
            inputStream = new BufferedInputStream(socket.getInputStream());
            outputStream = new BufferedOutputStream(socket.getOutputStream());
            
            // Handle requests in loop for persistent connections
            boolean keepAlive = true;
            while (keepAlive && !socket.isClosed() && socket.isConnected()) {
                keepAlive = handleRequest();
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error handling HTTP request", e);
        } finally {
            closeConnection();
        }
    }
    
    /**
     * Handles a single HTTP request.
     * 
     * @return true if connection should be kept alive, false otherwise
     * @throws IOException if an I/O error occurs
     */
    private boolean handleRequest() throws IOException {
        // Parse HTTP request line
        String requestLine = readLine(inputStream);
        if (requestLine == null || requestLine.isEmpty()) {
            return false; // Connection closed or invalid
        }
        
        String[] parts = requestLine.split(" ", 3);
        if (parts.length != 3) {
        	LOGGER.warning("Invalid request line: " + requestLine);
            sendBadRequest();
            return false;
        }
        
        String method = parts[0];
        String uri = parts[1];
        String protocol = parts[2];
        
        // Parse headers
        Map<String, String> headers = parseHeaders(inputStream);
        
        // Check for connection keep-alive
        String connectionHeader = headers.get("Connection");
        boolean keepAlive = "HTTP/1.1".equals(protocol) && 
                           !"close".equalsIgnoreCase(connectionHeader);
        
        // Create exchange object
        HttpServerDispatchExchange exchange = new HttpServerDispatchExchange(
            socket, inputStream, outputStream, method, uri, protocol, headers, keepAlive);
        
        try {
            if (handler != null) {
                handler.handle(exchange);
            } else {
                exchange.sendResponseHeaders(404, -1);
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in handler", e);
            try {
                exchange.sendResponseHeaders(500, -1);
            } catch (IOException ioException) {
                // Ignore, connection might be closed
            }
        }
        
        // Ensure response is sent
        try {
            outputStream.flush();
        } catch (IOException e) {
            // Connection might be closed
            return false;
        }
        
        return keepAlive && !exchange.isUpgradeRequested();
    }
    
    /**
     * Reads a line from the input stream (CRLF terminated).
     * 
     * @param inputStream the input stream
     * @return the line or null if end of stream
     * @throws IOException if an I/O error occurs
     */
    private String readLine(BufferedInputStream inputStream) throws IOException {
        StringBuilder sb = new StringBuilder();
        int b;
        while ((b = inputStream.read()) != -1) {
            if (b == '\r') {
                // Look ahead for \n
                inputStream.mark(1);
                int next = inputStream.read();
                if (next == '\n') {
                    break; // CRLF found
                } else {
                    inputStream.reset(); // Put back the byte
                    sb.append((char) b);
                }
            } else {
                sb.append((char) b);
            }
        }
        return b == -1 && sb.length() == 0 ? null : sb.toString();
    }
    
    /**
     * Parses HTTP headers from the input stream.
     * 
     * @param inputStream the input stream
     * @return map of headers
     * @throws IOException if an I/O error occurs
     */
    private Map<String, String> parseHeaders(BufferedInputStream inputStream) throws IOException {
        Map<String, String> headers = new HashMap<>();
        while (true) {
        	String line = readLine(inputStream);
        	if (line == null || line.isEmpty()) {
        		break;
        	}
            int colonIndex = line.indexOf(':');
            if (colonIndex > 0) {
                String name = line.substring(0, colonIndex).trim();
                String value = line.substring(colonIndex + 1).trim();
                headers.put(name, value);
            }
        }
        return headers;
    }
    
    private void sendBadRequest() throws IOException {
        String response = "HTTP/1.1 400 Bad Request\r\n" +
                         "Connection: close\r\n" +
                         "Content-Length: 0\r\n" +
                         "\r\n";
        outputStream.write(response.getBytes());
        outputStream.flush();
    }
    
    private void closeConnection() {
        try {
            if (inputStream != null) {
                inputStream.close();
            }
            if (outputStream != null) {
                outputStream.close();
            }
            if (socket != null && !socket.isClosed()) {
                socket.close();
            }
        } catch (IOException e) {
            // Ignore
        }
    }
}