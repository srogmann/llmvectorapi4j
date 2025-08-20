package org.rogmann.llmva4j.server.http;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Custom HTTP server implementation supporting HTTP/1.1 with chunked transfer encoding.
 * Supports WebSocket upgrade for future use.
 */
public class HttpServerDispatch {
    private static final Logger LOGGER = Logger.getLogger(HttpServerDispatch.class.getName());
    
    private final InetSocketAddress address;
    private final HttpHandler handler;
    private ServerSocket serverSocket;
    private Executor executor;
    private AtomicBoolean running = new AtomicBoolean(false);
    
    /**
     * Creates a new HTTP server dispatch.
     * 
     * @param address the address to bind to
     * @param handler the request handler
     */
    public HttpServerDispatch(InetSocketAddress address, HttpHandler handler) {
        this.address = address;
        this.handler = handler;
        this.executor = Executors.newCachedThreadPool();
    }
    
    /**
     * Sets the executor for handling requests.
     * 
     * @param executor the executor to use, or null for default
     */
    public void setExecutor(Executor executor) {
        this.executor = executor != null ? executor : Executors.newCachedThreadPool();
    }
    
    /**
     * Starts the HTTP server.
     * 
     * @throws IOException if an I/O error occurs
     */
    public void start() throws IOException {
        if (running.get()) {
            throw new IllegalStateException("Server already running");
        }
        
        serverSocket = new ServerSocket();
        serverSocket.bind(address);
        running.set(true);
        
        LOGGER.info("HttpServerDispatch started on " + address);
        
        // Accept connections in background thread
        new Thread(this::acceptConnections, "HttpServerDispatch-Acceptor").start();
    }
    
    /**
     * Stops the HTTP server.
     */
    public void stop(int delay) {
        running.set(false);
        try {
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
            }
        } catch (IOException e) {
        	LOGGER.log(Level.WARNING, "Error closing server socket", e);
        }
    }
    
    private void acceptConnections() {
        while (running.get()) {
            try {
                Socket clientSocket = serverSocket.accept();
                HttpServerDispatchThread thread = new HttpServerDispatchThread(clientSocket, handler);
                executor.execute(thread);
            } catch (IOException e) {
                if (running.get()) {
                    LOGGER.log(Level.WARNING, "Error accepting connection", e);
                }
                // If server is stopping, this is expected
            }
        }
    }

}