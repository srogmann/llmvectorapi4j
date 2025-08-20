package org.rogmann.llmva4j.server.http;

import java.io.IOException;

/**
 * Handler for HTTP requests.
 */
@FunctionalInterface
public interface HttpHandler {
    /**
     * Handles an HTTP request.
     * 
     * @param exchange the HTTP exchange
     * @throws IOException if an I/O error occurs
     */
    void handle(HttpServerDispatchExchange exchange) throws IOException;
}