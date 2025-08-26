package org.rogmann.llmva4j.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;

import com.sun.net.httpserver.HttpExchange;

/**
 * Decorator of {@link com.sun.net.httpserver.HttpExchange} to implement {@link IHttpExchange}.
 */
public class HttpExchangeDecorator implements IHttpExchange {

    /** decorated exchange */
    private final HttpExchange exchange;

    /**
     * Constructor
     * @param exhange HTTP exchange
     */
    public HttpExchangeDecorator(HttpExchange exchange) {
        this.exchange = exchange;
    }

    @Override
    public URI getRequestURI() {
        return exchange.getRequestURI();
    }

    @Override
    public String getRequestMethod() {
        return exchange.getRequestMethod();
    }

    @Override
    public IHeaders getRequestHeaders() {
        return new HeadersDecorator(exchange.getRequestHeaders());
    }

    @Override
    public InputStream getRequestBody() {
        return exchange.getRequestBody();
    }

    @Override
    public IHeaders getResponseHeaders() {
        return new HeadersDecorator(exchange.getResponseHeaders());
    }

    @Override
    public void sendResponseHeaders(int rCode, long responseLength) throws IOException {
        exchange.sendResponseHeaders(rCode, responseLength);
    }

    @Override
    public OutputStream getResponseBody() {
        return exchange.getResponseBody();
    }

    @Override
    public void close() {
        exchange.close();
    }
}
