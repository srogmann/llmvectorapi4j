package org.rogmann.llmva4j.server;

import com.sun.net.httpserver.Headers;

/**
 * Decorator of {@link Headers}.
 */
public class HeadersDecorator implements IHeaders {

    private final Headers headers;

    /**
     * Constructor
     * @param headers headers to be decorated
     */
    public HeadersDecorator(Headers headers) {
        this.headers = headers;
    }

    /** {@inheritDoc} */
    @Override
    public void add(String key, String value) {
        headers.add(key, value);
    }

    /** {@inheritDoc} */
    @Override
    public void set(String key, String value) {
        headers.set(key, value);
    }
}
