package org.rogmann.llmva4j.server;

/**
 * HTTP request and response headers are represented by this interface.
 */
public interface IHeaders {

    /**
     * Adds the given {@code value} to the list of headers for the given
     * {@code key}. If the mapping does not already exist, then it is created.
     *
     * @param key   the header name
     * @param value the value to add to the header
     */
    void add(String key, String value);

    /**
     * Sets the given {@code value} as the sole header value for the given
     * {@code key}. If the mapping does not already exist, then it is created.
     *
     * @param key   the header name
     * @param value the header value to set
     */
    void set(String key, String value);

}
