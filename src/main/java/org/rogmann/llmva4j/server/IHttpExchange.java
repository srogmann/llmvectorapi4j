package org.rogmann.llmva4j.server;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;

/**
 * Interface of a HTTP-exchange.
 */
public interface IHttpExchange {

    /**
     * Returns the request {@link URI}.
     *
     * @return the request {@code URI}
     */
    URI getRequestURI();

    /**
     * Returns the request method.
     *
     * @return the request method string
     */
    String getRequestMethod();

    /**
     * Returns an immutable {@link IHeaders} containing the HTTP headers that
     * were included with this request.
     *
     * <p> The keys in this {@code IHeaders} are the header names, while the
     * values are a {@link java.util.List} of Strings.</p>
     *
     * <p> The keys in {@code Headers} are case-insensitive.</p>
     *
     * @return a read-only {@code IHeaders} which can be used to access request
     *         headers.
     */
    IHeaders getRequestHeaders();

    /**
     * Returns a stream from which the request body can be read.
     * Multiple calls to this method will return the same stream.
     *
     * @return the stream from which the request body can be read
     */
    InputStream getRequestBody();

    /**
     * Returns a mutable {@link Headers} into which the HTTP response headers
     * can be stored and which will be transmitted as part of this response.
     *
     * <p> The keys in the {@code Headers} are the header names, while the
     * values must be a {@link java.util.List} of {@linkplain java.lang.String Strings}
     * containing each value that should be included multiple times (in the
     * order that they should be included).
     *
     * <p> The keys in {@code IHeaders} are case-insensitive.
     *
     * @return a writable {@code IHeaders} which can be used to set response
     *         headers.
     */
    IHeaders getResponseHeaders();

    /**
     * Starts sending the response back to the client using the current set of
     * response headers and the numeric response code as specified in this
     * method. The response body length is also specified as follows. If the
     * response length parameter is greater than {@code zero}, this specifies an
     * exact number of bytes to send and the application must send that exact
     * amount of data. If the response length parameter is {@code zero}, then
     * chunked transfer encoding is used and an arbitrary amount of data may be
     * sent. The application terminates the response body by closing the
     * {@link OutputStream}.
     *
     * @param rCode          the response code to send
     * @param responseLength if {@literal > 0}, specifies a fixed response body
     *                       length and that exact number of bytes must be written
     *                       to the stream acquired from {@link #getResponseCode()}
     *                       If {@literal == 0}, then chunked encoding is used,
     *                       and an arbitrary number of bytes may be written.
     *                       If {@literal <= -1}, then no response body length is
     *                       specified and no response body may be written.
     * @throws IOException   if the response headers have already been sent or an I/O error occurs
     */
    void sendResponseHeaders(int rCode, long responseLength) throws IOException;

    /**
     * Returns a stream to which the response body must be
     * written. {@link #sendResponseHeaders(int,long)}) must be called prior to
     * calling this method. Multiple calls to this method (for the same exchange)
     * will return the same stream. In order to correctly terminate each exchange,
     * the output stream must be closed, even if no response body is being sent.
     *
     * @return the stream to which the response body is written
     * @throws IOException in case of an IO-error
     */
    OutputStream getResponseBody() throws IOException;

    /**
     * Ends this exchange by doing the following in sequence:
     * <ol>
     *      <li> close the request {@link InputStream}, if not already closed.
     *      <li> close the response {@link OutputStream}, if not already closed.
     * </ol>
     * 
     * @throws IOException in case of an IO-error
     */
    void close() throws IOException;

}
