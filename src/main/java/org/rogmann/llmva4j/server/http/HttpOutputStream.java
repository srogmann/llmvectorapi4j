package org.rogmann.llmva4j.server.http;

import java.io.IOException;
import java.io.OutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * OutputStream wrapper that handles HTTP chunked transfer encoding.
 */
public class HttpOutputStream extends OutputStream {
    private static final Logger LOGGER = Logger.getLogger(HttpOutputStream.class.getName());
    
    private final OutputStream outputStream;
    private final boolean isChunked;
    private boolean closed = false;
    
    /**
     * Creates a new HTTP output stream.
     * 
     * @param outputStream the underlying output stream
     * @param isChunked <code>true</code> is response is chunked
     */
    public HttpOutputStream(OutputStream outputStream, boolean isChunked) {
        this.outputStream = outputStream;
        this.isChunked = isChunked;
    }
    
    @Override
    public void write(int b) throws IOException {
        if (closed) {
            throw new IOException("Stream closed");
        }
        if (isChunked) {
        	writeChunk(new byte[]{(byte) b}, 0, 1);
        } else {
        	outputStream.write(b);
        }
    }
    
    @Override
    public void write(byte[] b) throws IOException {
        write(b, 0, b.length);
    }
    
    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        if (closed) {
            throw new IOException("Stream closed");
        }
        if (len > 0) {
        	if (isChunked) {
        		writeChunk(b, off, len);
        	} else {
        		outputStream.write(b, off, len);
        	}
        }
    }
    
    private void writeChunk(byte[] b, int off, int len) throws IOException {
    	if (LOGGER.isLoggable(Level.FINER)) {
    		LOGGER.finer(String.format("write: off=%d, len=%d", off, len));
    	}
        // Write chunk size in hexadecimal followed by CRLF
        String chunkHeader = Integer.toHexString(len) + "\r\n";
        outputStream.write(chunkHeader.getBytes());
        
        // Write chunk data
        outputStream.write(b, off, len);
        
        // Write CRLF after chunk data
        outputStream.write("\r\n".getBytes());
    }
    
    @Override
    public void flush() throws IOException {
        outputStream.flush();
    }
    
    @Override
    public void close() throws IOException {
        if (!closed) {
            // Write final chunk of size 0
            outputStream.write("0\r\n\r\n".getBytes());
            outputStream.flush();
            closed = true;
        }
    }
}