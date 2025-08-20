package org.rogmann.llmva4j.server.http;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * InputStream wrapper that handles HTTP chunked transfer encoding.
 */
public class HttpInputStream extends InputStream {
    private static final Logger LOGGER = Logger.getLogger(HttpInputStream.class.getName());
    
    private final InputStream inputStream;
    private final boolean isChunked;
    private long remainingInChunk = -1;
    private boolean eof = false;
    private int contentLength;
    private int offset = 0;
    
    /**
     * Creates a new HTTP input stream.
     * 
     * @param inputStream the underlying input stream
     * @param headers the HTTP headers
     */
    public HttpInputStream(InputStream inputStream, Map<String, String> headers) {
        this.inputStream = inputStream;
        String transferEncoding = headers.get("Transfer-Encoding");
        this.isChunked = "chunked".equalsIgnoreCase(transferEncoding);
        String sContentLength = headers.get("Content-Length");
        if (LOGGER.isLoggable(Level.FINER)) {
        	LOGGER.finer("chunked: " + isChunked);
        }
        contentLength = -1;
        if (!isChunked) {
	        if (sContentLength == null) {
	        	throw new IllegalStateException("Missing content-length in none-chunking mode.");
	        }
	        contentLength = Integer.parseInt(sContentLength);
	        if (LOGGER.isLoggable(Level.FINER)) {
	        	LOGGER.finer("Content-Length: " + contentLength);
	        }
        }
    }
    
    @Override
    public int read() throws IOException {
        if (eof) {
            return -1;
        }
        
        if (isChunked) {
            return readChunked();
        }

        if (offset >= contentLength) {
        	return -1;
        }
        int b = inputStream.read();
        if (b != -1) {
        	offset ++;
        }
        return b;
    }
    
    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        if (eof) {
            return -1;
        }
        
        if (isChunked) {
            return readChunked(b, off, len);
        }
        
        if (offset >= contentLength) {
        	return -1;
        }

        int lenRead = inputStream.read(b, off, Math.min(len, contentLength - offset));
        if (lenRead >= 0) {
        	offset += lenRead;
        }
        return lenRead;
    }
    
    private int readChunked() throws IOException {
        if (remainingInChunk <= 0) {
            if (!readNextChunk()) {
                return -1;
            }
        }
        
        int b = inputStream.read();
        if (b != -1) {
            remainingInChunk--;
        }
        return b;
    }
    
    private int readChunked(byte[] b, int off, int len) throws IOException {
        if (remainingInChunk <= 0) {
            if (!readNextChunk()) {
                return -1;
            }
        }
        
        int toRead = (int) Math.min(len, remainingInChunk);
        int read = inputStream.read(b, off, toRead);
        if (read != -1) {
            remainingInChunk -= read;
            if (LOGGER.isLoggable(Level.FINEST)) {
            	LOGGER.finest(String.format("readChunked: toRead=%d, read=%d, txt=%s", toRead, read, new String(b, off, read)));
            }
        }
        return read;
    }
    
    private boolean readNextChunk() throws IOException {
        // Read chunk size line
        StringBuilder chunkSizeLine = new StringBuilder();
        int b;
        while ((b = inputStream.read()) != -1) {
            if (b == '\r') {
                int next = inputStream.read();
                if (next == '\n') {
                    break; // CRLF found
                } else {
                    chunkSizeLine.append((char) b);
                    if (next != -1) {
                        chunkSizeLine.append((char) next);
                    }
                }
            } else {
                chunkSizeLine.append((char) b);
            }
        }
        
        if (b == -1 && chunkSizeLine.length() == 0) {
            eof = true;
            return false;
        }
        
        // Parse chunk size (hexadecimal)
        String chunkSizeStr = chunkSizeLine.toString().trim();
        int semicolonIndex = chunkSizeStr.indexOf(';');
        if (semicolonIndex > 0) {
            chunkSizeStr = chunkSizeStr.substring(0, semicolonIndex);
        }
        
        try {
            int chunkSize = Integer.parseInt(chunkSizeStr, 16);
            remainingInChunk = chunkSize;
            
            if (chunkSize == 0) {
                // Last chunk - read trailing CRLF
                readCRLF();
                eof = true;
                return false;
            }
            
            return true;
        } catch (NumberFormatException e) {
            throw new IOException("Invalid chunk size: " + chunkSizeStr, e);
        }
    }
    
    private void readCRLF() throws IOException {
        int b1 = inputStream.read();
        int b2 = inputStream.read();
        if (b1 != '\r' || b2 != '\n') {
            throw new IOException(String.format("Expected CRLF after chunk data (0x%02x, 0x%02x)", b1 & 0xff, b2 & 0xff));
        }
    }
}