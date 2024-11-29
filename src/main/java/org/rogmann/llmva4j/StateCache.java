package org.rogmann.llmva4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.rogmann.llmva4j.Llama.Configuration;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;

public class StateCache {

    /** "GGSC": GGUF state cache */
    private static final int MAGIC_STATE_CACHE = 0x47475343;
    /** "Version 2 */
    private static final int MAGIC_STATE_VERSION = 0x02;

    private final Configuration config;
    private final int kvDim;    
    private final FloatTensor[] keyCache;
    private final FloatTensor[] valueCache;
    
    public StateCache(Configuration config, StateBase state) {
        this.config = config;

        if (state == null) {
            throw new IllegalStateException("No state to store.");
        }
        kvDim = state.kvDim;
        keyCache = state.keyCache;
        valueCache = state.valueCache;
    }
    
    public String saveKVCache(String userText, Path stateCacheFolder, List<TokenDetails> tokens) {
        Pattern pSaveName = Pattern.compile("/save:([A-Za-z0-9_][A-Za-z0-9_.-]+)");
        Matcher m = pSaveName.matcher(userText);
        if (!m.matches()) {
            throw new IllegalStateException("Invalid name to store key-value-cache, expected " + pSaveName);
        }
        if (stateCacheFolder == null) {
            throw new IllegalStateException("Missing argument --state-cache-folder to store key-value-cache.");
        }
        String baseName = m.group(1);
    
        final String fileName = baseName + ".ggsc";
        final File file = new File(stateCacheFolder.toFile(), fileName);
        try (FileOutputStream fos = new FileOutputStream(file);
                BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            serialize(bos, tokens);
        } catch (IOException e) {
            throw new RuntimeException(String.format("IO-error while writing %s", file), e);
        }
        return String.format("Wrote KV-cache (%d tokens) into file %s", tokens.size(), fileName);
    }

    void serialize(OutputStream os, List<TokenDetails> tokens) throws IOException {
        if (!(keyCache[0] instanceof ArrayFloatTensor)) {
            throw new UnsupportedOperationException("keyCache has unexpected type: " + keyCache.getClass());
        }   
        if (!(valueCache[0] instanceof ArrayFloatTensor)) {
            throw new UnsupportedOperationException("valueCache has unexpected type: " + valueCache.getClass());
        }
        byte[] bufName = config.modelGGUFName.getBytes(StandardCharsets.UTF_8);
        int[] sizes = { 24, bufName.length, tokens.size() * 4, kvDim * 4};
        ByteBuffer bb = ByteBuffer.allocate(Arrays.stream(sizes).max().getAsInt());
        bb.putInt(0, MAGIC_STATE_CACHE);
        bb.putInt(4, MAGIC_STATE_VERSION);
        bb.putInt(8, bufName.length);
        bb.putInt(12, tokens.size());
        bb.putInt(16, kvDim);
        bb.putInt(20, config.numberOfLayers);
        os.write(bb.array(), 0, 24);
        os.write(bufName);
        for (int i = 0; i < tokens.size(); i++) {
            bb.putInt(4 * i, tokens.get(i).token());
        }
        os.write(bb.array(), 0, 4 * tokens.size());
        for (int nLayer = 0; nLayer < config.numberOfLayers; nLayer++) {
            for (int i = 0; i < tokens.size(); i++) {
                for (int k = 0; k < kvDim; k++) {
                    bb.putFloat(4 * k, keyCache[nLayer].getFloat(k + i * kvDim));
                }
                os.write(bb.array(), 0, 4 * kvDim);
                for (int k = 0; k < kvDim; k++) {
                    bb.putFloat(4 * k, valueCache[nLayer].getFloat(k + i * kvDim));
                }
                os.write(bb.array(), 0, 4 * kvDim);
            }
        }
    }
    
    public int deserialize(InputStream is, Tokenizer tokenizer, List<Integer> tokens, boolean echo) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(24);
        read(is, bb, 24);
        check(bb, 0, MAGIC_STATE_CACHE, "MAGIC_STATE_CACHE");
        check(bb, 4, MAGIC_STATE_VERSION, "MAGIC_STATE_VERSION");
        int nameActualLength = bb.getInt(8);
        int numCachedTokens = bb.getInt(12);
        check(bb, 16, kvDim, "kvDim");
        check(bb, 20, config.numberOfLayers, "numberOfLayers");

        int[] sizes = { nameActualLength, numCachedTokens * 4, kvDim * 4 };
        bb = ByteBuffer.allocate(Arrays.stream(sizes).max().getAsInt());

        String nameExpected = config.modelGGUFName;
        read(is, bb, nameActualLength);
        String nameActual = new String(bb.array(), 0, nameActualLength, StandardCharsets.UTF_8);
        if (!nameActual.equals(nameExpected)) {
            throw new IllegalArgumentException(String.format("Invalid model-name in state-cache: expected='%s', actual='%s'", nameExpected, nameActual));
        }
        read(is, bb, 4 * numCachedTokens);
        int numTokensRead = 0;
        final List<Integer> cachedTokens = new ArrayList<>();
        for (int i = 0; i < numCachedTokens && i < tokens.size(); i++) {
            final int actual = bb.getInt(4 * i);
            cachedTokens.add(actual);
            if (i != numTokensRead) {
                continue;
            }
            int expected = tokens.get(i).intValue();
            if (actual == expected) {
                numTokensRead++;
            } else if (i < tokens.size() - 2) {
                System.out.printf("Reused %d of %d tokens in cache-file, actual=%d ('%s'), expected=%d ('%s')%n",
                        numTokensRead, tokens.size(),
                        expected, tokenizer.decode(Collections.singletonList(expected)), actual, tokenizer.decode(Collections.singletonList(actual)));
            }
        }
        if (echo) {
            System.out.println("Current tokens: " + tokens);
            System.out.println("Cached tokens:  " + cachedTokens);
        }
        for (int nLayer = 0; nLayer < config.numberOfLayers; nLayer++) {
            for (int i = 0; i < numCachedTokens; i++) {
                read(is, bb, 4 * kvDim);
                if (i < numTokensRead) {
                    for (int k = 0; k < kvDim; k++) {
                        keyCache[nLayer].setFloat(k + i * kvDim, bb.getFloat(4 * k));
                    }
                }
                read(is, bb, 4 * kvDim);
                if (i < numTokensRead) {
                    for (int k = 0; k < kvDim; k++) {
                        valueCache[nLayer].setFloat(k + i * kvDim, bb.getFloat(4 * k));
                    }
                }
            }
        }
        return numTokensRead;
    }
    
    static void check(ByteBuffer bb, int offset, int expected, String name) throws IOException {
        final int actual = bb.getInt(offset);
        if (actual != expected) {
            throw new IOException(String.format("Unexpected value '%s': actual=0x%x, expected=0x%x", name, expected, actual));
        }
    }

    static void check(ByteBuffer bb, int offset, int expected, String name, Tokenizer tokenizer) throws IOException {
        final int actual = bb.getInt(offset);
        if (actual != expected) {
            throw new IOException(String.format("Unexpected value '%s': actual=%d ('%s'), expected=%d ('%s')",
                    name, expected, tokenizer.decode(Collections.singletonList(expected)), actual, tokenizer.decode(Collections.singletonList(actual))));
        }
    }

    static void read(InputStream is, ByteBuffer bb, int size) throws IOException {
        byte[] buf = bb.array();
        int offset = 0;
        while (offset < size) {
            int len = is.read(buf, offset, size - offset);
            if (len == -1) {
                break;
            }
            offset += len;
        }
        if (offset < size) {
            throw new IOException(String.format("Unexpected end of stream: offset=%d, size=%d", offset, size));
        }
    }

}
