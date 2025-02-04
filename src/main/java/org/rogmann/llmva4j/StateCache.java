package org.rogmann.llmva4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.rogmann.llmva4j.ChatFormat.Message;
import org.rogmann.llmva4j.ChatFormat.MessageWithTokens;
import org.rogmann.llmva4j.ChatFormat.Role;
import org.rogmann.llmva4j.Llama.Configuration;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;

/**
 * Read or write the content of the KV-cache.
 *
 * <p>File format:</p>
 * <ul>
 * <li>Header (24 bytes)</li>
 * <li>model-name</li>
 * <li>messages of conversation</li>
 * <li>tokens of conversation</li>
 * <li>KV-cache</li>
 * </ul>
 */
public class StateCache {

    /** "GGSC": GGUF state cache */
    private static final int MAGIC_STATE_CACHE = 0x47475343;
    /** "Version 2 */
    private static final int MAGIC_STATE_VERSION = 0x04;

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
    
    public String saveKVCache(String userText, Path stateCacheFolder, List<TokenDetails> tokens, List<MessageWithTokens> conversation) {
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
            serialize(bos, tokens, conversation);
        } catch (IOException e) {
            throw new RuntimeException(String.format("IO-error while writing %s", file), e);
        }
        return String.format("Wrote KV-cache (%d tokens) into file %s", tokens.size(), fileName);
    }

    void serialize(OutputStream os, List<TokenDetails> tokens, List<MessageWithTokens> conversation) throws IOException {
        if (!(keyCache[0] instanceof ArrayFloatTensor)) {
            throw new UnsupportedOperationException("keyCache has unexpected type: " + keyCache.getClass());
        }   
        if (!(valueCache[0] instanceof ArrayFloatTensor)) {
            throw new UnsupportedOperationException("valueCache has unexpected type: " + valueCache.getClass());
        }
        int[] sizes = { 24, tokens.size() * 4, kvDim * 4};
        ByteBuffer bb = ByteBuffer.allocate(Arrays.stream(sizes).max().getAsInt());
        bb.putInt(0, MAGIC_STATE_CACHE);
        bb.putInt(4, MAGIC_STATE_VERSION);
        bb.putInt(8, config.numberOfLayers);
        bb.putInt(12, kvDim);
        bb.putInt(16, conversation.size());
        bb.putInt(20, tokens.size());
        assert (24 == sizes[0]);
        os.write(bb.array(), 0, 24);
        writeString(os, bb, config.modelGGUFName);
        for (MessageWithTokens msg : conversation) {
            writeString(os, bb, msg.role().name());
            writeString(os, bb, msg.content());
            writeInt(os, bb, msg.tokens().size());
            for (int token : msg.tokens()) {
                writeInt(os, bb, token);
            }
        }

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
    
    public enum DeserializeMode {
        /** read header and messages only */
        SCAN_FILE,
        /** fill KV-cache and check given tokens */
        DESERIALIZE_CHECK_TOKENS,
        /** fill KV-cache and fill token-list */
        DESERIALIZE_FILL_TOKENS
    }

    public int deserialize(InputStream is, Tokenizer tokenizer, List<MessageWithTokens> conversation, List<Integer> tokens, boolean echo) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(24);
        read(is, bb, 24);
        check(bb, 0, MAGIC_STATE_CACHE, "MAGIC_STATE_CACHE");
        check(bb, 4, MAGIC_STATE_VERSION, "MAGIC_STATE_VERSION");
        check(bb, 8, config.numberOfLayers, "numberOfLayers");
        check(bb, 12, kvDim, "kvDim");
        int numMessages = bb.getInt(16);
        int numCachedTokens = bb.getInt(20);

        String nameExpected = config.modelGGUFName;
        String nameActual = readString(is, bb);
        if (!nameActual.equals(nameExpected)) {
            throw new IllegalArgumentException(String.format("Invalid model-name in state-cache: expected='%s', actual='%s'", nameExpected, nameActual));
        }

        if (conversation == null) {
            // We read the header only.
            return 0;
        }

        for (int i = 0; i < numMessages; i++) {
            String roleName = readString(is, bb);
            String content = readString(is, bb);
            int numTokens = readInt(is, bb);
            List<Integer> msgTokens = new ArrayList<>();
            for (int j = 0; j < numTokens; j++) {
                msgTokens.add(readInt(is, bb));
            }
            conversation.add(new ChatFormat.MessageWithTokens(new Role(roleName), content, msgTokens));
        }

        if (tokens == null || tokenizer == null) {
            // We read header and conversation only.
            return 0;
        }

        int[] sizes = { 4, numCachedTokens * 4, kvDim * 4 };
        bb = ByteBuffer.allocate(Arrays.stream(sizes).max().getAsInt());

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
                        expected, tokenizer.decode(Collections.singletonList(expected)),
                        actual, tokenizer.decode(Collections.singletonList(actual)));
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

    record StateCacheFile(File file, List<MessageWithTokens> messages) { }

    public StateCacheFile searchConversationLog(Path stateCacheFolder, List<Message> requestMessages) {
        File[] files = stateCacheFolder.toFile().listFiles();
        if (files == null) {
            throw new IllegalArgumentException(String.format("State-cache folder (%s) is missing", stateCacheFolder));
        }
        int maxMsgs = 0;
        StateCacheFile fileMaxMsgs = null;
        for (File file : files) {
            if (!file.isFile() || !file.getName().endsWith(".ggsc")) {
                continue;
            }
            List<MessageWithTokens> msgsFile = new ArrayList<>();
            try (InputStream is = Files.newInputStream(file.toPath())) {
                deserialize(is, null, msgsFile, null, false);
            } catch (IllegalArgumentException e) {
                System.err.println(String.format("Can't use state-cache (%s): %s", file.getName(), e));
                continue;
            } catch (IOException e) {
                System.err.println(String.format("IO-error while reading state-cache (%s): %s", file, e));
                continue;
            }
            int numMsgsEqual = 0;
            for (int i = 0; i < msgsFile.size() && i < requestMessages.size(); i++) {
                if (!msgsFile.get(i).asMessage().equals(requestMessages.get(i))) {
                    break;
                }
                numMsgsEqual++;
            }
            if (numMsgsEqual > maxMsgs) {
                maxMsgs = numMsgsEqual;
                fileMaxMsgs = new StateCacheFile(file, msgsFile);
            }
        }
        return fileMaxMsgs;
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

    static int readInt(InputStream is, ByteBuffer bb) throws IOException {
        read(is, bb, 4);
        return bb.getInt(0);
    }

    static String readString(InputStream is, ByteBuffer bb) throws IOException {
        read(is, bb, 4);
        int size = bb.getInt(0);
        if (size < 0) {
            throw new IllegalArgumentException("Illegal string-size: " + size);
        }
        final byte[] buf = new byte[size];
        read(is, ByteBuffer.wrap(buf), size);
        return new String(buf, StandardCharsets.UTF_8);
    }

    private static void writeInt(OutputStream os, ByteBuffer bb, int value) throws IOException {
        bb.putInt(0, value);
        os.write(bb.array(), 0, 4);
    }

    private static void writeString(OutputStream os, ByteBuffer bb, String text) throws IOException {
        byte[] bufText = text.getBytes(StandardCharsets.UTF_8);
        bb.putInt(0, bufText.length);
        os.write(bb.array(), 0, 4);
        os.write(bufText);
    }


}
