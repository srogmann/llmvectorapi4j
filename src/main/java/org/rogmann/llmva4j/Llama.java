///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector
//MAIN com.llama4j.Llama3

// Author: Alfonso² Peterssen, extensions by Sascha Rogmann
// Based on Andrej Karpathy's llama2.c and minbpe projects
//
// 2024-11-11: Parallel prompt processing (better TTFT)
// 2024-11-11: Decoding of UTF-8 byte-sequences (display of emoticons)
//
// Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
package org.rogmann.llmva4j;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;
import sun.misc.Unsafe;

import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Reader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.rogmann.llmva4j.Llama.Options;

record Llama(String modelName, Configuration configuration, Tokenizer tokenizer, Weights weights) {
    /** batch-size used in prompt evaluation */
    static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
            float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo) {

        static final int DEFAULT_MAX_TOKENS = 512;

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Llama3.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> (optional) system prompt");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.1");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
            out.println();
            out.println("Examples:");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Reply concisely, in French\" --prompt \"Who was Marie Curie?\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Answer concisely\" --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Keep max context length small for low-memory devices.
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1; // skip arg
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo);
        }
    }

    public static final class Configuration {
        public final int dim; // transformer dimension
        public final int hiddenDim; // for ffn layers
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
        public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
        public final int contextLength; // max sequence length
        public final boolean sharedWeights;
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this; // no change
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength,
                    this.sharedWeights, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, boolean sharedWeights, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.sharedWeights = sharedWeights;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size)
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)

        public final FloatTensor[] q_bias; // (layer, dim)
        public final FloatTensor[] k_bias; // (layer, kv_dim)
        public final FloatTensor[] v_bias; // (layer, kv_dim)

        public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
        // weights for ffn
        public final FloatTensor[] w1; // (layer, hidden_dim, dim)
        public final FloatTensor[] w2; // (layer, dim, hidden_dim)
        public final FloatTensor[] w3; // (layer, hidden_dim, dim)
        // public final rmsnorm
        public final FloatBuffer rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
        public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatTensor wcls; // (vocab_size, dim)

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight,
                FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                FloatTensor[] q_bias, FloatTensor[] k_bias, FloatTensor[] v_bias,
                FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;

            this.q_bias = q_bias;
            this.k_bias = k_bias;
            this.v_bias = v_bias;

            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }
    }

    public static final class State {

        // current wave of activations
        public final int batchsize;
        public final FloatTensor[] x; // activation at current time stamp (dim,)
        public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
        public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] q; // query (dim,)
        public final FloatTensor[] k; // key (dim,)
        public final FloatTensor[] v; // value (dim,)
        public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits
        // kv cache
        public final FloatTensor[] keyCache; // (n_layer, seq_len, kv_dim)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

        public int latestToken;

        /**
         * Interface for classes that consume attention data.
         */
        public interface AttentionConsumer {

            /**
             * Accepts attention data.
             *
             * @param idxToken     index of the token
             * @param layer        layer index
             * @param head         attention-head
             * @param att          attention-tensor
             * @param attOffset    offset in the attention-tensor
             * @param attLength    size of the current block in the attention-tensor
             */
            void accept(int idxToken, int layer, int head, FloatTensor att, int attOffset, int attLength);
        }
        
        record AttentionDetail(int idxToken, int layer, int head, int idx, float attValue) { };

        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }
}

class LlamaHttpServer {
    record LlamaHttpSession(String sessionKey, Llama model, Sampler sampler, Options options, Llama.State state, List<Integer> conversationTokens) { ; }

    static enum JsonFormat {
        LLAMA_CPP,
        OPENAI
    }

    /**
     * Starts a HTTP-server to server requests in Llama.cpp-style.
     * @param model model
     * @param sampler sampler
     * @param optionsGlobal default options
     * @param host host-name or ip-address to bind the http-server
     * @param port port of http-server
     */
    static void runHttpServer(Llama model, Sampler sampler, Options optionsGlobal, String host, int port) {
        InetSocketAddress addr = new InetSocketAddress(host, port);
        int backlog = 0;
        String rootpath = "/";
        System.out.println(String.format("Start server at %s", addr));
        final AtomicLong reqCounter = new AtomicLong();
        final ConcurrentMap<String, LlamaHttpSession> mapSessions = new ConcurrentHashMap<>();

        AtomicReference<HttpServer> refServer = new AtomicReference<>();
        HttpHandler handler = exchange -> {
            System.out.format("httpserver: request of %s by %s%n", exchange.getRequestURI(), exchange.getRemoteAddress());
            if ("GET".equals(exchange.getRequestMethod())) {
                String pathReq = exchange.getRequestURI().getPath();
                String pathBase = System.getProperty("llm.server.path", "public");
                if ("/".equals(pathReq)) {
                    pathReq = "index.html";
                }
                if (!Pattern.matches("/?[A-Za-z0-9_.-]*", pathReq)) {
                    System.err.format("Invalid path: %s%n", pathReq);
                    byte[] buf = "Invalid path".getBytes(StandardCharsets.UTF_8);
                    exchange.setAttribute("Content-Type", "application/html");
                    exchange.sendResponseHeaders(404, buf.length);
                    exchange.getResponseBody().write(buf);
                    exchange.close();
                    return;
                }
                final File file = new File(pathBase, pathReq);
                if (!file.isFile()) {
                    System.err.println("No such file: " + file);
                    byte[] buf = "File not found".getBytes(StandardCharsets.UTF_8);
                    exchange.setAttribute("Content-Type", "application/html");
                    exchange.sendResponseHeaders(404, buf.length);
                    exchange.getResponseBody().write(buf);
                    exchange.close();
                    return;
                }
                exchange.getRequestBody().close();
                String contentType = switch (pathReq.replaceFirst(".*[.]", "")) {
                    case "html" -> "text/html";
                    case "css" -> "text/css";
                    case "js", "mjs" -> "text/javascript";
                    case "ico" -> "image";
                    default -> "application/octet-stream";
                };
                exchange.getResponseHeaders().set("Content-type", contentType);
                byte[] buf = Files.readAllBytes(file.toPath());
                exchange.sendResponseHeaders(200, buf.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(buf);
                }
                exchange.close();
                return;
            }
            List<ChatFormat.Message> chatMessages = new ArrayList<>();
            final Map<String, Object> mapRequest;
            try (InputStream is = exchange.getRequestBody();
                    InputStreamReader isr = new InputStreamReader(is);
                    BufferedReader br = new BufferedReader(isr);
                    TeeBufferedReader tbr = new TeeBufferedReader(br)) {
                try {
                    readChar(tbr, true, '{');
                    mapRequest = parseJsonDict(tbr);

                    List<Map<String, Object>> messages = getJsonArrayDicts(mapRequest, "messages");
                    String prompt = getJsonValue(mapRequest, "prompt", String.class);
                    if (prompt != null) {
                        // llama.cpp chat sends the whole chat as a long string :-/.
                        Pattern pLlamaCppChatDefault = Pattern.compile(".*\nUser: (.*)\nLlama:", Pattern.DOTALL);
                        Matcher m = pLlamaCppChatDefault.matcher(prompt);
                        if (m.matches()) {
                            prompt = m.group(1);
                        }
                    }

                    String systemPrompt = optionsGlobal.systemPrompt();
                    if (messages != null) {
                        for (Map<String, Object> msg : messages) {
                            String role = getJsonValue(msg, "role", String.class);
                            String content = getJsonValue(msg, "content", String.class);
                            if (role == null) {
                                throw new IllegalArgumentException("role is missing in incoming message.");
                            }
                            if (content == null) {
                                throw new IllegalArgumentException("content is missing in incoming message.");
                            }
                            if ("system".equals(role)) {
                                if (systemPrompt != null) {
                                    throw new IllegalArgumentException("Can't overwrite system-prompt.");
                                }
                                systemPrompt = content;
                            }
                            else if ("user".equals(role)) {
                                prompt = content;
                            }
                            else {
                                throw new IllegalArgumentException("Unexpected role in message: " + role);
                            }
                        }
                    }
                    if (prompt == null) {
                        System.out.println("Map: " + mapRequest);
                        throw new IllegalArgumentException("Prompt is missing in request");
                    }
                    if ("STOP".equalsIgnoreCase(prompt)) {
                        refServer.get().stop(0);
                        throw new IllegalArgumentException("Server is stopping");
                    }
                    if (systemPrompt != null) {
                        chatMessages.add(new ChatFormat.Message(ChatFormat.Role.SYSTEM, systemPrompt));
                    }
                    chatMessages.add(new ChatFormat.Message(ChatFormat.Role.USER, prompt));
                }
                catch (RuntimeException e) {
                    System.out.println("JSON-Prefix: " + tbr.sb);
                    e.printStackTrace();
                    Map<String, Object> mapError = new HashMap<>();
                    mapError.put("errormsg", "Invalid request: " + e.getMessage());
                    mapError.put("jsonProcessed", tbr.sb.toString());
                    var sb = new StringBuilder();
                    dumpJson(sb, mapError);
                    byte[] bufError = sb.toString().getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(400, bufError.length);
                    exchange.setAttribute("Content-Type", "application/json");
                    exchange.getResponseBody().write(bufError);
                    exchange.close();
                    return;
                }
            }
            catch (IOException e) {
                e.printStackTrace();
                exchange.sendResponseHeaders(500, 0);
                exchange.close();
                return;
            }

            JsonFormat format = mapRequest.containsKey("messages" ) ? JsonFormat.OPENAI : JsonFormat.LLAMA_CPP;

            try {
                List<String> lCookies = exchange.getRequestHeaders().get("Cookie");
                String cookie = (lCookies != null) ? lCookies.get(0) : null;
                LlamaHttpSession httpSession = null;
                {
                    String sessionKey = null;
                    if (cookie != null && cookie.startsWith("LLAMA_SESS_ID=")) {
                        sessionKey = cookie.replaceFirst("LLAMA_SESS_ID=([^;]*).*", "$1");
                        httpSession = mapSessions.get(sessionKey);
                        if (httpSession == null) {
                            System.err.format("Llama-HTTP-session (%s) doesn't exist (any more)%n", sessionKey);
                            sessionKey = null;
                        }
                    }
                    if (httpSession != null && httpSession.conversationTokens().size() > 0) {
                        if (ChatFormat.Role.SYSTEM.equals(chatMessages.get(0).role())) {
                            // System-prompt at begin only.
                            chatMessages.remove(0);
                        }
                    }
                    if (httpSession == null) {
                        // We build a new HTTP-session.
                        final Llama.State state = model.createNewState(Llama.BATCH_SIZE);
                        sessionKey = "SESS-" + reqCounter.get() + "-" + UUID.randomUUID().toString();
                        exchange.getResponseHeaders().add("Set-Cookie", "LLAMA_SESS_ID=" + sessionKey);

                        float temperature = readFloat(mapRequest, "temperature", optionsGlobal.temperature());
                        float topP = readFloat(mapRequest, "top_p", optionsGlobal.topp());
                        int maxLlamaCpp = readInt(mapRequest, "n_predict", optionsGlobal.maxTokens());
                        int maxTokensOld = readInt(mapRequest, "max_tokens", maxLlamaCpp);
                        int maxComplTokens = readInt(mapRequest, "max_completion_tokens", maxTokensOld);
                        long seed = readLong(mapRequest, "seed", optionsGlobal.seed());
                        boolean stream = readBoolean(mapRequest, "stream", optionsGlobal.stream());
                        Options optionsReq = new Options(optionsGlobal.modelPath(), "", optionsGlobal.systemPrompt(), true,
                                temperature, topP, seed, maxComplTokens, stream, optionsGlobal.echo());
                        System.out.format("New HTTP-Session (%s) for (%s), temp=%f, top_p=%f, n=%d, seed=%d%n", sessionKey, exchange.getRemoteAddress(),
                                temperature, topP, maxComplTokens, seed);
                        final List<Integer> conversationTokens = new ArrayList<>();
                        httpSession = new LlamaHttpSession(sessionKey, model, sampler, optionsReq, state, conversationTokens);
                        mapSessions.put(sessionKey, httpSession);
                    }
                }
                final String sessionKey = httpSession.sessionKey();
                final Options options = httpSession.options();
                final List<Integer> conversationTokens = httpSession.conversationTokens();
                int startPosition = conversationTokens.size();

                ChatFormat chatFormat = new ChatFormat(model.tokenizer());
                chatMessages.stream().map(m -> String.format("[%s]> %s", m.role(), m.content())).forEach(System.out::println);
                chatMessages.stream().map(chatFormat::encodeMessage).forEach(conversationTokens::addAll);
                conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
                //System.out.format("Tokens (start-pos %d): %s%n", startPosition, conversationTokens);
                //System.out.println("Text: " + model.tokenizer().decode(conversationTokens).replace("\n", "\\n"));
                Set<Integer> stopTokens = chatFormat.getStopTokens();

                if (options.stream()) {
                    // We use server-side events (SSE) for streaming.
                    exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                    exchange.getResponseHeaders().add("Cache-Control", "no-cache");
                    exchange.sendResponseHeaders(200, 0);
                }

                final long tsCreation = System.currentTimeMillis();
                List<Integer> responseTokens = Llama3.generateTokens(model, httpSession.state(), startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                    if (options.stream()) {
                        if (!model.tokenizer().isSpecialToken(token)) {
                            String sToken = model.tokenizer().decode(List.of(token));
                            System.out.print(sToken);

                            Integer stopToken = null;
                            Map<String, Object> mapResponse = createResponse(model, reqCounter, format, tsCreation,
                                    stopToken, true, sToken);

                            var sbOut = new StringBuilder();
                            dumpJson(sbOut, mapResponse);
                            byte[] buf = String.format("data: %s\n\n", sbOut).getBytes(StandardCharsets.UTF_8);
                            try {
                                exchange.getResponseBody().write(buf);
                                exchange.getResponseBody().flush();
                            } catch (IOException e) {
                                System.err.format("%nRemove session (%s)%n", sessionKey);
                                mapSessions.remove(sessionKey);
                                throw new IllegalStateException("IO-error while sending response", e);
                            }
                        }
                    }
                });
                // Include stop token in the prompt history, but not in the response displayed to the user.
                conversationTokens.addAll(responseTokens);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                    stopToken = responseTokens.getLast();
                    responseTokens.removeLast();
                }
                String responseText = "";
                if (!options.stream()) {
                    responseText = model.tokenizer().decode(responseTokens);
                    System.out.println(responseText);
                }
                Map<String, Object> mapResponse = createResponse(model, reqCounter, format, tsCreation,
                        stopToken, options.stream(), responseText);
                if (stopToken == null) {
                    System.err.println("Ran out of context length...");
                }
                var sbOut = new StringBuilder();
                dumpJson(sbOut, mapResponse);
                byte[] buf;
                if (options.stream()) {
                    buf = String.format("data: %s\n\n", sbOut).getBytes(StandardCharsets.UTF_8);
                } else {
                    buf = String.format("%s\n", sbOut).getBytes(StandardCharsets.UTF_8);
                    exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                    exchange.sendResponseHeaders(200, buf.length);
                }
                exchange.getResponseBody().write(buf);
                exchange.close();
            } catch (Exception e) {
                System.err.println("Error while creating response: " + e.getMessage());
                e.printStackTrace();

                Map<String, Object> mapError = new HashMap<>();
                mapError.put("errormsg", "Error while creating response");
                var sb = new StringBuilder();
                dumpJson(sb, mapError);
                byte[] bufError = sb.toString().getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(400, bufError.length);
                exchange.setAttribute("Content-Type", "application/json");
                exchange.getResponseBody().write(bufError);
                exchange.close();
            }
        };
        try {
            final HttpServer server = HttpServer.create(addr, backlog, rootpath, handler);
            refServer.set(server);
            server.start();
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Couldn't start LLM-server");
        }
    }

    private static Map<String, Object> createResponse(Llama model, final AtomicLong reqCounter,
            JsonFormat format, final long tsCreation, Integer stopToken,
            boolean isDelta, String responseText) {
        Map<String, Object> mapResponse = new LinkedHashMap<>();
        switch (format) {
        case LLAMA_CPP:
            mapResponse.put("content", responseText);
            mapResponse.put("stop", Boolean.valueOf(stopToken != null));
            break;
        case OPENAI:
            createResponseOpenAI(model, reqCounter, tsCreation, stopToken,
                    mapResponse, isDelta, responseText);
            break;
        default:
            throw new IllegalArgumentException("format " + format);
        }
        return mapResponse;
    }

    private static void createResponseOpenAI(Llama model, final AtomicLong reqCounter,
            final long tsCreation, Integer stopToken,
            Map<String, Object> mapResponse, boolean isDelta, String content) {
        mapResponse.put("id", "cc-" + reqCounter.incrementAndGet());
        mapResponse.put("object", "chat.completion");
        mapResponse.put("created", Long.toString(tsCreation / 1000L));
        mapResponse.put("model", model.modelName());
        List<Object> choices = new ArrayList<>();
        Map<String, Object> choice0 = new LinkedHashMap<>();
        choice0.put("index", "0");
        Map<String, Object> respMsg = new LinkedHashMap<>();
        respMsg.put("role", "assistant");
        respMsg.put("content", content);
        choice0.put(isDelta ? "delta" : "message", respMsg);
        choice0.put("logprobs", null);
        String finishReason = null;
        if (!isDelta) {
            finishReason = (stopToken == null) ? "length" : "stop";
        }
        choice0.put("finishReason", finishReason);
        choices.add(choice0);
        mapResponse.put("choices", choices);
    }

    @SuppressWarnings("unchecked")
    private static void dumpJson(StringBuilder sb, Map<String, Object> map) {
        sb.append('{');
        String as = "";
        for (Entry<String, Object> entry : map.entrySet()) {
            sb.append(as);
            dumpString(sb, entry.getKey());
            sb.append(':');
            var value = entry.getValue();
            if (value == null) {
                sb.append("null");
            }
            else if (value instanceof String s) {
                dumpString(sb, s);
            }
            else if (value instanceof List) {
                dumpJson(sb, (List<Object>) value);
            }
            else if (value instanceof Map) {
                dumpJson(sb, (Map<String, Object>) value);
            }
            else if (value instanceof Boolean b) {
                sb.append(b);
            }
            else {
                throw new IllegalArgumentException("Unexpected value of type " + value.getClass());
            }
            as = ",";
        }
        sb.append('}');
    }

    @SuppressWarnings("unchecked")
    private static void dumpJson(StringBuilder sb, List<Object> list) {
        sb.append('[');
        String as = "";
        for (Object value : list) {
            sb.append(as);
            if (value == null) {
                sb.append("null");
            }
            else if (value instanceof String s) {
                dumpString(sb, s);
            }
            else if (value instanceof List) {
                sb.append(value);
            }
            else if (value instanceof Map) {
                dumpJson(sb, (Map<String, Object>) value);
            }
            else if (value instanceof Boolean b) {
                sb.append(b);
            }
            else {
                throw new IllegalArgumentException("Unexpected value of type " + value.getClass());
            }
            as = ",";
        }
        sb.append(']');
    }

    private static void dumpString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            final char c = s.charAt(i);
            if (c == '"') {
                sb.append("\\\"");
            } else if ((c >= ' ' && c < 0x7f) || (c >= 0xa1 && c <= 0xff)) {
                sb.append(c);
            } else if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\r') {
                sb.append("\\r");
            } else if (c == '\t') {
                sb.append("\\t");
            } else {
                sb.append("\\u");
                final String sHex = Integer.toHexString(c);
                for (int j = sHex.length(); j < 4; j++) {
                    sb.append('0');
                }
                sb.append(sHex);
            }
        }
        sb.append('"');
    }

    static class TeeBufferedReader extends BufferedReader {
        final StringBuilder sb = new StringBuilder();
        /**
         * Constructor
         * @param in stream to be copied and read
         */
        public TeeBufferedReader(Reader in) {
            super(in);
        }

        public int read() throws IOException {
            int c = super.read();
            if (c >= 0) {
                sb.append((char) c);
            }
            return c;
        }
    }

    private static List<Object> parseJsonArray(BufferedReader br) throws IOException {
        // The '[' has been read already.
        List<Object> list = new ArrayList<>();
        boolean needComma = false;
        while (true) {
            char c = readChar(br, true);
            if (c == ']') {
                break;
            }
            if (needComma) {
                if (c != ',') {
                    throw new IllegalArgumentException(String.format("Missing comma but (%c), list: %s", c, list));
                }
                c = readChar(br, true);
            }
            Object value;
            if (c == '"') {
                value = readString(br);
            }
            else if (c == '{') {
                value = parseJsonDict(br);
            }
            else if (c == '[') {
                value = parseJsonArray(br);
            }
            else {
                var sb = new StringBuilder();
                while (true) {
                    if (c == '}' || c == ',') {
                        break;
                    }
                    if ((c >= 'a' && c <= 'z') || (c == 'E') || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+') {
                        sb.append(c);
                        c = readChar(br, false);
                    } else {
                        throw new IllegalArgumentException("Illegal value character: " + c);
                    }
                }
                if (sb.length() == 0) {
                    throw new IllegalArgumentException("Missing value");
                }
                value = parseJsonValue(sb.toString());
            }
            list.add(value);
            needComma = (c != ',');
        }
        return list;
    }

    /**
     * This is a simple (not complete, but without dependency) JSON-parser used to parse llama.cpp-responses.
     * Use a parser of https://json.org/ to get a complete implementation.
     * @param br reader containing a JSON document
     * @return map from key to value
     * @throws IOException in case of an IO error
     */
    private static Map<String, Object> parseJsonDict(BufferedReader br) throws IOException {
        // The '{' has been read already.
        Map<String, Object> map = new LinkedHashMap<>();
        boolean needComma = false;
        while (true) {
            char c;
            try {
                c = readChar(br, true);
            } catch (IllegalArgumentException e) {
                System.err.println("Map(part): " + map);
                throw e;
            }
            if (c == '}') {
                break;
            }
            if (needComma) {
                if (c != ',') {
                    throw new IllegalArgumentException("Missing comma: " + c);
                }
                c = readChar(br, true);
            }
            if (c != '"') {
                throw new IllegalArgumentException("Illegal key: " + c);
            }
            String key = readString(br);
            c = readChar(br, true);
            if (c != ':') {
                throw new IllegalArgumentException("Illegal character after key: " + c);
            }
            c = readChar(br, true);
            Object value;
            if (c == '"') {
                value = readString(br);
            }
            else if (c == '{') {
                value = parseJsonDict(br);
            }
            else if (c == '[') {
                value = parseJsonArray(br);
            }
            else {
                var sb = new StringBuilder();
                while (true) {
                    if (c == '}' || c == ',') {
                        break;
                    }
                    if ((c >= 'a' && c <= 'z') || (c == 'E') || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+') {
                        sb.append(c);
                        c = readChar(br, false);
                    } else if ((c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
                        break;
                    } else {
                        throw new IllegalArgumentException(String.format("Illegal value character (\\u%04x, '%c')", (int) c, c));
                    }
                }
                if (sb.length() == 0) {
                    throw new IllegalArgumentException("Missing value of key " + key);
                }
                value = parseJsonValue(sb.toString());
                if (c == '}') {
                    map.put(key, value);
                    break;
                }
            }
            map.put(key, value);
            needComma = (c != ',');
        }
        return map;
    }

    private static Object parseJsonValue(String value) {
        if ("null".equals(value)) {
            return null;
        }
        if ("true".equals(value)) {
            return Boolean.TRUE;
        }
        if ("false".equals(value)) {
            return Boolean.FALSE;
        }
        // value has to be a JSON-number.
        BigDecimal bd = new BigDecimal(value); // We accept some more values, e.g. "+5" instead of "5".
        if (bd.scale() == 0 && BigDecimal.valueOf(Integer.MAX_VALUE).compareTo(bd) >= 0
                && BigDecimal.valueOf(Integer.MIN_VALUE).compareTo(bd) <= 0) {
            return Integer.valueOf(bd.intValueExact());
        }
        return bd;
    }

    /**
     * Gets a JSON-value, if it exists.
     * @param <V> type of the expected value
     * @param map JSON-dictionary
     * @param key key
     * @param clazz class of the expected value
     * @return value or <code>null</code>
     */
    @SuppressWarnings("unchecked")
    static <V> V getJsonValue(Map<String, Object> map, String key, Class<V> clazz) {
        Object o = map.get(key);
        if (o == null) {
            return null;
        }
        if (clazz.isInstance(o)) {
            return (V) o;
        }
        throw new IllegalArgumentException(String.format("Unexpeted value-type (%s) of value (%s) at key (%s)", o.getClass(), o, key));
    }

    /**
     * Gets a JSON-array, if it exists.
     * @param map JSON-dictionary
     * @param key key
     * @return JSON-array or <code>null</code>
     */
    @SuppressWarnings("unchecked")
    static List<Object> getJsonArray(Map<String, Object> map, String key) {
        Object o = map.get(key);
        if (o == null) {
            return null;
        }
        if (!(o instanceof List)) {
            throw new IllegalArgumentException(String.format("Unexpected value-type (%s) of value (%s) at key (%s), expected json-array", o.getClass(), o, key));
        }
        return (List<Object>) o;
    }

    /**
     * Gets a JSON-array of dictionaries, if it exists.
     * @param map JSON-dictionary
     * @param key key
     * @return JSON-array or <code>null</code>
     */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> getJsonArrayDicts(Map<String, Object> map, String key) {
        List<Object> listObj = getJsonArray(map, key);
        if (listObj == null) {
            return null;
        }
        for (Object o : listObj) {
            if (!(o instanceof Map)) {
                throw new IllegalArgumentException(String.format("Unexpected value-type (%s) of value (%s) in list of key (%s), expected json-array with dictionaries", o.getClass(), o, key));
            }
        }
        return (List<Map<String, Object>>) (Object) listObj;
    }

    private static String readString(BufferedReader br) throws IOException {
        var sb = new StringBuilder();
        while (true) {
            char c = readChar(br, false);
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                c = readChar(br, false);
                if (c == '"') {
                    ;
                }
                else if (c == 't') {
                    c = '\t';
                }
                else if (c == 'n') {
                    c = '\n';
                }
                else if (c == 'r') {
                    c = '\r';
                }
                else if (c == 'b') {
                    c = '\b';
                }
                else if (c == 'f') {
                    c = '\f';
                }
                else if (c == '/') {
                    ;
                }
                else if (c == 'u') {
                    char[] cBuf = new char[4];
                    for (int i = 0; i < 4; i++) {
                        cBuf[i] = readChar(br, false);
                    }
                    try {
                        c = (char) Integer.parseInt(new String(cBuf), 16);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("Unexpected unicode-escape: " + new String(cBuf));
                    }
                }
                else {
                    throw new IllegalArgumentException("Unexpected escape character: " + c);
                }
                sb.append(c);
                continue;
            }
            sb.append(c);
        }
        return sb.toString();
    }

    private static char readChar(BufferedReader br, boolean ignoreWS) throws IOException {
        while (true) {
            int c = br.read();
            if (c == -1) {
                throw new IllegalArgumentException("Unexpected end of stream");
            }
            if (ignoreWS && (c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
                continue;
            }
            return (char) c;
        }
    }

    private static void readChar(BufferedReader br, boolean ignoreWS, char expected) throws IOException {
        while (true) {
            int c = br.read();
            if (c == -1) {
                throw new IllegalArgumentException(String.format("Unexpected end of stream, expected '%c', U+%04x", expected, (int) expected));
            }
            if (ignoreWS && (c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
                continue;
            }
            if (c == expected) {
                break;
            }
            throw new IllegalArgumentException(String.format("Unexpected character '%c' (0x%04x) instead of '%c'",
                        c, c, expected));
        }
    }

    private static float readFloat(Map<String, Object> map, String key, float defaultValue) {
        Object oValue = map.get(key);
        if (oValue == null) {
            return defaultValue;
        }
        if (oValue instanceof Integer iValue) {
            return iValue;
        }
        if (oValue instanceof BigDecimal bd) {
            return bd.floatValue();
        }
        throw new IllegalStateException(String.format("Unexpected type (%s / %s) at key (%s), expected float", oValue.getClass(), oValue, key));
    }

    private static int readInt(Map<String, Object> map, String key, int defaultValue) {
        Object oValue = map.get(key);
        if (oValue == null) {
            return defaultValue;
        }
        if (oValue instanceof Integer iValue) {
            return iValue;
        }
        if (oValue instanceof BigDecimal bd) {
            return bd.intValueExact();
        }
        throw new IllegalStateException(String.format("Unexpected type (%s / %s) at key (%s), expected int", oValue.getClass(), oValue, key));
    }

    private static long readLong(Map<String, Object> map, String key, long defaultValue) {
        Object oValue = map.get(key);
        if (oValue == null) {
            return defaultValue;
        }
        if (oValue instanceof Integer iValue) {
            return iValue;
        }
        if (oValue instanceof BigDecimal bd) {
            return bd.longValueExact();
        }
        throw new IllegalStateException(String.format("Unexpected type (%s / %s) at key (%s), expected long", oValue.getClass(), oValue, key));
    }

    private static boolean readBoolean(Map<String, Object> map, String key, boolean defaultValue) {
        Object oValue = map.get(key);
        if (oValue == null) {
            return defaultValue;
        }
        if (oValue instanceof Boolean bValue) {
            return bValue;
        }
        throw new IllegalStateException(String.format("Unexpected type (%s / %s) at key (%s), expected boolean", oValue.getClass(), oValue, key));
    }

}

final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;
    private long tensorDataOffset;
    private MemorySegment tensorData; // memory mapped tensor data
    private Map<String, GGMLTensorEntry> tensorEntries;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

    public Map<String, GGMLTensorEntry> getTensorEntries() {
        return tensorEntries;
    }

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath);
             var ignored = Timer.log("Parse " + modelPath)) {
            GGUF gguf = new GGUF();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }

    enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        UINT8(1),
        // The value is a 8-bit signed integer.
        INT8(1),
        // The value is a 16-bit unsigned little-endian integer.
        UINT16(2),
        // The value is a 16-bit signed little-endian integer.
        INT16(2),
        // The value is a 32-bit unsigned little-endian integer.
        UINT32(4),
        // The value is a 32-bit signed little-endian integer.
        INT32(4),
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32(4),
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        BOOL(1),
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        STRING(-8),
        // The value is an array of other values, with the length and type prepended.
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        ARRAY(-8),
        // The value is a 64-bit unsigned little-endian integer.
        UINT64(8),
        // The value is a 64-bit signed little-endian integer.
        INT64(8),
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64(8);

        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        public int byteSize() {
            return byteSize;
        }
    }

    @SuppressWarnings("preview")
    private void loadModelImpl(FileChannel fileChannel) throws IOException {
        // The header of the file.
        readHeader(fileChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        //long _padding = -fileChannel.position() & (ALIGNMENT - 1);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = fileChannel.position();
        Arena arena = Arena.ofAuto();
        this.tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        this.tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUF.GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUF.GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(FileChannel fileChannel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }

    void readHeader(FileChannel fileChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); // uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(FileChannel fileChannel) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (value_type) {
            case UINT8, INT8 -> {
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(fileChannel);
                }
                return bytes;
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(fileChannel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(fileChannel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(fileChannel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(fileChannel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(fileChannel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(fileChannel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }

    private byte readByte(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }

    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    private short readShort(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }

    private int readInt(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }

    private long readLong(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }

    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}

final class ModelLoader {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_LLAMA_3_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public static Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            GGUF gguf = GGUF.loadModel(ggufPath);
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            int modelContextLength = (int) metadata.get("llama.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            Llama.Configuration config = new Llama.Configuration(
                    (int) metadata.get("llama.embedding_length"),
                    (int) metadata.get("llama.feed_forward_length"),
                    (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),

                    metadata.containsKey("llama.attention.head_count_kv")
                            ? (int) metadata.get("llama.attention.head_count_kv")
                            : (int) metadata.get("llama.attention.head_count"),

                    vocabulary.size(),
                    contextLength,
                    false,
                    (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)
            );

            boolean ropeScaling = "Meta-Llama-3.1".equals(metadata.get("general.basename"));
            float scaleFactor = 8;
            float loFreqFactor = 1;
            float hiFreqFactor = 3;
            int oldContextLength = 8192;
            Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                    ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
            float[] ropeFreqsReal = ropeFreqs.first();
            float[] ropeFreqsImag = ropeFreqs.second();

            Map<String, GGMLTensorEntry> tensorEntries = gguf.getTensorEntries();
            GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
            Llama.Weights qw = new Llama.Weights(
                    loadQuantized(tokenEmbeddings),
                    loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                    null, null, null,
                    loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                    loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                    toFloatBuffer(tensorEntries.get("output_norm.weight")),
                    FloatBuffer.wrap(ropeFreqsReal),
                    FloatBuffer.wrap(ropeFreqsImag),
                    // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
                    // This is commonly referred as "tie word embeddings".
                    loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
            );

            return new Llama(ggufPath.getFileName().toString().replaceFirst("[.]gguf$", ""), config, tokenizer, qw);
        }
    }

    private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        // int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens, null);
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}

/** mask of a byte-sequence in UTF-8 encoding */
record Utf8Mask(int mask, int pattern, int len) {
    static final Utf8Mask[] MASKS = {
            new Utf8Mask(0b11100000, 0b11000000, 2),
            new Utf8Mask(0b11110000, 0b11100000, 3),
            new Utf8Mask(0b11111000, 0b11110000, 4)
    };
}

/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Tokenizer {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenTypes;

    /** buffer to store incomplete UTF-8 sequence */
    private final byte[] bufUtf8 = new byte[4];
    /** index in UTF-8 buffer */
    private int currUtf8Index = 0;
    /** current UTF-8 mask */
    private Utf8Mask currUtf8Mask;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    public int getTokenType(int tokenIndex) {
        if (tokenTypes == null) {
            throw new IllegalStateException("Tokenizer hasn't been constructed using tokenTypes");
        }
        return tokenTypes[tokenIndex];
    }

    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern,
            Map<String, Integer> specialTokens, int[] tokenTypes) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        this.tokenTypes = tokenTypes;
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
    List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length + 3];
        int indexRawByte = 0;
    loopDecoded:
        for (int i = 0; i < decoded.length(); i++) {
            byte b = (byte) decodedBytesAsInts[i];
            if (currUtf8Index == 0) {
                for (Utf8Mask utf8Mask : Utf8Mask.MASKS) {
                    if ((b & utf8Mask.mask()) == utf8Mask.pattern()) {
                        currUtf8Mask = utf8Mask;
                        bufUtf8[currUtf8Index++] = b;
                        continue loopDecoded;
                    }
                }
            }
            if (currUtf8Index > 0 && currUtf8Mask != null) {
                bufUtf8[currUtf8Index++] = b;
                if (currUtf8Index == currUtf8Mask.len()) {
                    System.arraycopy(bufUtf8, 0, rawBytes, indexRawByte, currUtf8Mask.len());
                    indexRawByte += currUtf8Mask.len();
                    currUtf8Index = 0;
                    currUtf8Mask = null;
                }
                continue;
            }
            rawBytes[indexRawByte++] = b;
        }
        return new String(rawBytes, 0, indexRawByte, StandardCharsets.UTF_8);
    }
}

final class JsonProcessing {
    public static String escapeString(String s) {
        var sb = new StringBuilder(s.length() + 10);
        dumpString(sb, s);
        return sb.toString();
    }
    
    private static void dumpString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            final char c = s.charAt(i);
            if (c == '"') {
                sb.append("\\\"");
            } else if ((c >= ' ' && c < 0x7f) || (c >= 0xa1 && c <= 0xff)) {
                sb.append(c);
            } else if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\r') {
                sb.append("\\r");
            } else if (c == '\t') {
                sb.append("\\t");
            } else {
                sb.append("\\u");
                final String sHex = Integer.toHexString(c);
                for (int j = sHex.length(); j < 4; j++) {
                    sb.append('0');
                }
                sb.append(sHex);
            }
        }
        sb.append('"');
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (endExclusive - startInclusive == 1) {
            action.accept(startInclusive);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

final class Float16 {
    public static final int BYTES = 2;
}

enum GGMLType {
    F32(Float.BYTES),
    F16(Float16.BYTES),
    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),
    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES);

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(int numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return Math.toIntExact(t / getBlockSize());
    }

    public static final int QK_K = 256; // or 64?

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
abstract class FloatTensor {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    // static final ValueLayout.OfFloat JAVA_FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    // static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
   
    // The use of Unsafe in this file is a temporary workaround to support native-image.
    static final Unsafe UNSAFE;
    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static short readShort(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    static byte readByte(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES = USE_VECTOR_API
            ? VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class)
            : null;

    abstract int size();

    abstract float getFloat(int index);

    abstract void setFloat(int index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1));
        });
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int argmax() {
        return argmax(0, size());
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }

    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }

    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(int thisOffset, int size) {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_0} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
final class Q4_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant;
        int modIndex = index % GGMLType.Q4_0.getBlockSize();
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = size / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = sum0.add(sum2).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q8_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class F32FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment segment;

    F32FloatTensor(int size, MemorySegment segment) {
        this.size = size;
        this.segment = segment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public float getFloat(int index) {
        return segment.get(OfFloat.JAVA_FLOAT, index * Float.BYTES);
    }

    @Override
    public void setFloat(int index, float value) {
        segment.set(OfFloat.JAVA_FLOAT, index * Float.BYTES, value);
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("getFloatVector is not yet implemented.");
    }
}

final class ArrayFloatTensor extends FloatTensor {

    final float[] values;

    ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return values[index];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
}

final class RoPE {
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
        boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        // nothing to do: freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }
}

@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(FloatTensor logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1; // in case of rounding errors
    }
}

final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        Comparator<Integer> comparator = Comparator.comparingDouble(logits::getFloat).reversed();

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break; // we've exceeded topp by including lastIndex
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex]; // in case of rounding errors
    }
}

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;

    public ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return Set.of(endOfText, endOfTurn);
    }

    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (ChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    public record Message(ChatFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
        public static ChatFormat.Role USER = new ChatFormat.Role("user");
        public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}
