///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+

// Practical Qwen2 inference in a single Java file
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Also please check the sibling projects:
//  - https://github.com/mukel/llama3.java
//  - https://github.com/mukel/mistral.java
//
// Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
//
package org.rogmann.llmva4j;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.rogmann.llmva4j.Llama.Configuration;
import org.rogmann.llmva4j.Qwen2Llama.State;
import org.rogmann.llmva4j.Qwen2Llama.Weights;

/**
 * Java implementation of Qwen-2 model using Vector API.
 */
public class Qwen2 {

    static void runInteractive(Qwen2Llama model, Sampler sampler, Options options) {
        State state = null;
        Qwen2ChatMLFormat chatFormat = new Qwen2ChatMLFormat(model.tokenizer());
        List<Integer> conversationTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        try (Scanner in = new Scanner(System.in)) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                if (state == null) {
                    // State allocation can take some time for large context sizes,
                    // allocate the model state only after printing the user '>' prompt.
                    state = model.createNewState();
                }
                String userText = in.nextLine();
                if (List.of("quit", "exit").contains(userText)) {
                    break;
                }
                conversationTokens.addAll(chatFormat.encodeMessage(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.USER, userText)));
                conversationTokens.addAll(chatFormat.encodeHeader(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.ASSISTANT, "")));
                Set<Integer> stopTokens = chatFormat.getStopTokens();
                List<Integer> responseTokens = Qwen2Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                    if (options.stream()) {
                        int tokenType = model.tokenizer().getTokenType(token);
                        if (tokenType == 1 || tokenType == 6) {
                            System.out.print(model.tokenizer().decode(List.of(token)));
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
                if (!options.stream()) {
                    String responseText = model.tokenizer().decode(responseTokens);
                    System.out.println(responseText);
                }
                if (stopToken == null) {
                    System.err.println("Ran out of context length...");
                    break;
                }
            }
        }
    }

    static void runInstructOnce(Qwen2Llama model, Sampler sampler, Options options) {
        State state = model.createNewState();
        Qwen2ChatMLFormat chatFormat = new Qwen2ChatMLFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Qwen2Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if (options.stream()) {
                int tokenType = model.tokenizer().getTokenType(token);
                if (tokenType == 1 || tokenType == 6) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }

    record Options(Path modelPath, String prompt, String systemPrompt, String suffix, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo) {

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
            out.println("Usage:  jbang Qwen2.java [options]");
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
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default 512");
            out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
            out.println();
            out.println("Examples:");
            out.println("  jbang Qwen2.java --model qwen2-7b-q4_0.gguf --chat");
            out.println("  jbang Qwen2.java --model qwen2-7b-q4_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Qwen2.java --model qwen2-7b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            String suffix = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Mistral models have a rather large context (> 32k)
            // Cap max context length at 512 to run out-of-the-box on low memory devices
            int maxTokens = 512;
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
                            case "--suffix" -> suffix = nextArg;
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
            return new Options(modelPath, prompt, systemPrompt, suffix, interactive, temperature, topp, seed, maxTokens, stream, echo);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Qwen2Llama model = Qwen2ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        Sampler sampler = Llama3.selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}

final class Qwen2ModelLoader {
    private static final String TOKENIZER_QWEN2_7B_MODEL = "gpt2";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_QWEN2_7B_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_QWEN2_7B_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public static Qwen2Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Qwen2 model")) {
            GGUF gguf = GGUF.loadModel(ggufPath);
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadVocabulary(metadata);
            Qwen2Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            int modelContextLength = (int) metadata.get("qwen2.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            Llama.Configuration config = new Llama.Configuration(
                    (int) metadata.get("qwen2.embedding_length"),
                    (int) metadata.get("qwen2.feed_forward_length"),
                    (int) metadata.get("qwen2.block_count"),
                    (int) metadata.get("qwen2.attention.head_count"),

                    metadata.containsKey("qwen2.attention.head_count_kv")
                            ? (int) metadata.get("qwen2.attention.head_count_kv")
                            : (int) metadata.get("qwen2.attention.head_count"),

                    vocabulary.size(),
                    contextLength,
                    false,
                    (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon"),
                    (float) metadata.get("qwen2.rope.freq_base")
            );

            Map<String, GGMLTensorEntry> tensorEntries = gguf.getTensorEntries();

            Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                    false, 8, 1, 3, 8192);
            float[] ropeFreqsReal = ropeFreqs.first();
            float[] ropeFreqsImag = ropeFreqs.second();


            FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensorEntries.get("token_embd.weight"));
            Weights qw = new Weights(
                    tokenEmbeddingTable,
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),

                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),

                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                    ModelLoader.loadQuantized(tensorEntries.get("output_norm.weight")),
                    new ArrayFloatTensor(ropeFreqsReal),
                    new ArrayFloatTensor(ropeFreqsImag),
                    tensorEntries.containsKey("output.weight")
                            ? ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                            : tokenEmbeddingTable // weights are shared
            );

            return new Qwen2Llama(config, tokenizer, qw);
        }
    }

    private final static String QWEN2_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Qwen2Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = vocabulary.getIndex("<|endoftext|>").orElseThrow(); // assume all tokens after the base ones are special.
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

        return new Qwen2Tokenizer(vocabulary, merges, QWEN2_PATTERN, specialTokens, tokenTypes);
    }

}

record Qwen2Llama(Configuration configuration, Qwen2Tokenizer tokenizer, Weights weights) {
    public State createNewState() {
        State state = new State(configuration());
        state.latestToken = tokenizer.getSpecialTokens().get("<|im_start|>");
        return state;
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatTensor[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size)
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)

        public final FloatTensor[] q_bias; // (layer, dim)
        public final FloatTensor[] k_bias; // (layer, kv_dim)
        public final FloatTensor[] v_bias; // (layer, kv_dim)

        public final FloatTensor[] rms_ffn_weight; // (layer, dim)
        // weights for ffn
        public final FloatTensor[] w1; // (layer, hidden_dim, dim)
        public final FloatTensor[] w2; // (layer, dim, hidden_dim)
        public final FloatTensor[] w3; // (layer, hidden_dim, dim)
        // public final rmsnorm
        public final FloatTensor rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatTensor freq_cis_real; // (seq_len, head_size/2)
        public final FloatTensor freq_cis_imag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatTensor wcls; // (vocab_size, dim)

        public Weights(FloatTensor token_embedding_table, FloatTensor[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] q_bias, FloatTensor[] k_bias, FloatTensor[] v_bias, FloatTensor[] wo, FloatTensor[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatTensor rms_final_weight, FloatTensor freq_cis_real, FloatTensor freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;

            this.q_bias = q_bias;
            this.k_bias = k_bias;
            this.v_bias = v_bias;

            this.wo = wo;
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
        public final FloatTensor x; // activation at current time stamp (dim,)
        public final FloatTensor xb; // same, but inside a residual branch (dim,)
        public final FloatTensor xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor q; // query (dim,)
        public final FloatTensor k; // key (kvDim,)
        public final FloatTensor v; // value (kvDim,)
        public final FloatTensor att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits
        // kv cache
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

        public int latestToken;

        State(Configuration config) {
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.x = ArrayFloatTensor.allocate(config.dim);
            this.xb = ArrayFloatTensor.allocate(config.dim);
            this.xb2 = ArrayFloatTensor.allocate(config.dim);
            this.hb = ArrayFloatTensor.allocate(config.hiddenDim);
            this.hb2 = ArrayFloatTensor.allocate(config.hiddenDim);
            this.q = ArrayFloatTensor.allocate(config.dim);
            this.k = ArrayFloatTensor.allocate(kvDim);
            this.v = ArrayFloatTensor.allocate(kvDim);
            this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatTensor weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.getFloat(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Qwen2Llama model, State state, int token, int position) {
        // a few convenience variables
        Llama.Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            if (weights.q_bias != null && weights.q_bias[l] != null) {
                state.q.addInPlace(weights.q_bias[l]);
            }
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            if (weights.k_bias != null && weights.k_bias[l] != null) {
                state.k.addInPlace(weights.k_bias[l]);
            }
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);
            if (weights.v_bias != null && weights.v_bias[l] != null) {
                state.v.addInPlace(weights.v_bias[l]);
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads; ++h) {
                int rotn = h < config.numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + ic);
                    float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + ic);
                    for (int v = 0; v < rotn; v++) {
                        FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + headSize/2);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + headSize/2, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

        return state.logits;
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Qwen2Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        int totalTokens = promptIndex + generatedTokens.size();
        System.err.printf("%n%.2f tokens/s (%d)%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens);

        return generatedTokens;
    }
}

/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Qwen2Tokenizer {
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
        return tokenTypes[tokenIndex];
    }

    public Qwen2Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens, int[] tokenTypes) {
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

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = Tokenizer.findAll(compiledPattern, text);
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
            ids = Tokenizer.merge(ids, pair, idx);
        }
        return ids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(Tokenizer.BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(Tokenizer.BYTE_DECODER::get).toArray();
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

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
class Qwen2ChatMLFormat {

    protected final Qwen2Tokenizer tokenizer;
    protected final int imStart;
    protected final int endOfText;
    protected final int imEnd;

    public Qwen2ChatMLFormat(Qwen2Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.imStart = specialTokens.get("<|im_start|>");
        this.imEnd = specialTokens.get("<|im_end|>");
        this.endOfText = specialTokens.get("<|endoftext|>");
    }

    public Qwen2Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return Set.of(imEnd, endOfText);
    }

    public List<Integer> encodeHeader(Qwen2ChatMLFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(Qwen2ChatMLFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(imEnd);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Qwen2ChatMLFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        for (Qwen2ChatMLFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new Qwen2ChatMLFormat.Message(Qwen2ChatMLFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    public record Message(Qwen2ChatMLFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static Qwen2ChatMLFormat.Role SYSTEM = new Qwen2ChatMLFormat.Role("system");
        public static Qwen2ChatMLFormat.Role USER = new Qwen2ChatMLFormat.Role("user");
        public static Qwen2ChatMLFormat.Role ASSISTANT = new Qwen2ChatMLFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}
