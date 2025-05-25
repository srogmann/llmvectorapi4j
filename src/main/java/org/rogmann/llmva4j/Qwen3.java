///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+

// Practical Qwen3 inference in a single Java file
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

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.rogmann.llmva4j.AttentionCollector.AttentionConsumer;
import org.rogmann.llmva4j.ChatFormat.ChatTokens;
import org.rogmann.llmva4j.ChatFormat.Message;
import org.rogmann.llmva4j.ChatFormat.MessageWithTokens;
import org.rogmann.llmva4j.ChatFormat.Role;
import org.rogmann.llmva4j.Llama.Options;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;
import org.rogmann.llmva4j.Qwen3Llama.State;
import org.rogmann.llmva4j.Qwen3Llama.Weights;

/**
 * Java implementation of Qwen-3 model using Vector API.
 */
public class Qwen3 {

    static void runInteractive(Qwen3Llama model, Sampler sampler, Options options) {
        State state = null;
        ChatFormat chatFormat = model.chatFormat();
        List<MessageWithTokens> conversation = new ArrayList<>();
        List<TokenDetails> conversationTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            Message msgSystem = new Message(Role.SYSTEM, options.systemPrompt());
            List<Integer> tokens = chatFormat.encodeMessage(msgSystem);
            conversation.add(new MessageWithTokens(msgSystem.role(), msgSystem.content(), tokens));
            conversationTokens.addAll(chatFormat.toTokenDetails(tokens, conversationTokens.size()));
        }

        int startPosition = 0;
        try (Scanner in = new Scanner(System.in)) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                if (state == null) {
                    // State allocation can take some time for large context sizes,
                    // allocate the model state only after printing the user '>' prompt.
                    state = model.createNewState(Llama.BATCH_SIZE);
                }
                String userText = in.nextLine();
                if (List.of("quit", "exit").contains(userText)) {
                    break;
                }
                if (userText.startsWith("/save:")) {
                    StateCache stateCache = new StateCache(model.configuration(), state);
                    try {
                        String msg = stateCache.saveKVCache(userText, options.stateCacheFolder(), conversationTokens, conversation);
                        System.out.println(msg);
                    } catch (IllegalStateException e) {
                        System.err.println(e.getMessage());
                    }
                    continue;
                }
                Message msgUser = new Message(Role.USER, userText);
                List<Integer> tokensUser = chatFormat.encodeMessage(msgUser);
                conversationTokens.addAll(chatFormat.toTokenDetails(tokensUser, conversationTokens.size()));
                List<Integer> tokensHeader = chatFormat.encodeHeader(new Message(Role.ASSISTANT, ""));
                conversationTokens.addAll(chatFormat.toTokenDetails(tokensHeader, conversationTokens.size()));
                conversation.add(new MessageWithTokens(msgUser.role(), msgUser.content(), tokensUser));
                Set<Integer> stopTokens = chatFormat.getStopTokens();
                List<Integer> promptTokens = conversationTokens.subList(startPosition, conversationTokens.size()).stream().map(TokenDetails::token).toList();
                StringBuilder sbResponse = new StringBuilder(200);
                List<Integer> tokensResponse = new ArrayList<>(tokensHeader);
                List<TokenDetails> responseTokens = Llama.generateTokens(model, state, startPosition, promptTokens
                        , stopTokens, options.maxTokens(), sampler,
                        options.stateCache(), options.echo(), tokenDetail -> {
                    if (options.stream()) {
                        int tokenType = model.tokenizer().getTokenType(tokenDetail.token());
                        if (tokenType == 1 || tokenType == 6) {
                            String sToken = model.tokenizer().decode(List.of(tokenDetail.token()));
                            System.out.print(sToken);
                            sbResponse.append(sbResponse);
                        }
                    }
                }, options.attentionTrace());
                // Include stop token in the prompt history, but not in the response displayed to the user.
                conversationTokens.addAll(responseTokens);
                responseTokens.stream().mapToInt(TokenDetails::token).forEach(tokensResponse::add);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast().token())) {
                    stopToken = responseTokens.getLast().token();
                    responseTokens.removeLast();
                }
                if (!options.stream()) {
                    String responseText = model.tokenizer().decode(responseTokens.stream().map(TokenDetails::token).toList());
                    System.out.println(responseText);
                    sbResponse.append(responseText);
                }

                conversation.add(new MessageWithTokens(Role.ASSISTANT, sbResponse.toString(), tokensResponse));
                if (stopToken == null) {
                    System.err.println("Ran out of context length...");
                    break;
                }
            }
        }
        String attTraceFile = System.getProperty("llama.attentionTrace.file");
        if (attTraceFile != null) {
            AttentionCollector.writeAttentionsIntoFile(new File(attTraceFile).toPath(), model, conversationTokens);
        }
    }

    static void runInstructOnce(Qwen3Llama model, Sampler sampler, Options options) {
        State state = model.createNewState(Llama.BATCH_SIZE);
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> promptTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new Message(Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new Message(Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<TokenDetails> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler,
                options.stateCache(), options.echo(), tokenDetail -> {
            if (options.stream()) {
                int tokenType = model.tokenizer().getTokenType(tokenDetail.token());
                if (tokenType == 1 || tokenType == 6) {
                    System.out.print(model.tokenizer().decode(List.of(tokenDetail.token())));
                }
            }
        }, options.attentionTrace());
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast().token())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens.stream().map(TokenDetails::token).toList());
            System.out.println(responseText);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Qwen3Llama model = Qwen3ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        Sampler sampler = Llama.selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.serverPath() != null) {
            LlamaHttpServer.runHttpServer(model, sampler, options, options.serverHost(), options.serverPort());
        } else if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}

final class Qwen3ModelLoader {
    private static final String TOKENIZER_QWEN3_7B_MODEL = "gpt2";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_QWEN3_7B_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_QWEN3_7B_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    static String valueToString(Object o) {
        if (o instanceof String s) {
            String sEscaped = s.replace("\\",  "\\\\").replace("\n", "\\n");
            return sEscaped.length() < 80 ? s : sEscaped.substring(0, 80) + "[...]";
        }
        if (o instanceof int[] arr) {
            return arr.length < 10 ? Arrays.toString(arr) : Arrays.toString(Arrays.copyOfRange(arr, 0, 10)) + "[...]";
        }
        if (o instanceof String[] arr) {
            return arr.length < 10 ? Arrays.toString(arr) : Arrays.toString(Arrays.copyOfRange(arr, 0, 10)) + "[...]";
        }
        if (o instanceof GGMLTensorEntry tensor) {
            return String.format("GGMLTE Shape %s, Type %s", Arrays.toString(tensor.shape()), tensor.ggmlType());
        }
        return o.toString();
    }
    public static Qwen3Llama loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load Qwen3 model")) {
            GGUF gguf = GGUF.loadModel(ggufPath);
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadVocabulary(metadata);
            boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);

            int modelContextLength = (int) metadata.get("qwen3.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            String modelName = ggufPath.getFileName().toString();
            Llama.Configuration config = new Llama.Configuration(
                    modelName,
                    (int) metadata.get("qwen3.embedding_length"),
                    (int) metadata.get("qwen3.feed_forward_length"),
                    (int) metadata.get("qwen3.block_count"),
                    (int) metadata.get("qwen3.attention.head_count"),

                    metadata.containsKey("qwen3.attention.head_count_kv")
                            ? (int) metadata.get("qwen3.attention.head_count_kv")
                            : (int) metadata.get("qwen3.attention.head_count"),
                   (int) metadata.get("qwen3.attention.key_length"),
                   (int) metadata.get("qwen3.attention.value_length"),

                    vocabulary.size(),
                    modelContextLength, contextLength,
                    false,
                    (float) metadata.get("qwen3.attention.layer_norm_rms_epsilon"),
                    (float) metadata.get("qwen3.rope.freq_base")
            );

            Map<String, GGMLTensorEntry> tensorEntries = gguf.getTensorEntries();

            Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLengthModel, config.numberOfHeadsKey, config.ropeTheta);
            float[] ropeFreqsReal = ropeFreqs.first();
            float[] ropeFreqsImag = ropeFreqs.second();


            FloatTensor tokenEmbeddingTable = ModelLoader.loadQuantized(tensorEntries.get("token_embd.weight"));
            Weights qw = new Weights(
                    tokenEmbeddingTable,
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),

                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),

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

            // Qwen2.5-coder uses <|endoftext|> as stop-token.
            ChatTokens chatTokens = isDeepSeekR1DistillQwen ?
                    new ChatTokens( "<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "") :
                    new ChatTokens( "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
            return new Qwen3Llama(ggufPath.getFileName().toString().replaceFirst("[.]gguf$", ""), config, tokenizer, qw, chatTokens);
        }
    }

    private final static String QWEN3_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary, boolean isDeepSeekR1DistillQwen) {
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
        String firstSpecialToken = isDeepSeekR1DistillQwen ? "<｜end▁of▁sentence｜>" : "<|endoftext|>";
        int baseTokens = vocabulary.getIndex(firstSpecialToken).orElseThrow(); // assume all tokens after the base ones are special.
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
        if (isDeepSeekR1DistillQwen) {
            specialTokens.remove("<think>");
            specialTokens.remove("</think>");
        }

        return new Tokenizer(vocabulary, merges, QWEN3_PATTERN, specialTokens, tokenTypes);
    }

}

class Qwen3Llama extends Llama<Qwen3Llama.State, Qwen3Llama.Weights> {

    private ChatTokens chatTokens;

    public Qwen3Llama(String modelName, Configuration configuration, Tokenizer tokenizer, Weights weights, ChatTokens chatTokens) {
        super(modelName, configuration, tokenizer, weights, new Qwen3ChatMLFormat(tokenizer, chatTokens));
        this.chatTokens = chatTokens;
    }

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatTokens.tStartHeader());
        return state;
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatTensor[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size); {n_embd, n_embd_head_k * n_head}
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size); {n_embd, n_embd_gqa}
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size); {n_embd, n_embd_gqa}
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim); {n_embd_head_k * n_head, n_embd}

        public final FloatTensor[] attnKNorm; // (layer, n_embd_head_k);
        public final FloatTensor[] attnQNorm; // (layer, n_embd_head_q);

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

        public Weights(FloatTensor token_embedding_table, FloatTensor[] rms_att_weight,
                FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
                FloatTensor[] attnKNorm, FloatTensor[] attnQNorm,
                FloatTensor[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatTensor rms_final_weight, FloatTensor freq_cis_real, FloatTensor freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;

            this.attnKNorm = attnKNorm;
            this.attnQNorm = attnQNorm;

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

    public static final class State extends StateBase {

        // current wave of activations
        public final FloatTensor[] x; // activation at current time stamp (dim,)
        public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
        public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] q; // query (dim,)
        public final FloatTensor[] k; // key (kvDim,)
        public final FloatTensor[] v; // value (kvDim,)
        public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor[] kq; // key (kvDim,)

        State(Configuration config, int batchsize) {
            super(config, batchsize);

            int nHeadKv = config.numberOfKeyValueHeads; // n_head_kv = numberOfKeyValueHeads
            int nEmbdHeadK = config.numberOfHeadsKey; // n_embd_head_k = n_embd / n_head; %s.attention.key_length
            int nEmbdKGqa = nEmbdHeadK * nHeadKv; // n_embd_k_gqa = n_embd_head_k * n_head_kv
            this.x = Llama.allocate(batchsize, config.dim);
            this.xb = Llama.allocate(batchsize, nEmbdHeadK * config.numberOfHeads);
            this.xb2 = Llama.allocate(batchsize, config.dim);
            this.hb = Llama.allocate(batchsize, config.hiddenDim);
            this.hb2 = Llama.allocate(batchsize, config.hiddenDim);
            this.q = Llama.allocate(batchsize, nEmbdHeadK * config.numberOfHeads);
            this.k = Llama.allocate(batchsize, nEmbdKGqa);
            this.v = Llama.allocate(batchsize, nEmbdKGqa);
            this.kq = Llama.allocate(batchsize, config.numberOfHeads, 32, 15);
            this.att = Llama.allocate(batchsize, config.numberOfHeads, config.contextLength);
        }
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatTensor weight, int offset, int size, float rmsNormEps) {
        if (offset + size > out.size() || offset + size > x.size() || size > weight.size()) {
            throw new IllegalArgumentException(String.format("rmsnorm: out.size=%d, x.size=%d, weight.size=%d, offset=%d, size=%d",
                    out.size(), x.size(), weight.size(), offset, size));
        }
        // calculate sum of squares
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(offset, size, (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    public FloatTensor forward(State state, int[] tokens, int position, boolean computeLogits, AttentionConsumer attentionConsumer) {
        // a few convenience variables
        Llama.Configuration config = configuration();
        Weights weights = weights();
        int dim = config.dim;
        int nHeadKv = config.numberOfKeyValueHeads; // n_head_kv = numberOfKeyValueHeads
        int nEmbdHeadK = config.numberOfHeadsKey; // n_embd_head_k = n_embd / n_head; %s.attention.key_length
        int nEmbdHeadV = config.numberOfHeadsValue; // n_embd_head_v = n_embd / n_head; %s.attention.value_length
        //int nEmbdKGqa = nEmbdHeadK * nHeadKv; // n_embd_k_gqa = n_embd_head_k * n_head_kv
        int nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        int nEmbdHead = nEmbdHeadV;
        int nEmbdGqa = nEmbdVGqa;
        int gqa = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(nEmbdHead);

        // We need states at each token.
        final int nTokens = tokens.length;

        // copy the token embedding into x
        Parallel.parallelFor(0, nTokens, t ->
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps)
            );

            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, nEmbdHeadK * config.numberOfHeads, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, nEmbdGqa, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, nEmbdGqa, dim);

            Parallel.parallelFor(0, nTokens, t -> {
                // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                for (int i = 0; i < config.numberOfHeads; i++) {
                    rmsnorm(state.q[t], state.q[t], weights.attnQNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps);
                }
                // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                for (int i = 0; i < config.numberOfKeyValueHeads; i++) {
                    rmsnorm(state.k[t], state.k[t], weights.attnKNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps);
                }
            });

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            Parallel.parallelFor(0, nTokens, t -> {
                for (int h = 0; h < config.numberOfHeads; ++h) {
                    int rotn = h < config.numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    int poffset = h * nEmbdHead;
                    int nComplEmbdHead = nEmbdHead / 2;
                    for (int ic = 0; ic < nComplEmbdHead; ic++) {
                        float fcr = weights.freq_cis_real.getFloat((position + t) * nComplEmbdHead + ic);
                        float fci = weights.freq_cis_imag.getFloat((position + t) * nComplEmbdHead + ic);
                        for (int vi = 0; vi < rotn; vi++) {
                            FloatTensor vec = (vi == 0) ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                            float v0 = vec.getFloat(poffset + ic);
                            float v1 = vec.getFloat(poffset + ic + nComplEmbdHead);
                            vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                            vec.setFloat(poffset + ic + nComplEmbdHead, v0 * fci + v1 * fcr);
                        }
                    }
                }
            });

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * nEmbdGqa, nEmbdGqa);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * nEmbdGqa, nEmbdGqa);
            });

            // multihead attention. iterate over all heads
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int idxToken = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * nEmbdHead;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position + idxToken; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * nEmbdGqa + (h / gqa) * nEmbdHead;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[idxToken].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, nEmbdHeadK);
                    state.kq[idxToken].setFloat(h * nTokens + t, score);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[idxToken].setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[idxToken].softmaxInPlace(attOffset, position + idxToken + 1);
                
                // Optional analysis of the attention.
                if (attentionConsumer != null) {
                    int vOffsetBase = (h / gqa) * nEmbdHeadV;
                    attentionConsumer.accept(position + idxToken, curLayer, h,
                            state.att[idxToken], attOffset, position + idxToken + 1,
                            state.valueCache[curLayer], vOffsetBase);
                }

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * nEmbdHeadV;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[idxToken].fillInPlace(xbOffset, nEmbdHeadV, 0f);

                for (int t = 0; t <= position + idxToken; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;C
                    int vOffset = /* loff + */ t * nEmbdGqa + (h / gqa) * nEmbdHeadV;
                    // get the attention weight for this timestep
                    float a = state.att[idxToken].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[idxToken].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, nEmbdHeadV, a);
                }
            });
            // final matmul to get the output of the attention
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, nEmbdHeadK * config.numberOfHeads);

            // residual connection back into x
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb2[t]);
            });

            // ffn rmsnorm
            Parallel.parallelFor(0, nTokens, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps);
            });

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

                // elementwise multiply with w3(x)
                state.hb[t].multiplyInPlace(state.hb2[t]);
            });

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
        }

        // final rmsnorm
        Parallel.parallelFor(0, nTokens, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, 0, dim, config.rmsNormEps);
        });

        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);

        return state.logits;
    }

}

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
class Qwen3ChatMLFormat extends ChatFormat {

    protected final int imStart; // beginOfText
    protected final int imEnd; // endOfText

    protected final int fimPrefix;
    protected final int fimSuffix;
    protected final int fimMiddle;

    public Qwen3ChatMLFormat(Tokenizer tokenizer, ChatTokens chatTokens) {
        super(tokenizer, "", chatTokens.tStartHeader(), chatTokens.tEndHeader(), chatTokens.tEndOfTurn(), chatTokens.tEndOfText(), "", chatTokens.tEndOfTextFim());

        imStart = super.startHeader;
        imEnd = super.endHeader;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        if (imEnd == -1 && endOfText == -1) {
            throw new IllegalStateException("No stop token is defined.");
        }
        if (imEnd == -1) {
            return Set.of(endOfText);
        }
        return Set.of(imEnd, endOfText, endOfTextFim);
    }

    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (endHeader == -1) {
            // DeepSeek-R1
            String sToken = switch (message.role().name()) {
            case "system" -> null;
            case "user" -> "<｜User｜>";
            case "assistant" -> "<｜Assistant｜>";
            case "fim_prefix" -> "<|fim_prefix|>";
            case "fim_middle" -> "<|fim_middle|>";
            case "fim_suffix" -> "<|fim_suffix|>";
            default -> null;
            };
            if (sToken != null) {
                Integer token = tokenizer.getSpecialTokens().get("<｜User｜>");
                if (token == null) {
                    throw new IllegalStateException(String.format("Unknown token '%s'", sToken));
                }
                tokens.add(token);
            }
        } else if (Role.FIM_PREFIX.equals(message.role())) {
            // fill-in-the-middle, token fim_prefix.
            tokens.add(fimPrefix);
        } else if (Role.FIM_SUFFIX.equals(message.role())) {
            tokens.add(fimSuffix);
        } else if (Role.FIM_MIDDLE.equals(message.role())) {
            tokens.add(fimMiddle);
        } else {
            tokens.add(imStart);
            tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
            tokens.addAll(this.tokenizer.encodeAsList("\n"));
        }
        return tokens;
    }

    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        boolean isFim = Role.FIM_PREFIX.equals(message.role())
                || Role.FIM_SUFFIX.equals(message.role())
                || Role.FIM_MIDDLE.equals(message.role());
        if (imEnd != -1 && !isFim) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        for (Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new Message(Role.ASSISTANT, "")));
        }
        return tokens;
    }

}
