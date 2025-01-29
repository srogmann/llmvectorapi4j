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
import org.rogmann.llmva4j.ChatFormat.Role;
import org.rogmann.llmva4j.Llama.Options;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;
import org.rogmann.llmva4j.Qwen2Llama.State;
import org.rogmann.llmva4j.Qwen2Llama.Weights;

/**
 * Java implementation of Qwen-2 model using Vector API.
 */
public class Qwen2 {

    static void runInteractive(Qwen2Llama model, Sampler sampler, Options options) {
        State state = null;
        ChatFormat chatFormat = model.chatFormat();
        List<ChatFormat.Message> conversation = new ArrayList<>();
        List<TokenDetails> conversationTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            Message msgSystem = new Message(Role.SYSTEM, options.systemPrompt());
            conversation.add(msgSystem);
            conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeMessage(msgSystem)));
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
                Message messageUser = new Message(Role.USER, userText);
                conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeMessage(messageUser)));
                conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeHeader(new Message(Role.ASSISTANT, ""))));
                conversation.add(messageUser);
                Set<Integer> stopTokens = chatFormat.getStopTokens();
                List<Integer> promptTokens = conversationTokens.subList(startPosition, conversationTokens.size()).stream().map(TokenDetails::token).toList();
                StringBuilder sbResponse = new StringBuilder(200);
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
                conversation.add(new Message(Role.ASSISTANT, sbResponse.toString()));
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

    static void runInstructOnce(Qwen2Llama model, Sampler sampler, Options options) {
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
        Qwen2Llama model = Qwen2ModelLoader.loadModel(options.modelPath(), options.maxTokens());
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
            boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);

            int modelContextLength = (int) metadata.get("qwen2.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            String modelName = ggufPath.getFileName().toString();
            Llama.Configuration config = new Llama.Configuration(
                    modelName,
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

            ChatTokens chatTokens = isDeepSeekR1DistillQwen ?
                    new ChatTokens( "<｜end▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>") :
                    new ChatTokens( "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>");
            return new Qwen2Llama(ggufPath.getFileName().toString().replaceFirst("[.]gguf$", ""), config, tokenizer, qw, chatTokens);
        }
    }

    private final static String QWEN2_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

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

        return new Tokenizer(vocabulary, merges, QWEN2_PATTERN, specialTokens, tokenTypes);
    }

}

class Qwen2Llama extends Llama<Qwen2Llama.State, Qwen2Llama.Weights> {

    private ChatTokens chatTokens;

    public Qwen2Llama(String modelName, Configuration configuration, Tokenizer tokenizer, Weights weights, ChatTokens chatTokens) {
        super(modelName, configuration, tokenizer, weights, new Qwen2ChatMLFormat(tokenizer, chatTokens));
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

        public Weights(FloatTensor token_embedding_table, FloatTensor[] rms_att_weight,
                FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] q_bias, FloatTensor[] k_bias, FloatTensor[] v_bias, FloatTensor[] wo, FloatTensor[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatTensor rms_final_weight, FloatTensor freq_cis_real, FloatTensor freq_cis_imag, FloatTensor wcls) {
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

        State(Configuration config, int batchsize) {
            super(config, batchsize);

            this.x = Llama.allocate(batchsize, config.dim);
            this.xb = Llama.allocate(batchsize, config.dim);
            this.xb2 = Llama.allocate(batchsize, config.dim);
            this.hb = Llama.allocate(batchsize, config.hiddenDim);
            this.hb2 = Llama.allocate(batchsize, config.hiddenDim);
            this.q = Llama.allocate(batchsize, config.dim);
            this.k = Llama.allocate(batchsize, kvDim);
            this.v = Llama.allocate(batchsize, kvDim);
            this.att = Llama.allocate(batchsize, config.numberOfHeads, config.contextLength);
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

    public FloatTensor forward(State state, int[] tokens, int position, boolean computeLogits, AttentionConsumer attentionConsumer) {
        // a few convenience variables
        Llama.Configuration config = configuration();
        Weights weights = weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

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
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);
            if ((weights.q_bias != null && weights.q_bias[curLayer] != null)
                || (weights.k_bias != null && weights.k_bias[curLayer] != null)
                || (weights.v_bias != null && weights.v_bias[curLayer] != null)) {
                Parallel.parallelFor(0, nTokens, t -> {
                    if (weights.q_bias != null && weights.q_bias[curLayer] != null) {
                        state.q[t].addInPlace(weights.q_bias[curLayer]);
                    }
                    if (weights.k_bias != null && weights.k_bias[curLayer] != null) {
                        state.k[t].addInPlace(weights.k_bias[curLayer]);
                    }
                    if (weights.v_bias != null && weights.v_bias[curLayer] != null) {
                        state.v[t].addInPlace(weights.v_bias[curLayer]);
                    }
                });
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            Parallel.parallelFor(0, nTokens, t -> {
                for (int h = 0; h < config.numberOfHeads; ++h) {
                    int rotn = h < config.numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    int poffset = h * headSize;
                    for (int i0 = 0; i0 < headSize; i0 += 2) {
                        int ic = i0 / 2;
                        float fcr = weights.freq_cis_real.getFloat((position + t) * (headSize / 2) + ic);
                        float fci = weights.freq_cis_imag.getFloat((position + t) * (headSize / 2) + ic);
                        for (int vi = 0; vi < rotn; vi++) {
                            FloatTensor vec = (vi == 0) ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                            float v0 = vec.getFloat(poffset + ic);
                            float v1 = vec.getFloat(poffset + ic + headSize/2);
                            vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                            vec.setFloat(poffset + ic + headSize/2, v0 * fci + v1 * fcr);
                        }
                    }
                }
            });

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            // multihead attention. iterate over all heads
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int idxToken = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position + idxToken; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[idxToken].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[idxToken].setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[idxToken].softmaxInPlace(attOffset, position + idxToken + 1);
                
                // Optional analysis of the attention.
                if (attentionConsumer != null) {
                    int vOffsetBase = (h / kvMul) * headSize;
                    attentionConsumer.accept(position + idxToken, curLayer, h,
                            state.att[idxToken], attOffset, position + idxToken + 1,
                            state.valueCache[curLayer], vOffsetBase);
                }

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[idxToken].fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position + idxToken; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;C
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att[idxToken].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[idxToken].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);

            // residual connection back into x
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb2[t]);
            });

            // ffn rmsnorm
            Parallel.parallelFor(0, nTokens, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
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
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });

        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);

        return state.logits;
    }

}

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
class Qwen2ChatMLFormat extends ChatFormat {

    protected final int imStart; // beginOfText
    protected final int imEnd; // endOfText

    public Qwen2ChatMLFormat(Tokenizer tokenizer, ChatTokens chatTokens) {
        super(tokenizer, "", chatTokens.tStartHeader(), chatTokens.tEndHeader(), chatTokens.tEndOfTurn(), chatTokens.tEndOfText(), "");

        imStart = super.startHeader;
        imEnd = super.endHeader;
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
        return Set.of(imEnd, endOfText);
    }

    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (endHeader == -1) {
            // DeepSeek-R1
            String sToken = switch (message.role().name()) {
            case "system" -> null;
            case "user" -> "<｜User｜>";
            case "assistant" -> "<｜Assistant｜>";
            default -> null;
            };
            if (sToken != null) {
                Integer token = tokenizer.getSpecialTokens().get("<｜User｜>");
                if (token == null) {
                    throw new IllegalStateException(String.format("Unknown token '%s'", sToken));
                }
                tokens.add(token);
            }
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
        if (imEnd != -1) {
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
