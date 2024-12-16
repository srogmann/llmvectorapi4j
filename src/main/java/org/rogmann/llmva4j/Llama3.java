package org.rogmann.llmva4j;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.rogmann.llmva4j.AttentionCollector.AttentionConsumer;
import org.rogmann.llmva4j.Llama.State;
import org.rogmann.llmva4j.Llama.Weights;

public class Llama3 extends Llama<State, Weights> {

    public Llama3(String modelName, Configuration configuration, Tokenizer tokenizer, Weights weights) {
        super(modelName, configuration, tokenizer, weights, new Llama3ChatFormat(tokenizer));
    }

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer().getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    static void runInteractive(Llama3 model, Sampler sampler, Options options) {
        Llama.State state = null;
        List<TokenDetails> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = model.chatFormat();
        conversationTokens.addAll(chatFormat.toTokenDetails(Collections.singletonList(chatFormat.beginOfText)));
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt()))));
        }
        int startPosition = 0;
        try (Scanner in = new Scanner(System.in)) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = in.nextLine();
                if (List.of("/quit", "/exit").contains(userText)) {
                    break;
                }
                if ("/context".equals(userText)) {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(),
                            options.maxTokens(),
                            options.maxTokens() - conversationTokens.size());
                    continue;
                }
                if (userText.startsWith("/save:")) {
                    StateCache stateCache = new StateCache(model.configuration(), state);
                    try {
                        String msg = stateCache.saveKVCache(userText, options.stateCacheFolder(), conversationTokens);
                        System.out.println(msg);
                    } catch (IllegalStateException e) {
                        System.err.println(e.getMessage());
                    }
                    continue;
                }
                if (state == null) {
                    state = model.createNewState(Llama.BATCH_SIZE);
                }
                conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText))));
                conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, ""))));
                Set<Integer> stopTokens = chatFormat.getStopTokens();
                List<Integer> promptTokens = conversationTokens.subList(startPosition, conversationTokens.size()).stream().map(TokenDetails::token).toList();
                List<TokenDetails> responseTokens = generateTokens(model, state, startPosition, promptTokens, stopTokens, options.maxTokens(), sampler,
                        options.stateCache(), options.echo(), tokenDetail -> {
                    if (options.stream()) {
                        if (!model.tokenizer().isSpecialToken(tokenDetail.token())) {
                            System.out.print(model.tokenizer().decode(List.of(tokenDetail.token())));
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
                }
                if (stopToken == null) {
                    System.err.println("Ran out of context length...");
                    break;
                }
            }
        }
    }
    
    public FloatTensor forward(State state, int[] tokens, int position, boolean computeLogits, AttentionConsumer attentionConsumer) {
        // a few convenience variables
        Configuration config = configuration();
        Weights weights = weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // We need states at each token.
        final int nTokens = tokens.length;
        // copy the token embedding into x
        Parallel.parallelFor(0, nTokens,
                t -> weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim));

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            // rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);

            final int curLayer = l;
            Parallel.parallelFor(0, nTokens,
                    t -> Llama.rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            // If the logits are not required, the attention and FFN of the last layer can be skipped entirely.
            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                // state.idxPrevBlock = nTokens - 1;
                return null;
            }

            // multihead attention. iterate over all heads
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position + token; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[token].setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[token].softmaxInPlace(attOffset, position + token + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position + token; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att[token].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
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
                Llama.rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
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
            Llama.rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });

        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);

        return state.logits;
    }

    static void runInstructOnce(Llama3 model, Sampler sampler, Options options) {
        Llama.State state = model.createNewState(Llama.BATCH_SIZE);
        ChatFormat chatFormat = new Llama3ChatFormat(model.tokenizer());

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<TokenDetails> responseTokens = generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler,
                options.stateCache(), options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token.token())) {
                    System.out.print(model.tokenizer().decode(List.of(token.token())));
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
        Llama3 model = Llama3ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        String host = System.getProperty("llm.server.host");
        int port = Integer.parseInt(System.getProperty("llm.server.port", "8089"));
        if (host != null) {
            LlamaHttpServer.runHttpServer(model, sampler, options, host, port);
        } else  if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }

    public static class Llama3ChatFormat extends ChatFormat {
        public Llama3ChatFormat(Tokenizer tokenizer) {
            super(tokenizer, "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                    "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>");
        }
    }

}

final class Llama3ModelLoader {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_LLAMA_3_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public static Llama3 loadModel(Path ggufPath, int contextLength) throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            GGUF gguf = GGUF.loadModel(ggufPath);
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = ModelLoader.createTokenizer(metadata, vocabulary);

            int modelContextLength = (int) metadata.get("llama.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            String modelName = ggufPath.getFileName().toString();
            Llama.Configuration config = new Llama.Configuration(
                    modelName,
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
                    ModelLoader.loadQuantized(tokenEmbeddings),
                    ModelLoader.loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                    null, null, null,
                    ModelLoader.loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                    ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                    ModelLoader.toFloatBuffer(tensorEntries.get("output_norm.weight")),
                    FloatBuffer.wrap(ropeFreqsReal),
                    FloatBuffer.wrap(ropeFreqsImag),
                    // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
                    // This is commonly referred as "tie word embeddings".
                    ModelLoader.loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
            );

            return new Llama3(ggufPath.getFileName().toString().replaceFirst("[.]gguf$", ""), config, tokenizer, qw);
        }
    }

}