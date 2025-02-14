package org.rogmann.llmva4j;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import org.rogmann.llmva4j.AttentionCollector.AttentionDetail;
import org.rogmann.llmva4j.ChatFormat.Message;
import org.rogmann.llmva4j.ChatFormat.MessageWithTokens;
import org.rogmann.llmva4j.ChatFormat.Role;
import org.rogmann.llmva4j.Llama.Options;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;
import org.rogmann.llmva4j.StateCache.StateCacheFile;

import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

/**
 * HTTP-server implementation to execute POST-requests (Llama.cpp-format or OpenAI-format).
 * Optionally static HTML-pages of a GUI can be served.
 */
class LlamaHttpServer {
    record LlamaHttpSession<S extends StateBase, W>(String sessionKey, Llama<S, W> model, Sampler sampler,
            Options options, S state, List<TokenDetails> conversationTokens, List<MessageWithTokens> conversation) { ; }

    /**
     * Starts a HTTP-server to server requests in Llama.cpp-style.
     * @param model model
     * @param sampler sampler
     * @param optionsGlobal default options
     * @param host host-name or ip-address to bind the http-server
     * @param port port of http-server
     */
    static <S extends StateBase, W> void runHttpServer(Llama<S, W> model, Sampler sampler, Options optionsGlobal, String host, int port) {
        InetSocketAddress addr = new InetSocketAddress(host, port);
        int backlog = 0;
        String rootpath = "/";
        System.out.println(String.format("Start server at %s", addr));
        final AtomicLong reqCounter = new AtomicLong();
        final ConcurrentMap<String, LlamaHttpSession<S, W>> mapSessions = new ConcurrentHashMap<>();

        AtomicReference<HttpServer> refServer = new AtomicReference<>();
        HttpHandler handler = exchange -> {
            System.out.format("httpserver: request of %s by %s%n", exchange.getRequestURI(), exchange.getRemoteAddress());
            if ("GET".equals(exchange.getRequestMethod())) {
                String pathReq = exchange.getRequestURI().getPath();
                String pathBase = optionsGlobal.serverPath();
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
                final File fileCheckGz = new File(pathBase, pathReq + ".gz");
                final File file = fileCheckGz.isFile() ? fileCheckGz : new File(pathBase, pathReq);
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
                final byte[] buf = (file == fileCheckGz) ? readGzip(file) : Files.readAllBytes(file.toPath());
                exchange.sendResponseHeaders(200, buf.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(buf);
                }
                exchange.close();
                return;
            }
            List<Message> requestMessages = new ArrayList<>();
            final Map<String, Object> mapRequest;
            try (InputStream is = exchange.getRequestBody();
                    InputStreamReader isr = new InputStreamReader(is);
                    BufferedReader br = new BufferedReader(isr);
                    TeeBufferedReader tbr = new TeeBufferedReader(br)) {
                try {
                    LightweightJsonHandler.readChar(tbr, true, '{');
                    mapRequest = LightweightJsonHandler.parseJsonDict(tbr);

                    List<Map<String, Object>> messages = LightweightJsonHandler.getJsonArrayDicts(mapRequest, "messages");
                    if (messages == null) {
                        throw new IllegalArgumentException("The request doesn't contain messages.");
                    }
                    for (Map<String, Object> msg : messages) {
                        String roleName = LightweightJsonHandler.getJsonValue(msg, "role", String.class);
                        String content = LightweightJsonHandler.getJsonValue(msg, "content", String.class);
                        if (roleName == null) {
                            throw new IllegalArgumentException("role is missing in incoming message.");
                        }
                        if (content == null) {
                            throw new IllegalArgumentException("content is missing in incoming message.");
                        }
                        if ("/stop".equalsIgnoreCase(content)) {
                            refServer.get().stop(0);
                            throw new IllegalArgumentException("Server is stopping");
                        }
                        requestMessages.add(new Message(new Role(roleName), content));
                    }
                }
                catch (RuntimeException e) {
                    System.out.println("JSON-Prefix: " + tbr.sb);
                    e.printStackTrace();
                    Map<String, Object> mapError = new HashMap<>();
                    mapError.put("errormsg", "Invalid request: " + e.getMessage());
                    mapError.put("jsonProcessed", tbr.sb.toString());
                    var sb = new StringBuilder();
                    LightweightJsonHandler.dumpJson(sb, mapError);
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

            try {
                List<String> lCookies = exchange.getRequestHeaders().get("Cookie");
                String cookie = (lCookies != null) ? lCookies.get(0) : null;
                LlamaHttpSession<S, W> httpSession = null;
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
                    if (httpSession == null) {
                        // We build a new HTTP-session.
                        final S state = model.createNewState(Llama.BATCH_SIZE);
                        sessionKey = "SESS-" + reqCounter.get() + "-" + UUID.randomUUID().toString();
                        exchange.getResponseHeaders().add("Set-Cookie", "LLAMA_SESS_ID=" + sessionKey + "; path=/; SameSite=Strict");

                        float temperature = LightweightJsonHandler.readFloat(mapRequest, "temperature", optionsGlobal.temperature());
                        float topP = LightweightJsonHandler.readFloat(mapRequest, "top_p", optionsGlobal.topp());
                        int maxLlamaCpp = LightweightJsonHandler.readInt(mapRequest, "n_predict", optionsGlobal.maxTokens());
                        int maxTokensOld = LightweightJsonHandler.readInt(mapRequest, "max_tokens", maxLlamaCpp);
                        int maxComplTokens = LightweightJsonHandler.readInt(mapRequest, "max_completion_tokens", maxTokensOld);
                        long seed = LightweightJsonHandler.readLong(mapRequest, "seed", optionsGlobal.seed());
                        boolean stream = LightweightJsonHandler.readBoolean(mapRequest, "stream", optionsGlobal.stream());
                        Options optionsReq = new Options(optionsGlobal.modelPath(), "", optionsGlobal.systemPrompt(), true,
                                temperature, topP, seed, maxComplTokens, stream,
                                optionsGlobal.echo(),
                                optionsGlobal.serverHost(), optionsGlobal.serverPort(), optionsGlobal.serverPath(),
                                optionsGlobal.stateCacheFolder(), optionsGlobal.stateCache(), optionsGlobal.stateCacheAutoSavePrefix(),
                                optionsGlobal.attentionTrace());
                        System.out.format("New HTTP-Session (%s) for (%s), temp=%f, top_p=%f, n=%d, seed=%d%n", sessionKey, exchange.getRemoteAddress(),
                                temperature, topP, maxComplTokens, seed);
                        final List<TokenDetails> conversationTokens = new ArrayList<>();
                        final List<MessageWithTokens> conversation = new ArrayList<>();
                        httpSession = new LlamaHttpSession<>(sessionKey, model, sampler, optionsReq, state, conversationTokens, conversation);
                        mapSessions.put(sessionKey, httpSession);
                    }
                }
                final String sessionKey = httpSession.sessionKey();
                final Options options = httpSession.options();
                final List<TokenDetails> conversationTokens = httpSession.conversationTokens();
                final List<MessageWithTokens> conversation = httpSession.conversation();
                int startPosition = conversationTokens.size();
                
                String systemMessage = null;
                if (requestMessages.getLast().role().equals(ChatFormat.Role.USER)
                        && requestMessages.getLast().content().startsWith("/save:")) {
                    StateCache stateCache = new StateCache(model.configuration(), httpSession.state);
                    try {
                        systemMessage = stateCache.saveKVCache(requestMessages.getLast().content(), options.stateCacheFolder(),
                                conversationTokens, conversation);
                    } catch (IllegalStateException e) {
                        System.err.println(e.getMessage());
                    }
                }

                final long tsCreation = System.currentTimeMillis();
                Integer stopToken = null;
                String responseText = "";
                List<TokenDetails> tokenDetails = new ArrayList<>();
                if (systemMessage != null) {
                    responseText = "#SYSTEM: " + systemMessage;
                } else if (requestMessages.isEmpty()) {
                    responseText = "#SYSTEM: no request message";
                } else {
                    if (options.systemPrompt() != null && conversation.isEmpty() && !Role.SYSTEM.name().equals(requestMessages.getFirst().role().name())) {
                        // We add a system-prompt.
                        requestMessages.add(0, new Message(Role.SYSTEM, options.systemPrompt()));
                    }

                    if (options.stream()) {
                        // We use server-side events (SSE) for streaming.
                        exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                        exchange.getResponseHeaders().add("Cache-Control", "no-cache");
                        exchange.sendResponseHeaders(200, 0);
                    }

                    // Does the current KV Cache fit the message list in the request?
                    List<MessageWithTokens> listMsgsKnown = IntStream.range(0, Math.min(requestMessages.size() - 1, conversation.size()))
                            .mapToObj(idx -> new Pair<>(requestMessages.get(idx), conversation.get(idx)))
                            .takeWhile(p -> p.first().role().equals(p.second().role())
                                    && p.first().content().equals(p.second().content()))
                            .map(p -> p.second()).toList();
                    ChatFormat chatFormat = model.chatFormat();
                    boolean isChatModified = false;
                    if (listMsgsKnown.isEmpty() && options.stateCacheFolder() != null) {
                        // Do we have a stored KV-cache?
                        StateCache stateCache = new StateCache(model.configuration(), httpSession.state);
                        StateCacheFile fileGgsc = stateCache.searchConversationLog(options.stateCacheFolder(), requestMessages);
                        if (fileGgsc != null) {
                            List<MessageWithTokens> msgsFile = fileGgsc.messages();
                            final int numMsgsToCheck = Math.min(msgsFile.size(), requestMessages.size());
                            List<MessageWithTokens> msgsFileKnown = IntStream.range(0, numMsgsToCheck)
                                    .mapToObj(idx -> new Pair<>(requestMessages.get(idx), msgsFile.get(idx)))
                                    .takeWhile(p -> p.first().role().equals(p.second().role())
                                            && p.first().content().equals(p.second().content()))
                                    .map(p -> p.second()).toList();
                            try (InputStream is = Files.newInputStream(fileGgsc.file().toPath())) {
                                final List<Integer> reqTokens = new ArrayList<>();
                                // We can read the tokens and KV-caches of the first numMsgsToCopy.size() messages.
                                conversation.clear();
                                conversationTokens.clear();
                                AtomicInteger refPosition = new AtomicInteger();
                                // We take tokens from msgsFile.
                                msgsFileKnown.stream().limit(numMsgsToCheck).map(MessageWithTokens::tokens).forEach(tokens -> tokens.stream().forEach(token -> {
                                    reqTokens.add(token);
                                    conversationTokens.add(new TokenDetails(refPosition.getAndIncrement(), token, false, 0f,
                                            model.tokenizer.decodeOneToken(token).getBytes(StandardCharsets.UTF_8), null));
                                }));
                                String statTokens = Arrays.toString(msgsFileKnown.stream().map(m -> String.format("%1.1s%d[%8.8s...]",
                                        m.role().name(), m.tokens().size(), LightweightJsonHandler.escapeString(m.content()))).toArray());
                                startPosition = stateCache.deserialize(is, model.tokenizer(), conversation, reqTokens, options.echo());
                                // How many messages can we copy from KV-cache?
                                int numMsgsToCopy = msgsFileKnown.size();
                                System.out.format("// Start-position via %s is %d (%d %s, %s)%n", fileGgsc.file().getName(), startPosition,
                                        numMsgsToCopy, numMsgsToCopy == 1 ? "msg" : "msgs", statTokens);
                                listMsgsKnown = msgsFileKnown.subList(0, numMsgsToCopy);
                                isChatModified = true;
                            } catch (IOException e) {
                                throw new RuntimeException("IO-error while reading " + fileGgsc, e);
                            }
                        }
                    } else if (listMsgsKnown.size() < conversation.size()) {
                        //System.out.println("listKnown: " + listMsgsKnown);
                        //System.out.println("conv: " + conversation);
                        //System.out.println("Req: " + requestMessages);
                        int startPositionOld = startPosition;
                        startPosition = listMsgsKnown.stream().mapToInt(msg -> msg.tokens().size()).sum();
                        System.out.format("// Trim conversation: %d -> %d msgs, pos %d -> %d%n", conversation.size(),
                                listMsgsKnown.size(), startPositionOld, startPosition);
                        isChatModified = true;
                    }
                    if (isChatModified) {
                        System.out.format("// chat modified, listMsgsKnown %d, conversation %d%n", listMsgsKnown.size(), conversation.size());
                        conversation.clear();
                        conversation.addAll(listMsgsKnown);
                        final int tokensSizeNew = startPosition;
                        conversationTokens.removeIf(token -> token.position() >= tokensSizeNew);
                    }

                    // Add in the KV-cache unknown parts of the chat and the next prompt.
                    for (int idx = listMsgsKnown.size(); idx < requestMessages.size(); idx++) {
                        Message msg = requestMessages.get(idx);
                        System.out.format("// %s: %s%n", msg.role(), msg.content());
                        List<Integer> msgTokens = chatFormat.encodeMessage(msg);
                        conversationTokens.addAll(chatFormat.toTokenDetails(msgTokens, conversationTokens.size()));
                        conversation.add(new MessageWithTokens(msg.role(), msg.content(), msgTokens));
                    }

                    List<TokenDetails> tokensAssistantHeader = chatFormat.toTokenDetails(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")), conversationTokens.size());
                    conversationTokens.addAll(tokensAssistantHeader);
                    //System.out.format("Tokens (start-pos %d): %s%n", startPosition, conversationTokens);
                    //System.out.println("Text: " + model.tokenizer().decode(conversationTokens).replace("\n", "\\n"));
                    final Set<Integer> stopTokens = chatFormat.getStopTokens();

                    final List<Integer> promptTokens = conversationTokens.subList(startPosition, conversationTokens.size()).stream().map(TokenDetails::token).toList();
                    StringBuilder sbResponse = new StringBuilder(200);
                    final int startPositionOutput = startPosition + promptTokens.size();
                    List<TokenDetails> responseTokens = Llama.generateTokens(model, httpSession.state(), startPosition, promptTokens, stopTokens, options.maxTokens(), sampler,
                            options.stateCache(), options.echo(), tokenDetail -> {
                        if (options.stream()) {
                            //System.out.format("tokDet: pos=%d, tok=%d%n", tokenDetail.position(), tokenDetail.token());
                            if (!model.tokenizer().isSpecialToken(tokenDetail.token()) || options.attentionTrace() > 0) {
                                String sToken = model.tokenizer().decode(List.of(tokenDetail.token()));
                                System.out.print(sToken);
                                final Integer iStopToken = stopTokens.contains(tokenDetail.token()) ? tokenDetail.token() : null;
                                if (tokenDetail.position() >= startPositionOutput && iStopToken == null) {
                                    sbResponse.append(sToken);
                                }

                                Map<String, Object> mapResponse = createResponse(model, reqCounter, tsCreation,
                                        startPositionOutput, iStopToken, true, sToken, List.of(tokenDetail));

                                var sbOut = new StringBuilder();
                                LightweightJsonHandler.dumpJson(sbOut, mapResponse);
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
                    }, options.attentionTrace());
                    // Include stop token in the prompt history, but not in the response displayed to the user.
                    List<Integer> tokensNew = new ArrayList<>();
                    tokensAssistantHeader.stream().map(TokenDetails::token).forEach(tokensNew::add);
                    responseTokens.stream().filter(token -> token.position() >= startPositionOutput).map(TokenDetails::token).forEach(tokensNew::add);
                    conversationTokens.addAll(responseTokens);
                    startPosition = conversationTokens.size();
                    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast().token())) {
                        stopToken = responseTokens.getLast().token();
                        responseTokens.removeLast();
                    }
                    if (!options.stream()) {
                        responseText = model.tokenizer().decode(responseTokens.stream().map(TokenDetails::token).toList());
                        sbResponse.append(responseText);
                        tokenDetails.addAll(responseTokens);
                        System.out.println(responseText);
                    }
                    conversation.add(new MessageWithTokens(Role.ASSISTANT, sbResponse.toString(), tokensNew));
                    if (options.stateCacheAutoSavePrefix() != null) {
                        StateCache stateCache = new StateCache(model.configuration(), httpSession.state);
                        try {
                            String cmdSaveFile = String.format("/save:%s", options.stateCacheAutoSavePrefix());
                            String sysMsg = stateCache.saveKVCache(cmdSaveFile, options.stateCacheFolder(),
                                    conversationTokens, conversation);
                            System.out.format("// %s%n", sysMsg);
                        } catch (IllegalStateException e) {
                            System.err.println(e.getMessage());
                        }
                    }
                }
                Map<String, Object> mapResponse = createResponse(model, reqCounter, tsCreation,
                        startPosition, stopToken, options.stream(), responseText, tokenDetails);
                if (stopToken == null) {
                    System.err.println("Ran out of context length...");
                }
                var sbOut = new StringBuilder();
                LightweightJsonHandler.dumpJson(sbOut, mapResponse);
                byte[] buf;
                if (options.stream() && systemMessage == null) {
                    buf = String.format("data: %s\n\n", sbOut).getBytes(StandardCharsets.UTF_8);
                } else {
                    buf = String.format("%s\n", sbOut).getBytes(StandardCharsets.UTF_8);
                    exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                    exchange.sendResponseHeaders(200, buf.length);
                }
                exchange.getResponseBody().write(buf);
                exchange.close();

                String attTraceFile = System.getProperty("llama.attentionTrace.file");
                if (attTraceFile != null) {
                    AttentionCollector.writeAttentionsIntoFile(new File(attTraceFile).toPath(), model, conversationTokens);
                }
            } catch (Exception e) {
                System.err.println("Error while creating response: " + e.getMessage());
                e.printStackTrace();

                Map<String, Object> mapError = new HashMap<>();
                mapError.put("errormsg", "Error while creating response");
                var sb = new StringBuilder();
                LightweightJsonHandler.dumpJson(sb, mapError);
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

    private static <S extends StateBase, W> Map<String, Object> createResponse(Llama<S, W> model, final AtomicLong reqCounter,
            final long tsCreation, int startPosition, Integer stopToken,
            boolean isDelta, String responseText, List<TokenDetails> tokenDetails) {
        Map<String, Object> mapResponse = new LinkedHashMap<>();
        createResponseChatCompletion(model, reqCounter, tsCreation, startPosition, stopToken,
                mapResponse, isDelta, responseText, tokenDetails);
        return mapResponse;
    }

    private static <S extends StateBase, W> void createResponseChatCompletion(Llama<S, W> model, final AtomicLong reqCounter,
            final long tsCreation, int startPosition, Integer stopToken,
            Map<String, Object> mapResponse, boolean isDelta, String content, List<TokenDetails> tokenDetails) {
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
        if (isDelta && tokenDetails != null && !tokenDetails.isEmpty()
                && (tokenDetails.getFirst().position() < startPosition
                        || Integer.valueOf(tokenDetails.getFirst().token()).equals(stopToken)
                        || !tokenDetails.getFirst().isInferred())) {
            respMsg.put("isHistory", true);
        }
        Map<String, Object> logprobs = new LinkedHashMap<>();
        List<Object> logprobscontent = new ArrayList<>();
        if (tokenDetails != null) {
            tokenDetails.forEach(detail -> {
                Map<String, Object> msgContTokn = new LinkedHashMap<>();
                msgContTokn.put("position", detail.position());
                msgContTokn.put("token", Integer.toString(detail.token()));
                msgContTokn.put("tokenText", new String(detail.bytes(), StandardCharsets.UTF_8));
                msgContTokn.put("logprob", detail.logprob());
                msgContTokn.put("bytes", detail.bytes());
                msgContTokn.put("top_logprobs", new ArrayList<>());
                if (detail.attentionDetails() != null && !detail.attentionDetails().isEmpty()) {
                    List<Object> listAtts = new ArrayList<>();
                    // Top 5 detail-entries sorted by length of partial value-vector.
                    List<AttentionDetail> top5 = detail.attentionDetails().stream().sorted((v1, v2) -> (int) Math.signum(v2.partValueLen() - v1.partValueLen())).limit(5)
                            .collect(Collectors.toList());
                    top5.forEach(attDet -> {
                        Map<String, Object> mapAtt = new LinkedHashMap<>();
                        mapAtt.put("position-ref", attDet.positionRef());
                        mapAtt.put("layer", attDet.layer());
                        mapAtt.put("head", attDet.head());
                        mapAtt.put("score", attDet.attValue());
                        mapAtt.put("valueLength", attDet.partValueLen());
                        listAtts.add(mapAtt);
                    });
                    msgContTokn.put("attentions", listAtts);
                }
                logprobscontent.add(msgContTokn);
            }); 
        }
        logprobs.put("content", logprobscontent);
        choice0.put("logprobs", logprobs);
        String finishReason = null;
        if (!isDelta) {
            finishReason = (stopToken == null) ? "length" : "stop";
        }
        choice0.put("finishReason", finishReason);
        choices.add(choice0);
        mapResponse.put("choices", choices);
    }

    private static byte[] readGzip(File file) throws IOException {
        final byte[] buf = new byte[16384];
        try (InputStream fis = new FileInputStream(file);
             InputStream gis = new GZIPInputStream(fis)) {
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream(16384)) {
                while (true) {
                    final int len = gis.read(buf);
                    if (len == -1) {
                        break;
                    }
                    baos.write(buf, 0, len);
                }
                return baos.toByteArray();
            }
        }
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

}