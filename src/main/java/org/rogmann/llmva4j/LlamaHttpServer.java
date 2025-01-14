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
import java.math.BigDecimal;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

import org.rogmann.llmva4j.AttentionCollector.AttentionDetail;
import org.rogmann.llmva4j.Llama.Options;
import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;

import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

/**
 * HTTP-server implementation to execute POST-requests (Llama.cpp-format or OpenAI-format).
 * Optionally static HTML-pages of a GUI can be served.
 */
class LlamaHttpServer {
    record LlamaHttpSession<S extends StateBase, W>(String sessionKey, Llama<S, W> model, Sampler sampler, Options options, S state, List<TokenDetails> conversationTokens) { ; }

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
            List<ChatFormat.Message> chatMessages = new ArrayList<>();
            final Map<String, Object> mapRequest;
            try (InputStream is = exchange.getRequestBody();
                    InputStreamReader isr = new InputStreamReader(is);
                    BufferedReader br = new BufferedReader(isr);
                    TeeBufferedReader tbr = new TeeBufferedReader(br)) {
                try {
                    LightweightJsonHandler.readChar(tbr, true, '{');
                    mapRequest = LightweightJsonHandler.parseJsonDict(tbr);

                    List<Map<String, Object>> messages = LightweightJsonHandler.getJsonArrayDicts(mapRequest, "messages");
                    String prompt = LightweightJsonHandler.getJsonValue(mapRequest, "prompt", String.class);
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
                            String role = LightweightJsonHandler.getJsonValue(msg, "role", String.class);
                            String content = LightweightJsonHandler.getJsonValue(msg, "content", String.class);
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
                    if (httpSession != null && httpSession.conversationTokens().size() > 0) {
                        if (ChatFormat.Role.SYSTEM.equals(chatMessages.get(0).role())) {
                            // System-prompt at begin only.
                            chatMessages.remove(0);
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
                                optionsGlobal.echo(), optionsGlobal.stateCacheFolder(), optionsGlobal.stateCache(),
                                optionsGlobal.attentionTrace());
                        System.out.format("New HTTP-Session (%s) for (%s), temp=%f, top_p=%f, n=%d, seed=%d%n", sessionKey, exchange.getRemoteAddress(),
                                temperature, topP, maxComplTokens, seed);
                        final List<TokenDetails> conversationTokens = new ArrayList<>();
                        httpSession = new LlamaHttpSession<>(sessionKey, model, sampler, optionsReq, state, conversationTokens);
                        mapSessions.put(sessionKey, httpSession);
                    }
                }
                final String sessionKey = httpSession.sessionKey();
                final Options options = httpSession.options();
                final List<TokenDetails> conversationTokens = httpSession.conversationTokens();
                int startPosition = conversationTokens.size();
                
                String systemMessage = null;
                if (chatMessages.size() == 1 && ChatFormat.Role.USER.equals(chatMessages.get(0).role())
                        && chatMessages.get(0).content().startsWith("/save:")) {
                    StateCache stateCache = new StateCache(model.configuration(), httpSession.state);
                    try {
                        systemMessage = stateCache.saveKVCache(chatMessages.get(0).content(), options.stateCacheFolder(), conversationTokens);
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
                } else {
                    ChatFormat chatFormat = model.chatFormat();
                    chatMessages.stream().map(m -> String.format("[%s]> %s", m.role(), m.content())).forEach(System.out::println);
                    chatMessages.stream().map(chatFormat::encodeMessage).map(chatFormat::toTokenDetails).forEach(conversationTokens::addAll);
                    conversationTokens.addAll(chatFormat.toTokenDetails(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, ""))));
                    //System.out.format("Tokens (start-pos %d): %s%n", startPosition, conversationTokens);
                    //System.out.println("Text: " + model.tokenizer().decode(conversationTokens).replace("\n", "\\n"));
                    Set<Integer> stopTokens = chatFormat.getStopTokens();
    
                    if (options.stream()) {
                        // We use server-side events (SSE) for streaming.
                        exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
                        exchange.getResponseHeaders().add("Cache-Control", "no-cache");
                        exchange.sendResponseHeaders(200, 0);
                    }
    
                    final Integer iStopToken = stopToken;
                    final List<Integer> promptTokens = conversationTokens.subList(startPosition, conversationTokens.size()).stream().map(TokenDetails::token).toList();
                    List<TokenDetails> responseTokens = Llama.generateTokens(model, httpSession.state(), startPosition, promptTokens, stopTokens, options.maxTokens(), sampler,
                            options.stateCache(), options.echo(), tokenDetail -> {
                        if (options.stream()) {
                            if (!model.tokenizer().isSpecialToken(tokenDetail.token())) {
                                String sToken = model.tokenizer().decode(List.of(tokenDetail.token()));
                                System.out.print(sToken);

                                Map<String, Object> mapResponse = createResponse(model, reqCounter, format, tsCreation,
                                        iStopToken, true, sToken, List.of(tokenDetail));

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
                    }, options.attentionTrace());
                    // Include stop token in the prompt history, but not in the response displayed to the user.
                    conversationTokens.addAll(responseTokens);
                    startPosition = conversationTokens.size();
                    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast().token())) {
                        stopToken = responseTokens.getLast().token();
                        responseTokens.removeLast();
                    }
                    if (!options.stream()) {
                        responseText = model.tokenizer().decode(responseTokens.stream().map(TokenDetails::token).toList());
                        tokenDetails.addAll(responseTokens);
                        System.out.println(responseText);
                    }
                }
                Map<String, Object> mapResponse = createResponse(model, reqCounter, format, tsCreation,
                        stopToken, options.stream(), responseText, tokenDetails);
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

    private static <S extends StateBase, W> Map<String, Object> createResponse(Llama<S, W> model, final AtomicLong reqCounter,
            JsonFormat format, final long tsCreation, Integer stopToken,
            boolean isDelta, String responseText, List<TokenDetails> tokenDetails) {
        Map<String, Object> mapResponse = new LinkedHashMap<>();
        switch (format) {
        case LLAMA_CPP:
            mapResponse.put("content", responseText);
            mapResponse.put("stop", Boolean.valueOf(stopToken != null));
            break;
        case OPENAI:
            createResponseOpenAI(model, reqCounter, tsCreation, stopToken,
                    mapResponse, isDelta, responseText, tokenDetails);
            break;
        default:
            throw new IllegalArgumentException("format " + format);
        }
        return mapResponse;
    }

    private static <S extends StateBase, W> void createResponseOpenAI(Llama<S, W> model, final AtomicLong reqCounter,
            final long tsCreation, Integer stopToken,
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
            else if (value instanceof Integer i) {
                sb.append(i);
            }
            else if (value instanceof Float f) {
                sb.append(f);
            }
            else if (value instanceof BigDecimal bd) {
                sb.append(bd);
            }
            else if (value instanceof byte[] buf) {
                sb.append(Arrays.toString(buf));
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
            else if (value instanceof Integer i) {
                sb.append(i);
            }
            else if (value instanceof Float f) {
                sb.append(f);
            }
            else if (value instanceof BigDecimal bd) {
                sb.append(bd);
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