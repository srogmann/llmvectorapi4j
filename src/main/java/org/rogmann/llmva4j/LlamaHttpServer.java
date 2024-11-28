package org.rogmann.llmva4j;

import java.io.BufferedReader;
import java.io.File;
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

import org.rogmann.llmva4j.Llama.Options;
import org.rogmann.llmva4j.Llama.State.AttentionConsumer;
import org.rogmann.llmva4j.Llama.StateBase;

import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

class LlamaHttpServer {
    record LlamaHttpSession<S extends StateBase, W>(String sessionKey, Llama<S, W> model, Sampler sampler, Options options, S state, List<Integer> conversationTokens) { ; }

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
                        exchange.getResponseHeaders().add("Set-Cookie", "LLAMA_SESS_ID=" + sessionKey);

                        float temperature = readFloat(mapRequest, "temperature", optionsGlobal.temperature());
                        float topP = readFloat(mapRequest, "top_p", optionsGlobal.topp());
                        int maxLlamaCpp = readInt(mapRequest, "n_predict", optionsGlobal.maxTokens());
                        int maxTokensOld = readInt(mapRequest, "max_tokens", maxLlamaCpp);
                        int maxComplTokens = readInt(mapRequest, "max_completion_tokens", maxTokensOld);
                        long seed = readLong(mapRequest, "seed", optionsGlobal.seed());
                        boolean stream = readBoolean(mapRequest, "stream", optionsGlobal.stream());
                        Options optionsReq = new Options(optionsGlobal.modelPath(), "", optionsGlobal.systemPrompt(), true,
                                temperature, topP, seed, maxComplTokens, stream,
                                optionsGlobal.echo(), optionsGlobal.stateCacheFolder(), optionsGlobal.stateCache());
                        System.out.format("New HTTP-Session (%s) for (%s), temp=%f, top_p=%f, n=%d, seed=%d%n", sessionKey, exchange.getRemoteAddress(),
                                temperature, topP, maxComplTokens, seed);
                        final List<Integer> conversationTokens = new ArrayList<>();
                        httpSession = new LlamaHttpSession<>(sessionKey, model, sampler, optionsReq, state, conversationTokens);
                        mapSessions.put(sessionKey, httpSession);
                    }
                }
                final String sessionKey = httpSession.sessionKey();
                final Options options = httpSession.options();
                final List<Integer> conversationTokens = httpSession.conversationTokens();
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
                if (systemMessage != null) {
                    responseText = "#SYSTEM: " + systemMessage;
                } else {
                    ChatFormat chatFormat = model.chatFormat();
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
    
                    final Integer iStopToken = stopToken;
                    AttentionConsumer attentionConsumer = null;;
                    List<Integer> responseTokens = Llama.generateTokens(model, httpSession.state(), startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler,
                            options.stateCache(), options.echo(), token -> {
                        if (options.stream()) {
                            if (!model.tokenizer().isSpecialToken(token)) {
                                String sToken = model.tokenizer().decode(List.of(token));
                                System.out.print(sToken);
    
                                Map<String, Object> mapResponse = createResponse(model, reqCounter, format, tsCreation,
                                        iStopToken, true, sToken);
    
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
                    }, attentionConsumer);
                    // Include stop token in the prompt history, but not in the response displayed to the user.
                    conversationTokens.addAll(responseTokens);
                    startPosition = conversationTokens.size();
                    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                        stopToken = responseTokens.getLast();
                        responseTokens.removeLast();
                    }
                    if (!options.stream()) {
                        responseText = model.tokenizer().decode(responseTokens);
                        System.out.println(responseText);
                    }
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

    private static <S extends StateBase, W> Map<String, Object> createResponse(Llama<S, W> model, final AtomicLong reqCounter,
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

    private static <S extends StateBase, W> void createResponseOpenAI(Llama<S, W> model, final AtomicLong reqCounter,
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