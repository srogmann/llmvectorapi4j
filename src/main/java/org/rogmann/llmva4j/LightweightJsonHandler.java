package org.rogmann.llmva4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A simple JSON-implementation to be used in this application only (to skip dependencies).
 * The recommendation is to use org.json or features like JAX-RS using DTOs in own projects.
 */
class LightweightJsonHandler {

    @SuppressWarnings("unchecked")
    static void dumpJson(StringBuilder sb, Map<String, Object> map) {
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
            } else if (c == '\\') {
                sb.append("\\\\");
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

    static String escapeString(String s) {
        var sb = new StringBuilder(s.length() + 10);
        dumpString(sb, s);
        return sb.toString();
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
    static Map<String, Object> parseJsonDict(BufferedReader br) throws IOException {
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
                else if (c == '\\') {
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

    static char readChar(BufferedReader br, boolean ignoreWS) throws IOException {
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

    static void readChar(BufferedReader br, boolean ignoreWS, char expected) throws IOException {
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

    static float readFloat(Map<String, Object> map, String key, float defaultValue) {
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

    static int readInt(Map<String, Object> map, String key, int defaultValue) {
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

    static long readLong(Map<String, Object> map, String key, long defaultValue) {
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

    static boolean readBoolean(Map<String, Object> map, String key, boolean defaultValue) {
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
