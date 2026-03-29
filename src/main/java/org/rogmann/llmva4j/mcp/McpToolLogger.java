package org.rogmann.llmva4j.mcp;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llmva4j.LightweightJsonHandler;

/**
 * This class logs calls of the example tools into a file.
 */
public class McpToolLogger {
    /** Logger */
    private static final Logger LOGGER = Logger.getLogger(McpToolLogger.class.getName());

    /** key of an optional log-file */
    private static final String PROP_MCP_LOGFILE = "mcp.logfile";
    
    /** key of the detail mode property */
    private static final String PROP_MCP_LOGFILE_DETAIL = "mcp.logfile.detail";
    
    /** length of the tail of a cut long argument value (non-detail mode) */
    private static final int LEN_TAIL = 100;
    
    /** maximum value of a dumped argument value (non-detail mode) */
    private static final int MAX_VALUE_LEN = Math.min(Integer.getInteger("mcp.logfile.maxValueLen", 2000), LEN_TAIL);
    
    /** detail mode: maximum length of a property value */
    private static final int DETAIL_MAX_LEN = 80;
    
    /** detail mode: prefix length for truncated values */
    private static final int DETAIL_PREFIX_LEN = 70;
    
    /** detail mode: suffix length for truncated values */
    private static final int DETAIL_SUFFIX_LEN = 10;
    
    /** date-time format */
    private static final DateTimeFormatter DTF_ISO = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    
    private static final AtomicBoolean IS_FIRST_ERROR = new AtomicBoolean(true);

    /**
     * Checks if detail mode is enabled.
     * @return true if detail mode is enabled (default: true)
     */
    private static boolean isDetailMode() {
        String sDetail = System.getProperty(PROP_MCP_LOGFILE_DETAIL);
        if (sDetail == null) {
            return true;
        }
        return Boolean.parseBoolean(sDetail);
    }
    
    /**
     * Truncates a string value for detail mode logging.
     * @param sbValue StringBuilder containing the value
     * @return the truncated string
     */
    private static String truncateForDetail(StringBuilder sbValue) {
        if (sbValue.length() <= DETAIL_MAX_LEN) {
            return sbValue.toString();
        }
        int len = sbValue.length();
        String prefix = sbValue.substring(0, DETAIL_PREFIX_LEN);
        String suffix = sbValue.substring(sbValue.length() - DETAIL_SUFFIX_LEN);
        return prefix + "[..len=%d..]".formatted(len) + suffix;
    }

    /**
     * Logs a tool-call, if the system-property "mcp.logfile" has been set.
     * @param type message type, e.g. "call", "resp"
     * @param toolName name of the tool
     * @param id id of the call
     * @param arguments arguments of the tool call
     */
    public static final void logCall(String type, String toolName, String id, Map<String, Object> arguments) {
        String sPath = System.getProperty(PROP_MCP_LOGFILE);
        if (sPath == null) {
            return;
        }
        boolean detailMode = isDetailMode();
        
        // Truncate id to 6 characters if detail mode is enabled
        String loggedId = detailMode && id.length() > 6 ? id.substring(0, 6) : id;
        
        AtomicInteger lenArgs = new AtomicInteger();
        StringBuilder sbArgs = new StringBuilder(100);
        var sbValue = new StringBuilder();
        arguments.entrySet().stream().sorted(Entry.comparingByKey()).forEach(entry -> {
            sbValue.setLength(0);
            LightweightJsonHandler.dumpJsonValue(sbValue, entry.getValue());
            lenArgs.addAndGet(sbValue.length());
            
            String valueStr;
            if (detailMode) {
                valueStr = truncateForDetail(sbValue);
            } else {
                if (sbValue.length() > MAX_VALUE_LEN) {
                    int len = sbValue.length();
                    String tail = sbValue.substring(sbValue.length() - LEN_TAIL);
                    sbValue.setLength(MAX_VALUE_LEN - LEN_TAIL);
                    sbValue.append("[...len=%d...]".formatted(len));
                    sbValue.append(tail);
                }
                valueStr = sbValue.toString();
            }
            
            if (!sbArgs.isEmpty()) {
                sbArgs.append(", ");
            }
            sbArgs.append(entry.getKey());
            sbArgs.append(':');
            sbArgs.append(valueStr);
        });
        Path pathLogfile = Path.of(sPath);
        try (BufferedWriter writer = Files.newBufferedWriter(pathLogfile, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            writer.write(String.format("%s %s %-20s (%6d): id %s, args %s%n", DTF_ISO.format(LocalDateTime.now()), type,
                    toolName, lenArgs.get(), loggedId, sbArgs));
        } catch (IOException e) {
            if (IS_FIRST_ERROR.getAndSet(false)) {
                LOGGER.log(Level.WARNING, "IO-error while logging tool-call to " + pathLogfile, e);
            } else {
                LOGGER.log(Level.WARNING, "IO-error while logging tool-call to " + pathLogfile + ": " + e.getMessage());
            }
        }
    }
}
