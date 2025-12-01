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
    
    /** length of the tail of a cut long argument value */
    private static final int LEN_TAIL = 100;
    
    /** maximum value of a dumped argument value */
    private static final int MAX_VALUE_LEN = Math.min(Integer.getInteger("mcp.logfile.maxValueLen", 2000), LEN_TAIL);
    
    /** date-time format */
    private static final DateTimeFormatter DTF_ISO = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    
    private static final AtomicBoolean IS_FIRST_ERROR = new AtomicBoolean(true);

    /**
     * Logs a tool-call, if the system-property "mcp.logfile" has been set.
     * @param toolName name of the tool
     * @param id id of the call
     * @param arguments arguments of the tool call
     */
    public static final void logCall(String toolName, String id, Map<String, Object> arguments) {
        String sPath = System.getProperty(PROP_MCP_LOGFILE);
        if (sPath == null) {
            return;
        }
        AtomicInteger lenArgs = new AtomicInteger();
        StringBuilder sbArgs = new StringBuilder(100);
        var sbValue = new StringBuilder();
        arguments.entrySet().stream().sorted(Entry.comparingByKey()).forEach(entry -> {
            sbValue.setLength(0);
            LightweightJsonHandler.dumpJsonValue(sbValue, entry.getValue());
            lenArgs.addAndGet(sbValue.length());
            if (sbValue.length() > MAX_VALUE_LEN) {
                String tail = sbValue.substring(sbValue.length() - LEN_TAIL, MAX_VALUE_LEN);
                sbValue.setLength(MAX_VALUE_LEN - LEN_TAIL);
                sbValue.append("[...]");
                sbValue.append(tail);
            }
            if (!sbArgs.isEmpty()) {
                sbArgs.append(", ");
            }
            sbArgs.append(entry.getKey());
            sbArgs.append(':');
            sbArgs.append(sbValue);
        });
        Path pathLogfile = Path.of(sPath);
        try (BufferedWriter writer = Files.newBufferedWriter(pathLogfile, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            writer.write(String.format("%s call %-20s (%6d): id %s, args %s%n", DTF_ISO.format(LocalDateTime.now()),
                    toolName, lenArgs.get(), id, sbArgs));
        } catch (IOException e) {
            if (IS_FIRST_ERROR.getAndSet(false)) {
                LOGGER.log(Level.WARNING, "IO-error while logging tool-call to " + pathLogfile, e);
            } else {
                LOGGER.log(Level.WARNING, "IO-error while logging tool-call to " + pathLogfile + ": " + e.getMessage());
            }
        }
    }
}
