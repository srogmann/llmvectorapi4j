package org.rogmann.llmva4j;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

import org.rogmann.llmva4j.Llama.StateBase;
import org.rogmann.llmva4j.Llama.TokenDetails;

public class AttentionCollector {

    /**
     * Interface for classes that consume attention data.
     */
    public interface AttentionConsumer {

        /**
         * Accepts attention data.
         *
         * @param idxToken     index of the token
         * @param layer        layer index
         * @param head         attention-head
         * @param att          attention-tensor
         * @param attOffset    offset in the attention-tensor
         * @param attLength    size of the current block in the attention-tensor
         */
        void accept(int idxToken, int layer, int head, FloatTensor att, int attOffset, int attLength);
    }

    /**
     * Details of an attention.
     *
     * @param position position of the token where the attention takes place in the conversation 
     * @param layer layer number within the model
     * @param head head number within the layer
     * @param positionRef position of the referenced token in the conversation
     * @param attValue value of the attention, indicating the strength of the interaction
     */
    record AttentionDetail(int position, int layer, int head, int positionRef, float attValue) { }

    public static <S extends StateBase, W> void writeAttentionsIntoFile(Path attTracePath, Llama<S, W> model,
            List<TokenDetails> conversationTokens) {
        if (attTracePath != null) {
            var file = attTracePath.toFile();
            System.out.format("Top attentions (JSON-output in %s):%n", file);
            try (OutputStream fos = new FileOutputStream(file);
                 OutputStreamWriter osw = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
                 BufferedWriter bw = new BufferedWriter(osw)) {
                IntFunction<String> token2Json = token -> JsonProcessing.escapeString(model.tokenizer().decode(List.of(token)));
                bw.write('[');
                String fs = "";
                for (int i = 0; i < conversationTokens.size(); i++) {
                    List<AttentionDetail> attDetails = conversationTokens.get(i).attentionDetails();
                    bw.write(fs);
                    fs = ", ";
                    bw.write(String.format("{\"position\": %d, \"tokenText\": %s, \"attentions\": [",
                            i, token2Json.apply(conversationTokens.get(i).token())));
                    if (attDetails != null) {
                        List<AttentionDetail> top5 = attDetails.stream().sorted((v1, v2) -> (int) Math.signum(v2.attValue() - v1.attValue())).limit(5)
                                .collect(Collectors.toList());
                        
                        List<String> top5Display = top5.stream().map(det -> String.format("Att[l=%2d, h=%2d, posRef=%3d(%-10.10s), score=%.4f]",
                                det.layer(), det.head(), det.positionRef(),
                                token2Json.apply(conversationTokens.get(det.positionRef()).token()), det.attValue(), Locale.US))
                                .collect(Collectors.toList());
                        System.out.format("  Position %d (%-10.10s): %s%n", i,
                                token2Json.apply(conversationTokens.get(i).token()), top5Display);
                        
                        String top5Json = top5.stream().map(det -> String.format(Locale.US, "{\"reference-token\": %d, \"layer\": %d, \"head\": %d, \"score\": %f, \"token-text\": %s}",
                                det.positionRef(), det.layer(), det.head(), det.attValue(), token2Json.apply(conversationTokens.get(det.positionRef()).token()))) 
                                .collect(Collectors.joining(", "));
                        bw.write(top5Json);
                    }
                    bw.write("]}");
                    bw.write(System.lineSeparator());
                }
                bw.write(']');
            } catch (IOException e) {
                throw new RuntimeException("IO-error while writing " + file, e);
            }
        }
    }

}
