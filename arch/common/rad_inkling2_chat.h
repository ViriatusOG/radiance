/* Inkling 2's reply format, as its chat template writes a turn.
 *
 * Every message is <|message_ROLE|><|content_KIND|>...<|end_message|>, and the model's turn is its
 * messages then <|content_model_end_sampling|> (the eos, which ends generation and is not read).
 * A reply that reasons, then answers, reads
 *
 *   <|message_model|><|content_thinking|>REASONING<|end_message|>
 *   <|message_model|><|content_text|>ANSWER<|end_message|>
 *
 * and one that calls tools writes each call as a message of its own whose header is followed by
 * the function's name rather than a content kind, its arguments in a JSON envelope that names it
 * again (the template sorts the arguments' keys and writes them compactly):
 *
 *   <|message_model|>get_weather<|content_invoke_tool_json|>
 *   {"name":"get_weather","args":{"city":"Lehi, Utah","units":"f"}}<|end_message|>
 *
 * So <|message_model|> is message_open: before <|content_thinking|> it is part of the reasoning
 * opener, before <|content_text|> (content_open) the answer's header, and both are dropped there;
 * before anything else it opens a call, so it is also name_prefix. The envelope is
 * object_args_key "args". With reasoning_anywhere a block the template interleaves later is
 * reasoning too, and the answer's own <|end_message|>, a close with no block open, is not part of
 * the answer.
 *
 * No grammar triggers. Every message opens with the same header, so a trigger word on it would
 * arm the call grammar on the reasoning and the answer as well (llama.cpp triggers on a pattern,
 * the header followed by something other than `<`, which trigger words cannot say). Calls are read
 * as the model writes them, unconstrained.
 */
#pragma once
#include "rad_builder.h"

namespace inkling2 {

inline const RadChatFormat* inkling2_chat_format() {
    static const RadChatFormat f = [] {
        RadChatFormat c{};
        c.struct_size         = (uint32_t)sizeof(RadChatFormat);
        c.name                = "plugin:inkling2";
        c.reasoning_open      = "<|content_thinking|>";
        c.reasoning_close     = "<|end_message|>";
        c.reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT;
        c.reasoning_anywhere  = 1;
        c.content_open        = "<|content_text|>";
        c.message_open        = "<|message_model|>";
        c.name_prefix         = "<|message_model|>";
        c.name_suffix         = "<|content_invoke_tool_json|>";
        c.call_end            = "<|end_message|>";
        c.args_form           = RAD_CHAT_ARGS_OBJECT;
        c.object_args_key     = "args";
        return c;
    }();
    return &f;
}

}  /* namespace inkling2 */
