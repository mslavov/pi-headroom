/**
 * Format bridge between Pi-AI Message[] and Headroom OpenAI message format.
 *
 * Pi-AI types:
 *   UserMessage    { role: "user", content: string | (TextContent | ImageContent)[], timestamp }
 *   AssistantMessage { role: "assistant", content: (TextContent | ThinkingContent | ToolCall)[], api, provider, model, usage, stopReason, timestamp, ... }
 *   ToolResultMessage { role: "toolResult", toolCallId, toolName, content: (TextContent | ImageContent)[], details?, isError, timestamp }
 *
 * Headroom OpenAI types:
 *   SystemMessage   { role: "system", content: string }
 *   UserMessage     { role: "user", content: string | ContentPart[] }
 *   AssistantMessage { role: "assistant", content: string | null, tool_calls?: ToolCall[] }
 *   ToolMessage     { role: "tool", content: string, tool_call_id: string }
 *
 * The OpenAI format has no room for provider signatures (ToolCall.thoughtSignature,
 * TextContent.textSignature, ThinkingContent.thinkingSignature), so the reverse
 * conversion restores them from the original Pi messages.
 */

import type {
  Message,
  UserMessage as PiUserMessage,
  AssistantMessage as PiAssistantMessage,
  ToolResultMessage as PiToolResultMessage,
  TextContent,
  ImageContent,
  ToolCall as PiToolCall,
} from "@mariozechner/pi-ai";

import type {
  OpenAIMessage,
  ToolCall as OpenAIToolCall,
} from "headroom-ai";

// ─── Pi-AI → OpenAI ────────────────────────────────────────────────────

/**
 * Convert Pi-AI Message[] to Headroom OpenAI format.
 *
 * - Strips ThinkingContent from assistant messages (opaque/encrypted, not useful for compression)
 * - Serializes Pi tool call arguments (Record<string,any>) to JSON strings
 * - Converts Pi ImageContent (base64) to OpenAI image_url content parts
 */
export function piToOpenAI(messages: Message[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    switch (msg.role) {
      case "user":
        result.push(convertUserMessage(msg));
        break;
      case "assistant":
        result.push(convertAssistantMessage(msg));
        break;
      case "toolResult":
        result.push(convertToolResultMessage(msg));
        break;
    }
  }

  return result;
}

function convertUserMessage(msg: PiUserMessage): OpenAIMessage {
  if (typeof msg.content === "string") {
    return { role: "user", content: msg.content };
  }

  // Check if there are any images
  const hasImages = msg.content.some((part) => part.type === "image");

  if (!hasImages) {
    // Text-only: join into a single string
    const text = msg.content
      .filter((p): p is TextContent => p.type === "text")
      .map((p) => p.text)
      .join("\n");
    return { role: "user", content: text };
  }

  // Mixed content: convert to OpenAI content parts
  const parts: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> = [];
  for (const part of msg.content) {
    if (part.type === "text") {
      parts.push({ type: "text", text: part.text });
    } else if (part.type === "image") {
      const imgPart = part as ImageContent;
      parts.push({
        type: "image_url",
        image_url: { url: `data:${imgPart.mimeType};base64,${imgPart.data}` },
      });
    }
  }
  return { role: "user", content: parts as any };
}

function convertAssistantMessage(msg: PiAssistantMessage): OpenAIMessage {
  // Extract text parts (skip ThinkingContent)
  const text = joinText(msg.content.filter(isTextContent));

  // Extract tool calls
  const toolCalls = msg.content.filter(isToolCall);

  const openaiMsg: any = {
    role: "assistant",
    content: text || null,
  };

  if (toolCalls.length > 0) {
    openaiMsg.tool_calls = toolCalls.map(
      (tc): OpenAIToolCall => ({
        id: tc.id,
        type: "function",
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.arguments),
        },
      }),
    );
  }

  return openaiMsg;
}

function convertToolResultMessage(msg: PiToolResultMessage): OpenAIMessage {
  const text = msg.content
    .filter((p): p is TextContent => p.type === "text")
    .map((p) => p.text)
    .join("\n");

  return {
    role: "tool",
    content: text,
    tool_call_id: msg.toolCallId,
  };
}

// ─── OpenAI → Pi-AI ────────────────────────────────────────────────────

/**
 * Convert compressed OpenAI messages back to Pi-AI Message[] format.
 *
 * Strategy: positional alignment with the original messages.
 * - If message counts match, copy structural metadata from originals, take text from compressed.
 * - If counts differ (compression merged/dropped messages), build fresh Pi messages,
 *   recovering assistant messages from the originals by tool call id where possible.
 *
 * Assistant messages keep their original api/provider/model and provider-specific
 * block fields (e.g. Gemini `thoughtSignature`), which providers require on replay.
 *
 * Note: The returned messages are used as a deep copy for a single LLM call,
 * so losing other metadata (timestamps, usage) is acceptable.
 */
export function openAIToPi(compressed: OpenAIMessage[], original: Message[]): Message[] {
  // If counts match, use positional alignment
  if (compressed.length === original.length) {
    return compressed.map((compMsg, i) => alignMessage(compMsg, original[i]));
  }

  // Counts differ: build fresh messages
  const assistantsByToolCallId = indexAssistantsByToolCallId(original);
  return compressed.map((compMsg) => buildFreshMessage(compMsg, assistantsByToolCallId));
}

/**
 * Align a compressed OpenAI message with its original Pi message,
 * preserving structural metadata from the original.
 */
function alignMessage(comp: OpenAIMessage, orig: Message): Message {
  switch (comp.role) {
    case "system":
    case "user":
      return alignUserMessage(comp, orig);
    case "assistant":
      return alignAssistantMessage(comp, orig);
    case "tool":
      return alignToolResultMessage(comp, orig);
    default:
      return buildFreshMessage(comp);
  }
}

function alignUserMessage(comp: OpenAIMessage & { role: "system" | "user" }, orig: Message): Message {
  const content = typeof comp.content === "string"
    ? comp.content
    : Array.isArray(comp.content)
      ? (comp.content as any[]).filter((p: any) => p.type === "text").map((p: any) => p.text).join("\n")
      : "";

  if (orig.role === "user") {
    return {
      ...orig,
      content: [{ type: "text", text: content }],
    };
  }

  // Role mismatch: build fresh
  return {
    role: "user",
    content: [{ type: "text", text: content }],
    timestamp: orig.timestamp ?? Date.now(),
  };
}

/**
 * Rebuild an assistant message from its compressed form while keeping the
 * original message's metadata (api/provider/model) and provider-specific block
 * fields such as `thoughtSignature` and `textSignature`. Providers like Gemini
 * reject replayed tool calls whose signatures were dropped, and pi-ai discards
 * signatures on messages whose api/provider/model don't match the current model.
 */
function alignAssistantMessage(comp: OpenAIMessage & { role: "assistant" }, orig: Message): Message {
  if (orig.role !== "assistant") return buildFreshAssistantMessage(comp);
  if (isAssistantUnchanged(comp, orig)) return orig;

  const thinkingParts = orig.content.filter((p) => p.type === "thinking");

  // A text signature is bound to the exact original text, so only keep it when the text is unchanged.
  const text = assistantText(comp.content);
  const origTextParts = orig.content.filter(isTextContent);
  const textParts: TextContent[] = text === joinText(origTextParts)
    ? origTextParts
    : text ? [{ type: "text", text }] : [];

  const origCallsById = new Map(orig.content.filter(isToolCall).map((tc) => [tc.id, tc]));
  const toolCallParts: PiToolCall[] = (comp.tool_calls ?? []).map((tc) => ({
    ...origCallsById.get(tc.id),
    type: "toolCall",
    id: tc.id,
    name: tc.function.name,
    arguments: safeJsonParse(tc.function.arguments),
  }));

  return {
    ...orig,
    content: [...thinkingParts, ...textParts, ...toolCallParts],
  };
}

/** True when compression left the assistant's text and tool calls untouched. */
function isAssistantUnchanged(comp: OpenAIMessage & { role: "assistant" }, orig: PiAssistantMessage): boolean {
  if (assistantText(comp.content) !== joinText(orig.content.filter(isTextContent))) return false;

  const origCalls = orig.content.filter(isToolCall);
  const compCalls = comp.tool_calls ?? [];
  if (compCalls.length !== origCalls.length) return false;

  // Compare parsed arguments: the proxy may re-serialize JSON with different whitespace.
  return compCalls.every((tc, i) =>
    tc.id === origCalls[i].id &&
    tc.function.name === origCalls[i].name &&
    JSON.stringify(safeJsonParse(tc.function.arguments)) === JSON.stringify(origCalls[i].arguments),
  );
}

function alignToolResultMessage(comp: OpenAIMessage & { role: "tool" }, orig: Message): Message {
  if (orig.role === "toolResult") {
    return {
      ...orig,
      content: [{ type: "text", text: comp.content }],
    };
  }

  // Role mismatch: build fresh
  return buildFreshToolResultMessage(comp);
}

// ─── Fresh message builders (when positional alignment fails) ───────────

type AssistantIndex = Map<string, PiAssistantMessage>;

function buildFreshMessage(comp: OpenAIMessage, assistantsByToolCallId: AssistantIndex = new Map()): Message {
  switch (comp.role) {
    case "system":
    case "user":
      return buildFreshUserMessage(comp);
    case "assistant": {
      const orig = comp.tool_calls
        ?.map((tc) => assistantsByToolCallId.get(tc.id))
        .find((msg) => msg !== undefined);
      return orig ? alignAssistantMessage(comp, orig) : buildFreshAssistantMessage(comp);
    }
    case "tool":
      return buildFreshToolResultMessage(comp);
    default:
      return {
        role: "user",
        content: [{ type: "text", text: String((comp as any).content ?? "") }],
        timestamp: Date.now(),
      };
  }
}

function buildFreshUserMessage(comp: { role: string; content: any }): PiUserMessage {
  const content = typeof comp.content === "string"
    ? comp.content
    : Array.isArray(comp.content)
      ? (comp.content as any[]).filter((p: any) => p.type === "text").map((p: any) => p.text).join("\n")
      : "";

  return {
    role: "user",
    content: [{ type: "text", text: content }],
    timestamp: Date.now(),
  };
}

function buildFreshAssistantMessage(comp: OpenAIMessage & { role: "assistant" }): PiAssistantMessage {
  const contentParts: PiAssistantMessage["content"] = [];

  if (typeof comp.content === "string" && comp.content) {
    contentParts.push({ type: "text", text: comp.content });
  }

  if (comp.tool_calls) {
    for (const tc of comp.tool_calls) {
      contentParts.push({
        type: "toolCall",
        id: tc.id,
        name: tc.function.name,
        arguments: safeJsonParse(tc.function.arguments),
      });
    }
  }

  return {
    role: "assistant",
    content: contentParts,
    api: "openai-completions",
    provider: "unknown",
    model: "unknown",
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function buildFreshToolResultMessage(comp: OpenAIMessage & { role: "tool" }): PiToolResultMessage {
  return {
    role: "toolResult",
    toolCallId: comp.tool_call_id,
    toolName: "unknown",
    content: [{ type: "text", text: comp.content }],
    isError: false,
    timestamp: Date.now(),
  };
}

// ─── Helpers ────────────────────────────────────────────────────────────

type AssistantPart = PiAssistantMessage["content"][number];

function isTextContent(part: AssistantPart): part is TextContent {
  return part.type === "text";
}

function isToolCall(part: AssistantPart): part is PiToolCall {
  return part.type === "toolCall";
}

/** Joins text blocks the same way `convertAssistantMessage` does, so results are comparable. */
function joinText(parts: TextContent[]): string {
  return parts.map((p) => p.text).join("");
}

function assistantText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.filter((p: any) => p?.type === "text").map((p: any) => p.text).join("");
  }
  return "";
}

function indexAssistantsByToolCallId(messages: Message[]): AssistantIndex {
  const index: AssistantIndex = new Map();
  for (const msg of messages) {
    if (msg.role !== "assistant") continue;
    for (const part of msg.content) {
      if (isToolCall(part)) index.set(part.id, msg);
    }
  }
  return index;
}

function safeJsonParse(str: string): Record<string, any> {
  try {
    return JSON.parse(str);
  } catch {
    return { _raw: str };
  }
}
