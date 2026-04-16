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
  const textParts = msg.content.filter((p): p is TextContent => p.type === "text");
  const text = textParts.map((p) => p.text).join("");

  // Extract tool calls
  const toolCalls = msg.content.filter((p): p is PiToolCall => p.type === "toolCall");

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
 * - If counts differ (compression merged/dropped messages), build fresh Pi messages.
 *
 * Note: The returned messages are used as a deep copy for a single LLM call,
 * so losing metadata (timestamps, usage) is acceptable.
 */
export function openAIToPi(compressed: OpenAIMessage[], original: Message[]): Message[] {
  // If counts match, use positional alignment
  if (compressed.length === original.length) {
    return compressed.map((compMsg, i) => alignMessage(compMsg, original[i]));
  }

  // Counts differ: build fresh messages
  return compressed.map((compMsg) => buildFreshMessage(compMsg));
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

function alignAssistantMessage(comp: OpenAIMessage & { role: "assistant" }, orig: Message): Message {
  const contentParts: PiAssistantMessage["content"] = [];

  // Add text content
  const text = typeof comp.content === "string" ? comp.content : null;
  if (text) {
    contentParts.push({ type: "text", text });
  }

  // Add tool calls
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

  // Preserve thinking content from original if it was an assistant message
  if (orig.role === "assistant") {
    const thinkingParts = orig.content.filter((p) => p.type === "thinking");
    return {
      ...orig,
      content: [...thinkingParts, ...contentParts],
    };
  }

  // Role mismatch: build fresh
  return buildFreshAssistantMessage(comp);
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

function buildFreshMessage(comp: OpenAIMessage): Message {
  switch (comp.role) {
    case "system":
    case "user":
      return buildFreshUserMessage(comp);
    case "assistant":
      return buildFreshAssistantMessage(comp);
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

function safeJsonParse(str: string): Record<string, any> {
  try {
    return JSON.parse(str);
  } catch {
    return { _raw: str };
  }
}
