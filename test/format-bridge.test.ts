import { test } from "node:test";
import assert from "node:assert/strict";
import type { Message, AssistantMessage, ToolResultMessage } from "@mariozechner/pi-ai";
import type { OpenAIMessage } from "headroom-ai";
import { piToOpenAI, openAIToPi } from "../src/format-bridge.ts";

const usage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function geminiAssistant(content: AssistantMessage["content"]): AssistantMessage {
  return {
    role: "assistant",
    content,
    api: "google-generative-ai",
    provider: "google",
    model: "gemini-3.1-pro-preview",
    usage,
    stopReason: "toolUse",
    timestamp: 1,
  };
}

function toolResult(toolCallId: string, toolName: string, text: string): ToolResultMessage {
  return {
    role: "toolResult",
    toolCallId,
    toolName,
    content: [{ type: "text", text }],
    isError: false,
    timestamp: 2,
  };
}

/** A user prompt followed by one assistant turn with parallel signed tool calls and their results. */
function parallelToolCallConversation(): Message[] {
  return [
    { role: "user", content: "read a.go and b.go", timestamp: 0 },
    geminiAssistant([
      { type: "thinking", thinking: "Reading both files", thinkingSignature: "sig-thinking" },
      { type: "text", text: "Reading files.", textSignature: "sig-text" },
      { type: "toolCall", id: "call_a", name: "read", arguments: { path: "a.go" }, thoughtSignature: "sig-a" },
      { type: "toolCall", id: "call_b", name: "read", arguments: { path: "b.go" } },
    ]),
    toolResult("call_a", "read", "package a\n".repeat(200)),
    toolResult("call_b", "read", "package b\n".repeat(200)),
  ];
}

/** Simulates the proxy compressing tool results only. */
function compressToolResults(messages: OpenAIMessage[]): OpenAIMessage[] {
  return messages.map((m) => (m.role === "tool" ? { ...m, content: "[compressed]" } : m));
}

function asAssistant(msg: Message): AssistantMessage {
  assert.equal(msg.role, "assistant");
  return msg as AssistantMessage;
}

test("unchanged assistant message round-trips with signatures and metadata intact", () => {
  const original = parallelToolCallConversation();
  const restored = openAIToPi(piToOpenAI(original), original);

  assert.deepEqual(restored[1], original[1]);
});

test("compressing tool results keeps thoughtSignature on tool calls", () => {
  const original = parallelToolCallConversation();
  const restored = openAIToPi(compressToolResults(piToOpenAI(original)), original);

  assert.deepEqual(restored[1], original[1]);
  assert.deepEqual((restored[2] as ToolResultMessage).content, [{ type: "text", text: "[compressed]" }]);
});

test("changed assistant text keeps tool call signatures and thinking, drops stale textSignature", () => {
  const original = parallelToolCallConversation();
  const compressed = piToOpenAI(original).map((m) =>
    m.role === "assistant" ? { ...m, content: "Reading." } : m,
  );
  const assistant = asAssistant(openAIToPi(compressed, original)[1]);

  assert.equal(assistant.provider, "google");
  assert.equal(assistant.api, "google-generative-ai");
  assert.equal(assistant.model, "gemini-3.1-pro-preview");
  assert.deepEqual(assistant.content, [
    { type: "thinking", thinking: "Reading both files", thinkingSignature: "sig-thinking" },
    { type: "text", text: "Reading." },
    { type: "toolCall", id: "call_a", name: "read", arguments: { path: "a.go" }, thoughtSignature: "sig-a" },
    { type: "toolCall", id: "call_b", name: "read", arguments: { path: "b.go" } },
  ]);
});

test("re-serialized tool call arguments are treated as unchanged", () => {
  const original = parallelToolCallConversation();
  const compressed = piToOpenAI(original).map((m) =>
    m.role === "assistant" && m.tool_calls
      ? {
          ...m,
          tool_calls: m.tool_calls.map((tc) => ({
            ...tc,
            function: { ...tc.function, arguments: JSON.stringify(JSON.parse(tc.function.arguments), null, 1) },
          })),
        }
      : m,
  );

  assert.deepEqual(openAIToPi(compressed, original)[1], original[1]);
});

test("when the message count changes, assistant messages are recovered by tool call id", () => {
  const original = parallelToolCallConversation();
  // Proxy dropped the user message, so positional alignment is not possible.
  const compressed = compressToolResults(piToOpenAI(original)).slice(1);
  const restored = openAIToPi(compressed, original);

  assert.equal(restored.length, 3);
  assert.deepEqual(restored[0], original[1]);
});

test("unmatched assistant messages on the fresh path still build a valid message", () => {
  const original = parallelToolCallConversation();
  const compressed: OpenAIMessage[] = [{ role: "assistant", content: "summary", tool_calls: undefined } as OpenAIMessage];
  const assistant = asAssistant(openAIToPi(compressed, original)[0]);

  assert.deepEqual(assistant.content, [{ type: "text", text: "summary" }]);
  assert.equal(assistant.provider, "unknown");
});

test("empty text blocks carrying a signature are preserved", () => {
  const original: Message[] = [
    { role: "user", content: "go", timestamp: 0 },
    geminiAssistant([
      { type: "text", text: "", textSignature: "sig-empty" },
      { type: "toolCall", id: "call_ls", name: "bash", arguments: { command: "ls" }, thoughtSignature: "sig-ls" },
    ]),
    toolResult("call_ls", "bash", "a.go\nb.go"),
  ];
  const restored = openAIToPi(compressToolResults(piToOpenAI(original)), original);

  assert.deepEqual(restored[1], original[1]);
});
