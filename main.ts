// free-model-aggregator-02: one server, two client wires, the best free model behind every answer.
//
// The server listens on an ephemeral port, registers with the env, and answers the six prompts the
// env sends it by dialling live OpenRouter free models through the grader's gateway, always on the
// Chat Completions route (the Anthropic wire is translated to Chat Completions and back; the
// gateway's Messages route is never dialled).
//
// Routing: the highest-rated free model that meets the request's requirements (JSON mode needs
// `json`, tools need `tools`, images need `vision`). An answer is only accepted when the provider's
// own text carries the prompt's token (and parses as JSON when JSON mode was asked); otherwise the
// next-best eligible model is tried, up to 4 attempts inside a 24 s per-prompt budget. Every
// refusal is reported on `x-makefun-upstream-failures` so the failover column can see it.
//
// Determinism: response ids are sha256(MAKEFUN_TRIAL_SEED + request counter + dialled model) and
// timestamps come from MAKEFUN_TRIAL_NOW, never the wall clock, so a replay reproduces the export.
//
// Node 22 standard library only; no vendor SDK, no dependencies. Run as `node main.ts` from the
// repo root. `node main.ts --list` prints the registered catalog and exits 0 before any server
// starts.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Per-prompt budget: the env waits 30 s per request, so stop starting new attempts after 24 s. */
const PROMPT_BUDGET_MS = 24_000;
/** Most upstream attempts spent on one prompt. */
const MAX_ATTEMPTS = 4;
/** Most refusal rows reported on one response. */
const MAX_FAILURE_ROWS = 8;
/** A prompt's token, as the brief spells it: WORD-1234. */
const TOKEN_RE = /[A-Z]{3,12}-[0-9]{4}/;
/** Fixed timestamps for the model lists: nothing in a reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? "").replace(/\/+$/, "");
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? "";
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? "default";
const TRIAL_NOW_MS = (() => {
  const parsed = Date.parse(process.env.MAKEFUN_TRIAL_NOW ?? "");
  return Number.isFinite(parsed) ? parsed : FIXED_OPENAI_CREATED * 1000;
})();
const TRIAL_NOW_SEC = Math.floor(TRIAL_NOW_MS / 1000);

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === "object" && !Array.isArray(v);
const asRecord = (v: unknown): Json | null => (isRecord(v) ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function shaId(prefix: string, counter: number, model: string): string {
  return `${prefix}${createHash("sha256").update(`${TRIAL_SEED}:${counter}:${model}`).digest("hex").slice(0, 24)}`;
}

// ---------------------------------------------------------------------------
// The brief: the free model list it carries, and the catalog it makes.
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/**
 * The house's sealed free catalog, in the brief's order, as a fallback when the brief file cannot
 * be read (the scored trial's brief carries the same fourteen rows; the brief wins when present).
 */
const FALLBACK_MODELS: ModelRow[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", provider: "openrouter", display_name: "Baidu: ERNIE-5.5-Flash-Preview (free)", rating_milli: 1150, context_length: 1024000, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/cohere/north-mini-code:free", provider: "openrouter", display_name: "Cohere: North Mini Code (free)", rating_milli: 1200, context_length: 256000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", provider: "openrouter", display_name: "Dots Studio: Dots3-Note Preview (free)", rating_milli: 1150, context_length: 512000, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", provider: "openrouter", display_name: "Google: Gemma 4 26B A4B (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/google/gemma-4-31b-it:free", provider: "openrouter", display_name: "Google: Gemma 4 31B (free)", rating_milli: 1210, context_length: 262144, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", provider: "openrouter", display_name: "inclusionAI: Ling 3.2 Flash (free)", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", provider: "openrouter", display_name: "LiquidAI: LFM3-1.7B (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3 Nano Omni (free)", rating_milli: 1150, context_length: 256000, tools: true, vision: true, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3 Ultra (free)", rating_milli: 1285, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3.5 Lightning (free)", rating_milli: 1221, context_length: 1000000, tools: true, vision: false, json: false, free: true },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", provider: "openrouter", display_name: "NVIDIA: Nemotron 3.5 Super VL (free)", rating_milli: 1303, context_length: 1000000, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/particle-ai/particle-1:free", provider: "openrouter", display_name: "Particle AI: Particle-1 (free)", rating_milli: 1150, context_length: 1048576, tools: true, vision: true, json: true, free: true },
  { id: "openrouter/recursive/flint-v1.0:free", provider: "openrouter", display_name: "Recursive: Flint (free)", rating_milli: 1150, context_length: 262144, tools: true, vision: false, json: true, free: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", provider: "openrouter", display_name: "Thinking Machines: Inkling 2 Small (free)", rating_milli: 1475, context_length: 1048576, tools: true, vision: true, json: false, free: true },
];

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === "\\") i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function readBriefModels(path: string): ModelRow[] {
  if (path === "") return FALLBACK_MODELS;
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    log(`cannot read brief ${path}: ${(err as Error).message}; using the sealed catalog`);
    return FALLBACK_MODELS;
  }
  const docs = documentsIn(text);
  for (const doc of docs) {
    if (typeof doc.schema !== "string" || !doc.schema.includes("free-model-aggregator-02/")) continue;
    for (const key of ["models", "ratings"]) {
      if (!Array.isArray(doc[key])) continue;
      const rows: ModelRow[] = [];
      for (const raw of doc[key] as unknown[]) {
        if (!isRecord(raw)) continue;
        const id = typeof raw.id === "string" && raw.id !== "" ? raw.id : typeof raw.model === "string" ? raw.model : "";
        if (id === "") continue;
        const provider = typeof raw.provider === "string" && raw.provider !== "" ? raw.provider : "openrouter";
        const full = provider === "openrouter" && !id.toLowerCase().startsWith("openrouter/") ? `openrouter/${id}` : id;
        rows.push({
          id: full,
          provider,
          display_name: typeof raw.display_name === "string" ? raw.display_name : full,
          rating_milli: typeof raw.rating_milli === "number" ? raw.rating_milli : 0,
          context_length: typeof raw.context_length === "number" ? raw.context_length : null,
          tools: typeof raw.tools === "boolean" ? raw.tools : null,
          vision: typeof raw.vision === "boolean" ? raw.vision : null,
          json: typeof raw.json === "boolean" ? raw.json : null,
          free: raw.free !== false,
        });
      }
      if (rows.length > 0) return rows;
    }
  }
  log(`the brief at ${path} holds no free-model document; using the sealed catalog`);
  return FALLBACK_MODELS;
}

const MODELS: ModelRow[] = readBriefModels(PROMPT_FILE).filter((m) => m.free);

/** The catalog `--list` prints and the register body carries: the same object, byte for byte. */
const CATALOG: Json = {
  providers: [
    {
      id: "openrouter",
      base_url: "https://openrouter.ai/api/v1",
      key_env: "OPENROUTER_API_KEY",
      wire: "openai",
      models: MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
    },
  ],
};

// ---------------------------------------------------------------------------
// The request: its requirements, its token, its declared tools.
// ---------------------------------------------------------------------------

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
}

/** The text of a string, of a block array's text blocks, or '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((b) => (isRecord(b) && b.type === "text" && typeof b.text === "string" ? [b.text] : []))
    .join("\n");
}

/** Every message text in a client body (either wire), for the token hunt. */
function requestTexts(body: Json): string[] {
  const out: string[] = [];
  if (body.system !== undefined) out.push(textOf(body.system));
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    out.push(textOf(raw.content));
  }
  return out;
}

function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (isRecord(block) && (block.type === "image" || block.type === "image_url")) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(format === "json_object" || format === "json_schema" ? { json: true } : {}),
  };
}

/** The prompt's token, read off the request's own text (the instruction names it). */
function tokenOf(body: Json): string | null {
  for (const text of requestTexts(body)) {
    const found = TOKEN_RE.exec(text);
    if (found !== null) return found[0];
  }
  return null;
}

/** The tool names the client declared, in the wire's own spelling. */
function declaredToolNames(wire: "openai" | "anthropic", body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (wire === "openai" && isRecord(raw.function) && typeof raw.function.name === "string") names.push(raw.function.name);
    else if (wire === "anthropic" && typeof raw.name === "string") names.push(raw.name);
  }
  return names;
}

function meetsRequirements(model: ModelRow, requires: Requirement): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  return true;
}

/** Eligible models, best first: rating descending, brief order breaking ties. */
function candidatesFor(requires: Requirement): ModelRow[] {
  return MODELS.map((m, i) => ({ m, i }))
    .filter(({ m }) => meetsRequirements(m, requires))
    .sort((a, b) => b.m.rating_milli - a.m.rating_milli || a.i - b.i)
    .map(({ m }) => m);
}

// ---------------------------------------------------------------------------
// The gateway: one Chat Completions dial, and the answer it gives back.
// ---------------------------------------------------------------------------

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === "") return "";
  return GATEWAY_URL.endsWith("/v1") ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

interface UpstreamAnswer {
  text: string;
  finishReason: string | null;
  usage: { input: number; output: number } | null;
  toolCalls: { id: string; name: string; arguments: string }[];
}

/** Text of a Chat Completions message: a string, or its text parts joined. */
function messageText(message: Json): string {
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((p) => (isRecord(p) && p.type === "text" && typeof p.text === "string" ? [p.text] : []))
      .join("");
  }
  return "";
}

function toolCallsOf(message: Json): UpstreamAnswer["toolCalls"] {
  const out: UpstreamAnswer["toolCalls"] = [];
  if (!Array.isArray(message.tool_calls)) return out;
  for (const raw of message.tool_calls) {
    if (!isRecord(raw) || !isRecord(raw.function)) continue;
    out.push({
      id: typeof raw.id === "string" ? raw.id : "",
      name: typeof raw.function.name === "string" ? raw.function.name : "",
      arguments: typeof raw.function.arguments === "string" ? raw.function.arguments : "",
    });
  }
  return out;
}

function usageOf(value: unknown): { input: number; output: number } | null {
  if (!isRecord(value)) return null;
  const input = value.prompt_tokens;
  const output = value.completion_tokens;
  if (typeof input !== "number" || typeof output !== "number") return null;
  return { input: Math.max(0, Math.floor(input)), output: Math.max(0, Math.floor(output)) };
}

/** One SSE text body as the answer it carries: deltas joined, usage and stop reason kept. */
function parseUpstreamStream(raw: string): UpstreamAnswer {
  let text = "";
  let finishReason: string | null = null;
  let usage: { input: number; output: number } | null = null;
  const toolCalls: UpstreamAnswer["toolCalls"] = [];
  for (const frame of raw.replace(/\r\n/g, "\n").split("\n\n")) {
    for (const line of frame.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "" || payload === "[DONE]") continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const found = usageOf(chunk.usage);
      if (found !== null) usage = found;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = asRecord(choices[0]);
      if (choice === null) continue;
      const delta = asRecord(choice.delta) ?? {};
      if (typeof delta.content === "string") text += delta.content;
      if (Array.isArray(delta.tool_calls)) toolCalls.push(...toolCallsOf({ tool_calls: delta.tool_calls }));
      if (typeof choice.finish_reason === "string") finishReason = choice.finish_reason;
    }
  }
  return { text, finishReason, usage, toolCalls };
}

function errorCodeOf(bodyText: string): string {
  try {
    const value: unknown = JSON.parse(bodyText);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === "string" && value.error.code !== "") return value.error.code.slice(0, 64);
      if (typeof value.error.type === "string" && value.error.type !== "") return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to report.
  }
  return "";
}

type DialResult =
  | { ok: true; answer: UpstreamAnswer }
  | { ok: false; failure: FailureRow };

async function dial(model: string, openAIBody: Json, timeoutMs: number): Promise<DialResult> {
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${GATEWAY_KEY}`, ...(openAIBody.stream === true ? { accept: "text/event-stream" } : {}) },
      body: JSON.stringify({ ...openAIBody, model }),
      signal: AbortSignal.timeout(Math.max(1000, timeoutMs)),
    });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === "TimeoutError";
    return { ok: false, failure: { status: timedOut ? 504 : 502, code: timedOut ? "timeout" : "upstream_unreachable", model } };
  }
  const raw = await res.text().catch(() => "");
  if (!res.ok || res.body === null) {
    return { ok: false, failure: { status: res.status, code: errorCodeOf(raw), model } };
  }
  if (openAIBody.stream === true) return { ok: true, answer: parseUpstreamStream(raw) };
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { ok: false, failure: { status: 502, code: "upstream_bad_json", model } };
  }
  if (!isRecord(value)) return { ok: false, failure: { status: 502, code: "upstream_bad_json", model } };
  const choice = asRecord(Array.isArray(value.choices) ? value.choices[0] : undefined) ?? {};
  const message = asRecord(choice.message) ?? {};
  return {
    ok: true,
    answer: {
      text: messageText(message),
      finishReason: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
      usage: usageOf(value.usage),
      toolCalls: toolCallsOf(message),
    },
  };
}

// ---------------------------------------------------------------------------
// Anthropic <-> Chat Completions translation (both ways).
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null otherwise. */
function partOf(block: Json): Json | null {
  if (block.type === "text" && typeof block.text === "string") return { type: "text", text: block.text };
  if (block.type === "image" && isRecord(block.source)) {
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      const media = typeof source.media_type === "string" ? source.media_type : "image/png";
      return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === "url" && typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== "" ? text : JSON.stringify(content);
  }
  return content === undefined ? "" : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as a Chat Completions request naming `model`: the system prompt
 * becomes a system message, image blocks become image_url parts, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === "assistant";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role: assistant ? "assistant" : "user", content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: { name: typeof block.name === "string" ? block.name : "", arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: "assistant", content: text !== "" ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({ role: "tool", tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "", content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== "string") return [];
      return [{ type: "function", function: { name: raw.name, description: typeof raw.description === "string" ? raw.description : "", parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === "auto") out.tool_choice = "auto";
    else if (choice.type === "any") out.tool_choice = "required";
    else if (choice.type === "tool" && typeof choice.name === "string") out.tool_choice = { type: "function", function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function anthropicStopReason(finish: string | null): string {
  if (finish === "length") return "max_tokens";
  if (finish === "tool_calls" || finish === "function_call") return "tool_use";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

/** OpenAI's finish_reason, or "stop" when the provider named none the API knows. */
function openAIStopReason(finish: string | null): string {
  return finish !== null && ["stop", "length", "tool_calls", "function_call", "content_filter"].includes(finish) ? finish : "stop";
}

function argumentsOf(value: string): Json {
  if (value.trim() === "") return {};
  try {
    const parsed: unknown = JSON.parse(value);
    if (isRecord(parsed)) return parsed;
  } catch {
    // Not JSON: an empty input.
  }
  return {};
}

/** Token counts for an answer: the provider's, or an estimate when it sent none (output >= 1). */
function usageFor(answer: UpstreamAnswer, requestJson: string): { input: number; output: number } {
  if (answer.usage !== null && answer.usage.output >= 1) return answer.usage;
  return {
    input: answer.usage?.input ?? Math.max(0, Math.ceil(requestJson.length / 4)),
    output: Math.max(1, Math.ceil(answer.text.length / 4)),
  };
}

// ---------------------------------------------------------------------------
// Serving one prompt: route, dial, accept only a token-carrying answer.
// ---------------------------------------------------------------------------

interface ServedAnswer {
  model: string;
  answer: UpstreamAnswer;
  failures: FailureRow[];
}

/** Whether an upstream answer may serve this request: token present, JSON valid, tools declared. */
function acceptable(answer: UpstreamAnswer, token: string | null, requires: Requirement, declared: string[]): boolean {
  if (token !== null ? !answer.text.includes(token) : answer.text === "") return false;
  if (requires.json === true) {
    try {
      JSON.parse(answer.text);
    } catch {
      return false;
    }
  }
  for (const call of answer.toolCalls) {
    if (!declared.includes(call.name)) return false;
  }
  return true;
}

async function servePrompt(wire: "openai" | "anthropic", body: Json): Promise<ServedAnswer | null> {
  const requires = requirementsFromBody(body);
  const token = tokenOf(body);
  const declared = declaredToolNames(wire, body);
  const candidates = candidatesFor(requires);
  const failures: FailureRow[] = [];
  const started = Date.now();
  const openAIBody = wire === "anthropic" ? anthropicToOpenAI(body, "") : { ...body };

  for (let attempt = 0; attempt < Math.min(MAX_ATTEMPTS, candidates.length); attempt++) {
    const remaining = PROMPT_BUDGET_MS - (Date.now() - started);
    if (attempt > 0 && remaining <= 1000) break;
    const model = candidates[attempt];
    const dialled = wire === "anthropic" ? { ...openAIBody, model: model.id } : { ...openAIBody, model: model.id };
    const result = await dial(model.id, dialled, Math.min(20_000, Math.max(1000, remaining)));
    if (!result.ok) {
      failures.push(result.failure);
      log(`attempt ${attempt + 1}: ${model.id} refused ${result.failure.status} ${result.failure.code}`);
      continue;
    }
    if (!acceptable(result.answer, token, requires, declared)) {
      log(`attempt ${attempt + 1}: ${model.id} answered without the token${requires.json === true ? " or valid JSON" : ""}; retrying`);
      continue;
    }
    return { model: model.id, answer: result.answer, failures };
  }
  return null;
}

// ---------------------------------------------------------------------------
// The answers, in the wire the client used. Every one names the dialled id.
// ---------------------------------------------------------------------------

function splitText(text: string): string[] {
  if (text === "") return [];
  const parts: string[] = [];
  const size = 64;
  for (let i = 0; i < text.length; i += size) parts.push(text.slice(i, i + size));
  return parts;
}

function openAIJsonAnswer(id: string, served: ServedAnswer, requestJson: string): Json {
  const usage = usageFor(served.answer, requestJson);
  const message: Json = { role: "assistant", content: served.answer.text };
  if (served.answer.toolCalls.length > 0) {
    message.tool_calls = served.answer.toolCalls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.arguments } }));
  }
  return {
    id,
    object: "chat.completion",
    created: TRIAL_NOW_SEC,
    model: served.model,
    choices: [{ index: 0, message, finish_reason: openAIStopReason(served.answer.finishReason) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function openAIStreamFrames(id: string, served: ServedAnswer, requestJson: string): string[] {
  const usage = usageFor(served.answer, requestJson);
  const frames: string[] = [];
  const parts = splitText(served.answer.text);
  const chunk = (delta: Json, finish: string | null, withUsage: boolean): string =>
    `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: TRIAL_NOW_SEC, model: served.model, choices: [{ index: 0, delta, finish_reason: finish }], ...(withUsage ? { usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } } : {}) })}\n\n`;
  if (parts.length === 0) {
    frames.push(chunk({ role: "assistant", content: "" }, null, false));
  } else {
    frames.push(chunk({ role: "assistant", content: parts[0] }, null, false));
    for (const part of parts.slice(1)) frames.push(chunk({ content: part }, null, false));
  }
  frames.push(chunk({}, openAIStopReason(served.answer.finishReason), true));
  frames.push("data: [DONE]\n\n");
  return frames;
}

function anthropicJsonAnswer(id: string, served: ServedAnswer, requestJson: string): Json {
  const usage = usageFor(served.answer, requestJson);
  const content: Json[] = [];
  if (served.answer.text !== "") content.push({ type: "text", text: served.answer.text });
  for (const call of served.answer.toolCalls) {
    content.push({ type: "tool_use", id: call.id !== "" ? call.id : "toolu_unknown", name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  return {
    id,
    type: "message",
    role: "assistant",
    model: served.model,
    content,
    stop_reason: anthropicStopReason(served.answer.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

function anthropicStreamFrames(id: string, served: ServedAnswer, requestJson: string): string[] {
  const usage = usageFor(served.answer, requestJson);
  const stop = anthropicStopReason(served.answer.finishReason);
  const event = (name: string, data: unknown): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    event("message_start", {
      type: "message_start",
      message: { id, type: "message", role: "assistant", model: served.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    event("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const part of splitText(served.answer.text)) {
    frames.push(event("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: part } }));
  }
  // Tool calls, if the provider made any, follow the text block as tool_use blocks.
  let index = 1;
  for (const call of served.answer.toolCalls) {
    frames.push(event("content_block_start", { type: "content_block_start", index, content_block: { type: "tool_use", id: call.id !== "" ? call.id : `toolu_${index}`, name: call.name, input: {} } }));
    if (call.arguments !== "") frames.push(event("content_block_delta", { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: call.arguments } }));
    frames.push(event("content_block_stop", { type: "content_block_stop", index }));
    index += 1;
  }
  frames.push(event("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(event("message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }));
  frames.push(event("message_stop", { type: "message_stop" }));
  return frames;
}

// ---------------------------------------------------------------------------
// HTTP: the routes, the errors, the register and the poll.
// ---------------------------------------------------------------------------

function failureHeaders(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { "x-makefun-upstream-failures": JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: "openai" | "anthropic", status: number, type: string, message: string, failures: FailureRow[] = []): void {
  const headers = failureHeaders(failures);
  if (wire === "anthropic") sendJson(res, status, { type: "error", error: { type, message } }, headers);
  else sendJson(res, status, { error: { message, type, code: null } }, headers);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers["x-api-key"] === "string") {
    sendJson(res, 200, {
      data: MODELS.map((m) => ({ type: "model", id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: "list",
      data: MODELS.map((m) => ({ id: m.id, object: "model", created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
    });
  }
}

let requestCounter = 0;

async function completion(req: IncomingMessage, res: ServerResponse, wire: "openai" | "anthropic"): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, "invalid_request_error", "the request body is not one JSON object");
    return;
  }
  requestCounter += 1;
  const counter = requestCounter;
  const stream = body.stream === true;
  const served = await servePrompt(wire, body);
  if (served === null) {
    sendError(res, wire, 502, wire === "anthropic" ? "api_error" : "upstream_error", "no free model served this prompt", []);
    return;
  }
  const headers = failureHeaders(served.failures);
  const requestJson = JSON.stringify(body);
  if (wire === "openai") {
    const id = shaId("chatcmpl-", counter, served.model);
    if (stream) {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", ...headers });
      for (const frame of openAIStreamFrames(id, served, requestJson)) res.write(frame);
      res.end();
    } else {
      sendJson(res, 200, openAIJsonAnswer(id, served, requestJson), headers);
    }
    return;
  }
  const id = shaId("msg_", counter, served.model);
  if (stream) {
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", ...headers });
    for (const frame of anthropicStreamFrames(id, served, requestJson)) res.write(frame);
    res.end();
  } else {
    sendJson(res, 200, anthropicJsonAnswer(id, served, requestJson), headers);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://entry");
  if (req.method === "GET" && url.pathname === "/v1/models") {
    models(req, res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
    await completion(req, res, "openai");
    return;
  }
  if (req.method === "POST" && url.pathname === "/v1/messages") {
    await completion(req, res, "anthropic");
    return;
  }
  sendError(res, url.pathname === "/v1/messages" ? "anthropic" : "openai", 404, "not_found", `no route ${req.method ?? ""} ${url.pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the server has no port"));
        return;
      }
      resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: CATALOG }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === "number") log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("register failed");
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes("--list")) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. Wait for stdout to flush so the tail is not lost.
    const text = `${JSON.stringify(CATALOG)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === "") throw new Error("MAKEFUN_ENV_URL is not set");
  if (chatCompletionsUrl() === "") throw new Error("OPENAI_BASE_URL is not set");
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith("/v1/messages") === true ? "anthropic" : "openai", 500, "internal_error", "the server failed to answer");
    });
  });
  const port = await listen(server);
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error("the OS picked 8080, which the env owns on one layout: restart the trial");
  log(`listening on 0.0.0.0:${port} with ${MODELS.length} free models`);
  await register(port);
  process.on("SIGTERM", () => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
