/**
 * telemetry — the layered turn recorder.
 *
 * Question it answers: when a teaching check arrives with no visible
 * teaching before it, WHERE did the prose go? Every layer a learner-facing
 * sentence crosses is recorded into one per-chat timeline, so a missing
 * paragraph can be traced from the HTTP request to the saved record and
 * the UI projection.
 *
 * Gating: env var only — LH_TELEMETRY=1 npm run dev. Off = no observers,
 * no files, no behavior change. Every hook here is a pure observer: it
 * returns undefined (pi's "keep going unchanged") and never throws into
 * the run.
 *
 * Artifact (one folder per chat):
 *   data/telemetry/README.md                       — the format reference
 *   data/telemetry/<chatId>/events.jsonl           — the layered timeline
 *   data/telemetry/<chatId>/provider-events.jsonl  — raw provider stream
 *                                                    (provider-events.ts)
 *   data/telemetry/<chatId>/blobs/<hash>.json      — content stored once
 *                                                    (system prompts, tool
 *                                                    schemas, request messages)
 *
 * Correlation: every line carries `t` (epoch ms), `boot` (this server
 * process), `seq` (per-chat, per-process, shared with the raw stream
 * file), `run` (one teacher/prober/practice run), and `req` (the model
 * request inside that run).
 */

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import path from "node:path";
import type { AgentEvent, AgentMessage, AgentTool, AfterToolCallContext, BeforeToolCallContext, AgentTurnContext, PrepareRequestContext } from "@earendil-works/pi-agent-core";
import type { ProviderResponse } from "@earendil-works/pi-ai";
import { PATHS } from "../../config.ts";
import { SEND_TO_USER_TOOL, userFacingTextOf, type LhMessage } from "../core/native-messages.ts";
import { QUESTION_TOOLS } from "../core/pending.ts";
import { normalizeQuizType } from "../core/question-core.ts";
import { projectView } from "../core/view.ts";
import { teachingBeforeCall } from "./teaching-guard.ts";

/** Test seam (the providerEventsDir idiom): redirect the output root. */
export const telemetryDir = { current: path.join(PATHS.data, "telemetry") };

/** True when recording is on (read per call, like providerEventsEnabled). */
export function telemetryEnabled(): boolean {
	return !!process.env.LH_TELEMETRY;
}

/** One chat's telemetry folder. Chat ids are app-minted; sanitize anyway. */
export function telemetryChatDir(chatId: string): string {
	return path.join(telemetryDir.current, chatId.replace(/[^a-zA-Z0-9_-]/g, "_"));
}

// ─── correlation state ──────────────────────────────────────────────────────

const BOOT = Date.now().toString(36);

interface ChatState {
	seq: number;
	run: string | null;
	runCount: number;
	req: number;
}

const chats = new Map<string, ChatState>();

function stateFor(chatId: string): ChatState {
	let state = chats.get(chatId);
	if (!state) {
		state = { seq: 0, run: null, runCount: 0, req: 0 };
		chats.set(chatId, state);
	}
	return state;
}

/** The correlation stamp every telemetry line carries — the raw stream
 *  recorder calls this too, so both files share one `seq` ordering. */
export function telemetryStamp(chatId: string): { t: number; boot: string; seq: number; run: string | null; req: number } {
	const state = stateFor(chatId);
	state.seq += 1;
	return { t: Date.now(), boot: BOOT, seq: state.seq, run: state.run, req: state.req };
}

// ─── writing ────────────────────────────────────────────────────────────────

function ensureReadme(): void {
	const file = path.join(telemetryDir.current, "README.md");
	if (fs.existsSync(file)) return;
	fs.mkdirSync(telemetryDir.current, { recursive: true });
	fs.writeFileSync(file, README, "utf-8");
}

/** Append one event to the chat's timeline. Never throws. */
export function telemetryEvent(chatId: string, layer: string, data: Record<string, unknown>): void {
	if (!telemetryEnabled()) return;
	try {
		ensureReadme();
		const dir = telemetryChatDir(chatId);
		fs.mkdirSync(dir, { recursive: true });
		fs.appendFileSync(path.join(dir, "events.jsonl"), `${safeJson({ ...telemetryStamp(chatId), layer, ...data })}\n`, "utf-8");
	} catch {
		// The recorder must never take a run down with it.
	}
}

function safeJson(value: unknown): string {
	try {
		return JSON.stringify(value);
	} catch {
		return JSON.stringify({ __nonSerializable: typeof value });
	}
}

const writtenBlobs = new Set<string>();

/** Store a value once under its content hash; returns the reference. */
function blob(chatId: string, value: unknown): { blob: string } {
	const text = safeJson(value);
	const hash = createHash("sha256").update(text).digest("hex").slice(0, 16);
	const file = path.join(telemetryChatDir(chatId), "blobs", `${hash}.json`);
	if (!writtenBlobs.has(file)) {
		try {
			if (!fs.existsSync(file)) {
				fs.mkdirSync(path.dirname(file), { recursive: true });
				fs.writeFileSync(file, text, "utf-8");
			}
			writtenBlobs.add(file);
		} catch {
			// A missing blob degrades to a dangling reference, never a crash.
		}
	}
	return { blob: hash };
}

/** Payload keys whose arrays are stored per element (each message once). */
const PER_ELEMENT_KEYS = new Set(["messages", "input", "contents"]);
/** Inline values up to this size; larger ones move to blobs. */
const INLINE_LIMIT = 2048;

/** The exact provider payload, with bulky content moved to blobs: the
 *  conversation per message, the system prompt and tool schemas whole. */
function dedupePayload(chatId: string, payload: unknown): unknown {
	if (!payload || typeof payload !== "object" || Array.isArray(payload)) return blob(chatId, payload);
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(payload as Record<string, unknown>)) {
		if (PER_ELEMENT_KEYS.has(key) && Array.isArray(value)) out[key] = value.map((element) => blob(chatId, element));
		else if (safeJson(value).length > INLINE_LIMIT) out[key] = blob(chatId, value);
		else out[key] = value;
	}
	return out;
}

// ─── message shapes ─────────────────────────────────────────────────────────

type Block = { type: string; [key: string]: unknown };

function blocksOf(message: AgentMessage): Block[] {
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") return [{ type: "text", text: content }];
	return Array.isArray(content) ? (content as Block[]) : [];
}

/** One content block, full text, with lengths precomputed for scanning. */
function blockRecord(block: Block): Record<string, unknown> {
	if (block.type === "text") {
		const text = String(block.text ?? "");
		return { type: "text", chars: text.length, text, ...(block.textSignature ? { textSignatureLength: String(block.textSignature).length } : {}) };
	}
	if (block.type === "thinking") {
		const thinking = String(block.thinking ?? "");
		return {
			type: "thinking",
			chars: thinking.length,
			thinking,
			signatureLength: typeof block.thinkingSignature === "string" ? block.thinkingSignature.length : 0,
			redacted: block.redacted === true,
		};
	}
	if (block.type === "toolCall") {
		return {
			type: "toolCall",
			id: block.id,
			name: block.name,
			arguments: block.arguments,
			...(block.argumentsError ? { argumentsError: block.argumentsError } : {}),
		};
	}
	if (block.type === "image") return { type: "image", mimeType: block.mimeType, bytes: String(block.data ?? "").length };
	return block;
}

/** Visible vs hidden character totals for one assistant message. */
function assistantTotals(message: AgentMessage): Record<string, number> {
	const blocks = blocksOf(message);
	let thinkingChars = 0, textChars = 0, sendToUserChars = 0, toolCalls = 0;
	for (const block of blocks) {
		if (block.type === "thinking") thinkingChars += String(block.thinking ?? "").length;
		else if (block.type === "text") textChars += String(block.text ?? "").length;
		else if (block.type === "toolCall") {
			toolCalls += 1;
			if (block.name === SEND_TO_USER_TOOL) sendToUserChars += String((block.arguments as { message?: unknown } | undefined)?.message ?? "").length;
		}
	}
	const visibleChars = userFacingTextOf(blocks).reduce((sum, text) => sum + text.length, 0);
	return { visibleChars, textChars, sendToUserChars, thinkingChars, toolCalls };
}

/** A whole message at message_end — role-specific, full content. */
function messageRecord(message: AgentMessage): Record<string, unknown> {
	const m = message as unknown as Record<string, unknown>;
	if (message.role === "assistant") {
		return {
			role: "assistant",
			provider: m.provider,
			model: m.model,
			api: m.api,
			responseModel: m.responseModel,
			responseId: m.responseId,
			thinkingLevel: m.thinkingLevel,
			providerThinkingLevel: m.providerThinkingLevel,
			stopReason: m.stopReason,
			errorMessage: m.errorMessage,
			diagnostics: m.diagnostics,
			usage: m.usage,
			totals: assistantTotals(message),
			blockOrder: blocksOf(message).map((b) => (b.type === "toolCall" ? `toolCall:${String(b.name)}` : b.type)),
			blocks: blocksOf(message).map(blockRecord),
		};
	}
	if (message.role === "toolResult") {
		return {
			role: "toolResult",
			toolName: m.toolName,
			toolCallId: m.toolCallId,
			isError: m.isError,
			content: blocksOf(message).map(blockRecord),
			details: m.details,
		};
	}
	if (message.role === "custom") {
		return { role: "custom", customType: m.customType, message: m };
	}
	return { role: message.role, content: blocksOf(message).map(blockRecord) };
}

// ─── the agent-level observers ──────────────────────────────────────────────

/** How the send_to_user registry gate resolved for this agent. */
export interface SendToUserGate {
	included: boolean;
	explicitRegistry: boolean;
	modelId: string;
}

export interface AgentTelemetry {
	prepareRequest(request: PrepareRequestContext): void;
	onPayload(payload: unknown, model: { provider?: string; id?: string }): undefined;
	onResponse(response: ProviderResponse, model: { provider?: string; id?: string }): void;
	beforeToolCall(context: BeforeToolCallContext): Promise<undefined>;
	afterToolCall(context: AfterToolCallContext): Promise<undefined>;
	finishTurn(turn: AgentTurnContext, ended: boolean): void;
	listener(event: AgentEvent): void;
}

function toolSchema(tool: AgentTool<any>): Record<string, unknown> {
	return { name: tool.name, description: tool.description, parameters: tool.parameters };
}

/**
 * The observers for one agent build, or undefined when disabled. Wired at
 * the shared construction point (teacher-agent's baseAgentOptions), so the
 * prober, teacher, and practitioner are all covered.
 */
export function agentTelemetry(chatId: string, agent: string, gate: SendToUserGate): AgentTelemetry | undefined {
	if (!telemetryEnabled()) return undefined;
	const state = stateFor(chatId);
	const record = (layer: string, data: Record<string, unknown>) => telemetryEvent(chatId, layer, { agent, ...data });
	record("agent.build", { sendToUser: gate });
	return {
		prepareRequest(request) {
			try {
				state.req += 1;
				const messages = request.context.messages;
				const system = messages.find((m) => m.role === "system");
				const tools = request.context.tools ?? [];
				record("request.prepare", {
					model: { provider: request.model.provider, id: request.model.id, api: request.model.api },
					thinkingLevel: request.thinkingLevel,
					messageCount: messages.length,
					roles: messages.map((m) => (m.role === "custom" ? `custom:${String((m as { customType?: unknown }).customType)}` : m.role)),
					systemPrompt: system ? blob(chatId, (system as { content?: unknown }).content) : null,
					toolNames: tools.map((t) => t.name),
					tools: blob(chatId, tools.map(toolSchema)),
					sendToUserOffered: tools.some((t) => t.name === SEND_TO_USER_TOOL),
				});
			} catch {
				// observer isolation
			}
		},
		onPayload(payload, model) {
			try {
				record("request.payload", { provider: model?.provider, model: model?.id, payload: dedupePayload(chatId, payload) });
			} catch {
				// observer isolation
			}
			return undefined;
		},
		onResponse(response, model) {
			record("response.http", { provider: model?.provider, model: model?.id, status: response.status, headers: response.headers });
		},
		async beforeToolCall(context) {
			try {
				const name = context.toolCall.name;
				const args = context.args as Record<string, unknown> | undefined;
				const question = (QUESTION_TOOLS as readonly string[]).includes(name);
				const questionType = question ? normalizeQuizType(args?.type) ?? null : undefined;
				const teaching = question ? teachingBeforeCall(context) : undefined;
				record("tool.before", {
					toolCallId: context.toolCall.id,
					name,
					args: context.args,
					...(question ? { questionType, nodeId: args?.nodeId, teachingBefore: teaching } : {}),
				});
				if (questionType === "check" && teaching && teaching.visibleChars === 0) {
					record("anomaly.check_without_teaching", {
						toolCallId: context.toolCall.id,
						name,
						nodeId: args?.nodeId,
						teachingBefore: teaching,
						currentMessageBlockOrder: blocksOf(context.assistantMessage as AgentMessage).map((b) => (b.type === "toolCall" ? `toolCall:${String(b.name)}` : b.type)),
					});
				}
			} catch {
				// observer isolation
			}
			return undefined;
		},
		async afterToolCall(context) {
			try {
				record("tool.after", {
					toolCallId: context.toolCall.id,
					name: context.toolCall.name,
					isError: context.isError,
					result: { content: context.result.content, details: context.result.details, terminate: (context.result as { terminate?: unknown }).terminate },
				});
			} catch {
				// observer isolation
			}
			return undefined;
		},
		finishTurn(turn, ended) {
			try {
				const errored = new Set(turn.toolResults.filter((r) => r.isError).map((r) => r.toolCallId));
				record("turn.finish", {
					ended,
					stopReason: turn.message.stopReason,
					toolCalls: blocksOf(turn.message)
						.filter((b) => b.type === "toolCall")
						.map((b) => ({ id: b.id, name: b.name, isError: errored.has(String(b.id)), question: (QUESTION_TOOLS as readonly string[]).includes(String(b.name)) })),
					toolResultCount: turn.toolResults.length,
				});
			} catch {
				// observer isolation
			}
		},
		listener(event) {
			try {
				if (event.type === "message_end") record("message.end", messageRecord(event.message));
				else if (event.type === "agent_start" || event.type === "agent_end" || event.type === "turn_start" || event.type === "turn_end") {
					record(`agent.${event.type}`, event.type === "agent_end" ? { messageCount: event.messages.length } : {});
				}
			} catch {
				// observer isolation
			}
		},
	};
}

// ─── run-level records (the chat-handler seam) ──────────────────────────────

/** A run begins: mint its id so every later line correlates to it. */
export function telemetryRunStart(chatId: string, info: Record<string, unknown>): void {
	if (!telemetryEnabled()) return;
	const state = stateFor(chatId);
	state.runCount += 1;
	state.run = `${BOOT}-${state.runCount}`;
	state.req = 0;
	telemetryEvent(chatId, "run.start", info);
}

/** A run settled: what was saved, and what the UI projection received. */
export function telemetryRunEnd(chatId: string, aborted: boolean, stamped: readonly LhMessage[], record: readonly LhMessage[]): void {
	if (!telemetryEnabled()) return;
	try {
		telemetryEvent(chatId, "record.persist", {
			aborted,
			recordMessageCount: record.length,
			saved: stamped.map((m) => ({
				id: m.id,
				role: m.role,
				...(m.role === "assistant" ? { totals: assistantTotals(m as AgentMessage), blockOrder: blocksOf(m as AgentMessage).map((b) => (b.type === "toolCall" ? `toolCall:${String(b.name)}` : b.type)) } : {}),
				...(m.role === "toolResult" ? { toolName: m.toolName, isError: m.isError } : {}),
			})),
		});
		const view = projectView(record);
		const last = [...view].reverse().find((m) => m.role === "assistant");
		const parts = last?.parts ?? [];
		telemetryEvent(chatId, "view.projection", {
			viewMessageCount: view.length,
			lastAssistantViewId: last?.id ?? null,
			lastAssistantPartCount: parts.length,
			// The tail of the run the learner is looking at, newest last.
			tailParts: parts.slice(-60).map((p) => ({
				type: p.type,
				...(typeof p.text === "string" ? { chars: p.text.length } : {}),
				...(p.toolCallId ? { toolCallId: p.toolCallId } : {}),
				...(p.state ? { state: p.state } : {}),
				...(p.type === "tool-quiz" || p.type === "tool-explain" ? { questionType: normalizeQuizType((p.input as { type?: unknown } | undefined)?.type) ?? null } : {}),
			})),
		});
	} catch {
		// observer isolation
	}
	telemetryEvent(chatId, "run.end", { aborted });
}

// ─── the format reference, written beside the data ──────────────────────────

const README = `# LearnHarness telemetry

Written when the dev server runs with \`LH_TELEMETRY=1\`. One folder per chat:

- \`<chatId>/events.jsonl\`: the layered timeline (below).
- \`<chatId>/provider-events.jsonl\`: every raw provider stream event, before
  the agent library normalizes it (the existing LH_PROVIDER_EVENTS recorder,
  redirected here; capped at ~20 MB per file).
- \`<chatId>/blobs/<hash>.json\`: content stored once and referenced as
  \`{"blob":"<hash>"}\` (system prompts, tool schemas, request messages).

Every line carries \`t\` (epoch ms), \`boot\` (server process), \`seq\` (order
within this chat and process, shared by both .jsonl files), \`run\` (one agent
run) and \`req\` (the model request within that run).

## Layers in events.jsonl, in the order a turn crosses them

| layer | when | key fields |
|---|---|---|
| run.start | a teacher/prober/practice run begins | agent, input (answer/steers/continue/retry…), model, thinkingLevel |
| agent.build | the agent is constructed | sendToUser gate: included, explicitRegistry, modelId |
| request.prepare | before each model request | thinkingLevel, roles, systemPrompt blob, toolNames, sendToUserOffered |
| request.payload | the exact body sent to the provider | payload (thinking settings inline; messages, system, tools as blobs) |
| response.http | HTTP response received | status, headers (includes the provider request id) |
| (provider-events.jsonl) | each raw stream event | event (e.g. content_block_start with a text or thinking block) |
| message.end | each finished message | assistant: blocks in order with full text/thinking, signatureLength, totals.visibleChars vs thinkingChars, usage, stopReason |
| tool.before | a tool call is about to run | name, args; question tools: questionType, nodeId, teachingBefore |
| anomaly.check_without_teaching | a check is posed with zero visible text since the learner's last input | teachingBefore, currentMessageBlockOrder |
| tool.after | a tool call finished | isError, result |
| turn.finish | the turn-stop rule ran | ended, toolCalls (isError, question) |
| record.persist | the run's messages were saved | saved: id, role, totals, blockOrder |
| view.projection | what the UI projection of the record holds | tailParts: type, chars, questionType |
| run.end | the run settled | aborted |

\`teachingBefore\` counts only learner-visible prose (text blocks and
successful send_to_user calls) between the learner's last input and the
question call. Thinking is counted separately and never as visible.

Start with: \`grep anomaly.check_without_teaching */events.jsonl\`, then read
that run's message.end lines and the matching provider-events.jsonl lines.
`;
