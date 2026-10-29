/**
 * teaching-guard — no question without visible teaching.
 *
 * The teaching doctrine (prompts/teacher/20-teaching.md) says "teach
 * first, check second", and a check with no prose before it is a failed
 * check. Nothing enforced it: the model could plan an explanation in its
 * thinking, record "I've explained…", and pose the check — the learner
 * saw only the question (telemetry, 2026-10-28: the raw provider stream
 * held thinking → thinking → tool_use, never a text block).
 *
 * The guard runs at the pre-tool-call boundary (pi's beforeToolCall) for
 * the TEACHER only, once teaching has begun. A guarded question with no
 * learner-visible prose since the learner's last input is blocked; pi
 * returns the reason as an error tool result, the question stopper keeps
 * the turn running on an errored pose, and the model retries — teaching
 * first. Owner rulings (2026-10-28):
 *
 * - Guarded: check quizzes/explains, ask_user_question, and a predict
 *   whose trigger was the learner's answer to a previous question (it
 *   owes a reaction). A predict after a typed message passes — it IS the
 *   step's opening move.
 * - Visible = any non-blank text block or successful send_to_user call.
 *   Thinking never counts; failed output calls never count.
 * - At most MAX_REJECTIONS blocks per agent run (one learner turn), then
 *   the question passes so the lesson never stalls.
 */

import type { AgentMessage, BeforeToolCallContext, BeforeToolCallResult } from "@earendil-works/pi-agent-core";
import { failedOutputCallIds, nativeInputCause, SEND_TO_USER_TOOL, userFacingTextOf, type LhMessage } from "../core/native-messages.ts";
import { normalizeQuizType } from "../core/question-core.ts";

/** Blocks per agent run before a question is let through. */
export const MAX_REJECTIONS = 3;

type Block = { type: string; [key: string]: unknown };

function blocksOf(message: AgentMessage): Block[] {
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") return [{ type: "text", text: content }];
	return Array.isArray(content) ? (content as Block[]) : [];
}

/** What the learner was shown before a tool call. */
export interface TeachingBefore {
	learnerBoundaryFound: boolean;
	/** "user", "answer:<question tool>", or "custom:<type>"; null when none. */
	learnerBoundaryKind: string | null;
	earlierAssistantMessages: number;
	visibleChars: number;
	visibleBlocks: number;
	visibleCharsInCurrentMessageBeforeCall: number;
	thinkingChars: number;
	sendToUserCalls: number;
	sendToUserFailed: number;
}

/** Measure the learner-visible prose between the learner's last input
 *  and this tool call: assistant messages since that input, plus the
 *  current message's blocks BEFORE the call. Shared with telemetry, so
 *  the recorder and the guard can never disagree. */
export function teachingBeforeCall(context: Pick<BeforeToolCallContext, "assistantMessage" | "toolCall" | "context">): TeachingBefore {
	const all = context.context.messages;
	let end = all.indexOf(context.assistantMessage as AgentMessage);
	if (end < 0) end = all.length;
	let boundary = -1;
	let boundaryKind: string | null = null;
	for (let i = end - 1; i >= 0; i--) {
		const message = all[i]!;
		if (nativeInputCause(message, all.slice(0, i)) === "learner") {
			boundary = i;
			const m = message as unknown as Record<string, unknown>;
			boundaryKind = message.role === "toolResult" ? `answer:${String(m.toolName)}` : message.role === "custom" ? `custom:${String(m.customType)}` : message.role;
			break;
		}
	}
	const between = all.slice(boundary + 1, end);
	const failed = failedOutputCallIds(between as unknown as LhMessage[]);
	let visibleChars = 0, visibleBlocks = 0, thinkingChars = 0, sendToUserCalls = 0, sendToUserFailed = 0, assistantMessages = 0;
	const tally = (blocks: Block[]) => {
		for (const text of userFacingTextOf(blocks, failed)) {
			if (!text.trim()) continue;
			visibleChars += text.length;
			visibleBlocks += 1;
		}
		for (const block of blocks) {
			if (block.type === "thinking") thinkingChars += String(block.thinking ?? "").length;
			if (block.type === "toolCall" && block.name === SEND_TO_USER_TOOL) {
				sendToUserCalls += 1;
				if (failed.has(String(block.id))) sendToUserFailed += 1;
			}
		}
	};
	for (const message of between) {
		if (message.role !== "assistant") continue;
		assistantMessages += 1;
		tally(blocksOf(message));
	}
	const current = blocksOf(context.assistantMessage as AgentMessage);
	const callIndex = current.findIndex((b) => b.type === "toolCall" && b.id === context.toolCall.id);
	const before = current.slice(0, callIndex < 0 ? current.length : callIndex);
	const visibleBeforeInCurrent = userFacingTextOf(before, failed).reduce((sum, text) => sum + text.trim().length, 0);
	tally(before);
	return {
		learnerBoundaryFound: boundary >= 0,
		learnerBoundaryKind: boundaryKind,
		earlierAssistantMessages: assistantMessages,
		visibleChars,
		visibleBlocks,
		visibleCharsInCurrentMessageBeforeCall: visibleBeforeInCurrent,
		thinkingChars,
		sendToUserCalls,
		sendToUserFailed,
	};
}

/** True once a nodeId-tagged quiz/explain exists — the teach seam
 *  (core/phases.ts), checked over the run's context plus this call. */
function teachingBegun(context: BeforeToolCallContext): boolean {
	const tagged = (block: Block) =>
		block.type === "toolCall" &&
		(block.name === "quiz" || block.name === "explain") &&
		typeof (block.arguments as { nodeId?: unknown } | undefined)?.nodeId === "string";
	if (tagged(context.toolCall as unknown as Block)) return true;
	return context.context.messages.some((m) => m.role === "assistant" && blocksOf(m).some(tagged));
}

type Rule = "check" | "predict-after-answer";

/** Which rule, if any, this call falls under. */
function ruleFor(context: BeforeToolCallContext, teaching: () => TeachingBefore): Rule | null {
	const name = context.toolCall.name;
	if (name === "ask_user_question") return "check";
	if (name !== "quiz" && name !== "explain") return null;
	// Same reading as the dock: everything that is not a predict is a check.
	if (normalizeQuizType((context.args as { type?: unknown } | undefined)?.type) !== "predict") return "check";
	return teaching().learnerBoundaryKind?.startsWith("answer:") ? "predict-after-answer" : null;
}

/** The rejection text the model reads (owner-approved wording). */
export function rejectionReason(rule: Rule, sendToUserAvailable: boolean): string {
	const channel = sendToUserAvailable ? "with send_to_user" : "as ordinary response text";
	if (rule === "predict-after-answer") {
		return (
			"Question not posed: the learner has seen no reaction to their last answer. Your thinking is private and never " +
			`reaches them. Resolve their answer now ${channel} (what was right or wrong, and why), then pose the prediction again.`
		);
	}
	return (
		"Question not posed: the learner has seen no text since their last input. Your thinking is private and never " +
		"reaches them, so anything you only thought has not been taught. Write the full explanation now " +
		`${channel}, then pose the question again.`
	);
}

/** One agent run's guard (fresh per construction — the agent is rebuilt
 *  every learner turn, so the rejection budget resets with it). */
export function createTeachingGuard(): (context: BeforeToolCallContext) => BeforeToolCallResult | undefined {
	let rejections = 0;
	return (context) => {
		if (!teachingBegun(context)) return undefined;
		let measured: TeachingBefore | undefined;
		const teaching = () => (measured ??= teachingBeforeCall(context));
		const rule = ruleFor(context, teaching);
		if (!rule || teaching().visibleChars > 0) return undefined;
		if (rejections >= MAX_REJECTIONS) return undefined;
		rejections += 1;
		const sendToUserAvailable = (context.context.tools ?? []).some((t) => t.name === SEND_TO_USER_TOOL);
		return { block: true, reason: rejectionReason(rule, sendToUserAvailable) };
	};
}
