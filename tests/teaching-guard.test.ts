/**
 * teaching-guard — no question without visible teaching.
 *
 * Pins the owner rulings (2026-10-28): during teaching, a check
 * (quiz/explain), ask_user_question, or a predict triggered by the
 * learner's answer is blocked when the learner has seen no visible prose
 * since their last input; thinking never counts; the model retries in the
 * same run; after MAX_REJECTIONS the question passes; the prober is never
 * guarded.
 */
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, describe, it } from "node:test";
import { createFauxCore, fauxAssistantMessage, fauxText, fauxThinking, fauxToolCall, type FauxResponseStep } from "@earendil-works/pi-ai";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { PiEngine } from "../src/agent/models.ts";

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "lh-teaching-guard-"));
const originalEnv = process.env;
const originalFetch = globalThis.fetch;
process.env = {
	...(originalEnv.NODE_TEST_CONTEXT ? { NODE_TEST_CONTEXT: originalEnv.NODE_TEST_CONTEXT } : {}),
	...(originalEnv.NODE_OPTIONS ? { NODE_OPTIONS: originalEnv.NODE_OPTIONS } : {}),
	NODE_ENV: "test",
	LH_DATA_DIR: path.join(scratch, "data"),
	LH_AUTH_FILE: path.join(scratch, "auth.json"),
	HOME: path.join(scratch, "home"),
};
globalThis.fetch = async () => { throw new Error("CLOSED: unexpected network request"); };
after(() => {
	process.env = originalEnv;
	globalThis.fetch = originalFetch;
	fs.rmSync(scratch, { recursive: true, force: true });
});

const { createProberAgent, createTeacherAgent } = await import("../src/agent/teacher-agent.ts");
const { MAX_REJECTIONS, rejectionReason } = await import("../src/agent/teaching-guard.ts");
const { chatStore } = await import("../src/store/chats.ts");
const { jobStore } = await import("../src/store/jobs.ts");

const quizBody = {
	question: "What is $2+2$?",
	options: [
		{ label: "$4$", value: "a" },
		{ label: "$5$", value: "b" },
	],
	explanation: "Count it out.",
	correctAnswer: "a",
};
const check = { type: "check", nodeId: "sums", ...quizBody };
const predict = { type: "predict", nodeId: "sums", ...quizBody };
const ask = { question: "Does that make sense?", options: [{ label: "Yes" }, { label: "No" }] };

function engineFor(id: string, steps: FauxResponseStep[]): { engine: PiEngine; calls: () => number } {
	const faux = createFauxCore({ provider: "anthropic", models: [{ id }] });
	faux.setResponses(steps);
	return { engine: { model: faux.getModel(), streamFn: faux.streamSimple }, calls: () => faux.state.callCount };
}

function toolUse(content: Parameters<typeof fauxAssistantMessage>[0]) {
	return fauxAssistantMessage(content, { stopReason: "toolUse" });
}

/** A history whose last input is the learner's answer to a tagged quiz —
 *  teaching has begun, and the run starts from that answer. */
function afterAnswer(): AgentMessage[] {
	return [
		{ role: "user", content: [{ type: "text", text: "Teach me sums." }], timestamp: 1 },
		toolUse([fauxText("Sums combine counts."), fauxToolCall("quiz", check, { id: "q0" })]),
		{
			role: "toolResult",
			toolCallId: "q0",
			toolName: "quiz",
			content: [{ type: "text", text: "Answered." }],
			details: { kind: "quiz", questionText: quizBody.question, dontKnow: false, answers: [{ index: 1, label: "$5$" }], answeredAt: "2026-10-28T00:00:00.000Z", correct: false },
			isError: false,
			timestamp: 2,
		},
	] as unknown as AgentMessage[];
}

/** The tool results the run produced, in order. */
function results(messages: readonly AgentMessage[]) {
	return messages.filter((m) => m.role === "toolResult").map((m) => ({
		name: (m as { toolName: string }).toolName,
		isError: (m as { isError: boolean }).isError,
		text: ((m as { content: Array<{ text?: string }> }).content[0]?.text) ?? "",
	}));
}

describe("teaching guard", () => {
	it("rejects a check posed with only thinking; the retry with teaching passes", async () => {
		const { engine, calls } = engineFor("claude-fixture", [
			toolUse([fauxThinking("I've explained sums."), fauxToolCall("quiz", check, { id: "q1" })]),
			toolUse([fauxText("Two and two make four."), fauxToolCall("quiz", check, { id: "q2" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Teach me sums.");
		assert.equal(calls(), 2);
		const [rejected, posed] = results(agent.state.messages);
		assert.equal(rejected!.isError, true);
		assert.equal(rejected!.text, rejectionReason("check", true));
		assert.match(rejected!.text, /with send_to_user/);
		assert.equal(posed!.isError, false);
	});

	it("counts a send_to_user message as teaching", async () => {
		const { engine, calls } = engineFor("claude-fixture", [
			toolUse([fauxToolCall("send_to_user", { message: "Two and two make four." }, { id: "s1" }), fauxToolCall("quiz", check, { id: "q1" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Teach me sums.");
		assert.equal(calls(), 1);
		assert.ok(results(agent.state.messages).every((r) => !r.isError));
	});

	it("a failed send_to_user is not teaching", async () => {
		const { engine } = engineFor("claude-fixture", [
			toolUse([fauxToolCall("send_to_user", { message: " " }, { id: "s1" })]),
			toolUse([fauxToolCall("quiz", check, { id: "q1" })]),
			toolUse([fauxText("Two and two make four."), fauxToolCall("quiz", check, { id: "q2" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Teach me sums.");
		const quizzes = results(agent.state.messages).filter((r) => r.name === "quiz");
		assert.deepEqual(quizzes.map((r) => r.isError), [true, false]);
	});

	it("tells models without send_to_user to write ordinary response text", async () => {
		const { engine } = engineFor("gpt-fixture", [
			toolUse([fauxToolCall("quiz", check, { id: "q1" })]),
			toolUse([fauxText("Two and two make four."), fauxToolCall("quiz", check, { id: "q2" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Teach me sums.");
		assert.equal(results(agent.state.messages)[0]!.text, rejectionReason("check", false));
		assert.match(results(agent.state.messages)[0]!.text, /as ordinary response text/);
	});

	it("guards ask_user_question once teaching has begun", async () => {
		const { engine, calls } = engineFor("claude-fixture", [
			toolUse([fauxThinking("I've explained it."), fauxToolCall("ask_user_question", ask, { id: "a1" })]),
			toolUse([fauxText("Five is one too many."), fauxToolCall("ask_user_question", ask, { id: "a2" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine, messages: afterAnswer() });
		await agent.continue();
		assert.equal(calls(), 2);
		assert.deepEqual(results(agent.state.messages).slice(1).map((r) => r.isError), [true, false]);
	});

	it("leaves ask_user_question alone during planning", async () => {
		const { engine, calls } = engineFor("claude-fixture", [toolUse([fauxToolCall("ask_user_question", ask, { id: "a1" })])]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Plan it differently.");
		assert.equal(calls(), 1);
		assert.equal(results(agent.state.messages)[0]!.isError, false);
	});

	it("rejects a predict that skips the reaction to the learner's answer", async () => {
		const { engine, calls } = engineFor("claude-fixture", [
			toolUse([fauxThinking("Off again; next step."), fauxToolCall("quiz", predict, { id: "p1" })]),
			toolUse([fauxText("Not five: two and two make four."), fauxToolCall("quiz", predict, { id: "p2" })]),
		]);
		const agent = createTeacherAgent(undefined, { engine, messages: afterAnswer() });
		await agent.continue();
		assert.equal(calls(), 2);
		const [rejected, posed] = results(agent.state.messages).slice(1);
		assert.equal(rejected!.text, rejectionReason("predict-after-answer", true));
		assert.equal(posed!.isError, false);
	});

	it("lets a predict after a typed message open the step", async () => {
		const { engine, calls } = engineFor("claude-fixture", [toolUse([fauxToolCall("quiz", predict, { id: "p1" })])]);
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Go ahead.");
		assert.equal(calls(), 1);
		assert.equal(results(agent.state.messages)[0]!.isError, false);
	});

	it(`lets the question through after ${MAX_REJECTIONS} rejections`, async () => {
		const stubborn = (id: string) => toolUse([fauxThinking("I've explained it."), fauxToolCall("quiz", check, { id })]);
		const { engine, calls } = engineFor("claude-fixture", Array.from({ length: MAX_REJECTIONS + 1 }, (_, i) => stubborn(`q${i}`)));
		const agent = createTeacherAgent(undefined, { engine });
		await agent.prompt("Teach me sums.");
		assert.equal(calls(), MAX_REJECTIONS + 1);
		assert.deepEqual(results(agent.state.messages).map((r) => r.isError), [...Array(MAX_REJECTIONS).fill(true), false]);
	});

	it("never guards the prober", async () => {
		const chatId = chatStore.createChat("Synthetic prober").id;
		const { engine, calls } = engineFor("claude-fixture", [toolUse([fauxThinking("Probe."), fauxToolCall("probe", quizBody, { id: "q1" })])]);
		const agent = createProberAgent({ chatId, chatStore, jobStore, deliver: () => {}, spawn: () => {} }, { engine });
		await agent.prompt("Teach me sums.");
		assert.equal(calls(), 1);
		assert.equal(results(agent.state.messages)[0]!.isError, false);
	});
});
