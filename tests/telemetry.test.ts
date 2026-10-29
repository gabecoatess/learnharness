/**
 * telemetry — the layered turn recorder pins.
 *
 * Off = zero surface (no observers, no folder). On = one folder per chat
 * under data/telemetry/, every layer in events.jsonl with the shared
 * correlation stamp, content-addressed blobs, and the
 * anomaly.check_without_teaching flag exactly when a check is posed with
 * no learner-visible prose since the learner's last input.
 */
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, afterEach, describe, it } from "node:test";
import { createFauxCore, fauxAssistantMessage, fauxText, fauxThinking, fauxToolCall } from "@earendil-works/pi-ai";
import type { PiEngine } from "../src/agent/models.ts";

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "lh-telemetry-"));
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

const { createTeacherAgent } = await import("../src/agent/teacher-agent.ts");
const { telemetryChatDir, telemetryDir, telemetryRunEnd, telemetryRunStart } = await import("../src/agent/telemetry.ts");
const { providerEventObserver } = await import("../src/agent/provider-events.ts");

afterEach(() => {
	delete process.env.LH_TELEMETRY;
});

const checkArgs = {
	type: "check",
	nodeId: "chain-rule",
	question: "A signal is $y=\\sin(2x)$ and $\\frac{dx}{dt}=3$. What is $\\frac{dy}{dt}$?",
	options: [
		{ label: "$6\\cos(2x)$", value: "a" },
		{ label: "$2\\cos(2x)$", value: "b" },
	],
	explanation: "Multiply the slopes.",
	correctAnswer: "a",
};

function engineFor(steps: Parameters<ReturnType<typeof createFauxCore>["setResponses"]>[0]): PiEngine {
	const faux = createFauxCore({ provider: "anthropic", models: [{ id: "claude-fixture" }] });
	faux.setResponses(steps);
	return { model: faux.getModel(), streamFn: faux.streamSimple };
}

function events(chatId: string): Array<Record<string, unknown>> {
	return fs
		.readFileSync(path.join(telemetryChatDir(chatId), "events.jsonl"), "utf-8")
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe("telemetry gating", () => {
	it("off: no observers and no folder", async () => {
		const chatId = "chat_off";
		const agent = createTeacherAgent(undefined, { chatId, engine: engineFor([fauxAssistantMessage([fauxText("Hi.")])]) });
		await agent.prompt("Synthetic request.");
		telemetryRunStart(chatId, {});
		assert.equal(providerEventObserver(chatId, "teacher"), undefined);
		assert.equal(fs.existsSync(telemetryDir.current), false);
	});
});

describe("telemetry layers", () => {
	it("records every agent layer in order, with the shared correlation stamp", async () => {
		process.env.LH_TELEMETRY = "1";
		const chatId = "chat_layers";
		telemetryRunStart(chatId, { agent: "teacher", input: "answer" });
		const agent = createTeacherAgent(undefined, {
			chatId,
			engine: engineFor([fauxAssistantMessage([fauxThinking("Plan the step."), fauxText("The teaching."), fauxToolCall("quiz", checkArgs, { id: "q1" })])]),
		});
		await agent.prompt("Synthetic request.");
		const lines = events(chatId);
		const layers = lines.map((l) => l.layer);
		for (const layer of ["run.start", "agent.build", "request.prepare", "message.end", "tool.before", "tool.after", "turn.finish"]) {
			assert.ok(layers.includes(layer), `missing ${layer}`);
		}
		assert.ok(layers.indexOf("request.prepare") < layers.indexOf("tool.before"));
		assert.ok(lines.every((l, i) => i === 0 || (l.seq as number) > (lines[i - 1]!.seq as number)), "seq is monotonic");
		assert.ok(lines.slice(1).every((l) => l.run === lines[0]!.run), "one run id");

		const prepare = lines.find((l) => l.layer === "request.prepare")!;
		assert.equal(prepare.req, 1);
		const blobRef = prepare.systemPrompt as { blob: string };
		assert.ok(fs.existsSync(path.join(telemetryChatDir(chatId), "blobs", `${blobRef.blob}.json`)), "system prompt stored as a blob");

		const answer = lines.find((l) => l.layer === "message.end" && l.role === "assistant")!;
		assert.deepEqual(answer.blockOrder, ["thinking", "text", "toolCall:quiz"]);
		assert.equal((answer.totals as { visibleChars: number }).visibleChars, "The teaching.".length);

		const before = lines.find((l) => l.layer === "tool.before")!;
		assert.equal(before.questionType, "check");
		assert.equal((before.teachingBefore as { visibleChars: number }).visibleChars, "The teaching.".length);
		assert.equal((before.teachingBefore as { learnerBoundaryKind: string }).learnerBoundaryKind, "user");
		assert.ok(!layers.includes("anomaly.check_without_teaching"), "taught check is not an anomaly");
		assert.equal(lines.find((l) => l.layer === "turn.finish")!.ended, true);
		assert.ok(fs.existsSync(path.join(telemetryDir.current, "README.md")));
	});

	it("flags a check posed with only thinking since the learner's input", async () => {
		process.env.LH_TELEMETRY = "1";
		const chatId = "chat_anomaly";
		const agent = createTeacherAgent(undefined, {
			chatId,
			engine: engineFor([fauxAssistantMessage([fauxThinking("Your prediction was off."), fauxToolCall("quiz", checkArgs, { id: "q1" })])]),
		});
		await agent.prompt("Synthetic request.");
		const anomaly = events(chatId).filter((l) => l.layer === "anomaly.check_without_teaching");
		assert.equal(anomaly.length, 1);
		const teaching = anomaly[0]!.teachingBefore as { visibleChars: number; thinkingChars: number };
		assert.equal(teaching.visibleChars, 0);
		assert.equal(teaching.thinkingChars, "Your prediction was off.".length);
		assert.deepEqual(anomaly[0]!.currentMessageBlockOrder, ["thinking", "toolCall:quiz"]);
	});

	it("predictions are never flagged", async () => {
		process.env.LH_TELEMETRY = "1";
		const chatId = "chat_predict";
		const agent = createTeacherAgent(undefined, {
			chatId,
			engine: engineFor([fauxAssistantMessage([fauxToolCall("quiz", { ...checkArgs, type: "predict" }, { id: "q1" })])]),
		});
		await agent.prompt("Synthetic request.");
		assert.ok(!events(chatId).some((l) => l.layer === "anomaly.check_without_teaching"));
	});

	it("records the saved record and its UI projection at run end", () => {
		process.env.LH_TELEMETRY = "1";
		const chatId = "chat_persist";
		const message = { ...fauxAssistantMessage([fauxText("Saved prose."), fauxToolCall("quiz", checkArgs, { id: "q1" })]), id: "a1" };
		telemetryRunEnd(chatId, false, [message] as never, [message] as never);
		const lines = events(chatId);
		const persist = lines.find((l) => l.layer === "record.persist")!;
		assert.deepEqual((persist.saved as Array<{ blockOrder: string[] }>)[0]!.blockOrder, ["text", "toolCall:quiz"]);
		const view = lines.find((l) => l.layer === "view.projection")!;
		const parts = view.tailParts as Array<{ type: string; chars?: number; questionType?: string }>;
		assert.ok(parts.some((p) => p.type === "text" && p.chars === "Saved prose.".length));
		assert.ok(parts.some((p) => p.type === "tool-quiz" && p.questionType === "check"));
		assert.equal(lines.at(-1)!.layer, "run.end");
	});

	it("routes the raw provider recorder into the chat folder with the stamp", () => {
		process.env.LH_TELEMETRY = "1";
		const chatId = "chat_raw";
		providerEventObserver(chatId, "teacher")!({ type: "content_block_start", content_block: { type: "text" } }, { provider: "anthropic", id: "claude-fixture" });
		const raw = fs.readFileSync(path.join(telemetryChatDir(chatId), "provider-events.jsonl"), "utf-8").trim().split("\n").map((l) => JSON.parse(l));
		assert.equal(raw.length, 1);
		assert.equal(typeof raw[0].seq, "number");
		assert.equal(raw[0].ctx, "teacher");
	});
});
