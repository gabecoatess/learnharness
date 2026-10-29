/** The per-request output-token ceiling: offline OpenRouter payloads only.
 * Unset, pi requests the catalog maximum and OpenRouter refuses any request
 * whose worst case exceeds the account's remaining credits (402). */
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { normalizeContext } from "@earendil-works/pi-ai";
import { MAX_OUTPUT_TOKENS, piEngineForSpec } from "../src/agent/models.ts";

/** Stream one offline request and return the JSON payload OpenRouter would receive. */
async function payloadFor(id: string, options: { maxTokens?: number } = {}): Promise<Record<string, unknown>> {
	const before = process.env;
	const originalFetch = globalThis.fetch;
	process.env = { NODE_ENV: "test", HOME: "/nonexistent/max-output", OPENROUTER_API_KEY: "dummy-max-output" };
	let payload: Record<string, unknown> | undefined;
	try {
		globalThis.fetch = async (input, init) => {
			payload = JSON.parse(await new Request(input, init).text());
			return new Response("{}", { status: 400, headers: { "content-type": "application/json" } });
		};
		const engine = piEngineForSpec({ provider: "openrouter", id });
		await engine.streamFn(engine.model, normalizeContext({ messages: [{ role: "user", content: "offline", timestamp: 1 }] }), { sessionId: "chat_max_output", ...options }).result();
	} finally {
		process.env = before;
		globalThis.fetch = originalFetch;
	}
	assert.ok(payload, "request was built");
	return payload;
}

const ceiling = (payload: Record<string, unknown>) => payload.max_tokens ?? payload.max_completion_tokens;

describe("output-token ceiling", { concurrency: false }, () => {
	it("caps a huge catalog maximum (Gemini 4 Argon, ~944k) at the ceiling", async () => {
		assert.equal(ceiling(await payloadFor("google/gemini-4-argon")), MAX_OUTPUT_TOKENS);
	});

	it("keeps a model's own smaller maximum", async () => {
		assert.equal(ceiling(await payloadFor("aion-labs/aion-3.0")), 32768);
	});

	it("an explicit caller value wins", async () => {
		assert.equal(ceiling(await payloadFor("google/gemini-4-argon", { maxTokens: 1234 })), 1234);
	});
});
