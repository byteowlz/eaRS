import { expect, test } from "bun:test";
import {
	MOONSHINE_RUNTIME,
	verifyRuntimeArchive,
} from "../scripts/stage-runtime.mjs";

test("runtime is digest-pinned to the German/split-frontend-compatible GitHub artifact, not npm", () => {
	expect(MOONSHINE_RUNTIME.url).toContain(
		"releases/download/v0.1.5/moonshine-voice-wasm.tar.gz",
	);
	expect(MOONSHINE_RUNTIME.sha256).toBe(
		"c515bf7691e12048f70a92cc82b3b0894c16c3773ffcacb7d48944fb150e4837",
	);
	expect(MOONSHINE_RUNTIME.wasmSha256).not.toBe(
		"d0e783f661f8be68f7af48d859d3bbc7a6381a980b35b4128fc36344218a6682",
	);
	expect(() => verifyRuntimeArchive(Buffer.from("unverified runtime"))).toThrow(
		"checksum mismatch",
	);
});
