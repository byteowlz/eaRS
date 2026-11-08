import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  MOONSHINE_RUNTIME,
  stageMoonshineRuntime,
  verifyRuntimeArchive,
} from "../scripts/stage-runtime.mjs";

test("default runtime is the vendored final-audio fix, not stock v0.1.5", () => {
  expect(MOONSHINE_RUNTIME.version).toBe("0.1.5-ears.1");
  expect(MOONSHINE_RUNTIME.coreRevision).toBe("5a1358d3253b81ef24ae2a23cb25fe47bb16b783");
  expect(MOONSHINE_RUNTIME.url).toBeUndefined();
  expect(() => verifyRuntimeArchive(Buffer.from("unverified runtime"))).toThrow("checksum mismatch");
});

test("ordinary staging works offline and records the patched runtime provenance", async () => {
  const temporary = mkdtempSync(path.join(os.tmpdir(), "ears-vendored-runtime-test-"));
  const fetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error("Vendored runtime attempted a download"); };
  try {
    await stageMoonshineRuntime(temporary);
    const manifest = JSON.parse(readFileSync(path.join(temporary, "runtime-manifest.json")));
    expect(manifest.version).toBe("0.1.5-ears.1");
    expect(manifest.wasmSha256).toBe(MOONSHINE_RUNTIME.wasmSha256);
  } finally {
    globalThis.fetch = fetch;
    rmSync(temporary, { recursive: true, force: true });
  }
});
