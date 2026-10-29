import { afterEach, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CommandTraceSink, sha256 } from "./command_trace";
const paths: string[] = [];
function root() { const path = mkdtempSync(join(tmpdir(), "ears-trace-")); paths.push(path); return path; }
afterEach(() => { for (const path of paths.splice(0)) rmSync(path, { recursive: true, force: true }); });
function records(sink: CommandTraceSink) { return readFileSync(sink.path, "utf8").trim().split("\n").map(line => JSON.parse(line)); }

describe("private command trace interchange", () => {
  test("preserves exact ordered model body bytes and hashes", () => {
    const sink = new CommandTraceSink(join(root(), "trace"));
    const body = '{"state":{"z":1,"a":2},"questions":{"b":{},"a":{}}}';
    sink.write({ kind: "model_result", decision_id: "fixture-1", payload: { request_utf8: body, request_sha256: sha256(body), label_source: "model_prediction" } }, { source: "synthetic" });
    const [record] = records(sink); sink.close();
    expect(record.payload.request_utf8).toBe(body);
    expect(record.payload.request_sha256).toBe(sha256(body));
    expect(record.training_eligible).toBe(false);
    expect(record.schema).toBe("byteowlz.command-trace/v1");
  });
  test("creates owner-only directory and file", () => {
    const dir = join(root(), "trace"); const sink = new CommandTraceSink(dir);
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    expect(statSync(sink.path).mode & 0o777).toBe(0o600); sink.close();
  });
  test("blocked and sensitive contexts leave no input or correlation fields", () => {
    const sink = new CommandTraceSink(join(root(), "trace"));
    const event = { kind: "routing_input", decision_id: "private-id", payload: { text: "private spoken content" } };
    for (const admission of [
      { source: "blocked", reason: "private reason" },
      { source: "owner_private", consented: false, secure_input: false, excluded_app: false },
      { source: "owner_private", consented: true, secure_input: true, excluded_app: false },
      { source: "owner_private", consented: true, secure_input: false, excluded_app: true },
    ] as const) expect(sink.write(event, admission).suppressed).toBe(true);
    sink.close(); const text = readFileSync(sink.path, "utf8");
    expect(text).not.toContain("private spoken content"); expect(text).not.toContain("private-id"); expect(text).not.toContain("private reason");
  });
  test("admits consented nonsensitive owner input but never promotes labels", () => {
    const sink = new CommandTraceSink(join(root(), "trace"));
    expect(sink.write({ kind: "owner_feedback", payload: { intended: "dictation", label_source: "owner_review" } }, { source: "owner_private", consented: true, secure_input: false, excluded_app: false }).suppressed).toBe(false);
    expect(records(sink)[0].training_eligible).toBe(false); sink.close();
  });
  test("suppresses secret fields, serialized JSON secrets and oversized records", () => {
    const sink = new CommandTraceSink(join(root(), "trace"));
    for (const payload of [{ api_key: "synthetic-value" }, { request_utf8: '{"state":{"password":"synthetic-value"}}' }, { text: "Bearer synthetic-token-value" }, { text: "a".repeat((1 << 20) + 1) }]) {
      expect(sink.write({ kind: "model_result", payload }, { source: "synthetic" }).suppressed).toBe(true);
    }
    expect(readFileSync(sink.path, "utf8")).not.toContain("synthetic-value"); sink.close();
  });
  test("refuses git paths and symlink routes into git", () => {
    const dir = root(); const repo = join(dir, "repo"); mkdirSync(repo); mkdirSync(join(repo, ".git"));
    expect(() => new CommandTraceSink(join(repo, "traces"))).toThrow("outside git");
    symlinkSync(repo, join(dir, "link")); expect(() => new CommandTraceSink(join(dir, "link", "traces"))).toThrow("outside git");
  });
  test("refuses shared directories; closes explicitly", () => {
    const shared = join(root(), "shared"); mkdirSync(shared); chmodSync(shared, 0o755);
    expect(() => new CommandTraceSink(shared)).toThrow("0700");
    const sink = new CommandTraceSink(join(root(), "trace")); sink.close();
    expect(() => sink.write({ kind: "model_result", payload: {} }, { source: "synthetic" })).toThrow("closed");
  });
});
