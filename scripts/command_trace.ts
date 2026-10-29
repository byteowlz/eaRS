// Private, synchronous trace sink for offline workers, not the live typing loop.
import { appendFileSync, closeSync, constants, existsSync, fsyncSync, mkdirSync, openSync, realpathSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { createHash, randomUUID } from "node:crypto";

export type TraceAdmission =
  | { source: "synthetic" }
  | { source: "owner_private"; consented: boolean; secure_input: boolean; excluded_app: boolean }
  | { source: "blocked"; reason: string };
export interface TraceEvent {
  kind: string;
  decision_id?: string;
  utterance_id?: string;
  payload: Record<string, unknown>;
}
const MAX_BYTES = 1 << 20;
const SECRET_KEYS = new Set(["authorization", "apikey", "accesstoken", "refreshtoken", "token", "password", "secret", "clientsecret", "cookie", "setcookie", "headers"]);
export const sha256 = (body: string) => createHash("sha256").update(body, "utf8").digest("hex");

function containsSecret(value: unknown, depth = 0, seen = new Set<object>()): boolean {
  if (depth > 20) return true;
  if (typeof value === "string") {
    if (/\bBearer\s+\S{8,}|\beavs[-_]\S{12,}|\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}|\bAKIA[A-Z0-9]{16}\b/i.test(value)) return true;
    if (/^\s*[\[{]/.test(value)) {
      let parsed: unknown;
      try { parsed = JSON.parse(value); } catch (error) {
        if (error instanceof SyntaxError) return false; // Ordinary text, not serialized JSON.
        throw error;
      }
      return containsSecret(parsed, depth + 1, seen);
    }
    return false;
  }
  if (!value || typeof value !== "object") return false;
  if (seen.has(value)) return true;
  seen.add(value);
  const secret = Object.entries(value).some(([key, item]) => SECRET_KEYS.has(key.toLowerCase().replace(/[_-]/g, "")) || containsSecret(item, depth + 1, seen));
  seen.delete(value);
  return secret;
}

function privateDirectory(directory: string): string {
  const requested = resolve(directory);
  let existing = requested;
  while (!existsSync(existing)) existing = dirname(existing);
  const canonical = join(realpathSync(existing), relative(existing, requested));
  for (let path = canonical; ; path = dirname(path)) {
    if (existsSync(join(path, ".git"))) throw new Error("Trace directory must be outside git repositories");
    if (dirname(path) === path) break;
  }
  mkdirSync(canonical, { recursive: true, mode: 0o700 });
  if (statSync(canonical).mode & 0o077) throw new Error("Trace directory must be owner-only (0700)");
  return canonical;
}

export class CommandTraceSink {
  readonly path: string;
  private fd: number;
  constructor(directory: string) {
    const root = privateDirectory(directory);
    this.path = join(root, `${new Date().toISOString().slice(0, 10)}-${process.pid}-${randomUUID()}.jsonl`);
    this.fd = openSync(this.path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_APPEND | (constants.O_NOFOLLOW ?? 0), 0o600);
  }
  write(event: TraceEvent, admission: TraceAdmission) {
    if (this.fd < 0) throw new Error("Trace sink is closed");
    const permitted = admission.source === "synthetic" || (admission.source === "owner_private" && admission.consented === true && admission.secure_input === false && admission.excluded_app === false);
    let reason = permitted ? null : "privacy_not_admitted";
    if (permitted && containsSecret(event)) reason = "possible_secret_or_unbounded_payload";
    if (!/^[a-z][a-z0-9_.-]{0,63}$/.test(event.kind) || [event.decision_id, event.utterance_id].some(id => id !== undefined && id.length > 256)) reason = "invalid_metadata";
    let data: string | undefined;
    if (!reason) {
      data = JSON.stringify(event.payload);
      if (Buffer.byteLength(data) > MAX_BYTES) reason = "record_too_large";
    }
    const record = {
      schema: "byteowlz.command-trace/v1", record_id: randomUUID(), at: new Date().toISOString(),
      kind: reason ? "suppressed" : event.kind,
      ...(!reason && event.decision_id ? { decision_id: event.decision_id } : {}),
      ...(!reason && event.utterance_id ? { utterance_id: event.utterance_id } : {}),
      privacy: { source: reason ? "suppressed" : admission.source, reason },
      training_eligible: false,
      ...(!reason ? { payload: JSON.parse(data!) } : {}),
    };
    appendFileSync(this.fd, JSON.stringify(record) + "\n", "utf8");
    fsyncSync(this.fd);
    return { record_id: record.record_id, persisted: true, suppressed: reason !== null };
  }
  close() { if (this.fd >= 0) { closeSync(this.fd); this.fd = -1; } }
}
