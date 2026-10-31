import {expect, test} from "bun:test";
import {createHash} from "node:crypto";
import {mkdtemp, mkdir, readFile, writeFile, appendFile, readdir, lstat, symlink, link, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";

const script = resolve(import.meta.dir, "install-macos.sh");
const mac = process.platform === "darwin";
const pin = "a".repeat(40);
const sha = (text: string) => createHash("sha256").update(text).digest("hex");
const json = (value: unknown) => JSON.stringify(value) + "\n";
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "ears-mac-installer-test-"));
  // Match the script's physical paths (macOS /var -> /private/var).
  const {realpath} = await import("node:fs/promises");
  const physical = await realpath(root);
  const prefix = join(physical, "Applications"), state = join(physical, "state");
  const env = {HOME: join(physical, "home"), XDG_STATE_HOME: join(physical, "xdg"), PATH: "/usr/bin:/bin:/usr/sbin:/sbin"};
  const args = ["--prefix", prefix, "--state-dir", state];
  const receipt = (current = pin, previous = "", previousHash = "") => ({
    schema: "ears.install/v2", path: join(prefix, "eaRS.app"), current_cdhash: current,
    previous_path: previous, previous_cdhash: previousHash,
  });
  const run = async (flags: string[], driver?: string[]) => {
    const child = Bun.spawn(driver ?? ["/bin/bash", script, ...flags], {cwd: physical, env, stdout: "pipe", stderr: "pipe"});
    const [code, out, error] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
    return {code, out, error};
  };
  return {root: physical, prefix, state, args, env, receipt, run, close: () => rm(physical, {recursive: true, force: true})};
}
function rejected(result: {code: number; error: string}, match?: RegExp) {
  expect(result.code).not.toBe(0);
  if (match) expect(result.error).toMatch(match);
}

// These parser tests also run off macOS; help and syntax admission precede OS checks.
test("mac installer help and duplicate/conflicting/malformed flags write nothing", async () => {
  const f = await fixture();
  try {
    expect((await f.run(["--help"])).code).toBe(0);
    for (const flags of [[], ["--app"], ["--app", "x"], ["--app", "x", "--app-cdhash", "bad"],
      ["--app", "x", "--app-cdhash", "a".repeat(39)], ["--uninstall", "--rollback"],
      ["--recover", "--recover"], ["--prefix", "a", "--prefix", "a"],
      ["--app", "x", "--app", "y", "--app-cdhash", pin],
      ["--app", "x", "--app-cdhash", pin, "--app-cdhash", pin],
      ["--uninstall", "--app", "x"], ["--rollback", "--app-cdhash", pin],
      ["--help", "--garbage"], ["--state-dir", "--uninstall"], ["--state-dir", ""]]) {
      expect((await f.run(flags)).code).toBe(2);
      expect(await readdir(f.root)).toEqual([]);
    }
  } finally { await f.close(); }
});

test.skipIf(!mac)("mac installer rejects unsafe paths before writes", async () => {
  const f = await fixture();
  try {
    for (const path of ["/", "relative", "/tmp/../x", "/tmp//x", "/tmp/./x", "/tmp/bad\nname", "/tmp/bad\tname", "/tmp/quote\"name", "/tmp/back\\slash"]) {
      expect((await f.run(["--recover", "--prefix", path, "--state-dir", f.state])).code).toBe(2);
      expect(await readdir(f.root)).toEqual([]);
    }
    expect((await f.run(["--recover", "--prefix", f.prefix, "--state-dir", join(f.prefix, "state")])).code).toBe(2);
    expect(await readdir(f.root)).toEqual([]);
  } finally { await f.close(); }
});

test.skipIf(!mac)("mac installer refuses unknown/legacy receipts even with destination absent", async () => {
  const f = await fixture();
  try {
    await mkdir(f.state);
    const path = join(f.state, "receipt.json");
    const valid = f.receipt();
    const receipts = [json({schema: "ears.install/v1", path: valid.path, rollback: "", version: "old"}),
      json({...valid, extra: true}), json({...valid, path: join(f.root, "other.app")}),
      json({...valid, current_cdhash: 123}), json({...valid, previous_path: "/unmanaged/old.app", previous_cdhash: pin}),
      json(valid).replace('"current_cdhash":', '"current_cdhash":"' + pin + '","current_cdhash":'),
      JSON.stringify(valid, null, 2) + "\n", " ".repeat(16385)];
    for (const body of receipts) {
      await writeFile(path, body, {mode: 0o600});
      const before = await lstat(path);
      rejected(await f.run(["--uninstall", ...f.args]), /receipt|Metadata|metadata/);
      expect(await readFile(path, "utf8")).toBe(body);
      expect((await lstat(path)).ino).toBe(before.ino);
      expect(await readdir(f.root)).toEqual(["state"]);
    }
    // Install must not overwrite an unknown receipt simply because dest is absent.
    const source = join(f.root, "untrusted.app");
    await mkdir(source);
    await writeFile(path, receipts[0]);
    rejected(await f.run(["--app", source, "--app-cdhash", pin, ...f.args]), /receipt/);
    expect(await readFile(path, "utf8")).toBe(receipts[0]);
    expect(await readdir(f.state)).toEqual(["receipt.json"]);
    await rm(path);
    await mkdir(join(f.state, "lock"));
    await writeFile(join(f.state, "lock/unknown"), "retain");
    rejected(await f.run(["--recover", ...f.args]), /Legacy\/unknown lock/);
    expect(await readFile(join(f.state, "lock/unknown"), "utf8")).toBe("retain");
  } finally { await f.close(); }
});

test.skipIf(!mac)("mac installer refuses receipt symlinks, directories and hardlinks without deletion", async () => {
  for (const type of ["symlink", "directory", "hardlink"]) {
    const f = await fixture();
    try {
      await mkdir(f.state);
      const target = join(f.root, "sentinel"), path = join(f.state, "receipt.json");
      await writeFile(target, json(f.receipt()), {mode: 0o600});
      if (type === "symlink") await symlink(target, path);
      if (type === "hardlink") await link(target, path);
      if (type === "directory") { await mkdir(path); await writeFile(join(path, "sentinel"), "retain"); }
      const before = await lstat(path);
      rejected(await f.run(["--uninstall", ...f.args]), /regular/);
      expect((await lstat(path)).ino).toBe(before.ino);
      expect(await readFile(target, "utf8")).toBe(json(f.receipt()));
      if (type === "directory") expect(await readFile(join(path, "sentinel"), "utf8")).toBe("retain");
    } finally { await f.close(); }
  }
});

test.skipIf(!mac)("mac installer never adopts unmanaged destination or malformed journal", async () => {
  const f = await fixture();
  try {
    await mkdir(join(f.prefix, "eaRS.app"), {recursive: true});
    await writeFile(join(f.prefix, "eaRS.app/sentinel"), "keep");
    rejected(await f.run(["--uninstall", ...f.args]), /Unmanaged destination/);
    expect(await readFile(join(f.prefix, "eaRS.app/sentinel"), "utf8")).toBe("keep");
    const journal = join(f.state, "journal.json");
    await writeFile(journal, '{"schema":"unknown"}\n', {mode: 0o600});
    const before = await readFile(journal, "utf8");
    rejected(await f.run(["--recover", ...f.args]), /journal|Recovery/);
    expect(await readFile(journal, "utf8")).toBe(before);
    expect(await readFile(join(f.prefix, "eaRS.app/sentinel"), "utf8")).toBe("keep");
  } finally { await f.close(); }
});

test.skipIf(!mac)("mac installer cancels a prepared, uncommitted WAL without deleting staged/unknown content", async () => {
  const f = await fixture();
  try {
    const tx = join(f.prefix, ".ears-install.ABC123"), next = join(f.state, ".ears-receipt.ABC123");
    await mkdir(join(tx, "incoming.app"), {recursive: true});
    await mkdir(f.state);
    await writeFile(join(tx, "incoming.app/synthetic"), "not a signed app; never adopted/executed");
    await writeFile(join(tx, "unknown"), "retain");
    await writeFile(next, json(f.receipt()), {mode: 0o600});
    const journal = {schema: "ears.install-journal/v1", path: join(f.prefix, "eaRS.app"), transaction: tx,
      old_cdhash: "", new_cdhash: pin, old_receipt_sha256: "", next_receipt: next,
      next_receipt_sha256: sha(json(f.receipt())), previous_path: "", previous_cdhash: ""};
    await writeFile(join(f.state, "journal.json"), json(journal), {mode: 0o600});
    const foreignReceipt = join(f.state, "receipt.json");
    await writeFile(foreignReceipt, '{"schema":"foreign"}\n', {mode: 0o600});
    rejected(await f.run(["--recover", ...f.args]), /receipt|Receipt/);
    expect(await readFile(foreignReceipt, "utf8")).toBe('{"schema":"foreign"}\n');
    expect(await readFile(join(f.state, "journal.json"), "utf8")).toBe(json(journal));
    await rm(foreignReceipt); // Remove only our synthetic fixture, never installer cleanup.
    await writeFile(next, "unrecognized replacement");
    rejected(await f.run(["--recover", ...f.args]));
    expect(await readFile(next, "utf8")).toBe("unrecognized replacement");
    expect(await readFile(join(f.state, "journal.json"), "utf8")).toBe(json(journal));
    await writeFile(next, json(f.receipt()));
    const result = await f.run(["--recover", ...f.args]);
    expect(result.code).toBe(0);
    expect(await readFile(join(tx, "unknown"), "utf8")).toBe("retain");
    expect(await readFile(join(tx, "incoming.app/synthetic"), "utf8")).toContain("never adopted");
    expect(await readFile(next, "utf8")).toBe(json(f.receipt()));
    expect(await readdir(f.state)).not.toContain("receipt.json");
    expect(await readdir(f.state)).not.toContain("journal.json");
  } finally { await f.close(); }
});

test.skipIf(!mac)("mac installer persistent lock serializes callers without unlinking its inode", async () => {
  const f = await fixture();
  try {
    expect((await f.run(["--recover", ...f.args])).code).toBe(0);
    for (const lock of [join(f.state, "installer.lock"), join(f.prefix, ".ears-installer.lock")]) {
      const inode = (await lstat(lock)).ino;
      const holder = Bun.spawn(["/bin/bash", "-c", 'exec 9<"$1"; /usr/bin/lockf -s -t 0 9 || exit; printf ready; read -r done', "holder", lock],
        {env: f.env, stdin: "pipe", stdout: "pipe", stderr: "pipe"});
      try {
        const reader = holder.stdout.getReader();
        const ready = await reader.read();
        expect(new TextDecoder().decode(ready.value)).toBe("ready");
        reader.releaseLock();
        expect((await f.run(["--recover", ...f.args])).code).toBe(3);
        if (lock.includes(".ears-installer.lock")) expect((await f.run(["--recover", "--prefix", f.prefix, "--state-dir", join(f.root, "other-state")])).code).toBe(3);
        expect((await lstat(lock)).ino).toBe(inode);
      } finally { holder.stdin.write("finish\n"); holder.stdin.end(); await holder.exited; }
      expect((await f.run(["--recover", ...f.args])).code).toBe(0);
      expect((await lstat(lock)).ino).toBe(inode);
    }
  } finally { await f.close(); }
});

test.skipIf(!mac)("unsigned same-ID bundle cannot authorize uninstall with a path-only receipt", async () => {
  const f = await fixture();
  try {
    const app = join(f.prefix, "eaRS.app");
    await mkdir(join(app, "Contents/MacOS"), {recursive: true});
    await mkdir(f.state);
    await writeFile(join(app, "Contents/Info.plist"), '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>dev.byteowlz.ears-ui</string><key>CFBundlePackageType</key><string>APPL</string><key>CFBundleExecutable</key><string>ears-ui</string></dict></plist>');
    await writeFile(join(app, "Contents/MacOS/ears-ui"), "synthetic bytes; not executable");
    await writeFile(join(app, "sentinel"), "retain");
    await writeFile(join(f.state, "receipt.json"), json(f.receipt()), {mode: 0o600});
    const receipt = await readFile(join(f.state, "receipt.json"), "utf8");
    rejected(await f.run(["--uninstall", ...f.args]));
    expect(await readFile(join(app, "sentinel"), "utf8")).toBe("retain");
    expect(await readFile(join(f.state, "receipt.json"), "utf8")).toBe(receipt);
  } finally { await f.close(); }
});

const appA = process.env.MACOS_INSTALLER_TEST_APP;
const hashA = process.env.MACOS_INSTALLER_TEST_APP_CDHASH;
const appB = process.env.MACOS_INSTALLER_TEST_APP_B;
const hashB = process.env.MACOS_INSTALLER_TEST_APP_B_CDHASH;
const nativeA = mac && !!appA && /^[0-9a-fA-F]{40}$/.test(hashA ?? "");
const nativeB = nativeA && !!appB && /^[0-9a-fA-F]{40}$/.test(hashB ?? "") && hashA!.toLowerCase() !== hashB!.toLowerCase();
if (!nativeA) console.error("UNTESTED: signed app/signature lifecycle; provide MACOS_INSTALLER_TEST_APP and externally trusted MACOS_INSTALLER_TEST_APP_CDHASH");
if (!nativeB) console.error("UNTESTED: distinct signed update, foreign same-ID rejection, SIGTERM/SIGKILL/ordinary-error rename-boundary recovery; provide distinct fixture B and trusted pin");
const install = (f: Awaited<ReturnType<typeof fixture>>, app: string, cdhash: string) => ["--app", app, "--app-cdhash", cdhash, ...f.args];
async function displayHash(path: string) {
  const child = Bun.spawn(["/usr/bin/codesign", "-d", "--verbose=4", path], {stdout: "pipe", stderr: "pipe"});
  const error = await new Response(child.stderr).text();
  expect(await child.exited).toBe(0);
  return /^CDHash=(.+)$/m.exec(error)?.[1];
}
async function ditto(from: string, to: string) {
  const child = Bun.spawn(["/usr/bin/ditto", from, to], {stdout: "ignore", stderr: "pipe"});
  const error = await new Response(child.stderr).text();
  expect(await child.exited, error).toBe(0);
}

test.skipIf(!nativeA)("signed fixture lifecycle pins, idempotency, uninstall retention and rollback", async () => {
  const f = await fixture();
  try {
    const sentinels = ["home/.config/ears/config.toml", "home/.local/share/ears/models/model.bin", "home/.local/state/ears/history.jsonl"];
    const {dirname} = await import("node:path");
    for (const path of sentinels) { await mkdir(dirname(join(f.root, path)), {recursive: true}); await writeFile(join(f.root, path), "user data stays"); }
    const wrong = hashA!.toLowerCase() === "0".repeat(40) ? "1".repeat(40) : "0".repeat(40);
    rejected(await f.run(install(f, appA!, wrong)), /CDHash/);
    expect((await f.run(install(f, appA!, hashA!))).code).toBe(0);
    const receipt = await readFile(join(f.state, "receipt.json"), "utf8");
    expect((await f.run(install(f, appA!, hashA!))).code).toBe(0);
    expect(await readFile(join(f.state, "receipt.json"), "utf8")).toBe(receipt);
    expect((await f.run(["--uninstall", ...f.args])).code).toBe(0);
    const removed = JSON.parse(await readFile(join(f.state, "receipt.json"), "utf8"));
    expect(removed.current_cdhash).toBe("");
    expect(await displayHash(removed.previous_path)).toBe(hashA!.toLowerCase());
    expect((await f.run(["--rollback", ...f.args])).code).toBe(0);
    expect(await displayHash(join(f.prefix, "eaRS.app"))).toBe(hashA!.toLowerCase());
    for (const path of sentinels) expect(await readFile(join(f.root, path), "utf8")).toBe("user data stays");
  } finally { await f.close(); }
}, 180000);

test.skipIf(!nativeB)("same-ID signed substitution cannot satisfy receipt current/previous CDHash", async () => {
  const {rename} = await import("node:fs/promises");
  const f = await fixture();
  try {
    expect((await f.run(install(f, appA!, hashA!))).code).toBe(0);
    expect((await f.run(install(f, appB!, hashB!))).code).toBe(0);
    const receipt = await readFile(join(f.state, "receipt.json"), "utf8");
    const value = JSON.parse(receipt);
    await rename(join(f.prefix, "eaRS.app"), join(f.root, "saved-current.app"));
    await ditto(appA!, join(f.prefix, "eaRS.app"));
    rejected(await f.run(["--uninstall", ...f.args]), /CDHash/);
    rejected(await f.run(["--rollback", ...f.args]), /CDHash/);
    expect(await readFile(join(f.state, "receipt.json"), "utf8")).toBe(receipt);
    await rename(join(f.prefix, "eaRS.app"), join(f.root, "foreign-current.app"));
    await rename(join(f.root, "saved-current.app"), join(f.prefix, "eaRS.app"));
    await rename(value.previous_path, join(f.root, "saved-previous.app"));
    await ditto(appB!, value.previous_path);
    rejected(await f.run(["--rollback", ...f.args]), /CDHash/);
    rejected(await f.run(["--uninstall", ...f.args]), /CDHash/);
    expect(await readFile(join(f.state, "receipt.json"), "utf8")).toBe(receipt);
    expect(await displayHash(join(f.root, "saved-previous.app"))).toBe(hashA!.toLowerCase());
  } finally { await f.close(); }
}, 180000);

test.skipIf(!nativeB)("corrupt signed update cannot replace the verified managed app", async () => {
  const f = await fixture();
  try {
    expect((await f.run(install(f, appA!, hashA!))).code).toBe(0);
    const receipt = await readFile(join(f.state, "receipt.json"), "utf8");
    const corrupted = join(f.root, "corrupt.app");
    await ditto(appB!, corrupted);
    await appendFile(join(corrupted, "Contents/Info.plist"), "\n<!-- synthetic signature corruption -->\n");
    rejected(await f.run(install(f, corrupted, hashB!)), /Info\.plist|sealed resource|signature|code.*sign/i);
    expect(await readFile(join(f.state, "receipt.json"), "utf8")).toBe(receipt);
    expect(await displayHash(join(f.prefix, "eaRS.app"))).toBe(hashA!.toLowerCase());
  } finally { await f.close(); }
}, 180000);

// Test-driver DEBUG traps only. Production has no fault flag, fake verifier, or PATH override.
const boundaryDriver = `
set -T
cut_phase=$1; cut_mode=$2; shift 2
cut() {
  local command=$1 match=false
  case "$cut_phase" in
    before-old) [[ "$command" == 'move_absent "$destination" "$transaction/old.app"' ]] && match=true ;;
    after-old) [[ "$command" == 'move_absent "$transaction/incoming.app" "$destination"' && -d "\${transaction:-/missing}/old.app" ]] && match=true ;;
    after-new) [[ "$command" == 'expect_receipt "$old_receipt"'  && -d "\${transaction:-/missing}/old.app" && -d "\${destination:-/missing}" ]] && match=true ;;
    after-receipt) [[ "$command" == recover_transaction && -f "\${journal:-/missing}" && -n "\${next_receipt:-}" && ! -e "$next_receipt" ]] && match=true ;;
  esac
  if $match; then trap - DEBUG; if [[ "$cut_mode" == error ]]; then exit 77; else /bin/kill -"$cut_mode" "$$"; fi; fi
  return 0
}
trap 'cut "$BASH_COMMAND"' DEBUG
source "$@"
`;

test.skipIf(!mac)("fault-driver sanity: DEBUG cuts match all four boundaries without a verifier override", async () => {
  const f = await fixture();
  try {
    const production = await readFile(script, "utf8");
    for (const command of ['move_absent "$destination" "$transaction/old.app"', 'move_absent "$transaction/incoming.app" "$destination"', 'expect_receipt "$old_receipt"', '\nrecover_transaction\n']) expect(production).toContain(command);
    const miniature = join(f.root, "driver-unit.sh");
    await writeFile(miniature, `set -eu
transaction="$1/transaction"; destination="$1/destination"; journal="$1/journal"; next_receipt="$1/missing-next"; old_receipt=fixture
/bin/mkdir -p "$transaction/old.app" "$destination"
printf journal > "$journal"
move_absent() { :; }; expect_receipt() { :; }; recover_transaction() { :; }
move_absent "$destination" "$transaction/old.app"
move_absent "$transaction/incoming.app" "$destination"
expect_receipt "$old_receipt"
recover_transaction
`);
    for (const phase of ["before-old", "after-old", "after-new", "after-receipt"]) {
      const result = await f.run([], ["/bin/bash", "-c", boundaryDriver, "driver-unit", phase, "error", miniature, f.root]);
      expect(result.code, `${phase}: ${result.error}`).toBe(77);
    }
  } finally { await f.close(); }
});

test.skipIf(!nativeB)("rename cuts recover old selection before commit and preserve completed commit", async () => {
  for (const [phase, mode] of [["before-old", "KILL"], ["after-old", "KILL"], ["after-new", "KILL"], ["after-receipt", "KILL"], ["after-old", "error"], ["after-new", "TERM"]]) {
    const f = await fixture();
    try {
      expect((await f.run(install(f, appA!, hashA!))).code).toBe(0);
      const before = await readFile(join(f.state, "receipt.json"), "utf8");
      const result = await f.run([], ["/bin/bash", "-c", boundaryDriver, "cut-test", phase, mode, script, ...install(f, appB!, hashB!)]);
      rejected(result);
      if (mode === "KILL") expect(await readdir(f.state)).toContain("journal.json");
      expect((await f.run(["--recover", ...f.args])).code).toBe(0);
      const after = await readFile(join(f.state, "receipt.json"), "utf8");
      if (phase === "after-receipt") {
        const committed = JSON.parse(after);
        expect(committed.current_cdhash).toBe(hashB!.toLowerCase());
        expect(await displayHash(committed.previous_path)).toBe(hashA!.toLowerCase());
      } else expect(after).toBe(before);
      expect(await displayHash(join(f.prefix, "eaRS.app"))).toBe(phase === "after-receipt" ? hashB!.toLowerCase() : hashA!.toLowerCase());
      expect(await readdir(f.state)).not.toContain("journal.json");
    } finally { await f.close(); }
  }
}, 180000);
