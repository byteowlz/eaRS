import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  cpSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Runtime code only, not ASR weights. No stock/CDN fallback.
const vendor = fileURLToPath(new URL("./vendor/moonshine/", import.meta.url));
export const MOONSHINE_RUNTIME = Object.freeze(
  JSON.parse(readFileSync(path.join(vendor, "manifest.json"), "utf8")),
);
export function verifyRuntimeArchive(bytes) {
  if (createHash("sha256").update(bytes).digest("hex") !== MOONSHINE_RUNTIME.sha256)
    throw new Error("Moonshine runtime checksum mismatch; refusing unverified assets");
}

/** Stage the self-contained pinned repair offline, preserving the worker tree. */
export async function stageMoonshineRuntime(destination) {
  const archive = path.join(vendor, "runtime.tar.gz");
  verifyRuntimeArchive(readFileSync(archive));
  const temporary = mkdtempSync(path.join(os.tmpdir(), "ears-moonshine-"));
  try {
    execFileSync("tar", ["-xzf", archive, "-C", temporary]);
    const source = path.join(temporary, "dist");
    const wasmDigest = createHash("sha256")
      .update(readFileSync(path.join(source, "moonshine.wasm"))).digest("hex");
    if (wasmDigest !== MOONSHINE_RUNTIME.wasmSha256)
      throw new Error("Moonshine WASM checksum mismatch");
    mkdirSync(destination, { recursive: true });
    for (const name of readdirSync(source))
      if (/\.(js|mjs|wasm)$/.test(name))
        cpSync(path.join(source, name), path.join(destination, name));
    cpSync(path.join(temporary, "README.md"), path.join(destination, "MOONSHINE-README.md"));
    cpSync(fileURLToPath(new URL("./MOONSHINE-LICENSE.txt", import.meta.url)),
      path.join(destination, "MOONSHINE-LICENSE.txt"));
    cpSync(fileURLToPath(new URL("./licenses/", import.meta.url)),
      path.join(destination, "licenses"), { recursive: true });
    writeFileSync(path.join(destination, "runtime-manifest.json"),
      JSON.stringify(MOONSHINE_RUNTIME, null, 2) + "\n");
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}
