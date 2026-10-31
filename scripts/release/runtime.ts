#!/usr/bin/env bun
import { mkdir, mkdtemp, open, rename, rm } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { options, required, sha256 } from "./common";

async function main() {
  const args = options(process.argv.slice(2), ["--platform", "--out"]);
  const platform = required(args, "--platform");
  if (!["macos_arm64", "linux_x64", "windows_x64"].includes(platform)) throw new Error("Unknown runtime profile");
  const spec = Bun.TOML.parse(await Bun.file("packaging/desktop.toml").text()) as any;
  const pin = spec.onnxruntime[platform];
  const root = resolve(required(args, "--out"));
  await mkdir(root, { recursive: true });
  const destination = join(root, basename(pin.archive));
  if (await Bun.file(destination).exists()) {
    if (await sha256(destination) !== pin.sha256) throw new Error("Existing runtime cache checksum mismatch; never overwrite it silently");
  } else {
    const transaction = await mkdtemp(join(root, ".ort-download-"));
    const temporary = join(transaction, "archive");
    try {
      const response = await fetch(`https://github.com/microsoft/onnxruntime/releases/download/v${spec.onnxruntime.version}/${pin.archive}`);
      if (!response.ok) throw new Error(`Runtime download failed: ${response.status}`);
      const size = Number(response.headers.get("content-length"));
      if (!Number.isFinite(size) || size <= 0 || size > 256 * 1024 * 1024) throw new Error("Unexpected runtime archive size");
      if (!response.body) throw new Error("Missing runtime response body");
      const file = await open(temporary, "wx", 0o600);
      let received = 0;
      try {
        for await (const chunk of response.body) {
          received += chunk.byteLength;
          if (received > size || received > 256 * 1024 * 1024) throw new Error("Runtime response exceeded size bound");
          await file.writeFile(chunk);
        }
      } finally { await file.close(); }
      if (received !== size) throw new Error("Incomplete runtime download");
      if (await sha256(temporary) !== pin.sha256) throw new Error("Runtime checksum mismatch");
      await rename(temporary, destination);
    } finally { await rm(transaction, { recursive: true, force: true }); }
  }
  console.log(JSON.stringify({ ok: true, path: destination, sha256: pin.sha256, version: spec.onnxruntime.version }));
}
if (import.meta.main) main().catch(error => { console.error(String(error)); process.exit(1); });
