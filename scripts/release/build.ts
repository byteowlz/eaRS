#!/usr/bin/env bun
// Build only an immutable git archive, never the caller's dirty/shared checkout.
import { cp, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { clientBuild, options, required, run, safeOutput, sha256, writeJson } from "./common";

export function buildJobs(value = "2") {
  if (!/^[1-4]$/.test(value)) throw new Error("Build jobs must be 1..4");
  return value;
}
export function sourceRevision(value: string) {
  if (!/^[a-f0-9]{40}$/.test(value)) throw new Error("A full immutable source commit is required");
  return value;
}
async function cargo(args: string[], cwd: string, env: Record<string, string | undefined>) {
  const child = Bun.spawn(["cargo", ...args], { cwd, env, stdin: "ignore", stdout: "inherit", stderr: "inherit" });
  if (await child.exited !== 0) throw new Error("Cargo build failed; retained output is not a release");
}
async function main() {
  const args = options(process.argv.slice(2), ["--out", "--source-commit", "--jobs"]);
  if (!((process.platform === "darwin" && process.arch === "arm64") || (["linux", "win32"].includes(process.platform) && process.arch === "x64"))) throw new Error("Unqualified desktop host profile");
  const revision = sourceRevision(required(args, "--source-commit"));
  if (await run("git", ["rev-parse", `${revision}^{commit}`]) !== revision) throw new Error("Source commit is unavailable");
  const jobs = buildJobs(args["--jobs"]);
  const output = safeOutput(required(args, "--out"));
  await mkdir(output); // Existing outputs are never merged or overwritten.
  const source = join(output, "source"), bin = join(output, "bin");
  await mkdir(source); await mkdir(bin);
  const archive = join(output, "source.tar");
  await run("git", ["archive", "--format=tar", "--output", archive, revision]);
  await run("tar", ["-xf", archive, "-C", source]);
  const spec = Bun.TOML.parse(await Bun.file(join(source, "packaging/desktop.toml")).text()) as any;
  if (await sha256(join(source, "apps/ears-ui/src/accepted_parametric_mark.rs")) !== spec.macos.accepted_renderer_sha256) throw new Error("Accepted renderer source hash mismatch");
  if (process.platform !== "darwin" && !(await Bun.file(join(source, "tools/desktop-installer/Cargo.toml")).exists())) throw new Error("This source revision predates the standalone installer");
  const env = { ...process.env, EARS_SOURCE_COMMIT: revision, CARGO_BUILD_JOBS: jobs,
    CMAKE_BUILD_PARALLEL_LEVEL: jobs, RAYON_NUM_THREADS: jobs,
    CARGO_TARGET_DIR: resolve(output, "target"), ...(process.platform === "darwin" ? { MACOSX_DEPLOYMENT_TARGET: "14.0" } : {}) };
  // No server/Metal/CUDA feature, model download, installation or application launch.
  await cargo(["build", "--locked", "--release", "-p", "eaRS", "--bin", "ears", "--bin", "ears-dictation", "--no-default-features", "--features", "client-vad"], source, env);
  await cargo(["build", "--locked", "--release", "-p", "ears-ui", "--no-default-features"], source, env);
  if (process.platform !== "darwin") await cargo(["build", "--locked", "--release", "--manifest-path", join(source, "tools/desktop-installer/Cargo.toml")], source, env);
  const suffix = process.platform === "win32" ? ".exe" : "";
  const hashes: Record<string, string> = {};
  const programs = ["ears", "ears-dictation", "ears-ui", ...(process.platform === "darwin" ? [] : ["ears-desktop-installer"])];
  for (const name of programs) {
    const destination = join(bin, name + suffix);
    await cp(join(env.CARGO_TARGET_DIR, "release", name + suffix), destination);
    hashes[name + suffix] = await sha256(destination);
  }
  const infos: Record<string, unknown> = {};
  for (const name of ["ears", "ears-dictation", "ears-ui"]) {
    const info = clientBuild(JSON.parse(await run(join(bin, name + suffix), ["--build-info"])), revision,
      process.platform === "darwin" ? "macos" : process.platform === "win32" ? "windows" : "linux",
      process.arch === "arm64" ? "aarch64" : "x86_64", name === "ears-ui");
    infos[name] = info;
  }
  await writeJson(join(output, "build-receipt.json"), { schema: "ears.desktop-build-receipt/v1", source_commit: revision,
    source_archive_sha256: await sha256(archive), host: { os: process.platform, arch: process.arch },
    rustc: await run("rustc", ["-vV"]), cargo: await run("cargo", ["--version"]), jobs: Number(jobs),
    binaries: hashes, capabilities: infos, models_loaded: false, microphone_started: false });
  console.log(JSON.stringify({ ok: true, bin, source_commit: revision, receipt: join(output, "build-receipt.json") }));
}
if (import.meta.main) main().catch(error => { console.error(String(error)); process.exit(1); });
