#!/usr/bin/env bun
// Native Linux/Windows candidate packages, not OS signing or desktop acceptance.
import { chmod, cp, mkdir } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { clientBuild, sourcePolicy, packageInputs, files, options, required, run, safeOutput, sha256, writeJson } from "./common";

export function portableProfile(os: string, arch: string) {
  if (arch !== "x64" || !["linux", "win32"].includes(os)) throw new Error("Native Linux/Windows x64 host required");
  return { os: os === "win32" ? "windows" : "linux", key: os === "win32" ? "windows_x64" : "linux_x64", suffix: os === "win32" ? ".exe" : "" };
}
async function main() {
  const args = options(process.argv.slice(2), ["--bin-dir", "--ort-archive", "--source-commit", "--out", "--vc-runtime-dir", "--vc-license"]);
  const profile = portableProfile(process.platform, process.arch);
  const revision = required(args, "--source-commit"), bin = resolve(required(args, "--bin-dir"));
  const spec = await sourcePolicy(revision);
  const archive = resolve(required(args, "--ort-archive")), pin = spec.onnxruntime[profile.key];
  if (await sha256(archive) !== pin.sha256) throw new Error("Runtime archive checksum mismatch");
  const cli = clientBuild(JSON.parse(await run(join(bin, "ears" + profile.suffix), ["--build-info"])), revision, profile.os, "x86_64", false);
  const ui = clientBuild(JSON.parse(await run(join(bin, "ears-ui" + profile.suffix), ["--build-info"])), revision, profile.os, "x86_64", true);
  const helper = clientBuild(JSON.parse(await run(join(bin, "ears-dictation" + profile.suffix), ["--build-info"])), revision, profile.os, "x86_64", false);
  if (cli.version !== ui.version || helper.version !== cli.version) throw new Error("Desktop client version mismatch");
  // App-local CRT avoids depending on the build VM's preinstalled VC runtime.
  // The operator must supply redistributable files and their approved licence.
  const crt = profile.os === "windows" ? resolve(required(args, "--vc-runtime-dir")) : "";
  const crtLicense = profile.os === "windows" ? resolve(required(args, "--vc-license")) : "";
  const output = safeOutput(required(args, "--out")); await mkdir(output);
  const inputs = await packageInputs(revision, output);
  const unpack = join(output, "runtime-source"); await mkdir(unpack);
  await run("tar", ["-xf", archive, "-C", unpack]);
  const runtimeRoot = join(unpack, `onnxruntime-${profile.os === "windows" ? "win" : "linux"}-x64-${spec.onnxruntime.version}`);
  const root = join(output, `eaRS-${cli.version}-${revision.slice(0, 8)}-${profile.os}-x64`);
  const payload = join(root, "payload"), programDir = join(payload, "bin"), licenses = join(payload, "licenses");
  await mkdir(programDir, { recursive: true }); await mkdir(licenses);
  for (const name of ["ears", "ears-dictation", "ears-ui"]) await cp(join(bin, name + profile.suffix), join(programDir, name + profile.suffix));
  const runtime = join(programDir, profile.os === "windows" ? "onnxruntime.dll" : "libonnxruntime.so");
  await cp(join(runtimeRoot, "lib", profile.os === "windows" ? "onnxruntime.dll" : `libonnxruntime.so.${spec.onnxruntime.version}`), runtime);
  if (profile.os === "linux") {
    // No package symlinks: materialize the SONAME alias as identical verified bytes.
    await cp(runtime, join(programDir, "libonnxruntime.so.1"));
    await cp(join(runtimeRoot, "lib/libonnxruntime_providers_shared.so"), join(programDir, "libonnxruntime_providers_shared.so"));
  } else {
    await cp(join(runtimeRoot, "lib/onnxruntime_providers_shared.dll"), join(programDir, "onnxruntime_providers_shared.dll"));
    for (const name of ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]) await cp(join(crt, name), join(programDir, name));
    await cp(crtLicense, join(licenses, "VC-Runtime-LICENSE.txt"));
  }
  await cp(join(inputs, "LICENSE"), join(licenses, "eaRS-LICENSE.txt"));
  for (const name of ["DepartureMono-OFL.txt", "JetBrainsMono-OFL.txt"]) await cp(join(inputs, "apps/ears-ui/assets/fonts", name), join(licenses, name));
  await cp(join(runtimeRoot, "LICENSE"), join(licenses, "ONNX-Runtime-LICENSE.txt"));
  await cp(join(runtimeRoot, "ThirdPartyNotices.txt"), join(licenses, "ONNX-Runtime-ThirdPartyNotices.txt"));
  const inventory = [];
  for (const name of await files(payload)) {
    const portable = name.replaceAll("\\", "/");
    const executable = ["ears", "ears-dictation", "ears-ui"].some(program => portable === `bin/${program}${profile.suffix}`);
    if (profile.os === "linux") await chmod(join(payload, name), executable ? 0o755 : 0o644);
    inventory.push({ path: portable, sha256: await sha256(join(payload, name)), executable });
  }
  await writeJson(join(payload, "manifest.json"), { schema: "ears.desktop-payload/v1", os: profile.os, arch: "x86_64",
    version: cli.version, source_commit: revision, client_vad: true, files: inventory });
  const manifestHash = await sha256(join(payload, "manifest.json"));
  const installer = "ears-desktop-installer" + profile.suffix;
  await cp(join(bin, installer), join(root, installer));
  if (profile.os === "windows") {
    for (const name of ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]) await cp(join(crt, name), join(root, name));
  }
  if (profile.os === "linux") await chmod(join(root, installer), 0o755);
  // Explicit runtime smoke test, no inference session/model/microphone. Not an install action.
  const checked = JSON.parse(await run(join(programDir, "ears" + profile.suffix), ["--runtime-check", runtime]));
  if (checked.runtime_available !== true || checked.models_loaded !== false || checked.microphone_started !== false) throw new Error("Packaged runtime check failed");
  await writeJson(join(root, "artifact.json"), { schema: "ears.desktop-artifact/v1", platform: profile.key,
    source_commit: revision, manifest_sha256: manifestHash, installer_sha256: await sha256(join(root, installer)),
    runtime_archive_sha256: pin.sha256, os_signed: false, vm_install_qualified: false, models_bundled: false });
  await Bun.write(join(root, "INSTALL.txt"), `Candidate only; not a signed/VM-qualified desktop release.\nVerify the archive checksum from a trusted distribution channel before executing the installer.\nDo not take a trust pin from an untrusted copy of this document.\n\nFrom this extracted directory, invoke the native ${installer} with:\n  install --package ./payload --manifest-sha256 ${manifestHash}\nUse --prefix with a trusted absolute per-user path for status, launch, rollback and uninstall.\nLaunch is explicit: launch ears-ui -- [literal args]. Installation never starts the app.\nKeep the installer at a stable absolute location outside managed payload snapshots.\nNo OS shortcut/service registration is implemented. Settings/models/history are retained.\n`);
  const archiveName = basename(root) + (profile.os === "windows" ? ".zip" : ".tar.gz"), packaged = join(output, archiveName);
  if (profile.os === "windows") await run("7z", ["a", "-tzip", packaged, basename(root)], output);
  else await run("tar", ["-czf", packaged, "-C", output, basename(root)]);
  await writeJson(join(output, "artifact.json"), { schema: "ears.desktop-artifact/v1", name: archiveName, sha256: await sha256(packaged),
    manifest_sha256: manifestHash, source_commit: revision, platform: profile.key, os_signed: false, vm_install_qualified: false });
  console.log(JSON.stringify({ ok: true, artifact: packaged, manifest_sha256: manifestHash, candidate_only: true }));
}
if (import.meta.main) main().catch(error => { console.error(String(error)); process.exit(1); });
