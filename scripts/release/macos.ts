#!/usr/bin/env bun
import { cp, mkdir, symlink } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { clientBuild, sourcePolicy, packageInputs, macDependencies, files, options, required, run, safeOutput, sha256, writeJson, xml, versionAtMost } from "./common";

const ID = "dev.byteowlz.ears-ui";
const executables = ["ears-ui", "ears", "ears-dictation"];

async function identity() {
  if (process.env.EARS_SIGN_IDENTITY) return process.env.EARS_SIGN_IDENTITY;
  const result = await run("security", ["find-identity", "-v", "-p", "codesigning"]);
  const identities = [...result.matchAll(/\b([A-F0-9]{40})\s+"Developer ID Application:[^\n]+"/g)];
  if (identities.length !== 1) throw new Error("Set EARS_SIGN_IDENTITY to one Developer ID Application identity");
  return identities[0][1];
}

async function inspectBinary(path: string, revision: string, ui: boolean) {
  return clientBuild(JSON.parse(await run(path, ["--build-info"])), revision, "macos", "aarch64", ui);
}

async function icon(output: string, inputs: string) {
  const source = join(inputs, "logo/ears_logo_mac_white_on_black.png");
  const dimensions=await run("sips",["-g","pixelWidth","-g","pixelHeight",source]);
  const width=dimensions.match(/pixelWidth: (\d+)/)?.[1],height=dimensions.match(/pixelHeight: (\d+)/)?.[1];
  if(!width||!height)throw new Error("Cannot determine supplied icon dimensions");
  let input=source;
  if(width!==height) {
    // Preserve the full composition: pad (never crop/stretch) to the native square canvas.
    input=join(output,"AppIcon-fit.png");const edge=String(Math.max(Number(width),Number(height)));
    await run("sips",["-p",edge,edge,source,"--out",input]);
  }
  const set = join(output, "AppIcon.iconset");
  await mkdir(set);
  for (const size of [16, 32, 128, 256, 512]) {
    for (const scale of [1, 2]) {
      const destination = join(set, `icon_${size}x${size}${scale === 2 ? "@2x" : ""}.png`);
      await run("sips", ["-z", String(size * scale), String(size * scale), input, "--out", destination]);
    }
  }
  const path = join(output, "AppIcon.icns");
  await run("iconutil", ["-c", "icns", "-o", path, set]);
  return path;
}

async function assemble(values: Record<string, string>) {
  const bin = resolve(required(values, "--bin-dir"));
  const revision = required(values, "--source-commit");
  const spec = await sourcePolicy(revision);
  const archive=resolve(required(values,"--ort-archive"));
  const archiveHash=await sha256(archive);
  if(archiveHash!==spec.onnxruntime.macos_arm64.sha256)throw new Error("Runtime archive checksum does not match the qualified release");
  const output = safeOutput(required(values, "--out"));
  if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error("An immutable full source commit is required");
  // Refuse to merge a new release with a stale bundle or overwrite another build.
  await mkdir(output, { recursive: false });
  const inputs = await packageInputs(revision, output);
  const unpack=join(output,"runtime-source");await mkdir(unpack);
  await run("tar",["-xzf",archive,"-C",unpack]);
  const runtimeRoot=join(unpack,`onnxruntime-osx-arm64-${spec.onnxruntime.version}`);
  const runtime=join(runtimeRoot,"lib",`libonnxruntime.${spec.onnxruntime.version}.dylib`);
  const app = join(output, "eaRS.app"), contents = join(app, "Contents");
  const macos = join(contents, "MacOS"), frameworks = join(contents, "Frameworks"), resources = join(contents, "Resources");
  await Promise.all([macos, frameworks, resources].map(path => mkdir(path, { recursive: true })));
  const info = await inspectBinary(join(bin, "ears-ui"), revision, true);
  if (info.source_commit !== revision) throw new Error("Binary source revision does not match package revision");
  const cli = await inspectBinary(join(bin, "ears"), revision, false);
  if (cli.source_commit !== revision) throw new Error("CLI source revision differs from UI");
  const helper = await inspectBinary(join(bin, "ears-dictation"), revision, false);
  if (helper.version !== cli.version || info.version !== cli.version) throw new Error("Desktop client versions differ");
  for (const name of executables) await cp(join(bin, name), join(macos, name));
  if(!(await run("lipo",["-archs",runtime])).split(/\s+/).includes("arm64"))throw new Error("ARM64 runtime required");
  await cp(runtime, join(frameworks, "libonnxruntime.dylib"));
  const dependencies = await run("otool", ["-L", join(frameworks, "libonnxruntime.dylib")]);
  if (dependencies.includes("/opt/homebrew/") || dependencies.includes("/usr/local/")) {
    throw new Error("Runtime has non-portable Homebrew dependencies; use the qualified upstream runtime");
  }
  await run("install_name_tool", ["-id", "@rpath/libonnxruntime.dylib", join(frameworks, "libonnxruntime.dylib")]);
  await cp(await icon(output, inputs), join(resources, "AppIcon.icns"));
  await cp(join(inputs, "LICENSE"), join(resources, "LICENSE.txt"));
  for (const name of ["DepartureMono-OFL.txt", "JetBrainsMono-OFL.txt"]) await cp(join(inputs, "apps/ears-ui/assets/fonts", name), join(resources, name));
  await cp(join(runtimeRoot,"LICENSE"), join(resources, "ONNX-Runtime-LICENSE.txt"));
  await cp(join(runtimeRoot,"ThirdPartyNotices.txt"), join(resources, "ONNX-Runtime-ThirdPartyNotices.txt"));
  const minimum=spec.macos.minimum_version;
  for(const payload of [...executables.map(name=>join(macos,name)),join(frameworks,"libonnxruntime.dylib")]) {
    const load=await run("otool",["-L",payload]);
    macDependencies(load);
    const build=await run("vtool",["-show-build",payload]);
    const target=build.match(/minos\s+(\d+\.\d+)/)?.[1];
    if(!target||!versionAtMost(target,minimum))throw new Error(`Payload requires macOS ${target??"unknown"} beyond declared ${minimum}`);
  }
  const version = String(info.version);
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error("Numeric release version required");
  await Bun.write(join(contents, "Info.plist"), `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>${ID}</string>
<key>CFBundleName</key><string>eaRS</string>
<key>CFBundleDisplayName</key><string>eaRS</string>
<key>CFBundleExecutable</key><string>ears-ui</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>${xml(version)}</string>
<key>CFBundleVersion</key><string>${xml(version)}</string>
<key>CFBundleIconFile</key><string>AppIcon.icns</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>LSMinimumSystemVersion</key><string>${minimum}</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>eaRS uses your microphone for dictation only when you start it or explicitly enable wake detection.</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>\n`);
  await run("plutil", ["-lint", join(contents, "Info.plist")]);
  await writeJson(join(resources, "build.json"), { schema: "ears.desktop-build/v1", source_commit: revision, info, runtime: { name: "onnxruntime", version: spec.onnxruntime.version, upstream_archive_sha256: archiveHash }, models_bundled: false });
  return { app, output, revision, version };
}

async function notarize(path: string, profile: string, log: string) {
  const result = JSON.parse(await run("xcrun", ["notarytool", "submit", path, "--keychain-profile", profile, "--wait", "--output-format", "json"]));
  await writeJson(log, result);
  if (result.status !== "Accepted") throw new Error("Apple did not accept the notarization submission; inspect the private result");
  return result.id;
}

async function main() {
  if (process.platform !== "darwin") throw new Error("Run macOS packaging on macOS");
  const values = options(process.argv.slice(2), ["--bin-dir", "--ort-archive", "--out", "--source-commit", "--notary-profile"]);
  const signer = await identity();
  const profile = required(values, "--notary-profile");
  const built = await assemble(values);
  const entitlements = join(built.output, "entitlements.plist");
  await Bun.write(entitlements, '<?xml version="1.0"?><plist version="1.0"><dict><key>com.apple.security.device.audio-input</key><true/></dict></plist>');
  for (const path of await files(join(built.app, "Contents", "Frameworks"))) {
    await run("codesign", ["--force", "--sign", signer, "--timestamp", "--options", "runtime", join(built.app, "Contents", "Frameworks", path)]);
  }
  for (const name of executables) await run("codesign", ["--force", "--sign", signer, "--timestamp", "--options", "runtime", "--entitlements", entitlements, join(built.app, "Contents", "MacOS", name)]);
  await run("codesign", ["--force", "--sign", signer, "--timestamp", "--options", "runtime", "--entitlements", entitlements, built.app]);
  await run("codesign", ["--verify", "--deep", "--strict", built.app]);
  const zip = join(built.output, "eaRS-notary.zip");
  await run("ditto", ["-c", "-k", "--keepParent", built.app, zip]);
  await notarize(zip, profile, join(built.output, "app-notary.json"));
  await run("xcrun", ["stapler", "staple", built.app]);
  await run("xcrun", ["stapler", "validate", built.app]);
  const stage = join(built.output, "dmg-root");
  await mkdir(stage); await cp(built.app, join(stage, "eaRS.app"), { recursive: true });
  await symlink("/Applications", join(stage, "Applications"));
  const dmg = join(built.output, `eaRS-${built.version}-${built.revision.slice(0, 8)}-macos-arm64.dmg`);
  await run("hdiutil", ["create", "-volname", "eaRS", "-srcfolder", stage, "-format", "UDZO", dmg]);
  await run("codesign", ["--sign", signer, "--timestamp", dmg]);
  await notarize(dmg, profile, join(built.output, "dmg-notary.json"));
  await run("xcrun", ["stapler", "staple", dmg]);
  await run("xcrun", ["stapler", "validate", dmg]);
  await run("spctl", ["--assess", "--type", "execute", "--verbose=2", built.app]);
  const signature = await run("codesign", ["-d", "--verbose=4", built.app], undefined, true);
  const cdhash = signature.match(/^CDHash=([a-f0-9]{40})$/m)?.[1];
  if (!cdhash) throw new Error("Cannot record the notarized application's CDHash trust pin");
  await writeJson(join(built.output, "artifact.json"), { app_cdhash: cdhash, schema: "ears.desktop-artifact/v1", platform: "macos-arm64", source_commit: built.revision, name: basename(dmg), sha256: await sha256(dmg), notarized: true, models_bundled: false });
  console.log(JSON.stringify({ ok: true, artifact: dmg, source_commit: built.revision, notarized: true }));
}
if (import.meta.main) main().catch(error => { console.error(String(error)); process.exit(1); });
