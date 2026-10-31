import { expect, test } from "bun:test";
import { buildJobs, sourceRevision } from "./build";
import { clientBuild, macDependencies, options } from "./common";
import { portableProfile } from "./portable";

const revision = "a".repeat(40);
const base = { source_commit: revision, os: "linux", arch: "x86_64", version: "0.6.0", client_vad: true, moonshine: false, parakeet_rs: false, experimental_overlays: false, renderer: "accepted-e034c69f" };
test("release profiles require exact source, platform, client VAD and accepted UI", () => {
  expect(clientBuild(base, revision, "linux", "x86_64", true)).toEqual(base);
  for (const changed of [{ source_commit: "b".repeat(40) }, { os: "windows" }, { arch: "aarch64" }, { client_vad: false }, { moonshine: true }, { parakeet_rs: true }, { version: "dev" }, { experimental_overlays: true }, { renderer: "experimental" }]) {
    expect(() => clientBuild({ ...base, ...changed }, revision, "linux", "x86_64", true)).toThrow();
  }
});
test("bounded builds and strict named options reject accidental release profiles", () => {
  for (const value of ["0", "5", "2;sh", "2.0"]) expect(() => buildJobs(value)).toThrow();
  expect(buildJobs()).toBe("2");
  expect(sourceRevision(revision)).toBe(revision);
  for (const value of ["HEAD", "main", "abc", "A".repeat(40)]) expect(() => sourceRevision(value)).toThrow();
  expect(() => options(["--unrecognized", "value"], ["--out"])).toThrow();
  expect(portableProfile("win32", "x64").suffix).toBe(".exe");
  expect(portableProfile("linux", "x64").os).toBe("linux");
  for (const profile of [["darwin", "arm64"], ["linux", "arm64"], ["win32", "ia32"]]) expect(() => portableProfile(...profile as [string, string])).toThrow();
});
test("Mach-O dependencies reject build-tree, Homebrew and unbundled rpath libraries", () => {
  const wrapped = (value: string) => `binary:\n\t${value} (compatibility version 1.0.0, current version 1.0.0)\n`;
  for (const value of ["/usr/lib/libSystem.B.dylib", "/System/Library/Frameworks/AppKit.framework/AppKit", "@rpath/libonnxruntime.dylib"]) expect(macDependencies(wrapped(value))).toEqual([value]);
  for (const value of ["/opt/homebrew/lib/private.dylib", "/usr/local/lib/private.dylib", "/tmp/build/private.dylib", "@rpath/unknown.dylib", "/Users/example/lib/private.dylib"]) expect(() => macDependencies(wrapped(value))).toThrow();
  expect(() => macDependencies("unexpected")).toThrow();
});
