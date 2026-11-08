import { expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");

test("public snapshots retain build-required CPack documentation", () => {
  const source = "vendor/sentencepiece-sys/source";
  const cmake = readFileSync(resolve(root, source, "CMakeLists.txt"), "utf8");
  const configPath = resolve(root, "publish.toml");
  // The policy stays private; the generated mirror must still carry the inputs.
  const policy = existsSync(configPath)
    ? Bun.TOML.parse(readFileSync(configPath, "utf8"))
    : undefined;
  const inputs = [...cmake.matchAll(/set\(CPACK_RESOURCE_FILE_\w+\s+"\$\{PROJECT_SOURCE_DIR\}\/([^"\n]+)"\)/g)];
  expect(inputs.length).toBeGreaterThan(0);
  for (const [, file] of inputs) {
    const path = `${source}/${file}`;
    expect(existsSync(resolve(root, path))).toBe(true);
    if (policy && file.endsWith(".md")) {
      expect(policy.docs).toContain(path);
    }
  }
});
