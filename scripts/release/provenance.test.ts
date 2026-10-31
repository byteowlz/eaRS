import { expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";

test("policy, artwork and licences come from the binary commit, never dirty resource files", async () => {
  const root = await mkdtemp(join(tmpdir(), "ears-source-inputs-"));
  const repo = join(root, "repo"), output = join(root, "package");
  await mkdir(repo); await mkdir(output);
  const env = { ...process.env, GIT_CONFIG_GLOBAL: join(root, "absent-global-config"), GIT_CONFIG_NOSYSTEM: "1",
    GIT_AUTHOR_NAME: "Fixture", GIT_AUTHOR_EMAIL: "fixture@example.invalid", GIT_COMMITTER_NAME: "Fixture", GIT_COMMITTER_EMAIL: "fixture@example.invalid" };
  async function run(args: string[]) {
    const process = Bun.spawn(args, { cwd: repo, env, stdout: "pipe", stderr: "pipe" });
    const [out, error, code] = await Promise.all([new Response(process.stdout).text(), new Response(process.stderr).text(), process.exited]);
    expect(code, error).toBe(0); return out.trim();
  }
  const resources = ["LICENSE", "logo/ears_logo_mac_white_on_black.png", "apps/ears-ui/assets/fonts/DepartureMono-OFL.txt", "apps/ears-ui/assets/fonts/JetBrainsMono-OFL.txt"];
  try {
    await run(["git", "init", "--initial-branch=main"]);
    for (const name of [...resources, "packaging/desktop.toml"]) await mkdir(dirname(join(repo, name)), { recursive: true });
    for (const name of resources) await writeFile(join(repo, name), "committed public fixture");
    await writeFile(join(repo, "packaging/desktop.toml"), 'schema="ears.desktop-release/v1"\n');
    await run(["git", "add", "."]); await run(["git", "-c", "commit.gpgSign=false", "commit", "-m", "synthetic fixture"]);
    const revision = await run(["git", "rev-parse", "HEAD"]);
    for (const name of resources) await writeFile(join(repo, name), "uncommitted synthetic content MUST NOT package");
    await writeFile(join(repo, "packaging/desktop.toml"), 'schema="dirty-and-wrong"\n');
    const common = resolve(import.meta.dir, "common.ts");
    const script = `import {sourcePolicy,packageInputs} from ${JSON.stringify(common)}; const policy=await sourcePolicy(process.argv[1]); const path=await packageInputs(process.argv[1],process.argv[2]); console.log(JSON.stringify({policy,path}));`;
    const result = JSON.parse(await run([process.execPath, "-e", script, revision, output]));
    expect(result.policy.schema).toBe("ears.desktop-release/v1");
    for (const name of resources) expect(await readFile(join(result.path, name), "utf8")).toBe("committed public fixture");
  } finally { await rm(root, { recursive: true, force: true }); }
});
