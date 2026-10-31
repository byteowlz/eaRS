import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, readdir } from "node:fs/promises";
import { resolve, join, parse } from "node:path";

export function options(args: string[], allowed?: string[]) {
  const values: Record<string, string> = {};
  for (let index = 0; index < args.length; index += 2) {
    if (!args[index]?.startsWith("--") || !args[index + 1] || args[index + 1].startsWith("--")) {
      throw new Error("Options require --name value pairs");
    }
    if (allowed && !allowed.includes(args[index])) throw new Error(`Unknown option ${args[index]}`);
    if (values[args[index]]) throw new Error(`Duplicate option ${args[index]}`);
    values[args[index]] = args[index + 1];
  }
  return values;
}
export function required(values: Record<string, string>, key: string) {
  if (!values[key]) throw new Error(`Required: ${key}`);
  return values[key];
}
export async function run(program: string, args: string[], cwd?: string, includeStderr = false) {
  const child = Bun.spawn([program, ...args], { cwd, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  if (code !== 0) throw new Error(`${program} failed (${code}): ${stderr.slice(0, 2000)}`);
  return (includeStderr ? `${stdout}\n${stderr}` : stdout).trim();
}
export async function sha256(path: string) {
  const metadata = await lstat(path);
  if (!metadata.isFile() || metadata.isSymbolicLink()) throw new Error("Hash input must be a regular non-symlink file");
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}
export function safeOutput(path: string) {
  const output = resolve(path);
  if (output === parse(output).root || output === process.env.HOME || output === process.cwd()) throw new Error("Unsafe output directory");
  return output;
}
export async function files(root: string, relative = ""): Promise<string[]> {
  const output: string[] = [];
  for (const entry of (await readdir(join(root, relative))).sort()) {
    const name = join(relative, entry), metadata = await lstat(join(root, name));
    if (metadata.isSymbolicLink()) throw new Error(`Unexpected symbolic link in package: ${name}`);
    if (metadata.isDirectory()) output.push(...await files(root, name));
    else if (metadata.isFile()) output.push(name);
    else throw new Error(`Unexpected file type: ${name}`);
  }
  return output;
}
export async function writeJson(path: string, value: unknown) {
  await mkdir(resolve(path, ".."), { recursive: true });
  await Bun.write(path, JSON.stringify(value, null, 2) + "\n");
}
export function versionAtMost(actual:string,maximum:string) {
  if(!/^\d+(\.\d+)*$/.test(actual)||!/^\d+(\.\d+)*$/.test(maximum))throw new Error("Numeric versions required");
  const a=actual.split(".").map(Number),b=maximum.split(".").map(Number);
  for(let i=0;i<Math.max(a.length,b.length);i++) {if((a[i]??0)!==(b[i]??0))return (a[i]??0)<(b[i]??0);}
  return true;
}
export async function sourcePolicy(revision: string) {
  if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error("Immutable source commit required");
  return Bun.TOML.parse(await run("git", ["show", `${revision}:packaging/desktop.toml`])) as any;
}
export async function packageInputs(revision: string, output: string) {
  const root = join(output, "source-resources");
  await mkdir(root);
  const archive = join(output, "source-resources.tar");
  // Exact public resources from the binary's commit, never dirty working files.
  await run("git", ["archive", "--format=tar", "--output", archive, revision,
    "LICENSE", "packaging/desktop.toml", "logo/ears_logo_mac_white_on_black.png",
    "apps/ears-ui/assets/fonts/DepartureMono-OFL.txt", "apps/ears-ui/assets/fonts/JetBrainsMono-OFL.txt"]);
  await run("tar", ["-xf", archive, "-C", root]);
  return root;
}
export function macDependencies(output: string) {
  const dependencies = output.split("\n").slice(1).filter(line => line.trim()).map(line => line.trim().split(" (compatibility version")[0]);
  if (!dependencies.length) throw new Error("Cannot inspect Mach-O dependencies");
  for (const dependency of dependencies) {
    if (!dependency.startsWith("/System/Library/") && !dependency.startsWith("/usr/lib/") && dependency !== "@rpath/libonnxruntime.dylib") {
      throw new Error("Unbundled/non-system Mach-O dependency");
    }
  }
  return dependencies;
}
export function clientBuild(info: any, revision: string, os: string, arch: string, ui: boolean) {
  if (!/^[a-f0-9]{40}$/.test(revision) || !info || info.source_commit !== revision) throw new Error("Build source mismatch");
  if (info.os !== os || info.arch !== arch || info.client_vad !== true) throw new Error("Native client-VAD capabilities required");
  if (info.moonshine !== false || info.parakeet_rs !== false) throw new Error("Server/local-ASR features are not a desktop release profile");
  if (!/^\d+\.\d+\.\d+$/.test(info.version)) throw new Error("Numeric version required");
  if (ui && (info.experimental_overlays !== false || info.renderer !== "accepted-e034c69f")) throw new Error("Accepted release renderer required");
  return info;
}
export function xml(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
