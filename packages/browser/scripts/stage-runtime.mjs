import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	cpSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	readdirSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// npm 0.1.5 embeds an older WASM core: it rejects split frontends and German.
// The same-version GitHub release is different. Pin bytes, not just a version.
export const MOONSHINE_RUNTIME = {
	version: "0.1.5",
	url: "https://github.com/moonshine-ai/moonshine/releases/download/v0.1.5/moonshine-voice-wasm.tar.gz",
	sha256: "c515bf7691e12048f70a92cc82b3b0894c16c3773ffcacb7d48944fb150e4837",
	wasmSha256:
		"22fca4a5b2dc50fe36dc68e4d25fab73b0250b71f744d9fea511c3a308b2420a",
};
export function verifyRuntimeArchive(bytes) {
	if (
		createHash("sha256").update(bytes).digest("hex") !==
		MOONSHINE_RUNTIME.sha256
	) {
		throw new Error(
			"Moonshine runtime checksum mismatch; refusing to stage unverified assets",
		);
	}
}

/** Build-time GET only; verified archive caching supports offline rebuilds.
 * Keep the upstream ESM/worker/pthread tree intact; never rebundle moonshine.mjs.
 */
export async function stageMoonshineRuntime(destination) {
	const cache = path.join(
		process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache"),
		"ears-browser",
	);
	mkdirSync(cache, { recursive: true });
	const archive = path.join(
		cache,
		`moonshine-${MOONSHINE_RUNTIME.sha256}.tar.gz`,
	);
	if (!existsSync(archive)) {
		const response = await fetch(MOONSHINE_RUNTIME.url);
		if (!response.ok)
			throw new Error(`Moonshine runtime download failed (${response.status})`);
		const bytes = Buffer.from(await response.arrayBuffer());
		verifyRuntimeArchive(bytes);
		writeFileSync(archive, bytes);
	}
	verifyRuntimeArchive(readFileSync(archive));
	const temporary = mkdtempSync(path.join(os.tmpdir(), "ears-moonshine-"));
	try {
		execFileSync("tar", ["-xzf", archive, "-C", temporary]);
		const source = path.join(temporary, "dist");
		const wasmDigest = createHash("sha256")
			.update(readFileSync(path.join(source, "moonshine.wasm")))
			.digest("hex");
		if (wasmDigest !== MOONSHINE_RUNTIME.wasmSha256)
			throw new Error("Moonshine WASM checksum mismatch");
		mkdirSync(destination, { recursive: true });
		for (const name of readdirSync(source)) {
			if (/\.(js|mjs|wasm)$/.test(name))
				cpSync(path.join(source, name), path.join(destination, name));
		}
		cpSync(
			path.join(temporary, "README.md"),
			path.join(destination, "MOONSHINE-README.md"),
		);
		cpSync(
			fileURLToPath(new URL("./MOONSHINE-LICENSE.txt", import.meta.url)),
			path.join(destination, "MOONSHINE-LICENSE.txt"),
		);
		cpSync(
			fileURLToPath(new URL("./licenses/", import.meta.url)),
			path.join(destination, "licenses"),
			{ recursive: true },
		);
		writeFileSync(
			path.join(destination, "runtime-manifest.json"),
			`${JSON.stringify(MOONSHINE_RUNTIME, null, 2)}\n`,
		);
	} finally {
		rmSync(temporary, { recursive: true, force: true });
	}
}
