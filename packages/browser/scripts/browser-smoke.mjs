import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stageMoonshineRuntime } from "./stage-runtime.mjs";

// Public/consented fixtures stay outside git. No microphone or speech service.
const [english, german] = process.argv.slice(2);
if (!english || !german)
	throw new Error(
		"Usage: bun scripts/browser-smoke.mjs <english.wav> <german.wav>",
	);
const fixtures = { en: path.resolve(english), de: path.resolve(german) };
const directory = mkdtempSync(path.join(os.tmpdir(), "ears-browser-smoke-"));
await stageMoonshineRuntime(path.join(directory, "runtime"));
const sdk = fileURLToPath(new URL("../dist/index.js", import.meta.url));
const session = `ears-smoke-${crypto.randomUUID()}`;
const networkGuard = `
const __earsOriginalFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = (input, options) => {
  const method = options?.method ?? (input instanceof Request ? input.method : 'GET');
  if (method.toUpperCase() !== 'GET' || options?.body) throw new Error('Audio egress attempt');
  return __earsOriginalFetch(input, options);
};
globalThis.WebSocket = class { constructor() { throw new Error('Local recognition attempted a WebSocket'); } };
`;
const script = `
import { Recognizer } from '/sdk.js';
${networkGuard}
window.runProbe = async (language) => {
  const result = window.probeResult = { language, isolated: crossOriginIsolated, previews: 0, completed: [] };
  const recognizer = new Recognizer();
  const start = performance.now();
  recognizer.subscribe(event => {
    if (event.type === 'preview') result.previews++;
    if (event.type === 'completed') result.completed.push({ id: event.id, text: event.text });
  });
  let context;
  try {
    await recognizer.prepare({ provider: 'moonshine', language, sampleRate: 16000, runtimeBaseUrl: '/runtime/' });
    result.prepareMs = performance.now() - start;
    context = new AudioContext({ sampleRate: 16000 });
    const buffer = await context.decodeAudioData(await (await fetch('/' + language + '.wav')).arrayBuffer());
    const audio = buffer.getChannelData(0);
    result.audioSeconds = audio.length / buffer.sampleRate;
    for (let offset = 0; offset < audio.length; offset += 3200) {
      recognizer.feed(audio.subarray(offset, offset + 3200));
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    await recognizer.finish();
    result.totalMs = performance.now() - start;
  } catch(error) { result.error = error.message; }
  finally { recognizer.dispose(); await context?.close(); result.done = true; }
};`;
const server = Bun.serve({
	hostname: "127.0.0.1",
	port: 0,
	async fetch(request) {
		const url = new URL(request.url);
		const headers = {
			"Cross-Origin-Opener-Policy": "same-origin",
			"Cross-Origin-Embedder-Policy": "require-corp",
			"Cache-Control": "no-store",
		};
		if (request.method !== "GET")
			return new Response("No audio upload endpoint", { status: 405, headers });
		if (url.pathname === "/")
			return new Response(
				'<!doctype html><html><body><script type="module" src="/probe.js"></script></body></html>',
				{ headers: { ...headers, "Content-Type": "text/html" } },
			);
		if (url.pathname === "/probe.js")
			return new Response(script, {
				headers: { ...headers, "Content-Type": "text/javascript" },
			});
		if (url.pathname === "/sdk.js")
			return new Response(Bun.file(sdk), { headers });
		if (url.pathname === "/en.wav" || url.pathname === "/de.wav")
			return new Response(Bun.file(fixtures[url.pathname.slice(1, 3)]), {
				headers,
			});
		if (url.pathname === "/runtime/stt-worker.js")
			return new Response(
				`${networkGuard}\n${await Bun.file(path.join(directory, url.pathname)).text()}`,
				{ headers: { ...headers, "Content-Type": "text/javascript" } },
			);
		if (url.pathname.startsWith("/runtime/") && !url.pathname.includes(".."))
			return new Response(Bun.file(path.join(directory, url.pathname)), {
				headers,
			});
		return new Response("Not found", { status: 404, headers });
	},
});
async function browser(...args) {
	const process = Bun.spawn(
		["agent-browser", "--session", session, "--json", ...args],
		{ stdout: "pipe", stderr: "pipe" },
	);
	const [text, errors, exit] = await Promise.all([
		new Response(process.stdout).text(),
		new Response(process.stderr).text(),
		process.exited,
	]);
	if (exit) throw new Error(`Browser command failed: ${errors || text}`);
	const response = JSON.parse(text);
	if (!response.success) throw new Error(JSON.stringify(response));
	return response.data;
}
try {
	await browser("open", `http://127.0.0.1:${server.port}`);
	// Page request capture complements hard egress guards in both page and STT worker.
	await browser("network", "route", "**/*");
	const results = [];
	for (const language of ["en", "de"]) {
		await browser(
			"eval",
			`void window.runProbe(${JSON.stringify(language)}); true`,
		);
		const deadline = Date.now() + 240000;
		let result;
		while (Date.now() < deadline) {
			await new Promise((resolve) => setTimeout(resolve, 500));
			const data = await browser("eval", "window.probeResult");
			result = data.result;
			if (result?.done) break;
		}
		if (
			!result?.done ||
			result.error ||
			!result.completed?.some((line) => line.text.trim())
		)
			throw new Error(`Recognition failed: ${JSON.stringify(result)}`);
		if (
			new Set(result.completed.map((line) => line.id)).size !==
			result.completed.length
		)
			throw new Error("Duplicate line completion");
		results.push(result);
	}
	const network = await browser("network", "requests");
	const requests = network.requests ?? [];
	if (requests.some((request) => request.method !== "GET"))
		throw new Error("Unexpected non-GET traffic in local recognition");
	console.log(
		JSON.stringify(
			{
				results,
				network: {
					pageRequests: requests.length,
					pageMethods: [...new Set(requests.map((request) => request.method))],
					pageAndSttWorkerEgressGuards: true,
				},
			},
			null,
			2,
		),
	);
} finally {
	await browser("close").catch((error) => console.error(error.message));
	server.stop(true);
	rmSync(directory, { recursive: true, force: true });
}
