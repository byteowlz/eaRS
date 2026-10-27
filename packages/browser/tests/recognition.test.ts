import { describe, expect, test } from "bun:test";
import { PcmResampler, pcmFloat32LE } from "../src/audio";
import { Recognizer } from "../src/index";
import type { MoonshineHost } from "../src/moonshine";
import type { RecognitionEvent } from "../src/types";

class Host implements MoonshineHost {
	closed = false;
	loaded?: Parameters<MoonshineHost["loadTranscriber"]>[0];
	listener?: Parameters<MoonshineHost["setListener"]>[1];
	audio: Float32Array[] = [];
	onProgress?: MoonshineHost["onProgress"];
	async loadTranscriber(
		config: Parameters<MoonshineHost["loadTranscriber"]>[0],
	) {
		this.loaded = config;
		this.onProgress?.("ears", 1, 2, "encoder.ort");
	}
	async createStream() {}
	setListener(
		_id: string,
		listener: Parameters<MoonshineHost["setListener"]>[1],
	) {
		this.listener = listener;
	}
	async start() {}
	addAudio(_id: string, audio: Float32Array) {
		this.audio.push(audio.slice());
	}
	queuedSeconds() {
		return 0;
	}
	async stop() {
		this.line("9007199254740993123", "finished", true);
	}
	close() {
		this.closed = true;
	}
	line(id: string, text: string, isComplete = false) {
		const event = { line: { id, text, isComplete } };
		if (isComplete) this.listener?.onLineCompleted(event);
		else this.listener?.onLineTextChanged(event);
	}
}
class Socket {
	readyState = 0;
	binaryType = "";
	bufferedAmount = 0;
	onopen?: () => void;
	onmessage?: (event: { data: string }) => void;
	onerror?: () => void;
	onclose?: () => void;
	sent: (string | ArrayBuffer)[] = [];
	send(data: string | ArrayBuffer) {
		this.sent.push(data);
	}
	close() {
		this.readyState = 3;
		this.onclose?.();
	}
	open() {
		this.readyState = 1;
		this.onopen?.();
	}
	message(message: object) {
		this.onmessage?.({ data: JSON.stringify(message) });
	}
}
const local = {
	provider: "moonshine",
	language: "de",
	sampleRate: 48000,
	runtimeBaseUrl: "/runtime/",
} as const;
const remote = {
	provider: "ears",
	language: "de",
	sampleRate: 48000,
	url: "wss://example.test/recognize",
} as const;
const completed = (events: RecognitionEvent[]) =>
	events.filter((event) => event.type === "completed");
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function fakeLocal() {
	const host = new Host();
	const events: RecognitionEvent[] = [];
	const recognizer = new Recognizer({
		loadMoonshineHost: async () => host,
		socketFactory: () => {
			throw new Error("Audio egress attempted");
		},
	});
	recognizer.subscribe((event) => events.push(event));
	return { host, events, recognizer };
}
function fakeRemote() {
	const socket = new Socket();
	const events: RecognitionEvent[] = [];
	const recognizer = new Recognizer({
		socketFactory: () => socket as unknown as WebSocket,
	});
	recognizer.subscribe((event) => events.push(event));
	return { socket, events, recognizer };
}

describe("audio format", () => {
	test("resampling is invariant under chunking at fractional rates", () => {
		const audio = Float32Array.from({ length: 997 }, (_, i) => Math.sin(i / 9));
		const whole = new PcmResampler(44100, 16000);
		const a = [...whole.feed(audio), ...whole.finish()];
		const chunks = new PcmResampler(44100, 16000);
		const b: number[] = [];
		for (let i = 0; i < audio.length; i += 37)
			b.push(...chunks.feed(audio.subarray(i, i + 37)));
		b.push(...chunks.finish());
		expect(b).toEqual(a);
		expect(b.length).toBe(Math.ceil((audio.length * 16000) / 44100));
	});
	test("wire is little endian and honors subarray offsets", () => {
		const buffer = pcmFloat32LE(Float32Array.of(0, -0.5, 1).subarray(1));
		expect(buffer.byteLength).toBe(8);
		expect(new DataView(buffer).getFloat32(0, true)).toBe(-0.5);
	});
});

describe("local recognition lifecycle", () => {
	test("pins German Tiny, runs off-thread, previews replace and completions dedupe by opaque ID", async () => {
		const { host, recognizer, events } = fakeLocal();
		await recognizer.prepare(local);
		expect(host.loaded?.modelArch).toBe(2);
		expect(host.loaded?.source.files["encoder.ort"]).toContain(
			"tiny-streaming-de/quantized_26_08_24/",
		);
		expect(events.some((event) => event.type === "progress")).toBe(true);
		host.line("9007199254740993123", "wrong");
		host.line("9007199254740993123", "revised");
		expect(completed(events)).toHaveLength(0);
		const audio = Float32Array.of(0.3, 0.6, 0.9);
		recognizer.feed(audio);
		expect(audio.length).toBe(3);
		expect(host.audio[0]?.length).toBe(1);
		const finish = recognizer.finish();
		expect(recognizer.finish()).toBe(finish);
		await finish;
		host.line("9007199254740993123", "finished", true);
		expect(completed(events)).toHaveLength(1);
		expect(completed(events)[0]?.id).toEndWith(":9007199254740993123");
		expect(host.closed).toBe(true);
	});
	test("explicit Small uses the matching architecture and model bundle", async () => {
		const { host, recognizer } = fakeLocal();
		await recognizer.prepare({ ...local, modelSize: "small" });
		expect(host.loaded?.modelArch).toBe(4);
		expect(host.loaded?.source.files["encoder.ort"]).toContain(
			"small-streaming-de/quantized_26_08_24/",
		);
		recognizer.cancel();
	});
	test("explicit local Small override keeps the Small architecture", async () => {
		const { host, recognizer } = fakeLocal();
		await recognizer.prepare({ ...local, modelSize: "small", modelBaseUrl: "https://example.test/models/" });
		expect(host.loaded?.modelArch).toBe(4);
		expect(host.loaded?.source.files["encoder.ort"]).toBe("https://example.test/models/encoder.ort");
		recognizer.cancel();
	});
	test("rejects unsupported size without starting a worker", async () => {
		const { host, recognizer } = fakeLocal();
		await expect(recognizer.prepare({ ...local, modelSize: "medium" } as never)).rejects.toThrow("Unsupported Moonshine model size");
		expect(host.loaded).toBeUndefined();
	});
	test("cancel drops provisional and late results; never uploads or commits a preview", async () => {
		const { host, recognizer, events } = fakeLocal();
		await recognizer.prepare(local);
		host.line("a", "tentative");
		recognizer.cancel();
		host.line("a", "late", true);
		expect(completed(events)).toHaveLength(0);
		expect(() => recognizer.feed(Float32Array.of(0))).toThrow("not ready");
	});
	test("cancel during import settles immediately and closes a late-created worker", async () => {
		const host = new Host();
		let load!: (host: MoonshineHost) => void;
		const recognizer = new Recognizer({
			loadMoonshineHost: () =>
				new Promise((resolve) => {
					load = resolve;
				}),
		});
		const prepare = recognizer.prepare(local);
		recognizer.cancel();
		await expect(prepare).rejects.toMatchObject({ name: "AbortError" });
		load(host);
		await tick();
		expect(host.closed).toBe(true);
	});
	test("failure does not fall back to a remote service and can be retried", async () => {
		let attempts = 0;
		const recognizer = new Recognizer({
			loadMoonshineHost: async () => {
				if (++attempts === 1) throw new Error("model unavailable");
				return new Host();
			},
			socketFactory: () => {
				throw new Error("Silent fallback");
			},
		});
		await expect(recognizer.prepare(local)).rejects.toThrow(
			"model unavailable",
		);
		await recognizer.prepare(local);
		recognizer.cancel();
	});
	test("rejects nonfinite audio and unsupported language/rate", async () => {
		const { recognizer } = fakeLocal();
		await expect(
			recognizer.prepare({ ...local, sampleRate: 0 }),
		).rejects.toThrow("sample rate");
		await recognizer.prepare(local);
		expect(() => recognizer.feed(Float32Array.of(Number.NaN))).toThrow(
			"finite",
		);
	});
});

describe("explicit eaRS adapter", () => {
	test("confirms language, requests PCM and boundaries, resamples at 24kHz", async () => {
		const { socket, recognizer } = fakeRemote();
		const prepare = recognizer.prepare(remote);
		socket.open();
		expect(
			socket.sent.map((data) =>
				typeof data === "string" ? JSON.parse(data).type : "audio",
			),
		).toEqual(["setcodec", "setboundaryvad", "setlanguage", "getstatus"]);
		socket.message({ type: "languagechanged", lang: "de-DE" });
		await prepare;
		recognizer.feed(Float32Array.of(0.2, 0.4, 0.6, 0.8));
		expect((socket.sent.at(-1) as ArrayBuffer).byteLength).toBe(8);
		recognizer.cancel();
	});
	test("revisable interim replaces words; speech boundaries plus cumulative Final do not duplicate", async () => {
		const { socket, recognizer, events } = fakeRemote();
		const prepare = recognizer.prepare(remote);
		socket.open();
		socket.message({ type: "status", lang: "de" });
		await prepare;
		socket.message({ type: "word", word: "old" });
		socket.message({ type: "interim", text: "corrected" });
		socket.message({ type: "speech", active: false });
		socket.message({ type: "word", word: "tail" });
		const finish = recognizer.finish();
		socket.message({
			type: "final",
			text: "old tail",
			words: [{ word: "old" }, { word: "tail" }],
		});
		await finish;
		expect(completed(events).map((event) => event.text)).toEqual([
			"corrected",
			"tail",
		]);
	});
	test("legacy Final works; Pause never commits words", async () => {
		const { socket, recognizer, events } = fakeRemote();
		const prepare = recognizer.prepare(remote);
		socket.open();
		socket.message({ type: "status", lang: "de" });
		await prepare;
		socket.message({ type: "word", word: "hello" });
		socket.message({ type: "pause" });
		expect(completed(events)).toHaveLength(0);
		const finish = recognizer.finish();
		socket.message({ type: "final", text: "hello", words: [] });
		await finish;
		expect(completed(events).map((event) => event.text)).toEqual(["hello"]);
	});
	test("fails closed on unsupported language and late messages after cancel", async () => {
		const { socket, recognizer, events } = fakeRemote();
		const prepare = recognizer.prepare({ ...remote, connectTimeoutMs: 5 });
		socket.open();
		socket.message({ type: "status", lang: "en-US" });
		await expect(prepare).rejects.toThrow("confirm");
		socket.message({ type: "final", text: "late" });
		expect(completed(events)).toHaveLength(0);
	});
	test("backpressure is an error, not dropped audio", async () => {
		const { socket, recognizer } = fakeRemote();
		const prepare = recognizer.prepare(remote);
		socket.open();
		socket.message({ type: "status", lang: "de" });
		await prepare;
		socket.bufferedAmount = 300000;
		expect(() => recognizer.feed(Float32Array.of(0, 0))).toThrow(
			"backpressure",
		);
		expect(socket.readyState).toBe(3);
	});
	test("finish timeout never promotes a provisional preview", async () => {
		const { socket, recognizer, events } = fakeRemote();
		const prepare = recognizer.prepare({ ...remote, finishTimeoutMs: 5 });
		socket.open();
		socket.message({ type: "status", lang: "de" });
		await prepare;
		socket.message({ type: "interim", text: "not final" });
		await expect(recognizer.finish()).rejects.toThrow("timed out");
		expect(completed(events)).toHaveLength(0);
	});
});
