import { PcmResampler } from "./audio";
import type { Emit, RecognitionAdapter, RecognitionOptions } from "./types";

type MoonshineOptions = Extract<RecognitionOptions, { provider: "moonshine" }>;
type Line = { id: string; text: string; isComplete: boolean };
type Listener = {
	onLineStarted(event: { line: Line }): void;
	onLineTextChanged(event: { line: Line }): void;
	onLineCompleted(event: { line: Line }): void;
	onError(event: { error: Error }): void;
};
/** Narrow structural view of the pinned upstream host, not a second worker protocol. */
export interface MoonshineHost {
	onProgress?: (
		id: string,
		loaded: number,
		total: number | undefined,
		file: string,
	) => void;
	loadTranscriber(config: {
		transcriberId: string;
		modelArch: number;
		source: { kind: "urls"; files: Record<string, string> };
	}): Promise<void>;
	createStream(
		transcriberId: string,
		streamId: string,
		options: { updateInterval: number },
	): Promise<void>;
	setListener(streamId: string, listener: Listener): void;
	start(streamId: string): Promise<void>;
	addAudio(streamId: string, audio: Float32Array, sampleRate: number): void;
	queuedSeconds(streamId: string): number;
	stop(streamId: string): Promise<void>;
	close(): void;
}
export type MoonshineHostLoader = (base: string) => Promise<MoonshineHost>;

const MODEL_FILES = [
	"adapter.ort",
	"cross_kv.ort",
	"decoder_kv.ort",
	"encoder.ort",
	"frontend.model.ort",
	"frontend.weights.ort",
	"streaming_config.json",
	"tokenizer.bin",
];
export const MOONSHINE_MODEL_BASES = {
	en: "https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/",
	de: "https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/",
} as const;

export async function loadMoonshineHost(
	runtimeBaseUrl: string,
): Promise<MoonshineHost> {
	if (
		!globalThis.isSecureContext ||
		!globalThis.crossOriginIsolated ||
		typeof Worker === "undefined"
	) {
		throw new Error(
			"Local speech requires HTTPS and cross-origin isolation (COOP/COEP). No audio was sent to a server.",
		);
	}
	const base = new URL(runtimeBaseUrl, location.href);
	if (base.origin !== location.origin)
		throw new Error(
			"Moonshine runtime assets must be hosted on the application origin",
		);
	const moduleUrl = new URL("stt-worker-host.js", base).href;
	// Preserve the upstream module tree: its worker/pthread paths use import.meta.url.
	const module = await import(/* @vite-ignore */ moduleUrl);
	return new module.SttWorkerHost(base.href) as MoonshineHost;
}

export class MoonshineAdapter implements RecognitionAdapter {
	private host?: MoonshineHost;
	private closed = false;
	private readonly resampler: PcmResampler;
	constructor(
		private options: MoonshineOptions,
		private emit: Emit,
		private loadHost: MoonshineHostLoader = loadMoonshineHost,
	) {
		this.resampler = new PcmResampler(options.sampleRate, 16000);
	}
	private assertOpen() {
		if (this.closed) throw new Error("Recognition cancelled");
	}
	async prepare() {
		const host = await this.loadHost(this.options.runtimeBaseUrl);
		if (this.closed) {
			host.close();
			this.assertOpen();
		}
		this.host = host;
		host.onProgress = (_id, loaded, total, file) =>
			this.emit({ type: "progress", loaded, total, file });
		const base =
			this.options.modelBaseUrl ?? MOONSHINE_MODEL_BASES[this.options.language];
		const files = Object.fromEntries(
			MODEL_FILES.map((file) => [file, new URL(file, base).href]),
		);
		await host.loadTranscriber({
			transcriberId: "ears",
			modelArch: 2,
			source: { kind: "urls", files },
		});
		this.assertOpen();
		await host.createStream("ears", "audio", { updateInterval: 0.3 });
		this.assertOpen();
		const preview = ({ line }: { line: Line }) => {
			if (!this.closed && !line.isComplete)
				this.emit({ type: "preview", id: line.id, text: line.text });
		};
		host.setListener("audio", {
			onLineStarted: preview,
			onLineTextChanged: preview,
			onLineCompleted: ({ line }) => {
				if (!this.closed)
					this.emit({ type: "completed", id: line.id, text: line.text });
			},
			onError: ({ error }) =>
				this.emit({ type: "error", message: error.message }),
		});
		await host.start("audio");
		this.assertOpen();
	}
	feed(audio: Float32Array) {
		this.assertOpen();
		if (!this.host) throw new Error("Moonshine is not ready");
		if (this.host.queuedSeconds("audio") > 5)
			throw new Error(
				"Local recognition is falling behind; stopped rather than dropping audio",
			);
		const samples = this.resampler.feed(audio);
		if (samples.length) this.host.addAudio("audio", samples, 16000);
	}
	async finish() {
		this.assertOpen();
		const host = this.host;
		if (!host) throw new Error("Moonshine is not ready");
		const tail = this.resampler.finish();
		if (tail.length) host.addAudio("audio", tail, 16000);
		// Upstream stop forces a final pass and line completion before its RPC ack.
		await host.stop("audio");
		this.cancel();
	}
	cancel() {
		this.closed = true;
		this.host?.close();
	}
}
