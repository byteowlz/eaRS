import { EarsAdapter, type SocketFactory } from "./ears";
import { MoonshineAdapter, type MoonshineHostLoader } from "./moonshine";
import type {
	RecognitionAdapter,
	RecognitionEvent,
	RecognitionOptions,
	RecognitionState,
} from "./types";

export type {
	RecognitionEvent,
	RecognitionLanguage,
	RecognitionOptions,
	RecognitionProvider,
	RecognitionState,
} from "./types";
export { MOONSHINE_MODEL_BASES } from "./moonshine";

/** No microphone, UI, command execution, gateway or implicit provider fallback. */
export class Recognizer {
	private adapter?: RecognitionAdapter;
	private generation = 0;
	private state: RecognitionState = "idle";
	private completed = new Set<string>();
	private listeners = new Set<(event: RecognitionEvent) => void>();
	private cancelPending?: () => void;
	private finishPromise?: Promise<void>;
	constructor(
		private dependencies: {
			socketFactory?: SocketFactory;
			loadMoonshineHost?: MoonshineHostLoader;
		} = {},
	) {}
	subscribe(listener: (event: RecognitionEvent) => void): () => void {
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}
	private emit(event: RecognitionEvent) {
		for (const listener of this.listeners) listener(event);
	}
	private setState(state: RecognitionState) {
		this.state = state;
		this.emit({ type: "state", state });
	}
	private fail(error: unknown) {
		const message = error instanceof Error ? error.message : String(error);
		this.cancelPending?.();
		this.adapter?.cancel();
		this.setState("error");
		this.emit({ type: "error", message });
	}
	private async run(
		work: Promise<void>,
		generation: number,
		timeoutMs: number,
	) {
		let timer: ReturnType<typeof setTimeout> | undefined;
		const cancelled = new Promise<never>((_resolve, reject) => {
			this.cancelPending = () =>
				reject(new DOMException("Recognition cancelled", "AbortError"));
			timer = setTimeout(
				() => reject(new Error("Recognition operation timed out")),
				timeoutMs,
			);
		});
		try {
			await Promise.race([work, cancelled]);
		} finally {
			clearTimeout(timer);
			if (generation === this.generation) this.cancelPending = undefined;
		}
		if (generation !== this.generation)
			throw new DOMException("Recognition cancelled", "AbortError");
	}
	async prepare(options: RecognitionOptions) {
		this.cancel();
		if (
			!["moonshine", "ears"].includes(options.provider) ||
			!["en", "de"].includes(options.language)
		) {
			throw new Error("Unsupported recognition provider or language");
		}
		if (
			!Number.isFinite(options.sampleRate) ||
			options.sampleRate < 8000 ||
			options.sampleRate > 96000
		) {
			throw new Error("PCM sample rate must be between 8000 and 96000 Hz");
		}
		const generation = this.generation;
		this.completed.clear();
		this.finishPromise = undefined;
		const receive = (event: RecognitionEvent) => {
			if (
				generation !== this.generation ||
				this.state === "error" ||
				this.state === "idle"
			)
				return;
			if (event.type === "error") {
				this.fail(new Error(event.message));
				return;
			}
			if (event.type === "completed" || event.type === "preview") {
				// Scope opaque provider IDs to this preparation, even when providers reuse IDs.
				const id = `${generation}:${event.id}`;
				if (this.completed.has(id)) return;
				if (event.type === "completed") this.completed.add(id);
				this.emit({ ...event, id });
			} else this.emit(event);
		};
		this.setState("preparing");
		this.adapter =
			options.provider === "moonshine"
				? new MoonshineAdapter(
						options,
						receive,
						this.dependencies.loadMoonshineHost,
					)
				: new EarsAdapter(options, receive, this.dependencies.socketFactory);
		try {
			await this.run(this.adapter.prepare(), generation, 180000);
			if ((this.state as RecognitionState) === "error")
				throw new Error("Recognition preparation failed");
			this.setState("ready");
		} catch (error) {
			if (
				generation === this.generation &&
				(this.state as RecognitionState) !== "error"
			)
				this.fail(error);
			throw error;
		}
	}
	feed(audio: Float32Array) {
		if (this.state !== "ready")
			throw new Error("Recognition is not ready for audio");
		try {
			if (
				!(audio instanceof Float32Array) ||
				audio.some((sample) => !Number.isFinite(sample) || Math.abs(sample) > 1)
			) {
				throw new Error("Audio must be mono normalized finite Float32 PCM");
			}
			this.adapter?.feed(audio);
		} catch (error) {
			this.fail(error);
			throw error;
		}
	}
	finish(): Promise<void> {
		if (this.finishPromise) return this.finishPromise;
		if (this.state !== "ready" || !this.adapter)
			return Promise.reject(new Error("Recognition is not ready to finish"));
		const generation = this.generation;
		this.setState("finishing");
		this.finishPromise = this.run(this.adapter.finish(), generation, 20000)
			.then(() => {
				this.setState("idle");
			})
			.catch((error) => {
				if (
					generation === this.generation &&
					(this.state as RecognitionState) !== "error"
				)
					this.fail(error);
				throw error;
			});
		return this.finishPromise;
	}
	cancel() {
		this.generation++;
		this.cancelPending?.();
		this.cancelPending = undefined;
		this.adapter?.cancel();
		this.adapter = undefined;
		this.finishPromise = undefined;
		this.setState("idle");
	}
	dispose() {
		this.cancel();
		this.listeners.clear();
	}
}
