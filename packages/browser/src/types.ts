export type RecognitionLanguage = "en" | "de";
export type RecognitionProvider = "moonshine" | "ears";
export type RecognitionState =
	| "idle"
	| "preparing"
	| "ready"
	| "finishing"
	| "error";
export type RecognitionEvent =
	| { type: "state"; state: RecognitionState }
	| { type: "progress"; loaded: number; total?: number; file: string }
	| { type: "preview"; id: string; text: string }
	| { type: "completed"; id: string; text: string }
	| { type: "speech"; active: boolean }
	| { type: "error"; message: string };

export type RecognitionOptions = {
	language: RecognitionLanguage;
	/** Format of every feed() call: mono normalized Float32 PCM. */
	sampleRate: number;
} & (
	| { provider: "moonshine"; runtimeBaseUrl: string; modelBaseUrl?: string }
	| {
			provider: "ears";
			url: string;
			connectTimeoutMs?: number;
			finishTimeoutMs?: number;
	  }
);

/** Internal inference seam; callers supply audio, not a microphone. */
export interface RecognitionAdapter {
	prepare(): Promise<void>;
	feed(audio: Float32Array): void;
	finish(): Promise<void>;
	cancel(): void;
}
export type Emit = (event: RecognitionEvent) => void;
