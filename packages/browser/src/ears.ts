import { PcmResampler, pcmFloat32LE } from "./audio";
import type { Emit, RecognitionAdapter, RecognitionOptions } from "./types";

type EarsOptions = Extract<RecognitionOptions, { provider: "ears" }>;
export type SocketFactory = (url: string) => WebSocket;

/** eaRS has no result IDs on the wire: assign ordinal IDs per connection/segment.
 * Speech(false) closes a segment; Final.words is the cumulative session snapshot.
 * Never commit a preview merely because a UI silence timer elapsed.
 */
export class EarsAdapter implements RecognitionAdapter {
	private socket?: WebSocket;
	private closed = false;
	private finishing = false;
	private segment = 0;
	private words: string[] = [];
	private preview = "";
	private interim = false;
	private committedWords = 0;
	private boundarySeen = false;
	private resampler: PcmResampler;
	private rejectPending?: (error: Error) => void;
	private finishPromise?: Promise<void>;
	private finalReceived?: () => void;

	constructor(
		private options: EarsOptions,
		private emit: Emit,
		private socketFactory: SocketFactory = (url) => new WebSocket(url),
	) {
		this.resampler = new PcmResampler(options.sampleRate, 24000);
	}
	async prepare(): Promise<void> {
		const url = new URL(this.options.url);
		if (!["ws:", "wss:"].includes(url.protocol))
			throw new Error("eaRS requires a ws:// or wss:// URL");
		await new Promise<void>((resolve, reject) => {
			const timer = setTimeout(
				() => fail(new Error("eaRS did not confirm the requested language")),
				this.options.connectTimeoutMs ?? 15000,
			);
			let ready = false;
			const fail = (error: Error) => {
				clearTimeout(timer);
				this.rejectPending = undefined;
				reject(error);
			};
			this.rejectPending = fail;
			const socket = this.socketFactory(url.href);
			this.socket = socket;
			socket.binaryType = "arraybuffer";
			socket.onopen = () => {
				if (this.closed) return;
				this.command({ type: "setcodec", codec: "pcm" });
				this.command({ type: "setboundaryvad", enabled: true });
				this.command({ type: "setlanguage", lang: this.options.language });
				this.command({ type: "getstatus" });
			};
			socket.onmessage = (event) => {
				if (this.closed) return;
				try {
					const message = JSON.parse(String(event.data));
					if (
						!ready &&
						(message.type === "languagechanged" || message.type === "status")
					) {
						const language = String(message.lang ?? "")
							.toLowerCase()
							.split(/[-_]/)[0];
						if (language === this.options.language) {
							ready = true;
							clearTimeout(timer);
							this.rejectPending = undefined;
							resolve();
						}
					}
					if (message.type === "error")
						throw new Error(String(message.message ?? "eaRS error"));
					if (ready) this.handleMessage(message);
				} catch (error) {
					const failure =
						error instanceof Error ? error : new Error(String(error));
					if (!ready) fail(failure);
					this.emit({ type: "error", message: failure.message });
				}
			};
			socket.onerror = () => {
				const error = new Error("eaRS WebSocket failed");
				if (!ready) fail(error);
				else this.emit({ type: "error", message: error.message });
			};
			socket.onclose = () => {
				if (!ready) fail(new Error("eaRS disconnected during preparation"));
				else if (!this.closed)
					this.emit({
						type: "error",
						message: "eaRS disconnected before recognition completed",
					});
			};
		});
	}
	private command(command: object) {
		this.socket?.send(JSON.stringify(command));
	}
	private update(text: string) {
		this.preview = text;
		this.emit({ type: "preview", id: String(this.segment), text });
	}
	private complete(text: string) {
		if (text.trim())
			this.emit({
				type: "completed",
				id: String(this.segment),
				text: text.trim(),
			});
		this.segment++;
		this.words = [];
		this.preview = "";
		this.interim = false;
	}
	private handleMessage(message: {
		type: string;
		word?: string;
		text?: string;
		active?: boolean;
		words?: { word: string }[];
	}) {
		switch (message.type) {
			case "word":
				if (typeof message.word !== "string" || !message.word.trim()) return;
				this.words.push(message.word.trim());
				if (!this.interim) this.update(this.words.join(" "));
				break;
			case "interim":
				if (typeof message.text !== "string") return;
				this.interim = true;
				this.update(message.text);
				break;
			case "speech":
				if (typeof message.active !== "boolean") return;
				this.boundarySeen = true;
				this.emit({ type: "speech", active: message.active });
				if (!message.active) {
					this.committedWords += this.words.length;
					this.complete(this.preview || this.words.join(" "));
				}
				break;
			case "final": {
				// In boundary mode the final is a full-session summary, not a new turn.
				const text = this.boundarySeen
					? Array.isArray(message.words)
						? message.words
								.slice(this.committedWords)
								.map((word) => word.word)
								.join(" ")
						: this.preview
					: (message.text ?? this.preview);
				this.complete(text);
				this.finalReceived?.();
				break;
			}
		}
	}
	private send(audio: Float32Array) {
		if (!audio.length) return;
		const socket = this.socket;
		if (!socket || socket.readyState !== 1)
			throw new Error("eaRS is not connected");
		if (socket.bufferedAmount + audio.byteLength > 256 * 1024) {
			throw new Error(
				"eaRS audio backpressure exceeded; recognition stopped rather than dropping audio",
			);
		}
		socket.send(pcmFloat32LE(audio));
	}
	feed(audio: Float32Array) {
		if (this.closed || this.finishing)
			throw new Error("eaRS recognition is not accepting audio");
		this.send(this.resampler.feed(audio));
	}
	finish(): Promise<void> {
		if (this.finishPromise) return this.finishPromise;
		this.finishing = true;
		this.finishPromise = new Promise<void>((resolve, reject) => {
			const timer = setTimeout(() => {
				this.rejectPending = undefined;
				reject(
					new Error(
						"eaRS final transcript timed out; provisional text was not committed",
					),
				);
			}, this.options.finishTimeoutMs ?? 15000);
			this.rejectPending = (error) => {
				clearTimeout(timer);
				reject(error);
			};
			this.finalReceived = () => {
				clearTimeout(timer);
				this.rejectPending = undefined;
				this.finalReceived = undefined;
				this.closed = true;
				this.socket?.close();
				resolve();
			};
			try {
				this.send(this.resampler.finish());
				this.command({ type: "stop" });
			} catch (error) {
				clearTimeout(timer);
				reject(error);
			}
		});
		return this.finishPromise;
	}
	cancel() {
		this.closed = true;
		this.rejectPending?.(new Error("Recognition cancelled"));
		this.rejectPending = undefined;
		this.finalReceived = undefined;
		this.socket?.close();
	}
}
