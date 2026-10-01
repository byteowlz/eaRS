/** Streaming area-average resampler. Fractional bins survive chunk boundaries.
 * Downsampling averages input samples rather than aliasing by decimation.
 * This is a modest ASR resampler, not a studio-quality reconstruction filter.
 */
export class PcmResampler {
	private readonly width: number;
	private filled = 0;
	private sum = 0;
	constructor(inputRate: number, outputRate: number) {
		this.width = inputRate / outputRate;
	}
	feed(audio: Float32Array): Float32Array {
		const result: number[] = [];
		for (const sample of audio) {
			let remaining = 1;
			while (remaining > 1e-10) {
				const weight = Math.min(remaining, this.width - this.filled);
				this.sum += sample * weight;
				this.filled += weight;
				remaining -= weight;
				if (this.filled >= this.width - 1e-10) {
					result.push(this.sum / this.width);
					this.sum = 0;
					this.filled = 0;
				}
			}
		}
		return Float32Array.from(result);
	}
	finish(): Float32Array {
		const tail = this.filled
			? Float32Array.of(this.sum / this.filled)
			: new Float32Array();
		this.sum = 0;
		this.filled = 0;
		return tail;
	}
}

export function pcmFloat32LE(audio: Float32Array): ArrayBuffer {
	const buffer = new ArrayBuffer(audio.length * 4);
	const view = new DataView(buffer);
	audio.forEach((sample, index) => view.setFloat32(index * 4, sample, true));
	return buffer;
}
