#!/usr/bin/env python3
"""Benchmark eaRS streaming ASR against reference transcriptions from a manifest."""
import argparse, asyncio, json, os, sys, time, wave, struct
import numpy as np

try:
    import websockets
except ImportError:
    print("pip3 install websockets")
    sys.exit(1)


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_error_rate(hyp: str, ref: str) -> float:
    """Simple WER using edit distance on words."""
    hyp_w = normalize_text(hyp).split()
    ref_w = normalize_text(ref).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    n, m = len(ref_w), len(hyp_w)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_w[i-1] == hyp_w[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / len(ref_w)


def char_error_rate(hyp: str, ref: str) -> float:
    hyp_c = list(normalize_text(hyp))
    ref_c = list(normalize_text(ref))
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    n, m = len(ref_c), len(hyp_c)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_c[i-1] == hyp_c[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / len(ref_c)


def count_duplicated_words(text: str) -> int:
    """Count immediately repeated words."""
    words = normalize_text(text).split()
    if len(words) < 2:
        return 0
    count = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            count += 1
    return count


def read_wav(path: str, target_sr: int = 24000) -> tuple:
    """Read WAV file, return (samples_f32, sample_rate)."""
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        # Convert to float32
        if wf.getsampwidth() == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif wf.getsampwidth() == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


async def stream_file(ws_url: str, audio_path: str, engine: str, chunk_ms: int = 80) -> dict:
    """Stream a WAV file to eaRS server and collect results."""
    samples, sr = read_wav(audio_path)
    duration = len(samples) / sr
    
    chunk_samples = int(sr * chunk_ms / 1000)
    
    words_received = []
    final_text = ""
    first_word_time = None
    t_start = time.time()
    
    async with websockets.connect(ws_url, max_size=None) as ws:
        # Switch engine
        await ws.send(json.dumps({"type": "set_engine", "engine": engine}))
        
        # Stream audio in chunks
        for offset in range(0, len(samples), chunk_samples):
            chunk = samples[offset:offset + chunk_samples]
            # Convert to f32 LE bytes (server expects f32 PCM, not int16)
            pcm = chunk.astype(np.float32).tobytes()
            await ws.send(pcm)
            await asyncio.sleep(chunk_ms / 1000 * 0.5)  # Slightly faster than real-time
        
        # Send stop as text message (binary "stop" would be treated as audio)
        await ws.send("stop")
        
        # Collect results
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                if data.get("type") == "word":
                    if first_word_time is None:
                        first_word_time = time.time() - t_start
                    words_received.append(data.get("word", ""))
                elif data.get("type") == "final":
                    final_text = data.get("text", "")
                    break
                elif data.get("type") == "engine_changed":
                    pass  # OK
                elif data.get("type") == "error":
                    print(f"  Server error: {data.get('message', '')}")
                    break
            except asyncio.TimeoutError:
                break
    
    t_end = time.time()
    
    return {
        "transcript": final_text or " ".join(words_received),
        "words": len(words_received),
        "duplicated_words": count_duplicated_words(final_text or " ".join(words_received)),
        "first_word_latency": round(first_word_time, 3) if first_word_time else None,
        "total_latency": round(t_end - t_start, 3),
        "rtf": round((t_end - t_start) / duration, 3) if duration > 0 else 0,
        "audio_duration": round(duration, 2),
    }


async def benchmark_manifest(manifest_path: str, ws_url: str, engines: list, chunk_ms: int):
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    base_dir = os.path.dirname(manifest_path)
    
    # Group by language
    by_lang = {}
    for entry in manifest:
        lang = entry["language"]
        by_lang.setdefault(lang, []).append(entry)
    
    all_results = {}
    
    for lang, entries in sorted(by_lang.items()):
        print(f"\n{'='*60}")
        print(f"  Language: {lang.upper()} ({len(entries)} samples)")
        print(f"{'='*60}")
        
        for engine in engines:
            print(f"\n  --- {engine.upper()} ---")
            lang_results = []
            
            for entry in entries:
                audio_path = os.path.join(base_dir, entry["file"])
                if not os.path.exists(audio_path):
                    print(f"    SKIP {entry['file']} (not found)")
                    continue
                
                ref_text = entry["text"]
                try:
                    result = await stream_file(ws_url, audio_path, engine, chunk_ms)
                except Exception as e:
                    print(f"    ERROR {entry['file']}: {e}")
                    continue
                
                wer = word_error_rate(result["transcript"], ref_text)
                cer = char_error_rate(result["transcript"], ref_text)
                dups = result["duplicated_words"]
                
                lang_results.append({
                    "file": entry["file"],
                    "wer": wer,
                    "cer": cer,
                    "dups": dups,
                    "words": result["words"],
                    "rtf": result["rtf"],
                    "latency": result["first_word_latency"],
                })
                
                print(f"    {entry['file']}: WER={wer:.1%} CER={cer:.1%} dups={dups} words={result['words']} RTF={result['rtf']}x")
                print(f"      REF: {ref_text[:80]}")
                print(f"      HYP: {result['transcript'][:80]}")
            
            if lang_results:
                avg_wer = sum(r["wer"] for r in lang_results) / len(lang_results)
                avg_cer = sum(r["cer"] for r in lang_results) / len(lang_results)
                avg_rtf = sum(r["rtf"] for r in lang_results) / len(lang_results)
                total_dups = sum(r["dups"] for r in lang_results)
                print(f"\n  {engine.upper()} AVG: WER={avg_wer:.1%} CER={avg_cer:.1%} RTF={avg_rtf:.2f}x dups={total_dups}")
            
            all_results[f"{lang}_{engine}"] = lang_results
    
    # Save full results
    results_path = os.path.join(base_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark eaRS against reference transcriptions")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--ws", default="ws://[::1]:8765", help="WebSocket URL")
    parser.add_argument("--engines", nargs="+", default=["parakeet"], choices=["parakeet", "kyutai", "both"])
    parser.add_argument("--chunk-ms", type=int, default=80, help="Audio chunk size in ms")
    args = parser.parse_args()
    
    engines = ["parakeet", "kyutai"] if "both" in args.engines else args.engines
    asyncio.run(benchmark_manifest(args.manifest, args.ws, engines, args.chunk_ms))


if __name__ == "__main__":
    main()
