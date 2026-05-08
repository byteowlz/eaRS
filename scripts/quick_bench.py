#!/usr/bin/env python3
"""Quick Parakeet streaming benchmark across all languages (3 files per lang)."""
import asyncio, json, time, wave, re, sys
import numpy as np
try:
    import websockets
except ImportError:
    print("pip3 install websockets"); sys.exit(1)

def read_wav(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def wer(hyp, ref):
    hw = normalize_text(hyp).split()
    rw = normalize_text(ref).split()
    if not rw: return 0.0 if not hw else 1.0
    n, m = len(rw), len(hw)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if rw[i-1] == hw[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / len(rw)

async def test_file(ws_url, audio_path):
    samples, sr = read_wav(audio_path)
    duration = len(samples) / sr
    chunk_ms = 80
    chunk_samples = int(sr * chunk_ms / 1000)
    
    words = []
    final_text = ""
    t0 = time.time()
    
    async with websockets.connect(ws_url, max_size=None) as ws:
        for offset in range(0, len(samples), chunk_samples):
            chunk = samples[offset:offset + chunk_samples]
            pcm = chunk.astype(np.float32).tobytes()
            await ws.send(pcm)
            await asyncio.sleep(chunk_ms / 1000 * 0.5)
        await ws.send("stop")
        
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                data = json.loads(msg)
                if data.get("type") == "word":
                    words.append(data.get("word", ""))
                elif data.get("type") == "final":
                    final_text = data.get("text", "")
                    break
                elif data.get("type") == "error":
                    break
            except asyncio.TimeoutError:
                break
    
    t1 = time.time()
    return {
        "duration": duration,
        "words": len(words),
        "dups": sum(1 for i in range(1, len(words)) if words[i] == words[i-1]),
        "text": final_text,
        "rtf": (t1 - t0) / duration,
    }

async def main():
    with open("/Users/tommyfalkowski/byteowlz/eaRS/test_audio/hub/manifest.json") as f:
        manifest = json.load(f)
    
    ws_url = "ws://[::1]:8765"
    results = {}
    
    by_lang = {}
    for e in manifest:
        by_lang.setdefault(e['language'], []).append(e)
    
    for lang, entries in sorted(by_lang.items()):
        print(f"\n--- {lang.upper()} ---", flush=True)
        
        lang_results = []
        for entry in entries[:5]:
            path = f"/Users/tommyfalkowski/byteowlz/eaRS/test_audio/hub/{entry['file']}"
            r = await test_file(ws_url, path)
            w = wer(r['text'], entry['text'])
            lang_results.append({"wer": w, **r})
            print(f"  {entry['file']}: WER={w:.0%} words={r['words']} dups={r['dups']} RTF={r['rtf']:.2f}x", flush=True)
        
        avg_wer = sum(r['wer'] for r in lang_results) / len(lang_results)
        avg_rtf = sum(r['rtf'] for r in lang_results) / len(lang_results)
        total_dups = sum(r['dups'] for r in lang_results)
        print(f"  AVG: WER={avg_wer:.0%} RTF={avg_rtf:.2f}x dups={total_dups}", flush=True)
        results[lang] = {"avg_wer": avg_wer, "avg_rtf": avg_rtf, "total_dups": total_dups}
    
    print(f"\n{'='*60}", flush=True)
    print("  PARAKEET STREAMING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for lang, r in sorted(results.items()):
        print(f"  {lang:12s}: WER={r['avg_wer']:.0%} RTF={r['avg_rtf']:.2f}x dups={r['total_dups']}", flush=True)
    
    total_dups = sum(r['total_dups'] for r in results.values())
    print(f"\n  TOTAL DUPLICATED WORDS: {total_dups}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
