#!/usr/bin/env python3
"""Download Fleurs speech samples for eaRS benchmarking."""
import os, json, wave, sys
import numpy as np

OUTDIR = os.path.expanduser("~/byteowlz/eaRS/test_audio/fleurs")
os.makedirs(OUTDIR, exist_ok=True)

languages = {
    "en_us": "english",
    "es_419": "spanish", 
    "de_de": "german",
    "fr_fr": "french",
}

SAMPLES_PER_LANG = 10
MIN_DURATION = 5.0
MAX_DURATION = 30.0

manifest = []

for lang_code, lang_name in languages.items():
    print(f"\n=== Loading {lang_name} ({lang_code}) ===")
    sys.stdout.flush()
    try:
        from datasets import load_dataset
        ds = load_dataset("google/fleurs", lang_code, split="test", streaming=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.stdout.flush()
        continue
    
    collected = 0
    for i, sample in enumerate(ds):
        audio = sample["audio"]
        sr = audio["sampling_rate"]
        samples = np.array(audio["array"], dtype=np.float32)
        duration = len(samples) / sr
        
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue
        
        out_path = os.path.join(OUTDIR, f"{lang_name}_{collected:02d}.wav")
        tmp_path = out_path + ".tmp"
        
        # Write original rate as 16-bit PCM
        with wave.open(tmp_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            pcm = (samples * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        
        # Resample to 24k with ffmpeg
        ret = os.system(f"ffmpeg -y -i {tmp_path} -ar 24000 -ac 1 -sample_fmt s16 {out_path} 2>/dev/null")
        os.remove(tmp_path)
        if ret != 0:
            print(f"  ffmpeg failed for sample {i}")
            continue
        
        text = sample.get("transcription", sample.get("text", ""))
        entry = {
            "file": f"{lang_name}_{collected:02d}.wav",
            "text": text,
            "language": lang_name,
            "lang_code": lang_code,
            "duration_s": round(duration, 2),
        }
        manifest.append(entry)
        print(f"  [{lang_name}] #{collected} ({duration:.1f}s): {text[:70]}...")
        sys.stdout.flush()
        
        collected += 1
        if collected >= SAMPLES_PER_LANG:
            break
    
    print(f"  Collected {collected} for {lang_name}")
    sys.stdout.flush()

with open(os.path.join(OUTDIR, "manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {len(manifest)} samples in {OUTDIR}")
