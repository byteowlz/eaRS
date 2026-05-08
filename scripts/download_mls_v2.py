#!/usr/bin/env python3
"""Download MLS multilingual samples via HF datasets server API. v2"""
import os, json, subprocess, requests, sys

OUTDIR = os.path.expanduser("~/byteowlz/eaRS/test_audio/hub")
API = "https://datasets-server.huggingface.co"

languages = [
    ("german", "de"),
    ("french", "fr"),
    ("spanish", "es"),
    ("italian", "it"),
    ("portuguese", "pt"),
    ("dutch", "nl"),
    ("polish", "pl"),
]

# Load existing manifest
manifest_path = os.path.join(OUTDIR, "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
else:
    manifest = []

existing_langs = {e["language"] for e in manifest}

for lang_name, lang_code in languages:
    if lang_name in existing_langs:
        print(f"\n  {lang_name}: already downloaded, skipping")
        continue
        
    print(f"\n=== {lang_name.upper()} (MLS) ===")
    sys.stdout.flush()
    
    # Get more rows since some may be too short
    try:
        resp = requests.get(
            f"{API}/rows?dataset=facebook/multilingual_librispeech&config={lang_name}&split=test&offset=0&length=100",
            timeout=30
        )
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
        print(f"  Got {len(rows)} rows from API")
        sys.stdout.flush()
    except Exception as e:
        print(f"  API error: {e}")
        continue
    
    collected = 0
    for row in rows:
        if collected >= 10:
            break
        row_data = row.get("row", {})
        
        # MLS uses 'transcript' field
        text = row_data.get("transcript", row_data.get("text", ""))
        if not text or len(text.strip()) < 10:
            continue
        
        audio_info = row_data.get("audio", [])
        if isinstance(audio_info, list):
            audio_src = audio_info[0].get("src", "") if audio_info else ""
        elif isinstance(audio_info, dict):
            audio_src = audio_info.get("src", "")
        else:
            continue
        
        if not audio_src:
            continue
        
        # Check duration from metadata
        audio_dur = row_data.get("audio_duration", 0)
        if audio_dur and (audio_dur < 3.0 or audio_dur > 30.0):
            continue
        
        out_path = os.path.join(OUTDIR, f"{lang_name}_{collected:02d}.wav")
        tmp_path = out_path + ".tmp.opus"
        
        try:
            # Download audio
            r = requests.get(audio_src, timeout=60)
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                f.write(r.content)
            
            # Resample to 24k WAV
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_path],
                capture_output=True
            )
            os.remove(tmp_path)
            if result.returncode != 0:
                print(f"  ffmpeg failed: {result.stderr.decode()[:100]}")
                continue
            
            # Get actual duration
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", out_path],
                capture_output=True, text=True
            )
            duration = float(probe.stdout.strip())
            
            if duration < 3.0 or duration > 30.0:
                os.remove(out_path)
                continue
            
            entry = {
                "file": f"{lang_name}_{collected:02d}.wav",
                "text": text.strip(),
                "language": lang_name,
                "lang_code": lang_code,
                "duration_s": round(duration, 2),
            }
            manifest.append(entry)
            print(f"  {lang_name}_{collected:02d} ({duration:.1f}s): {text.strip()[:60]}...")
            sys.stdout.flush()
            collected += 1
        except Exception as e:
            print(f"  Skip: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Save manifest
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

# Summary
by_lang = {}
for e in manifest:
    by_lang[e["language"]] = by_lang.get(e["language"], 0) + 1
print(f"\nTotal: {len(manifest)} samples")
for lang, count in sorted(by_lang.items()):
    print(f"  {lang}: {count}")
