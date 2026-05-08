#!/usr/bin/env python3
"""Download test speech samples directly from HuggingFace Hub.
Uses LibriSpeech (English) and VoxPopuli (multilingual) via hf_hub.
"""
import os, json, subprocess, sys

OUTDIR = os.path.expanduser("~/byteowlz/eaRS/test_audio/hub")
os.makedirs(OUTDIR, exist_ok=True)

# Use hf command to download specific files
# LibriSpeech test-clean has speaker/chapter/file structure
# Let's use a simpler approach: download from parquet-based datasets directly

import requests
from pathlib import Path

def download_file(url, dest):
    """Download a file with progress."""
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloading: {pct:.0f}%", end="", flush=True)
    if total:
        print()

def get_hf_api(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

manifest = []

# ============================================================
# ENGLISH: LibriSpeech test-clean via HF datasets API
# ============================================================
print("\n=== English (LibriSpeech) ===")

# Use the HF datasets server API to get rows from parquet
API_BASE = "https://datasets-server.huggingface.co"

# First check what's available
try:
    info = get_hf_api(f"{API_BASE}/info?dataset=openslr/librispeech_asr")
    configs = info.get("dataset_info", {}).get("configs", [])
    print(f"  Available configs: {[c['config_name'] for c in configs[:5]]}")
except Exception as e:
    print(f"  API error: {e}")

# Get rows from test-clean split
try:
    rows = get_hf_api(f"{API_BASE}/rows?dataset=openslr/librispeech_asr&config=clean&split=test&offset=0&length=30")
    rows_list = rows.get("rows", [])
    print(f"  Got {len(rows_list)} rows from API")
except Exception as e:
    print(f"  Row fetch error: {e}")
    rows_list = []

collected_en = 0
for row in rows_list:
    if collected_en >= 10:
        break
    row_data = row.get("row", {})
    audio_info = row_data.get("audio", [])
    
    # audio field has src (URL) and sampling_rate
    if isinstance(audio_info, list):
        audio_src = audio_info[0].get("src", "") if audio_info else ""
    elif isinstance(audio_info, dict):
        audio_src = audio_info.get("src", "")
    else:
        continue
    
    text = row_data.get("text", "")
    if not audio_src or not text:
        continue
    
    # Download
    out_path = os.path.join(OUTDIR, f"english_{collected_en:02d}.wav")
    tmp_path = out_path + ".flac"
    
    try:
        download_file(audio_src, tmp_path)
        # Resample to 24k
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_path],
            capture_output=True, check=True
        )
        os.remove(tmp_path)
        
        # Get duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", out_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        
        if duration < 3.0 or duration > 30.0:
            os.remove(out_path)
            continue
        
        entry = {
            "file": f"english_{collected_en:02d}.wav",
            "text": text,
            "language": "english",
            "lang_code": "en",
            "duration_s": round(duration, 2),
        }
        manifest.append(entry)
        print(f"  english_{collected_en:02d} ({duration:.1f}s): {text[:60]}...")
        collected_en += 1
    except Exception as e:
        print(f"  Skip: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================
# GERMAN: VoxPopuli 
# ============================================================
print("\n=== German (VoxPopuli) ===")
try:
    rows = get_hf_api(f"{API_BASE}/rows?dataset=facebook/voxpopuli&config=de&split=test&offset=0&length=30")
    rows_list = rows.get("rows", [])
    print(f"  Got {len(rows_list)} rows from API")
except Exception as e:
    print(f"  Row fetch error: {e}")
    rows_list = []

collected_de = 0
for row in rows_list:
    if collected_de >= 10:
        break
    row_data = row.get("row", {})
    audio_info = row_data.get("audio", [])
    
    if isinstance(audio_info, list):
        audio_src = audio_info[0].get("src", "") if audio_info else ""
    elif isinstance(audio_info, dict):
        audio_src = audio_info.get("src", "")
    else:
        continue
    
    text = row_data.get("normalized_text", row_data.get("text", ""))
    if not audio_src or not text:
        continue
    
    out_path = os.path.join(OUTDIR, f"german_{collected_de:02d}.wav")
    tmp_path = out_path + ".tmp.wav"
    
    try:
        download_file(audio_src, tmp_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_path],
            capture_output=True, check=True
        )
        os.remove(tmp_path)
        
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", out_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        
        if duration < 3.0 or duration > 30.0:
            os.remove(out_path)
            continue
        
        entry = {
            "file": f"german_{collected_de:02d}.wav",
            "text": text,
            "language": "german",
            "lang_code": "de",
            "duration_s": round(duration, 2),
        }
        manifest.append(entry)
        print(f"  german_{collected_de:02d} ({duration:.1f}s): {text[:60]}...")
        collected_de += 1
    except Exception as e:
        print(f"  Skip: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================
# FRENCH: VoxPopuli
# ============================================================
print("\n=== French (VoxPopuli) ===")
try:
    rows = get_hf_api(f"{API_BASE}/rows?dataset=facebook/voxpopuli&config=fr&split=test&offset=0&length=30")
    rows_list = rows.get("rows", [])
    print(f"  Got {len(rows_list)} rows from API")
except Exception as e:
    print(f"  Row fetch error: {e}")
    rows_list = []

collected_fr = 0
for row in rows_list:
    if collected_fr >= 10:
        break
    row_data = row.get("row", {})
    audio_info = row_data.get("audio", [])
    
    if isinstance(audio_info, list):
        audio_src = audio_info[0].get("src", "") if audio_info else ""
    elif isinstance(audio_info, dict):
        audio_src = audio_info.get("src", "")
    else:
        continue
    
    text = row_data.get("normalized_text", row_data.get("text", ""))
    if not audio_src or not text:
        continue
    
    out_path = os.path.join(OUTDIR, f"french_{collected_fr:02d}.wav")
    tmp_path = out_path + ".tmp.wav"
    
    try:
        download_file(audio_src, tmp_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_path],
            capture_output=True, check=True
        )
        os.remove(tmp_path)
        
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", out_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        
        if duration < 3.0 or duration > 30.0:
            os.remove(out_path)
            continue
        
        entry = {
            "file": f"french_{collected_fr:02d}.wav",
            "text": text,
            "language": "french",
            "lang_code": "fr",
            "duration_s": round(duration, 2),
        }
        manifest.append(entry)
        print(f"  french_{collected_fr:02d} ({duration:.1f}s): {text[:60]}...")
        collected_fr += 1
    except Exception as e:
        print(f"  Skip: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================
# SPANISH: VoxPopuli
# ============================================================
print("\n=== Spanish (VoxPopuli) ===")
try:
    rows = get_hf_api(f"{API_BASE}/rows?dataset=facebook/voxpopuli&config=es&split=test&offset=0&length=30")
    rows_list = rows.get("rows", [])
    print(f"  Got {len(rows_list)} rows from API")
except Exception as e:
    print(f"  Row fetch error: {e}")
    rows_list = []

collected_es = 0
for row in rows_list:
    if collected_es >= 10:
        break
    row_data = row.get("row", {})
    audio_info = row_data.get("audio", [])
    
    if isinstance(audio_info, list):
        audio_src = audio_info[0].get("src", "") if audio_info else ""
    elif isinstance(audio_info, dict):
        audio_src = audio_info.get("src", "")
    else:
        continue
    
    text = row_data.get("normalized_text", row_data.get("text", ""))
    if not audio_src or not text:
        continue
    
    out_path = os.path.join(OUTDIR, f"spanish_{collected_es:02d}.wav")
    tmp_path = out_path + ".tmp.wav"
    
    try:
        download_file(audio_src, tmp_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_path],
            capture_output=True, check=True
        )
        os.remove(tmp_path)
        
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", out_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
        
        if duration < 3.0 or duration > 30.0:
            os.remove(out_path)
            continue
        
        entry = {
            "file": f"spanish_{collected_es:02d}.wav",
            "text": text,
            "language": "spanish",
            "lang_code": "es",
            "duration_s": round(duration, 2),
        }
        manifest.append(entry)
        print(f"  spanish_{collected_es:02d} ({duration:.1f}s): {text[:60]}...")
        collected_es += 1
    except Exception as e:
        print(f"  Skip: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Save manifest
with open(os.path.join(OUTDIR, "manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {len(manifest)} samples in {OUTDIR}")
print(f"  English: {collected_en}, German: {collected_de}, French: {collected_fr}, Spanish: {collected_es}")
