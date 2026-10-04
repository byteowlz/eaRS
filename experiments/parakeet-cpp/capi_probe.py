"""Real native ownership/error/streaming smoke tests; no production ctypes adapter."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import time

import numpy as np
import soundfile as sf

OPTIONS = json.dumps({"threshold": 0.5, "min_pause": 0.2, "min_speech": 0.1,
                      "speech_pad": 0, "probabilities": True}).encode()


class Native:
    def __init__(self, library, model):
        self.lib = ct.CDLL(str(library.resolve()))
        signatures = {
            "load": (ct.c_void_p, [ct.c_char_p]),
            "free": (None, [ct.c_void_p]),
            "free_string": (None, [ct.c_void_p]),
            "last_error": (ct.c_char_p, [ct.c_void_p]),
            "vad_path_json": (ct.c_void_p, [ct.c_void_p, ct.c_char_p, ct.c_char_p]),
            "transcribe_path_json": (ct.c_void_p, [ct.c_void_p, ct.c_char_p, ct.c_int]),
            "vad_stream_begin": (ct.c_void_p, [ct.c_void_p, ct.c_int, ct.c_char_p]),
            "vad_stream_free": (None, [ct.c_void_p]),
            "vad_stream_feed_json": (ct.c_void_p, [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int, ct.c_int]),
        }
        for name, (result, arguments) in signatures.items():
            fn = getattr(self.lib, "parakeet_capi_" + name)
            fn.restype, fn.argtypes = result, arguments
            setattr(self, name, fn)
        self.ctx = self.load(str(model.resolve()).encode())
        if not self.ctx:
            raise RuntimeError(f"Native load failed: {model}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free(self.ctx)
        self.ctx = None

    def document(self, pointer):
        if not pointer:
            raise RuntimeError(self.last_error(self.ctx).decode())
        try:
            return json.loads(ct.string_at(pointer))
        finally:
            self.free_string(pointer)

    def vad(self, path, options=OPTIONS):
        return self.document(self.vad_path_json(self.ctx, str(path.resolve()).encode(), options))


def reject(native, operation):
    pointer = operation()
    if pointer:
        native.free_string(pointer)
        raise AssertionError("Expected native rejection")
    message = native.last_error(native.ctx).decode()
    if not message:
        raise AssertionError("Native rejection must describe the error")
    return message


def streaming_probe(native, path, whole):
    stream = native.vad_stream_begin(native.ctx, 16000, OPTIONS)
    if not stream:
        return {"supported": False, "error": native.last_error(native.ctx).decode()}
    probabilities, events = [], []
    samples, sr = sf.read(path, dtype="float32")
    if sr != 16000 or samples.ndim != 1:
        native.vad_stream_free(stream)
        raise ValueError("Streaming probe requires mono 16kHz")
    try:
        for offset in range(0, len(samples), 511):
            chunk = np.ascontiguousarray(samples[offset:offset + 511])
            document = native.document(native.vad_stream_feed_json(
                stream, chunk.ctypes.data_as(ct.POINTER(ct.c_float)), len(chunk),
                int(offset + len(chunk) == len(samples))))
            probabilities.extend(document["probabilities"])
            events.extend(document["events"])
        if not np.allclose(probabilities, whole["probabilities"], atol=0.0001, rtol=0):
            raise AssertionError("Partitioned probabilities differ from clip-level output")
        return {"supported": True, "partition_probability_parity": True, "events": events}
    finally:
        native.vad_stream_free(stream)


def probe(args):
    with Native(args.library, args.model) as native:
        # The next successful call on the same context must work after each error.
        errors = {
            "bad_options": reject(native, lambda: native.vad_path_json(
                native.ctx, str(args.audio.resolve()).encode(), b'{"threshold":2}')),
            "missing_audio": reject(native, lambda: native.vad_path_json(
                native.ctx, b"/nonexistent/ears-spike.wav", OPTIONS)),
            "vad_only_asr": reject(native, lambda: native.transcribe_path_json(
                native.ctx, str(args.audio.resolve()).encode(), 2)),
        }
        whole = native.vad(args.audio)  # One untimed warm-up after errors.
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            result = native.vad(args.audio)
            timings.append((time.perf_counter() - start) * 1000)
            if result != whole:
                raise AssertionError("Repeat VAD output changed")
        stream = streaming_probe(native, args.audio, whole)
        return {"model": args.model.name, "audio": args.audio.name, "errors": errors,
                "recovery": True, "warm_median_ms": statistics.median(timings),
                "thread_budget": "native library default 8 (not CLI benchmark 4)",
                "streaming_vad": stream, "output": {key: value for key, value in whole.items()
                                                       if key != "probabilities"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args), indent=2))


if __name__ == "__main__":
    main()
