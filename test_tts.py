from pathlib import Path
import time
import wave
import os
import statistics

import psutil
from piper import PiperVoice


MODEL = Path("models/hindi/hi_IN-pratham-medium.onnx")
OUTPUT_DIR = Path("output/benchmark")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sentences = [
    "मदद की आवश्यकता है।",
    "कृपया तुरंत सहायता भेजें।",
    "मैं सुरक्षित हूं।",
    "मैं घायल हूं।",
    "हमारा स्थान सेक्टर पांच के पास है।",
    "आग लग गई है।",
    "तुरंत सहायता की आवश्यकता है।",
    "कृपया पुलिस को सूचित करें।",
    "कृपया नजदीकी अस्पताल से संपर्क करें।",
    "यह एक आपातकालीन संदेश है।",
]


process = psutil.Process(os.getpid())


# ----------------------------------------
# Model size
# ----------------------------------------

model_size_mb = MODEL.stat().st_size / (1024 * 1024)

print("=" * 55)
print("PSS TTS BENCHMARK")
print("=" * 55)

print(f"Model size: {model_size_mb:.2f} MB")


# ----------------------------------------
# Load model
# ----------------------------------------

print("\nLoading model...")

load_start = time.perf_counter()

voice = PiperVoice.load(str(MODEL))

load_time = time.perf_counter() - load_start

ram_after_load = process.memory_info().rss / (1024 * 1024)

print(f"Load time: {load_time:.3f} sec")
print(f"RAM after loading: {ram_after_load:.2f} MB")


# ----------------------------------------
# Benchmark
# ----------------------------------------

results = []

peak_ram = ram_after_load

print("\nRunning inference...\n")


for i, text in enumerate(sentences, start=1):

    output_file = OUTPUT_DIR / f"sentence_{i}.wav"

    start = time.perf_counter()

    with wave.open(str(output_file), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    tts_time = time.perf_counter() - start

    # Audio duration
    with wave.open(str(output_file), "rb") as wav_file:

        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

    audio_duration = frames / sample_rate

    # RTF
    rtf = tts_time / audio_duration

    # RAM
    ram = process.memory_info().rss / (1024 * 1024)

    peak_ram = max(peak_ram, ram)

    results.append({
        "tts_time": tts_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "ram": ram,
    })

    print(
        f"{i:02d} | "
        f"TTS: {tts_time:.3f}s | "
        f"Audio: {audio_duration:.3f}s | "
        f"RTF: {rtf:.3f} | "
        f"RAM: {ram:.1f} MB"
    )


# ----------------------------------------
# Statistics
# ----------------------------------------

tts_times = [r["tts_time"] for r in results]
rtfs = [r["rtf"] for r in results]
audio_durations = [r["audio_duration"] for r in results]


print("\n")
print("=" * 55)
print("SUMMARY")
print("=" * 55)

print(f"Sentences tested       : {len(results)}")

print(f"Average TTS time       : {statistics.mean(tts_times):.3f} sec")
print(f"Median TTS time        : {statistics.median(tts_times):.3f} sec")

print(f"Min TTS time           : {min(tts_times):.3f} sec")
print(f"Max TTS time           : {max(tts_times):.3f} sec")

print(f"Average audio duration : {statistics.mean(audio_durations):.3f} sec")

print(f"Average RTF            : {statistics.mean(rtfs):.3f}")
print(f"Best RTF               : {min(rtfs):.3f}")
print(f"Worst RTF              : {max(rtfs):.3f}")

print(f"RAM after loading      : {ram_after_load:.2f} MB")
print(f"Peak RAM               : {peak_ram:.2f} MB")

print("=" * 55)