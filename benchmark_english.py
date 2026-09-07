import os
import time
import wave
import psutil
from piper import PiperVoice

MODEL = r"D:\TTS\models\english\en_US-lessac-medium.onnx"
OUTPUT_DIR = r"D:\TTS\benchmarks\english"

sentences = [
    "I need help.",
    "Please send help immediately.",
    "I am safe.",
    "I am injured.",
    "Our location is near Sector Five.",
    "There is a fire.",
    "Immediate assistance is required.",
    "Please inform the police.",
    "Please contact the nearest hospital.",
    "This is an emergency message."
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

process = psutil.Process(os.getpid())

print("=" * 60)
print("ENGLISH TTS BENCHMARK")
print("=" * 60)

# Model size
model_size_mb = os.path.getsize(MODEL) / (1024 * 1024)
print(f"\nModel size: {model_size_mb:.2f} MB")

# RAM before loading
ram_before = process.memory_info().rss / (1024 * 1024)
print(f"RAM before loading: {ram_before:.2f} MB")

# Load model
start = time.perf_counter()
voice = PiperVoice.load(MODEL)
load_time = time.perf_counter() - start

ram_after_load = process.memory_info().rss / (1024 * 1024)

print(f"Model load time: {load_time:.3f} s")
print(f"RAM after loading: {ram_after_load:.2f} MB")

results = []

print("\nRunning 10 sentences...\n")

for i, text in enumerate(sentences, 1):

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{i:02d}.wav"
    )

    start = time.perf_counter()

    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    tts_time = time.perf_counter() - start

    with wave.open(output_file, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

    audio_duration = frames / sample_rate
    rtf = tts_time / audio_duration

    ram_now = process.memory_info().rss / (1024 * 1024)

    results.append({
        "sentence": i,
        "tts_time": tts_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "ram": ram_now
    })

    print(
        f"{i:02d}. "
        f"TTS: {tts_time:.3f}s | "
        f"Audio: {audio_duration:.3f}s | "
        f"RTF: {rtf:.3f} | "
        f"RAM: {ram_now:.2f} MB"
    )

# Summary
avg_tts = sum(x["tts_time"] for x in results) / len(results)
avg_audio = sum(x["audio_duration"] for x in results) / len(results)
avg_rtf = sum(x["rtf"] for x in results) / len(results)
peak_ram = max(x["ram"] for x in results)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Model size       : {model_size_mb:.2f} MB")
print(f"Load time        : {load_time:.3f} s")
print(f"Average TTS time : {avg_tts:.3f} s")
print(f"Average audio    : {avg_audio:.3f} s")
print(f"Average RTF      : {avg_rtf:.3f}")
print(f"Peak RAM         : {peak_ram:.2f} MB")

print("\nBenchmark complete.")
print(f"WAV files saved to: {OUTPUT_DIR}")