from pathlib import Path
import time
import wave
import statistics
import psutil

from piper import PiperVoice


MODEL = Path("models/marathi/mr_IN-google-medium.onnx")

SENTENCES = [
    "मला मदतीची आवश्यकता आहे.",
    "कृपया त्वरित मदत पाठवा.",
    "मी सुरक्षित आहे.",
    "मला दुखापत झाली आहे.",
    "आमचे ठिकाण सेक्टर पाचच्या जवळ आहे.",
    "आग लागली आहे.",
    "तातडीने मदतीची आवश्यकता आहे.",
    "कृपया पोलिसांना कळवा.",
    "कृपया जवळच्या रुग्णालयाशी संपर्क साधा.",
    "हा एक आपत्कालीन संदेश आहे.",
]


process = psutil.Process()


def get_ram_mb():
    return process.memory_info().rss / (1024 * 1024)


model_size = MODEL.stat().st_size / (1024 * 1024)

print("=" * 70)
print("PSS MARATHI TTS BENCHMARK")
print("=" * 70)

print(f"Model size: {model_size:.2f} MB")
print(f"RAM before loading: {get_ram_mb():.2f} MB")

print("\nLoading model...")

start = time.perf_counter()

voice = PiperVoice.load(str(MODEL))

load_time = time.perf_counter() - start
ram_after_load = get_ram_mb()

print(f"Load time: {load_time:.3f} sec")
print(f"RAM after loading: {ram_after_load:.2f} MB")

print("\nRunning inference...\n")

results = []

for i, sentence in enumerate(SENTENCES, start=1):

    output_dir = Path("output") / "benchmark" / "marathi"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{i:02d}.wav"

    start = time.perf_counter()

    with wave.open(str(output_file), "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file)

    tts_time = time.perf_counter() - start

    with wave.open(str(output_file), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

    audio_duration = frames / sample_rate
    rtf = tts_time / audio_duration
    ram = get_ram_mb()

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


tts_times = [r["tts_time"] for r in results]
audio_durations = [r["audio_duration"] for r in results]
rtfs = [r["rtf"] for r in results]
ram_values = [r["ram"] for r in results]

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

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
print(f"Peak RAM               : {max(ram_values):.2f} MB")

print("=" * 70)