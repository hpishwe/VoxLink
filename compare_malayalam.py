from pathlib import Path
import time
import wave
import statistics
import psutil

from piper import PiperVoice


MODELS = {
    "Arjun": Path("models/malayalam/ml_IN-arjun-medium.onnx"),
    "Meera": Path("models/malayalam/ml_IN-meera-medium.onnx"),
}


SENTENCES = [
    "എനിക്ക് സഹായം ആവശ്യമാണ്.",
    "ദയവായി ഉടൻ സഹായം അയയ്ക്കുക.",
    "ഞാൻ സുരക്ഷിതനാണ്.",
    "എനിക്ക് പരിക്കേറ്റു.",
    "ഞങ്ങളുടെ സ്ഥലം സെക്ടർ അഞ്ചിന് സമീപമാണ്.",
    "തീ പടർന്നിരിക്കുന്നു.",
    "ഉടൻ സഹായം ആവശ്യമാണ്.",
    "ദയവായി പോലീസിനെ അറിയിക്കുക.",
    "ദയവായി അടുത്തുള്ള ആശുപത്രിയുമായി ബന്ധപ്പെടുക.",
    "ഇത് ഒരു അടിയന്തര സന്ദേശമാണ്.",
]


process = psutil.Process()


def get_ram_mb():
    return process.memory_info().rss / (1024 * 1024)


def get_model_size_mb(model_path):
    return model_path.stat().st_size / (1024 * 1024)


all_results = {}


print("=" * 70)
print("PSS MALAYALAM TTS MODEL COMPARISON")
print("=" * 70)


for model_name, model_path in MODELS.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model_size = get_model_size_mb(model_path)

    print(f"Model size: {model_size:.2f} MB")
    print(f"RAM before loading: {get_ram_mb():.2f} MB")

    start = time.perf_counter()

    voice = PiperVoice.load(str(model_path))

    load_time = time.perf_counter() - start

    ram_after_load = get_ram_mb()

    print(f"Load time: {load_time:.3f} sec")
    print(f"RAM after loading: {ram_after_load:.2f} MB")
    print("\nRunning inference...\n")

    results = []

    for i, sentence in enumerate(SENTENCES, start=1):

        output_dir = Path("output") / "comparison" / model_name.lower()
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

    summary = {
        "model_size": model_size,
        "load_time": load_time,
        "ram_after_load": ram_after_load,
        "peak_ram": max(ram_values),
        "avg_tts": statistics.mean(tts_times),
        "median_tts": statistics.median(tts_times),
        "min_tts": min(tts_times),
        "max_tts": max(tts_times),
        "avg_audio": statistics.mean(audio_durations),
        "avg_rtf": statistics.mean(rtfs),
        "best_rtf": min(rtfs),
        "worst_rtf": max(rtfs),
    }

    all_results[model_name] = summary

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Average TTS time : {summary['avg_tts']:.3f} sec")
    print(f"Average RTF      : {summary['avg_rtf']:.3f}")
    print(f"Peak RAM         : {summary['peak_ram']:.2f} MB")


print("\n\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<12}"
    f"{'Size MB':>10}"
    f"{'Load s':>10}"
    f"{'Avg TTS':>10}"
    f"{'Avg RTF':>10}"
    f"{'Peak RAM':>12}"
)

print("-" * 70)

for model_name, s in all_results.items():

    print(
        f"{model_name:<12}"
        f"{s['model_size']:>10.2f}"
        f"{s['load_time']:>10.3f}"
        f"{s['avg_tts']:>10.3f}"
        f"{s['avg_rtf']:>10.3f}"
        f"{s['peak_ram']:>12.2f}"
    )

print("=" * 70)