import argparse
import os
import sys
import time
import threading

import numpy as np
import psutil
import soundfile as sf
import sherpa_onnx
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = "./models"

TARGET_SAMPLE_RATE = 16000

NUM_THREADS = 2

WARMUP_RUNS = 1

BENCHMARK_RUNS = 3

FEATURE_DIM = 80

PROVIDER = "cpu"


# ============================================================
# PROCESS / MEMORY
# ============================================================

process = psutil.Process(os.getpid())


def rss_mb():
    """
    Return current process RSS memory in MB.
    """

    return process.memory_info().rss / (1024 * 1024)


class MemoryMonitor:
    """
    Continuously monitors process RSS and records
    the maximum memory usage.
    """

    def __init__(self, interval=0.01):

        self.interval = interval

        self.running = False

        self.peak = 0

        self.thread = None

    def _monitor(self):

        while self.running:

            current = rss_mb()

            if current > self.peak:
                self.peak = current

            time.sleep(self.interval)

    def start(self):

        self.peak = rss_mb()

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join()

        self.peak = max(
            self.peak,
            rss_mb()
        )

        return self.peak


# ============================================================
# FILE UTILITIES
# ============================================================

def file_size_mb(path):

    return os.path.getsize(path) / (
        1024 * 1024
    )


def find_files(root, extension):

    """
    Recursively find all files with a given extension.
    """

    results = []

    if not os.path.exists(root):
        return results

    for current_root, dirs, files in os.walk(root):

        for filename in files:

            if filename.lower().endswith(extension):

                results.append(
                    os.path.join(
                        current_root,
                        filename
                    )
                )

    return sorted(results)


# ============================================================
# MODEL DISCOVERY
# ============================================================

def discover_models():

    """
    Automatically discover ONNX models inside models/.

    Every .onnx file is considered a possible model.
    """

    onnx_files = find_files(
        MODELS_DIR,
        ".onnx"
    )

    models = []

    for model_path in onnx_files:

        model_dir = os.path.dirname(
            model_path
        )

        model_filename = os.path.basename(
            model_path
        )

        # ----------------------------------------------------
        # Find tokens.txt
        # ----------------------------------------------------

        tokens_path = None

        # First look beside the model
        candidate = os.path.join(
            model_dir,
            "tokens.txt"
        )

        if os.path.exists(candidate):

            tokens_path = candidate

        else:

            # Search parent directories
            current = model_dir

            for _ in range(3):

                parent_candidate = os.path.join(
                    current,
                    "tokens.txt"
                )

                if os.path.exists(
                    parent_candidate
                ):

                    tokens_path = parent_candidate

                    break

                parent = os.path.dirname(
                    current
                )

                if parent == current:
                    break

                current = parent

        # ----------------------------------------------------
        # Model name
        # ----------------------------------------------------

        relative_model = os.path.relpath(
            model_path,
            MODELS_DIR
        )

        # Use model's parent folder as name
        model_name = os.path.basename(
            model_dir
        )

        # ----------------------------------------------------
        # Find WAV files near model
        # ----------------------------------------------------

        audio_files = find_files(
            model_dir,
            ".wav"
        )

        # ----------------------------------------------------
        # Create model config
        # ----------------------------------------------------

        config = {
            "name": model_name,
            "model": model_path,
            "tokens": tokens_path,
            "audio": (
                audio_files[0]
                if audio_files
                else None
            ),
            "relative_path": relative_model,
        }

        models.append(config)

    return models


# ============================================================
# PRINT DISCOVERED MODELS
# ============================================================

def print_available_models(models):

    print("\nAvailable models:")
    print("-" * 70)

    for index, config in enumerate(
        models,
        start=1
    ):

        print(
            f"{index}. "
            f"{config['name']}"
        )

        print(
            f"   Model: "
            f"{config['relative_path']}"
        )

        if config["tokens"]:

            print(
                f"   Tokens: "
                f"{os.path.relpath(config['tokens'])}"
            )

        else:

            print(
                "   Tokens: NOT FOUND"
            )

        if config["audio"]:

            print(
                f"   Audio: "
                f"{os.path.relpath(config['audio'])}"
            )

        else:

            print(
                "   Audio: NOT FOUND"
            )

        print()


# ============================================================
# FIND MODEL BY NAME
# ============================================================

def get_model(
    models,
    model_name
):

    model_name = model_name.lower()

    # --------------------------------------------------------
    # Exact model-name match
    # --------------------------------------------------------

    for config in models:

        if config["name"].lower() == model_name:

            return config

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    matches = []

    for config in models:

        if (
            model_name in
            config["name"].lower()
        ):

            matches.append(config)

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        print(
            f"\nMultiple models match '{model_name}':"
        )

        for config in matches:

            print(
                f"  - {config['name']}"
            )

        raise ValueError(
            "Please use a more specific model name."
        )

    raise ValueError(
        f"Model '{model_name}' not found."
    )


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path):

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{audio_path}"
        )

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    # --------------------------------------------------------
    # Stereo -> Mono
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = audio.mean(
            axis=1
        )

    return sample_rate, audio


# ============================================================
# RESAMPLE AUDIO
# ============================================================

def normalize_sample_rate(
    sample_rate,
    audio
):

    if sample_rate == TARGET_SAMPLE_RATE:

        return audio

    print(
        f"\nResampling audio:"
        f" {sample_rate} Hz"
        f" -> {TARGET_SAMPLE_RATE} Hz"
    )

    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        sample_rate
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_recognizer(
    model_path,
    tokens_path
):

    print("\nLoading model...")

    monitor = MemoryMonitor()

    monitor.start()

    start = time.perf_counter()

    recognizer = (
        sherpa_onnx
        .OfflineRecognizer
        .from_nemo_ctc(

            model=model_path,

            tokens=tokens_path,

            num_threads=NUM_THREADS,

            sample_rate=TARGET_SAMPLE_RATE,

            feature_dim=FEATURE_DIM,

            decoding_method="greedy_search",

            provider=PROVIDER,
        )
    )

    load_time = (
        time.perf_counter()
        - start
    )

    peak_memory = monitor.stop()

    return (
        recognizer,
        load_time,
        peak_memory
    )


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe(
    recognizer,
    audio
):

    stream = (
        recognizer
        .create_stream()
    )

    stream.accept_waveform(
        TARGET_SAMPLE_RATE,
        audio
    )

    recognizer.decode_stream(
        stream
    )

    return stream.result.text


# ============================================================
# BENCHMARK MODEL
# ============================================================

def benchmark(
    config,
    audio_override=None
):

    print("\n")
    print("=" * 70)

    print(
        f"ASR BENCHMARK: "
        f"{config['name']}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = config["model"]

    tokens_path = config["tokens"]

    audio_path = (
        audio_override
        if audio_override
        else config["audio"]
    )

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}"
        )

    if not tokens_path:

        raise FileNotFoundError(
            f"tokens.txt could not be found "
            f"for model:\n{model_path}"
        )

    if not os.path.exists(
        tokens_path
    ):

        raise FileNotFoundError(
            f"Tokens file not found:\n"
            f"{tokens_path}"
        )

    # --------------------------------------------------------
    # Validate audio
    # --------------------------------------------------------

    if not audio_path:

        raise FileNotFoundError(
            "No WAV file was automatically "
            "found for this model.\n"
            "Use --audio path/to/file.wav"
        )

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    model_size = file_size_mb(
        model_path
    )

    sample_rate, audio = load_audio(
        audio_path
    )

    audio = normalize_sample_rate(
        sample_rate,
        audio
    )

    audio_duration = (
        len(audio)
        / TARGET_SAMPLE_RATE
    )

    print(
        f"\nModel:"
        f"        {model_path}"
    )

    print(
        f"Tokens:"
        f"       {tokens_path}"
    )

    print(
        f"Audio:"
        f"        {audio_path}"
    )

    print(
        f"\nModel size:"
        f"       {model_size:.2f} MB"
    )

    print(
        f"Audio duration:"
        f"   {audio_duration:.3f} sec"
    )

    print(
        f"Sample rate:"
        f"      {TARGET_SAMPLE_RATE} Hz"
    )

    # --------------------------------------------------------
    # Baseline RAM
    # --------------------------------------------------------

    ram_before = rss_mb()

    print(
        f"\nProcess RSS before model:"
        f" {ram_before:.1f} MB"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        recognizer,
        load_time,
        peak_during_load
    ) = load_recognizer(
        model_path,
        tokens_path
    )

    ram_after_load = rss_mb()

    print(
        f"Model load time:"
        f"             {load_time:.4f} sec"
    )

    print(
        f"Process RSS after model:"
        f"     {ram_after_load:.1f} MB"
    )

    print(
        f"Peak RSS during model load:"
        f" {peak_during_load:.1f} MB"
    )

    print(
        f"RSS increase after model:"
        f"   {ram_after_load - ram_before:.1f} MB"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"\nWarmup "
        f"({WARMUP_RUNS} run)..."
    )

    for _ in range(
        WARMUP_RUNS
    ):

        transcribe(
            recognizer,
            audio
        )

    # --------------------------------------------------------
    # Benchmark inference
    # --------------------------------------------------------

    print(
        f"\nBenchmarking inference "
        f"({BENCHMARK_RUNS} runs)..."
    )

    inference_times = []

    peak_inference_rss = (
        ram_after_load
    )

    for i in range(
        BENCHMARK_RUNS
    ):

        monitor = MemoryMonitor()

        monitor.start()

        start = time.perf_counter()

        stream = (
            recognizer
            .create_stream()
        )

        stream.accept_waveform(
            TARGET_SAMPLE_RATE,
            audio
        )

        recognizer.decode_stream(
            stream
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        peak = monitor.stop()

        inference_times.append(
            elapsed
        )

        peak_inference_rss = max(
            peak_inference_rss,
            peak
        )

        print(
            f"Run {i + 1}: "
            f"{elapsed:.4f} sec"
            f" | Peak RSS: "
            f"{peak:.1f} MB"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_inference = (
        sum(inference_times)
        / len(inference_times)
    )

    fastest = min(
        inference_times
    )

    slowest = max(
        inference_times
    )

    rtf = (
        average_inference
        / audio_duration
    )

    realtime_speed = (
        audio_duration
        / average_inference
    )

    final_rss = rss_mb()

    # --------------------------------------------------------
    # Final transcription
    # --------------------------------------------------------

    transcript = transcribe(
        recognizer,
        audio
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nModel:"
        f"                  {config['name']}"
    )

    print(
        f"Model size:"
        f"             {model_size:.2f} MB"
    )

    print(
        f"Audio duration:"
        f"         {audio_duration:.3f} sec"
    )

    print(
        f"\nModel load time:"
        f"        {load_time:.4f} sec"
    )

    print("\nInference:")

    for i, t in enumerate(
        inference_times
    ):

        print(
            f"  Run {i + 1}:"
            f"             {t:.4f} sec"
        )

    print(
        f"  Average:"
        f"              {average_inference:.4f} sec"
    )

    print(
        f"  Fastest:"
        f"              {fastest:.4f} sec"
    )

    print(
        f"  Slowest:"
        f"              {slowest:.4f} sec"
    )

    print(
        f"\nRTF:"
        f"                    {rtf:.4f}"
    )

    print(
        f"Real-time speed:"
        f"        {realtime_speed:.2f}x"
    )

    print("\nMemory (Process RSS):")

    print(
        f"  Before model:"
        f"         {ram_before:.1f} MB"
    )

    print(
        f"  After model:"
        f"          {ram_after_load:.1f} MB"
    )

    print(
        f"  Increase:"
        f"             "
        f"{ram_after_load - ram_before:.1f} MB"
    )

    print(
        f"  Peak during load:"
        f"     {peak_during_load:.1f} MB"
    )

    print(
        f"  Peak during inference:"
        f" {peak_inference_rss:.1f} MB"
    )

    print(
        f"  Final RSS:"
        f"            {final_rss:.1f} MB"
    )

    print("\nTranscript:")

    print(transcript)

    print("\n" + "=" * 70)

    return {
        "model": config["name"],
        "model_path": model_path,
        "model_size_mb": model_size,
        "audio_duration_sec": audio_duration,
        "load_time_sec": load_time,
        "average_inference_sec": average_inference,
        "fastest_sec": fastest,
        "slowest_sec": slowest,
        "rtf": rtf,
        "realtime_speed": realtime_speed,
        "ram_before_mb": ram_before,
        "ram_after_load_mb": ram_after_load,
        "ram_increase_mb": (
            ram_after_load
            - ram_before
        ),
        "peak_load_mb": peak_during_load,
        "peak_inference_mb": peak_inference_rss,
        "final_rss_mb": final_rss,
        "transcript": transcript,
    }


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generic Sherpa-ONNX "
            "ASR benchmark"
        )
    )

    parser.add_argument(
        "model",
        nargs="?",
        help=(
            "Model name, or 'all' "
            "to benchmark every model"
        )
    )

    parser.add_argument(
        "--audio",
        help=(
            "WAV file to use instead "
            "of the model's default audio"
        )
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered models"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Discover models
    # --------------------------------------------------------

    models = discover_models()

    if not models:

        print(
            f"No ONNX models found "
            f"inside {MODELS_DIR}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # List models
    # --------------------------------------------------------

    if args.list:

        print_available_models(
            models
        )

        return

    # --------------------------------------------------------
    # Model argument required
    # --------------------------------------------------------

    if not args.model:

        print_available_models(
            models
        )

        print(
            "\nUsage:"
        )

        print(
            "  python benchmark_asr.py MODEL"
        )

        print(
            "  python benchmark_asr.py all"
        )

        print(
            "  python benchmark_asr.py --list"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Benchmark all
    # --------------------------------------------------------

    if args.model.lower() == "all":

        results = []

        for config in models:

            try:

                result = benchmark(
                    config,
                    args.audio
                )

                results.append(
                    result
                )

            except Exception as e:

                print("\n" + "=" * 70)

                print(
                    f"FAILED: "
                    f"{config['name']}"
                )

                print(
                    f"Reason: {e}"
                )

                print("=" * 70)

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if results:

            print("\n")
            print("=" * 100)
            print("BENCHMARK SUMMARY")
            print("=" * 100)

            print(
                f"{'Model':<30}"
                f"{'Size MB':>10}"
                f"{'Load s':>10}"
                f"{'Infer s':>10}"
                f"{'RTF':>10}"
                f"{'Speed':>10}"
                f"{'RAM MB':>10}"
            )

            print("-" * 100)

            for r in results:

                print(
                    f"{r['model'][:29]:<30}"
                    f"{r['model_size_mb']:>10.2f}"
                    f"{r['load_time_sec']:>10.3f}"
                    f"{r['average_inference_sec']:>10.3f}"
                    f"{r['rtf']:>10.4f}"
                    f"{r['realtime_speed']:>9.2f}x"
                    f"{r['ram_increase_mb']:>10.1f}"
                )

            print("=" * 100)

        return

    # --------------------------------------------------------
    # Benchmark selected model
    # --------------------------------------------------------

    try:

        config = get_model(
            models,
            args.model
        )

        benchmark(
            config,
            args.audio
        )

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()