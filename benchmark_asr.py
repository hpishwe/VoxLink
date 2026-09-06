import os
import sys
import time
import threading
import psutil
import soundfile as sf
import sherpa_onnx


NUM_THREADS = 2
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3

MODELS = {
    "parakeet": {
        "name": "Parakeet TDT-CTC 110M INT8",
        "model": "./models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/model.int8.onnx",
        "tokens": "./models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/tokens.txt",
        "audio": "./models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/test_wavs/0.wav",
    },

    "indicconformer": {
        "name": "IndicConformer Hindi Large INT8",
        "model": "./models/indicconformer-hi/model.int8.onnx",
        "tokens": "./models/indicconformer-hi/tokens.txt",
        "audio": "./hindi.wav",
    },
}


process = psutil.Process(os.getpid())


def rss_mb():
    """Current process RSS in MB."""
    return process.memory_info().rss / (1024 * 1024)


class MemoryMonitor:
    """Samples process RSS continuously and records peak memory."""

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

        # Capture one final reading
        self.peak = max(self.peak, rss_mb())

        return self.peak


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def benchmark(model_key):

    config = MODELS[model_key]

    print("=" * 70)
    print(f"ASR BENCHMARK: {config['name']}")
    print("=" * 70)

    model_path = config["model"]
    tokens_path = config["tokens"]
    audio_path = config["audio"]

    # ------------------------------------------------------------
    # File information
    # ------------------------------------------------------------

    model_size = file_size_mb(model_path)

    audio, sample_rate = sf.read(audio_path)
    audio_duration = len(audio) / sample_rate

    print(f"\nModel size:       {model_size:.2f} MB")
    print(f"Audio duration:   {audio_duration:.3f} sec")
    print(f"Sample rate:      {sample_rate} Hz")

    # ------------------------------------------------------------
    # Baseline RAM
    # ------------------------------------------------------------

    ram_before = rss_mb()

    print(f"\nProcess RSS before model: {ram_before:.1f} MB")

    # ------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------

    print("\nLoading model...")

    load_monitor = MemoryMonitor()

    load_monitor.start()

    load_start = time.perf_counter()

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model_path,
        tokens=tokens_path,
        num_threads=NUM_THREADS,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )

    load_time = time.perf_counter() - load_start

    peak_during_load = load_monitor.stop()

    ram_after_load = rss_mb()

    print(f"Model load time:             {load_time:.4f} sec")
    print(f"Process RSS after model:     {ram_after_load:.1f} MB")
    print(f"Peak RSS during model load: {peak_during_load:.1f} MB")
    print(f"RSS increase after model:   {ram_after_load - ram_before:.1f} MB")

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    print("\nWarmup...")

    for _ in range(WARMUP_RUNS):
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)

    # ------------------------------------------------------------
    # Benchmark inference
    # ------------------------------------------------------------

    print("\nBenchmarking inference...")

    inference_times = []
    peak_inference_rss = ram_after_load

    for i in range(BENCHMARK_RUNS):

        monitor = MemoryMonitor()

        monitor.start()

        start = time.perf_counter()

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)

        elapsed = time.perf_counter() - start

        peak = monitor.stop()

        inference_times.append(elapsed)
        peak_inference_rss = max(peak_inference_rss, peak)

        print(f"Run {i + 1}: {elapsed:.4f} sec | Peak RSS: {peak:.1f} MB")

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    average_inference = sum(inference_times) / len(inference_times)
    fastest = min(inference_times)
    slowest = max(inference_times)

    rtf = average_inference / audio_duration
    realtime_speed = audio_duration / average_inference

    final_rss = rss_mb()

    # ------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)

    transcript = stream.result.text

    # ------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nModel:                  {config['name']}")
    print(f"Model size:             {model_size:.2f} MB")
    print(f"Audio duration:         {audio_duration:.3f} sec")

    print(f"\nModel load time:        {load_time:.4f} sec")

    print("\nInference:")
    for i, t in enumerate(inference_times):
        print(f"  Run {i + 1}:             {t:.4f} sec")

    print(f"  Average:              {average_inference:.4f} sec")
    print(f"  Fastest:              {fastest:.4f} sec")
    print(f"  Slowest:              {slowest:.4f} sec")

    print(f"\nRTF:                    {rtf:.4f}")
    print(f"Real-time speed:        {realtime_speed:.2f}x")

    print("\nMemory (Process RSS):")
    print(f"  Before model:         {ram_before:.1f} MB")
    print(f"  After model:          {ram_after_load:.1f} MB")
    print(f"  Increase:             {ram_after_load - ram_before:.1f} MB")
    print(f"  Peak during load:     {peak_during_load:.1f} MB")
    print(f"  Peak during inference:{peak_inference_rss:.1f} MB")
    print(f"  Final RSS:            {final_rss:.1f} MB")

    print("\nTranscript:")
    print(transcript)

    print("\n" + "=" * 70)


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python benchmark_asr.py parakeet")
        print("  python benchmark_asr.py indicconformer")
        sys.exit(1)

    model_key = sys.argv[1].lower()

    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print("Choose: parakeet or indicconformer")
        sys.exit(1)

    benchmark(model_key)


if __name__ == "__main__":
    main()