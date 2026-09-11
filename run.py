import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


# ============================================================
# CONFIGURATION
# ============================================================

# Project root = directory containing this run.py file
PROJECT_DIR = Path(__file__).resolve().parent

# All models are inside this directory
MODELS_DIR = PROJECT_DIR / "models"

# Benchmark reports will be saved here
REPORTS_DIR = PROJECT_DIR / "benchmark_reports"

# Sherpa-ONNX configuration
NUM_THREADS = 2
PROVIDER = "cpu"


# ============================================================
# FIND MODELS
# ============================================================

def find_models():
    """
    Find all model.int8.onnx files inside models/.

    For every model, find its corresponding tokens.txt.

    Returns:
        List of dictionaries containing:
            model_path
            tokens_path
            name
    """

    models = []

    if not MODELS_DIR.exists():
        print(f"\nERROR: Models directory not found:")
        print(MODELS_DIR)
        return models

    # Search recursively for model.int8.onnx
    model_files = list(
        MODELS_DIR.rglob("model.int8.onnx")
    )

    for model_path in model_files:

        tokens_path = find_tokens(model_path)

        if tokens_path is None:
            print(
                f"\nWARNING: No tokens.txt found for:"
                f"\n{model_path}"
            )
            continue

        # Get path relative to models/
        relative_path = model_path.relative_to(MODELS_DIR)

        # Remove filename from path
        model_parts = relative_path.parts[:-1]

        # Example:
        #
        # indicconformer-hi
        #
        # indicconformer / marathi
        #
        if model_parts:
            name = " / ".join(model_parts)
        else:
            name = model_path.parent.name

        models.append(
            {
                "name": name,
                "model_path": model_path,
                "tokens_path": tokens_path,
            }
        )

    # Sort alphabetically
    models.sort(
        key=lambda x: x["name"].lower()
    )

    return models


# ============================================================
# FIND TOKENS
# ============================================================

def find_tokens(model_path):
    """
    Find the correct tokens.txt for the model.

    Rules:
    1. If tokens.txt exists beside the model, use it.
    2. IndicConformer language models use the shared:
       models/indicconformer/Tokens/tokens.txt
    3. Other models use tokens.txt from their own directory
       or immediate model folder.
    """

    # --------------------------------------------------------
    # 1. tokens.txt next to model
    # --------------------------------------------------------

    local_tokens = model_path.parent / "tokens.txt"

    if local_tokens.exists():
        return local_tokens

    # --------------------------------------------------------
    # 2. IndicConformer shared tokens
    # --------------------------------------------------------

    try:
        relative = model_path.relative_to(MODELS_DIR)

        # Example:
        #
        # indicconformer/marathi/model.int8.onnx
        #
        parts = relative.parts

        if len(parts) >= 2:
            family = parts[0]
            language = parts[1]

            if family == "indicconformer":

                shared_tokens = (
                    MODELS_DIR
                    / "indicconformer"
                    / "Tokens"
                    / "tokens.txt"
                )

                if shared_tokens.exists():
                    return shared_tokens

    except ValueError:
        pass

    # --------------------------------------------------------
    # 3. Search immediate parent directories
    # --------------------------------------------------------

    current = model_path.parent

    for _ in range(3):

        candidate = current / "tokens.txt"

        if candidate.exists():
            return candidate

        if current == current.parent:
            break

        current = current.parent

    return None

# ============================================================
# DISPLAY MODELS
# ============================================================

def display_models(models):

    print()
    print("=" * 70)
    print("AVAILABLE ASR MODELS")
    print("=" * 70)

    for index, model in enumerate(models, start=1):

        print()
        print(f"[{index}] {model['name']}")


    print()
    print("[0] Exit")
    print("=" * 70)


# ============================================================
# SELECT MODEL
# ============================================================

def select_model(models):

    while True:

        display_models(models)

        choice = input(
            "\nEnter model number: "
        ).strip()

        try:
            choice = int(choice)

        except ValueError:
            print(
                "\nPlease enter a valid number."
            )
            continue

        if choice == 0:
            return None

        if 1 <= choice <= len(models):
            return models[choice - 1]

        print(
            f"\nPlease enter a number between "
            f"0 and {len(models)}."
        )


# ============================================================
# SELECT AUDIO
# ============================================================

def select_audio():

    print()
    print("=" * 70)
    print("AUDIO FILE")
    print("=" * 70)

    print()
    print("Enter the path of the WAV file.")
    print()
    print("Examples:")
    print("  test.wav")
    print("  models/indicconformer/marathi/Marathi.wav")
    print("  /Users/yourname/Desktop/test.wav")
    print()

    while True:

        audio_input = input(
            "WAV file path: "
        ).strip()

        # Remove quotes if user pasted:
        # "file.wav"
        # or 'file.wav'
        audio_input = audio_input.strip("\"'")

        audio_path = Path(audio_input)

        # If relative path, make it relative
        # to the project directory.
        if not audio_path.is_absolute():
            audio_path = PROJECT_DIR / audio_path

        audio_path = audio_path.resolve()

        if not audio_path.exists():

            print(
                f"\nERROR: File does not exist:"
            )
            print(audio_path)
            print()

            continue

        if audio_path.suffix.lower() != ".wav":

            print(
                "\nERROR: Please provide a WAV file."
            )

            continue

        return audio_path


# ============================================================
# READ WAV
# ============================================================

def read_wav(audio_path):
    """
    Read a 16-bit PCM mono WAV file.

    Returns:
        samples
        sample_rate
        duration
    """

    with wave.open(
        str(audio_path),
        "rb"
    ) as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()

        # ----------------------------------------------------
        # Validate WAV
        # ----------------------------------------------------

        if channels != 1:

            raise ValueError(
                f"Audio must be MONO.\n"
                f"Detected channels: {channels}"
            )

        if sample_width != 2:

            raise ValueError(
                f"Audio must be 16-bit PCM.\n"
                f"Detected sample width: "
                f"{sample_width * 8}-bit"
            )

        raw_data = wav.readframes(
            frame_count
        )

    # Convert int16 -> float32
    samples = np.frombuffer(
        raw_data,
        dtype=np.int16
    ).astype(np.float32)

    samples = samples / 32768.0

    duration = (
        len(samples) / sample_rate
    )

    return (
        samples,
        sample_rate,
        duration
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_info):

    model_path = model_info["model_path"]
    tokens_path = model_info["tokens_path"]

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print()
    print(
        f"Model : {model_path}"
    )

    print(
        f"Tokens: {tokens_path}"
    )

    print()

    start_time = time.perf_counter()

    recognizer = (
        sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_path),
            tokens=str(tokens_path),
            num_threads=NUM_THREADS,
            provider=PROVIDER,
            decoding_method="greedy_search",
        )
    )

    load_time = (
        time.perf_counter() - start_time
    )

    print(
        f"Model loaded successfully."
    )

    print(
        f"Loading time: {load_time:.3f} seconds"
    )

    return (
        recognizer,
        load_time
    )


# ============================================================
# RUN ASR
# ============================================================

def run_asr(
    recognizer,
    audio_path
):

    print()
    print("=" * 70)
    print("RUNNING ASR")
    print("=" * 70)

    # --------------------------------------------------------
    # Read audio
    # --------------------------------------------------------

    print()
    print("Reading audio...")

    (
        samples,
        sample_rate,
        audio_duration
    ) = read_wav(audio_path)

    print(
        f"Sample rate : {sample_rate} Hz"
    )

    print(
        f"Duration    : {audio_duration:.3f} seconds"
    )

    print(
        f"Samples     : {len(samples)}"
    )

    # --------------------------------------------------------
    # Create recognition stream
    # --------------------------------------------------------

    stream = recognizer.create_stream()

    stream.accept_waveform(
        sample_rate,
        samples
    )

    # --------------------------------------------------------
    # Start benchmark timer
    # --------------------------------------------------------

    print()
    print("Transcribing...")

    start_time = time.perf_counter()

    recognizer.decode_stream(
        stream
    )

    inference_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Get transcription
    # --------------------------------------------------------

    transcription = stream.result.text

    # --------------------------------------------------------
    # Calculate benchmark metrics
    # --------------------------------------------------------

    if audio_duration > 0:

        # Real Time Factor
        rtf = (
            inference_time /
            audio_duration
        )

        # How many times faster than real-time
        realtime_speed = (
            audio_duration /
            inference_time
        )

    else:

        rtf = 0
        realtime_speed = 0

    return {
        "sample_rate": sample_rate,
        "audio_duration": audio_duration,
        "inference_time": inference_time,
        "rtf": rtf,
        "realtime_speed": realtime_speed,
        "transcription": transcription,
    }


# ============================================================
# PRINT BENCHMARK REPORT
# ============================================================

def print_report(
    model_info,
    audio_path,
    load_time,
    result
):

    print()
    print()
    print("=" * 70)
    print("                    ASR BENCHMARK REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("MODEL")
    print("-" * 70)

    print(
        f"Name            : {model_info['name']}"
    )

    print(
        f"Model           : "
        f"{model_info['model_path'].relative_to(PROJECT_DIR)}"
    )

    print(
        f"Tokens          : "
        f"{model_info['tokens_path'].relative_to(PROJECT_DIR)}"
    )

    print(
        f"Provider        : {PROVIDER}"
    )

    print(
        f"CPU Threads     : {NUM_THREADS}"
    )

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    print()
    print("AUDIO")
    print("-" * 70)

    print(
        f"File            : "
        f"{audio_path.relative_to(PROJECT_DIR)}"
    )

    print(
        f"Sample Rate     : "
        f"{result['sample_rate']} Hz"
    )

    print(
        f"Audio Duration  : "
        f"{result['audio_duration']:.3f} sec"
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    print()
    print("PERFORMANCE")
    print("-" * 70)

    print(
        f"Model Load Time : "
        f"{load_time:.3f} sec"
    )

    print(
        f"Inference Time  : "
        f"{result['inference_time']:.3f} sec"
    )

    print(
        f"RTF             : "
        f"{result['rtf']:.4f}"
    )

    print(
        f"Real-Time Speed : "
        f"{result['realtime_speed']:.2f}x"
    )

    # --------------------------------------------------------
    # Performance interpretation
    # --------------------------------------------------------

    print()
    print("PERFORMANCE INTERPRETATION")
    print("-" * 70)

    if result["rtf"] < 1:

        print(
            "✓ Faster than real-time"
        )

        print(
            f"  The model processes speech "
            f"{result['realtime_speed']:.2f}x faster "
            f"than real-time."
        )

    elif result["rtf"] == 1:

        print(
            "≈ Real-time performance"
        )

    else:

        print(
            "✗ Slower than real-time"
        )

        print(
            f"  The model requires "
            f"{result['rtf']:.2f} seconds "
            f"to process 1 second of speech."
        )

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    print()
    print("TRANSCRIPTION")
    print("-" * 70)

    if result["transcription"]:

        print(
            result["transcription"]
        )

    else:

        print(
            "[No transcription produced]"
        )

    print()
    print("=" * 70)


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    model_info,
    audio_path,
    load_time,
    result
):

    REPORTS_DIR.mkdir(
        exist_ok=True
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_model_name = (
        model_info["name"]
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    report_path = (
        REPORTS_DIR /
        f"{safe_model_name}_{timestamp}.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "=" * 70 + "\n"
        )

        report.write(
            "ASR BENCHMARK REPORT\n"
        )

        report.write(
            "=" * 70 + "\n\n"
        )

        # Model
        report.write(
            "MODEL\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        report.write(
            f"Name            : "
            f"{model_info['name']}\n"
        )

        report.write(
            f"Model           : "
            f"{model_info['model_path']}\n"
        )

        report.write(
            f"Tokens          : "
            f"{model_info['tokens_path']}\n"
        )

        report.write(
            f"Provider        : "
            f"{PROVIDER}\n"
        )

        report.write(
            f"CPU Threads     : "
            f"{NUM_THREADS}\n\n"
        )

        # Audio
        report.write(
            "AUDIO\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        report.write(
            f"File            : "
            f"{audio_path}\n"
        )

        report.write(
            f"Sample Rate     : "
            f"{result['sample_rate']} Hz\n"
        )

        report.write(
            f"Audio Duration  : "
            f"{result['audio_duration']:.3f} sec\n\n"
        )

        # Performance
        report.write(
            "PERFORMANCE\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        report.write(
            f"Model Load Time : "
            f"{load_time:.3f} sec\n"
        )

        report.write(
            f"Inference Time  : "
            f"{result['inference_time']:.3f} sec\n"
        )

        report.write(
            f"RTF             : "
            f"{result['rtf']:.4f}\n"
        )

        report.write(
            f"Real-Time Speed : "
            f"{result['realtime_speed']:.2f}x\n\n"
        )

        # Transcription
        report.write(
            "TRANSCRIPTION\n"
        )

        report.write(
            "-" * 70 + "\n"
        )

        report.write(
            result["transcription"] + "\n\n"
        )

        report.write(
            "=" * 70 + "\n"
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("          SHERPA-ONNX ASR BENCHMARK TOOL")
    print("=" * 70)

    # --------------------------------------------------------
    # Find models
    # --------------------------------------------------------

    print()
    print("Searching for ASR models...")

    models = find_models()

    if not models:

        print()
        print(
            "ERROR: No supported models found."
        )

        print()
        print(
            f"Expected models inside:"
        )

        print(
            MODELS_DIR
        )

        return

    print(
        f"Found {len(models)} model(s)."
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    model_info = select_model(
        models
    )

    if model_info is None:

        print(
            "\nExiting..."
        )

        return

    # --------------------------------------------------------
    # Select audio
    # --------------------------------------------------------

    audio_path = select_audio()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        recognizer, load_time = (
            load_model(model_info)
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("MODEL LOADING FAILED")
        print("=" * 70)

        print()
        print(error)

        print()

        return

    # --------------------------------------------------------
    # Run ASR
    # --------------------------------------------------------

    try:

        result = run_asr(
            recognizer,
            audio_path
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ASR EXECUTION FAILED")
        print("=" * 70)

        print()
        print(error)

        print()

        return

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(
        model_info,
        audio_path,
        load_time,
        result
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = save_report(
        model_info,
        audio_path,
        load_time,
        result
    )

    print()
    print(
        f"Benchmark report saved to:"
    )

    print(
        f"{report_path}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
