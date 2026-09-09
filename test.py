import argparse
import os
import time
import wave

import numpy as np
import sherpa_onnx
import sounddevice as sd
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

# Just change this name to test another model
#
# Examples:
#   "indicconformer-hi"
#   "malayalam"
#   "marathi"
#   "telugu"
#   "parakeet-en"
#
MODEL_NAME = "malayalam_16k.wav"

# Root directory containing all models
MODELS_DIR = "models"

# Target sample rate required by the models
TARGET_SAMPLE_RATE = 16000

# Number of CPU threads
NUM_THREADS = 4

# Sherpa execution provider
PROVIDER = "cpu"


# ============================================================
# AUTOMATIC MODEL PATHS
# ============================================================

MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

TOKENS = os.path.join(MODEL_DIR, "tokens.txt")

# Prefer INT8 model if available
MODEL_INT8 = os.path.join(MODEL_DIR, "model.int8.onnx")
MODEL_NORMAL = os.path.join(MODEL_DIR, "model.onnx")


def find_model():
    """
    Automatically find the ONNX model inside the selected
    model directory.
    """

    if os.path.exists(MODEL_INT8):
        return MODEL_INT8

    if os.path.exists(MODEL_NORMAL):
        return MODEL_NORMAL

    # Search for any .onnx file as a fallback
    if os.path.exists(MODEL_DIR):
        onnx_files = [
            file
            for file in os.listdir(MODEL_DIR)
            if file.endswith(".onnx")
        ]

        if len(onnx_files) == 1:
            return os.path.join(MODEL_DIR, onnx_files[0])

        if len(onnx_files) > 1:
            raise FileNotFoundError(
                f"Multiple ONNX models found in '{MODEL_DIR}'.\n"
                f"Please use model.int8.onnx or model.onnx."
            )

    raise FileNotFoundError(
        f"No ONNX model found inside:\n{MODEL_DIR}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generic Sherpa-ONNX speech-to-text model tester"
    )

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--audio",
        help="Path to WAV file"
    )

    source.add_argument(
        "--mic",
        action="store_true",
        help="Record audio from microphone"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Microphone recording duration in seconds"
    )

    return parser.parse_args()


# ============================================================
# CHOOSE AUDIO SOURCE
# ============================================================

def choose_source(args):

    if args.audio:
        return "file", args.audio

    if args.mic:
        return "mic", None

    print("\nChoose audio source:")
    print("  1. WAV file")
    print("  2. Microphone")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":

        filename = input(
            "Enter WAV filename: "
        ).strip()

        return "file", filename

    if choice == "2":

        duration = input(
            "Recording duration in seconds [10]: "
        ).strip()

        if duration:
            args.duration = float(duration)

        return "mic", None

    raise ValueError("Please choose 1 or 2")


# ============================================================
# LOAD WAV FILE
# ============================================================

def load_wav(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Audio file not found:\n{filename}"
        )

    with wave.open(filename, "rb") as wav_file:

        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        raw_audio = wav_file.readframes(
            wav_file.getnframes()
        )

    # Sherpa input will be float32
    # We currently support 16-bit PCM WAV
    if sample_width != 2:

        raise ValueError(
            "WAV must be 16-bit PCM.\n"
            "Convert it using ffmpeg if necessary."
        )

    # int16 -> float32
    audio = (
        np.frombuffer(
            raw_audio,
            dtype=np.int16
        )
        .astype(np.float32)
        / 32768.0
    )

    # Stereo/multi-channel -> mono
    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(axis=1)

    return sample_rate, audio


# ============================================================
# RECORD MICROPHONE
# ============================================================

def record_microphone(duration):

    print(
        f"\nRecording for {duration:.1f} seconds..."
    )

    print("Speak now...")

    recording = sd.rec(
        int(duration * TARGET_SAMPLE_RATE),
        samplerate=TARGET_SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("Recording complete.")

    return TARGET_SAMPLE_RATE, recording[:, 0]


# ============================================================
# RESAMPLE AUDIO
# ============================================================

def normalize_sample_rate(sample_rate, audio):

    if sample_rate == TARGET_SAMPLE_RATE:
        return audio

    print(
        f"\nResampling audio: "
        f"{sample_rate} Hz -> "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        sample_rate
    )

    return audio.astype(np.float32)


# ============================================================
# CHECK MODEL FILES
# ============================================================

def check_model_files(model_path):

    print("\nModel files:")

    print(
        f"  Model directory : {MODEL_DIR}"
    )

    print(
        f"  ONNX model      : {model_path}"
    )

    print(
        f"  Tokens           : {TOKENS}"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file does not exist:\n{model_path}"
        )

    if not os.path.exists(TOKENS):

        raise FileNotFoundError(
            f"tokens.txt does not exist:\n{TOKENS}"
        )


# ============================================================
# LOAD SHERPA-ONNX MODEL
# ============================================================

def load_recognizer(model_path):

    print("\nLoading model...")

    load_start = time.perf_counter()

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(

        model=model_path,

        tokens=TOKENS,

        num_threads=NUM_THREADS,

        sample_rate=TARGET_SAMPLE_RATE,

        feature_dim=80,

        decoding_method="greedy_search",

        provider=PROVIDER,
    )

    load_time = time.perf_counter() - load_start

    print(
        f"Model loaded in: "
        f"{load_time:.2f} sec"
    )

    return recognizer, load_time


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe(
    recognizer,
    audio
):

    stream = recognizer.create_stream()

    stream.accept_waveform(
        TARGET_SAMPLE_RATE,
        audio
    )

    print("\nRunning transcription...")

    start = time.perf_counter()

    recognizer.decode_stream(stream)

    inference_time = (
        time.perf_counter() - start
    )

    text = stream.result.text

    return text, inference_time


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    text,
    inference_time,
    audio_duration,
    load_time
):

    print("\n")
    print("=" * 60)
    print("TRANSCRIPTION")
    print("=" * 60)

    print(text)

    print("\n")
    print("=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    print(
        f"Audio duration : "
        f"{audio_duration:.2f} sec"
    )

    print(
        f"Inference time : "
        f"{inference_time:.2f} sec"
    )

    # Real-Time Factor
    rtf = inference_time / audio_duration

    # How many times faster/slower than real-time
    speed = audio_duration / inference_time

    print(
        f"RTF            : "
        f"{rtf:.4f}"
    )

    print(
        f"Real-time speed: "
        f"{speed:.2f}x"
    )

    print(
        f"Model load time: "
        f"{load_time:.2f} sec"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("=" * 60)
    print("GENERIC SHERPA-ONNX SPEECH-TO-TEXT TESTER")
    print("=" * 60)

    print(
        f"\nSelected model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Find model
    # --------------------------------------------------------

    model_path = find_model()

    check_model_files(model_path)

    # --------------------------------------------------------
    # Load recognizer
    # --------------------------------------------------------

    recognizer, load_time = load_recognizer(
        model_path
    )

    # --------------------------------------------------------
    # Choose audio
    # --------------------------------------------------------

    source_type, audio_filename = choose_source(
        args
    )

    # --------------------------------------------------------
    # Get audio
    # --------------------------------------------------------

    if source_type == "file":

        sample_rate, audio = load_wav(
            audio_filename
        )

    else:

        sample_rate, audio = record_microphone(
            args.duration
        )

    # --------------------------------------------------------
    # Normalize sample rate
    # --------------------------------------------------------

    audio = normalize_sample_rate(
        sample_rate,
        audio
    )

    audio_duration = (
        len(audio) / TARGET_SAMPLE_RATE
    )

    print("\nAudio information:")
    print(
        f"  Sample rate : "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print("  Channels    : 1")

    print("  Format      : float32")

    print(
        f"  Frames      : "
        f"{len(audio)}"
    )

    print(
        f"  Duration    : "
        f"{audio_duration:.2f} sec"
    )

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    text, inference_time = transcribe(
        recognizer,
        audio
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_results(
        text,
        inference_time,
        audio_duration,
        load_time
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()