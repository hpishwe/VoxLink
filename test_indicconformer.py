import time
import wave
import numpy as np
import sherpa_onnx


MODEL = "models/indicconformer-hi/model.int8.onnx"
TOKENS = "models/indicconformer-hi/tokens.txt"
AUDIO = "hindi.wav"


print("=" * 60)
print("IndicConformer Hindi INT8 - Sherpa-ONNX Test")
print("=" * 60)

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading model...")

load_start = time.perf_counter()

recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model=MODEL,
    tokens=TOKENS,
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
)

load_time = time.perf_counter() - load_start

print(f"Model loaded in: {load_time:.2f} sec")


# --------------------------------------------------
# Read WAV
# --------------------------------------------------

with wave.open(AUDIO, "rb") as f:
    sample_rate = f.getframerate()
    channels = f.getnchannels()
    sample_width = f.getsampwidth()
    n_frames = f.getnframes()

    raw_audio = f.readframes(n_frames)


audio_duration = n_frames / sample_rate

print("\nAudio:")
print(f"  Sample rate : {sample_rate} Hz")
print(f"  Channels    : {channels}")
print(f"  Sample width: {sample_width * 8}-bit")
print(f"  Frames      : {n_frames}")
print(f"  Duration    : {audio_duration:.2f} sec")


# --------------------------------------------------
# Convert PCM16 → float32
# --------------------------------------------------

audio = np.frombuffer(
    raw_audio,
    dtype=np.int16
).astype(np.float32) / 32768.0


# --------------------------------------------------
# Convert stereo → mono if necessary
# --------------------------------------------------

if channels > 1:
    audio = audio.reshape(-1, channels).mean(axis=1)


# --------------------------------------------------
# Create recognition stream
# --------------------------------------------------

stream = recognizer.create_stream()

stream.accept_waveform(
    sample_rate,
    audio
)


# --------------------------------------------------
# Run inference
# --------------------------------------------------

print("\nRunning transcription...")

start = time.perf_counter()

recognizer.decode_stream(stream)

inference_time = time.perf_counter() - start


# --------------------------------------------------
# Result
# --------------------------------------------------

text = stream.result.text

print("\n" + "=" * 60)
print("TRANSCRIPTION")
print("=" * 60)

print(text)

print("\n" + "=" * 60)
print("PERFORMANCE")
print("=" * 60)

rtf = inference_time / audio_duration
speed = audio_duration / inference_time

print(f"Audio duration : {audio_duration:.2f} sec")
print(f"Inference time : {inference_time:.2f} sec")
print(f"RTF            : {rtf:.4f}")
print(f"Real-time speed: {speed:.2f}x")

print("=" * 60)
