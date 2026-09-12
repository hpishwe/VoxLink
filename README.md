# VoxLink — Offline Multilingual Voice Communication

VoxLink is an **offline, low-bandwidth voice communication system** designed for emergency, remote-field, and mission-critical environments where network connectivity may be limited or unreliable.

Instead of transmitting raw voice, VoxLink performs **on-device Speech-to-Text (STT)**, optionally converts the message into a compact semantic representation, transmits it over an available communication link, and converts the received information back to speech using **offline Text-to-Speech (TTS)**.

## Architecture

```text
Voice
  ↓
Silero VAD
  ↓
Offline STT
  ↓
Text / Semantic Processing
  ↓
Adaptive Transmission
  ↓
Wi-Fi Direct / Bluetooth / Mesh / RF
  ↓
Packet Validation
  ↓
Semantic / Text Reconstruction
  ↓
Offline TTS
  ↓
Speaker

## Current ASR Models

| Language | Model                                | Format    | Runtime     |
| -------- | ------------------------------------ | --------- | ----------- |
| Hindi    | AI4Bharat IndicConformer Hindi Large | ONNX INT8 | Sherpa-ONNX |
| English  | NVIDIA Parakeet TDT-CTC 110M         | ONNX INT8 | Sherpa-ONNX |

**Audio:** 16 kHz, mono, PCM WAV

**VAD:** Silero VAD

The architecture is designed to expand to the required Indian languages:
**Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali, and English.**

## TTS

Offline TTS is based on **Piper ONNX voice models** with **Sherpa-ONNX** for native/local inference.

Initial benchmark results on a laptop CPU:

| Model              | Language  |     Size |    Load | Inference | RTF ↓ | Speed ↑ |
| ------------------ | --------- | -------: | ------: | --------: | ----: | ------: |
| Piper – Rohan      | Hindi     | 60.03 MB | 2.919 s |   0.231 s | 0.104 |   9.62× |
| Piper – Meera      | Malayalam | 60.03 MB | 2.816 s |   0.242 s | 0.104 |   9.62× |
| Piper – Google     | Marathi   | 73.21 MB | 3.754 s |   0.271 s | 0.118 |   8.47× |
| Piper – Padmavathi | Telugu    | 60.03 MB | 4.666 s |   0.247 s | 0.101 |   9.90× |
| Piper – Lessac     | English   | 60.27 MB | 4.936 s |   0.172 s | 0.136 |   7.35× |

> RTF = Inference Time / Audio Duration. Lower RTF is better.
> These are initial laptop benchmarks; Android performance will be benchmarked separately.

## Semantic Communication

VoxLink can extract important information from STT output:

```text
Intent
Entities
Priority
Location
Numbers
Units
Actions
Conditions
```

Example:

```text
"Please send help immediately, I am injured and I'm near Sector 5."
```

becomes:

```json
{
  "intent": "REQUEST_HELP",
  "priority": "EMERGENCY",
  "entities": {
    "condition": "INJURED",
    "location": "SECTOR 5"
  }
}
```

Compact representation:

```text
H|3|I=I|L=S5
```

The receiver can reconstruct:

```text
Emergency. Help is required.
Condition: INJURED.
Location: SECTOR 5.
```

### Adaptive Transmission

```text
Good Link       → Full Text
Limited Link    → Semantic Packet
Severe Link     → Critical Information Only
```

If semantic confidence is low, the system can fall back to transmitting the original text to avoid information loss.

## Communication

VoxLink is designed to support:

* **Wi-Fi Direct** — short-range, high-speed phone-to-phone communication
* **Bluetooth** — local device communication
* **Mesh** — multi-hop communication between devices
* **RF via ESP32** — low-bandwidth, infrastructure-independent communication

```text
Phone
  ↓
STT + Semantic Processing
  ↓
Compact Packet
  ↓
ESP32 / Wi-Fi / Bluetooth / Mesh
  ↓
Receiver
  ↓
Semantic Decoder
  ↓
TTS
  ↓
Speech
```

## Repository Setup

Clone the STT model branch:

```bash
git clone -b STT-models https://github.com/hpishwe/SwarSync.git
cd SwarSync
```

The repository currently uses Git LFS for large ONNX models:

```bash
git lfs install
git lfs pull
git lfs ls-files
```

Create the Python environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install sherpa-onnx soundfile
```

## Model Paths

### Hindi

```text
models/indicconformer-hi/model.int8.onnx
models/indicconformer-hi/tokens.txt
```

### English

```text
models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/model.int8.onnx
models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/tokens.txt
```

## Basic ASR Usage

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model="PATH_TO_MODEL",
    tokens="PATH_TO_TOKENS",
    num_threads=2,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
)

audio = sherpa_onnx.read_wave("test.wav")

stream = recognizer.create_stream()
stream.accept_waveform(audio.sample_rate, audio.samples)

recognizer.decode_stream(stream)

print(stream.result.text)
```

## Audio Conversion

Convert input audio to the required format:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

## Key Objectives

VoxLink is being evaluated on:

* **Efficiency:** model size, RAM, CPU, storage
* **Accuracy:** STT WER, semantic accuracy, TTS intelligibility
* **Latency:** STT latency, transmission latency, TTS latency, end-to-end latency
* **Bandwidth:** transmitted bytes under different link conditions
* **Robustness:** packet loss, latency, reduced bandwidth
* **Scalability:** phone-to-phone and phone-to-ESP32 communication



## Goal

> **VoxLink enables offline, multilingual, low-bandwidth voice communication that adapts to network conditions while preserving critical information.**

```
```
