 Offline ASR for Low-Bandwidth Emergency Communication

An offline Speech-to-Text (STT) system for emergency communication scenarios where network bandwidth may be limited.

The system processes speech locally on the device, converts it into text, and prepares the information for transmission over a low-bandwidth communication link. At the receiving end, the text can be converted back into speech using an offline TTS system.

ASR Models

Hindi — AI4Bharat IndicConformer

- IndicConformer Hindi Large
- ONNX INT8
- Offline inference
- 16 kHz mono audio
- Tested with Silero VAD

English — NVIDIA Parakeet TDT-CTC 110M

- Parakeet TDT-CTC 110M
- ONNX INT8
- Offline inference
- 16 kHz mono audio
- Tested with Silero VAD

Voice Activity Detection

Silero VAD is used to detect speech segments before sending audio to the ASR model.

text
Microphone
    ↓
Silero VAD
    ↓
Language Selection
    ↓
Hindi / English ASR
    ↓
Text
    ↓
Compression / Encoding
    ↓
Low-Bandwidth Communication Link
    ↓
Receiving Device
    ↓
Offline TTS
    ↓
Speaker

 SwarSync — Offline STT Models

This branch contains the Hindi and English offline Speech-to-Text models used by SwarSync for low-bandwidth emergency communication.

1. Clone the branch

bash
git clone -b STT-models https://github.com/hpishwe/SwarSync.git
cd SwarSync


2. Download models using Git LFS

The ONNX models are large binary files (~300 MB total), so we use Git LFS (Large File Storage) instead of storing them directly in normal Git history.

Install and initialize Git LFS:

bash
brew install git-lfs
git lfs install


Download the actual model files:

bash
git lfs pull


Check that the models were downloaded:

bash
git lfs ls-files


> If the .onnx files are only a few KB, run git lfs pull again. The actual models should be hundreds of MB.

3. Install Sherpa-ONNX

bash
pip install sherpa-onnx soundfile


4. Model Paths

Hindi:

text
models/indicconformer-hi/model.int8.onnx
models/indicconformer-hi/tokens.txt


English:

text
models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/model.int8.onnx
models/parakeet-en/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8/tokens.txt


5. Basic Python Usage

python
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


Replace PATH_TO_MODEL and PATH_TO_TOKENS with the Hindi or English paths above.

Audio Requirements

For the tested setup, use:

- 16 kHz sample rate
- Mono
- PCM WAV

Example:

bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav


Pipeline

text
Microphone → VAD → STT → Text → Compression → Low-Bandwidth Link


The STT models run locally/offline, so no internet or cloud API is required during inference.
