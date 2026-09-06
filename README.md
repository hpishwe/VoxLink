# Offline ASR for Low-Bandwidth Emergency Communication

An offline Speech-to-Text (STT) system for emergency communication scenarios where network bandwidth may be limited.

The system processes speech locally on the device, converts it into text, and prepares the information for transmission over a low-bandwidth communication link. At the receiving end, the text can be converted back into speech using an offline TTS system.

## ASR Models

### Hindi — AI4Bharat IndicConformer

- IndicConformer Hindi Large
- ONNX INT8
- Offline inference
- 16 kHz mono audio
- Tested with Silero VAD

### English — NVIDIA Parakeet TDT-CTC 110M

- Parakeet TDT-CTC 110M
- ONNX INT8
- Offline inference
- 16 kHz mono audio
- Tested with Silero VAD

## Voice Activity Detection

Silero VAD is used to detect speech segments before sending audio to the ASR model.

```text
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
