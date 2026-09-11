# Offline ASR Evaluation Report

1. Overview

As part of our offline emergency communication project, we evaluated two speech-to-text models that can run locally without depending on an internet connection:

- Parakeet TDT-CTC 110M INT8 for English
- AI4Bharat IndicConformer Hindi Large INT8 for Hindi

Both models were integrated with Sherpa-ONNX and tested using Silero VAD. The goal was to understand whether they are practical for a low-bandwidth emergency communication system, where speech can be converted into text locally and the text can then be transmitted instead of the original audio.

The evaluation focused on model size, loading time, inference speed, memory consumption, real microphone recognition, and VAD-based continuous speech recognition.

---

2. Test Environment

The experiments were performed on a MacBook Air using:

- macOS
- Apple Silicon
- Python 3.11.4
- Sherpa-ONNX 1.13.7
- ONNX Runtime 1.29.0
- CPU execution
- 2 ASR threads
- 16 kHz mono audio
- Silero VAD

The models were tested locally without requiring an external speech recognition API.

---

3. Models

#3.1 Parakeet TDT-CTC 110M INT8

The Parakeet model used in this evaluation is the Sherpa-ONNX CTC export of NVIDIA's Parakeet model.

Local model:

model.int8.onnx

Model size:

125.55 MB

It was selected as the primary English ASR candidate because it provides a relatively small model size and very fast inference.

#3.2 IndicConformer Hindi Large INT8

The Hindi model is the AI4Bharat IndicConformer Hindi Large model converted to an INT8 ONNX CTC model for use with Sherpa-ONNX.

Local model:

model.int8.onnx

Model size:

188.44 MB

It was evaluated as the Hindi ASR candidate because it is designed specifically for Indian-language speech recognition and showed strong results during Hindi microphone testing.

---

5. Performance Comparison

| Model | Language | Audio Duration | Load Time | Inference Time | RTF ↓ | Real-Time Speed ↑ |
|---|---|---:|---:|---:|---:|---:|
| IndicConformer | Telugu | 12.137 s | 0.334 s | 0.432 s | 0.0356 | 28.13× |
| IndicConformer | Malayalam | 21.625 s | 0.390 s | 0.865 s | 0.0400 | 25.00× |
| IndicConformer | Marathi | 21.625 s | 0.397 s | 0.859 s | 0.0397 | 25.18× |
| IndicConformer | Hindi | 22.453 s | 0.337 s | 0.317 s | 0.0576 | 17.37× |
| Parakeet | English | 17.48 s* | 0.622 s | 0.098 s | 0.0131 | 76.08× |

Both models run faster than real time on the test machine.

Parakeet showed a substantial advantage in inference speed and memory consumption. Its measured real-time speed was approximately 4.4 times higher than IndicConformer in the benchmark.

The memory measurements represent process RSS rather than model-only memory. They therefore include the model runtime, Python/Sherpa components, buffers, and other process allocations.

---

6. Real Microphone Test

The models were also tested using live speech captured from the MacBook Air microphone.

English

A natural English recording was created and processed using Parakeet.

The recording contained conversational speech about the project and the ASR system. Parakeet produced a largely correct transcription, including the overall sentence structure.

Some errors appeared around proper names and technical terms. For example, "Rishabh" was recognized incorrectly and "Sherpa ONNX" was also misrecognized.

This is expected to some extent because proper names and technical product names are more difficult for a general ASR model.

Hindi

A natural Hindi recording was processed using IndicConformer.

The spoken sentence was:

"नमस्ते मेरा नाम ऋषभ है और हम एक ऑफ़लाइन इमरजेंसी कम्युनिकेशन सिस्टम बना रहे हैं"

The model produced the same sentence with essentially exact recognition.

The result was encouraging because the recording was made using the laptop microphone rather than a clean pre-existing benchmark recording.

---

7. Silero VAD + ASR Test

Both models were tested with Sherpa-ONNX's microphone-based offline ASR pipeline using Silero VAD.

The VAD configuration used:

- Threshold: 0.5
- Minimum speech duration: 0.25 seconds
- Minimum silence duration: 0.5 seconds
- Maximum speech duration: 20 seconds
- Window size: 512 samples
- Sample rate: 16 kHz

#Parakeet + VAD

The system successfully detected multiple separate speech segments from the microphone and passed them to Parakeet automatically.

Recognized examples included:

- "This is reallyop"
- "I'm trying to taste this"
- "Model for our project"
- "I think this is working fine..."
- "So we are finalizing this model for English specific language..."

There were some recognition errors, particularly with technical terms and model names, but the VAD-to-ASR pipeline operated continuously and successfully.

IndicConformer + VAD

IndicConformer also successfully operated with the same VAD pipeline.

Recognized examples included:

- "क्या मेरी आवाज़ आ रही है"
- "मेरा नाम ऋषभ है"
- "और मैं इस मॉडल को ट्राई कर रहा हूँ..."

The Hindi recognition quality during this live test was strong, and the VAD successfully separated the individual utterances.

---

8. Observations

Several conclusions can be drawn from the tests.

First, both models are fast enough for real-time offline speech recognition on the development machine.

Second, Parakeet has a significant performance advantage for English. Its smaller model size, substantially lower inference time, and lower measured process memory make it attractive for resource-constrained deployments.

Third, IndicConformer is the stronger choice for Hindi in this evaluation. The model produced very good results on natural Hindi microphone input and is specifically targeted at Indian-language speech.

Fourth, Silero VAD works well as the front end for continuous microphone-based recognition. It prevents the ASR model from processing the entire microphone stream continuously and instead creates speech segments for recognition.

---

9. Recommended Model Assignment

Based on the current tests, the recommended model assignment is:

English → Parakeet TDT-CTC 110M INT

Hindi → IndicConformer Hindi Large INT

The two-model approach is preferable to forcing a single model to handle both languages when language-specific models provide better results and performance.

---

10. Proposed Communication Architecture

The evaluated ASR components fit into the planned communication architecture:

Microphone
    |
    v
Silero VAD
    |
    v
Language Selection / Detection
    |
    +--------------------+
    |                    |
    v                    v
Parakeet             IndicConformer
English                  Hindi
    |                    |
    +---------+----------+
              |
              v
          Text Message
              |
              v
      Compression / Encoding
              |
              v
        Low-bandwidth Link
              |
              v
        Receiving Device
              |
              v
             TTS
              |
              v
           Speaker

