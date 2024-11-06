import numpy as np
import onnxruntime
from text import text_to_sequence, sequence_to_text
import json
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import wave
from time import perf_counter

app = FastAPI()

# Constants
DEFAULT_SPEAKER_ID = os.environ.get("DEFAULT_SPEAKER_ID", default="quim")
DEFAULT_ACCENT = os.environ.get("DEFAULT_ACCENT", default="balear")
SAMPLE_RATE = 22050

# Load models and configs
MODEL_PATH_MATCHA_E2E = "matxa_multiaccent_wavenext_e2e.onnx"
CONFIG_PATH = "config.yaml"
SPEAKER_ID_DICT = "spk_to_id_3.json"

# Initialize models
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

device = "cpu"

if device == "cuda":
    print("loading in GPU")
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
else:
    print("loading in CPU")
    providers = ["CPUExecutionProvider"]

model_matcha_mel_all = onnxruntime.InferenceSession(
    str(MODEL_PATH_MATCHA_E2E),
    sess_options=sess_options,
    providers=providers
)

# Load speaker IDs and cleaners
speaker_id_dict = json.load(open(SPEAKER_ID_DICT))
cleaners = {
    "balear": "catalan_balear_cleaners",
    "nord-occidental": "catalan_occidental_cleaners",
    "valencia": "catalan_valencia_cleaners",
    "central": "catalan_cleaners"
}


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def process_text(text: str, cleaner: str):
    # Convert text to sequence and intersperse with 0
    text_sequence = text_to_sequence(text, [cleaner])
    interspersed_sequence = intersperse(text_sequence, 0)

    # Convert to NumPy array
    x = np.array(interspersed_sequence, dtype=np.int64)[None]
    x_lengths = np.array([x.shape[-1]], dtype=np.int64)
    x_phones = sequence_to_text(x.squeeze(0).tolist())

    print(x_phones)
    return x, x_lengths


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_SPEAKER_ID
    accent: str = DEFAULT_ACCENT
    temperature: float = 0.2
    length_scale: float = 0.89
    type: str = "text"


def create_wav_header(audio_data):
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        # Configure WAV settings
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)

        # Convert float32 audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    return wav_buffer


async def generate_audio(text: str, accent: str, spk_name: str,
                         temperature: float, length_scale: float):
    spk_id = speaker_id_dict[accent][spk_name]
    sid = np.array([int(spk_id)]) if spk_id is not None else None

    text_matcha, text_lengths = process_text(text, cleaner=cleaners[accent])

    inputs = {
        "x": text_matcha,
        "x_lengths": text_lengths,
        "scales": np.array([temperature, length_scale], dtype=np.float32),
        "spks": sid
    }

    t0 = perf_counter()
    _, wav = model_matcha_mel_all.run(None, inputs)
    infer_secs = perf_counter() - t0
    sr_out = 22050

    wav_length = wav.shape[1]  # num of samples
    wav_secs = wav_length / sr_out
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"Overall RTF: {rtf}")

    return wav.squeeze(0)


async def stream_wav_generator(wav_buffer):
    CHUNK_SIZE = 4096  # Adjust chunk size as needed
    wav_buffer.seek(0)

    while True:
        data = wav_buffer.read(CHUNK_SIZE)
        if not data:
            break
        yield data


@app.get("/health")
def health_check():
    return "Running"


@app.post("/api/tts")
async def tts(request: TTSRequest):
    if len(request.text) > 500:
        return Response(content="Text too long. Maximum 500 characters allowed.",
                        status_code=400)

    try:
        # Generate audio
        audio_data = await generate_audio(
            request.text,
            request.accent,
            request.voice,
            request.temperature,
            request.length_scale
        )

        # Create WAV file in memory
        wav_buffer = create_wav_header(audio_data)

        # Return streaming response
        return StreamingResponse(
            stream_wav_generator(wav_buffer),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=audio.wav"
            }
        )

    except Exception as e:
        return Response(content=str(e), status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
