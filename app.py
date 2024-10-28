import numpy as np
import onnxruntime
from text import text_to_sequence, sequence_to_text
from scipy.fft import irfft
from scipy.signal.windows import hann
import yaml
import json
import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
import wave
import struct

app = FastAPI()

# Constants
DEFAULT_SPEAKER_ID = os.environ.get("DEFAULT_SPEAKER_ID", default="quim")
DEFAULT_ACCENT = os.environ.get("DEFAULT_ACCENT", default="balear")
SAMPLE_RATE = 22050

# Load models and configs
MODEL_PATH_MATCHA_MEL_ALL = "matcha_multispeaker_cat_all_opset_15_10_steps.onnx"
MODEL_PATH_VOCOS = "mel_spec_22khz_cat.onnx"
CONFIG_PATH = "config.yaml"
SPEAKER_ID_DICT = "spk_to_id_3.json"

# Initialize models
sess_options = onnxruntime.SessionOptions()
model_matcha_mel_all = onnxruntime.InferenceSession(
    str(MODEL_PATH_MATCHA_MEL_ALL), 
    sess_options=sess_options, 
    providers=["CPUExecutionProvider"]
)
model_vocos = onnxruntime.InferenceSession(
    str(MODEL_PATH_VOCOS), 
    sess_options=sess_options, 
    providers=["CPUExecutionProvider"]
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

# inference w/o torch 
# source https://github.com/OpenVoiceOS/ovos-tts-plugin-matxa-multispeaker-cat/blob/dev/ovos_tts_plugin_matxa_multispeaker_cat/tts.py 
def vocos_inference(mel, denoise=True):

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    params = config["feature_extractor"]["init_args"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    win_length = n_fft

    # ONNX inference
    mag, x, y = model_vocos.run(None, {"mels": mel})

    # Complex spectrogram from vocos output
    spectrogram = mag * (x + 1j * y)
    window = hann(win_length, sym=False)

    if denoise:
        # Vocoder bias
        mel_rand = np.zeros_like(mel)
        mag_bias, x_bias, y_bias = model_vocos.run(
            None,
            {
                "mels": mel_rand.astype(np.float32)
            },
        )

        # Complex spectrogram from vocos output
        spectrogram_bias = mag_bias * (x_bias + 1j * y_bias)

        # Denoising
        spec = np.stack([np.real(spectrogram), np.imag(spectrogram)], axis=-1)
        # Get magnitude of vocos spectrogram
        mag_spec = np.sqrt(np.sum(spec ** 2, axis=-1))

        # Get magnitude of bias spectrogram
        spec_bias = np.stack([np.real(spectrogram_bias), np.imag(spectrogram_bias)], axis=-1)
        mag_spec_bias = np.sqrt(np.sum(spec_bias ** 2, axis=-1))

        # Subtract
        strength = 0.0025
        mag_spec_denoised = mag_spec - mag_spec_bias * strength
        mag_spec_denoised = np.clip(mag_spec_denoised, 0.0, None)

        # Return to complex spectrogram from magnitude
        angle = np.arctan2(np.imag(spectrogram), np.real(spectrogram))
        spectrogram = mag_spec_denoised * (np.cos(angle) + 1j * np.sin(angle))

    # Inverse STFT
    pad = (win_length - hop_length) // 2
    B, N, T = spectrogram.shape

    # Inverse FFT
    ifft = irfft(spectrogram, n=n_fft, axis=1)
    ifft *= window[None, :, None]

    # Overlap and Add
    output_size = (T - 1) * hop_length + win_length
    y = np.zeros((B, output_size))
    for b in range(B):
        for t in range(T):
            y[b, t * hop_length:t * hop_length + win_length] += ifft[b, :, t]

    # Window envelope
    window_sq = np.expand_dims(window ** 2, axis=0)
    window_envelope = np.zeros((B, output_size))
    for b in range(B):
        for t in range(T):
            window_envelope[b, t * hop_length:t * hop_length + win_length] += window_sq[0]

    # Normalize
    if np.any(window_envelope <= 1e-11):
        print("Warning: Some window envelope values are very small.")

    y /= np.maximum(window_envelope, 1e-11)  # Prevent division by very small values

    return y

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
    
    mel, mel_lengths = model_matcha_mel_all.run(None, inputs)
    wavs_vocos = vocos_inference(mel, denoise=True)
    
    return wavs_vocos.squeeze(0)

async def stream_wav_generator(wav_buffer):
    CHUNK_SIZE = 4096  # Adjust chunk size as needed
    wav_buffer.seek(0)
    
    while True:
        data = wav_buffer.read(CHUNK_SIZE)
        if not data:
            break
        yield data

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