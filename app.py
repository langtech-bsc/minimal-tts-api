import numpy as np
import onnxruntime
from text import text_to_sequence, sequence_to_text
import torch
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
    x = torch.tensor(
        intersperse(text_to_sequence(text, [cleaner]), 0),
        dtype=torch.long,
        device="cpu",
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")
    return x.numpy(), x_lengths.numpy()

def vocos_inference(mel, denoise=True):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    params = config["feature_extractor"]["init_args"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    win_length = n_fft

    mag, x, y = model_vocos.run(None, {"mels": mel})
    spectrogram = mag * (x + 1j * y)
    window = torch.hann_window(win_length)

    if denoise:
        mel_rand = torch.zeros_like(torch.tensor(mel))
        mag_bias, x_bias, y_bias = model_vocos.run(None, {"mels": mel_rand.float().numpy()})
        spectrogram_bias = mag_bias * (x_bias + 1j * y_bias)
        
        spec = torch.view_as_real(torch.tensor(spectrogram))
        mag_spec = torch.sqrt(spec.pow(2).sum(-1))
        
        spec_bias = torch.view_as_real(torch.tensor(spectrogram_bias))
        mag_spec_bias = torch.sqrt(spec_bias.pow(2).sum(-1))
        
        strength = 0.0025
        mag_spec_denoised = mag_spec - mag_spec_bias * strength
        mag_spec_denoised = torch.clamp(mag_spec_denoised, 0.0)
        
        angle = torch.atan2(spec[..., -1], spec[..., 0])
        spectrogram = torch.complex(mag_spec_denoised * torch.cos(angle), 
                                  mag_spec_denoised * torch.sin(angle))

    pad = (win_length - hop_length) // 2
    spectrogram = torch.tensor(spectrogram)
    B, N, T = spectrogram.shape

    ifft = torch.fft.irfft(spectrogram, n_fft, dim=1, norm="backward")
    ifft = ifft * window[None, :, None]

    output_size = (T - 1) * hop_length + win_length
    y = torch.nn.functional.fold(
        ifft, 
        output_size=(1, output_size), 
        kernel_size=(1, win_length), 
        stride=(1, hop_length),
    )[:, 0, 0, pad:-pad]

    window_sq = window.square().expand(1, T, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq, 
        output_size=(1, output_size), 
        kernel_size=(1, win_length), 
        stride=(1, hop_length),
    ).squeeze()[pad:-pad]

    assert (window_envelope > 1e-11).all()
    y = y / window_envelope
    
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
    
    return wavs_vocos.squeeze(0).numpy()

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