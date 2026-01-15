import torch
import soundfile as sf
import librosa
import numpy as np

import soundfile as sf
import librosa
import numpy as np

def read_audio_16k(path):
    audio, sr = sf.read(path)

    # mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample if needed
    if sr != 16000:
        audio = librosa.resample(
            audio.astype("float32"),
            orig_sr=sr,
            target_sr=16000
        )

    return audio.astype("float32"),sr


vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",model="silero_vad",force_reload=False)

(get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks) = utils

def apply_vad_to_wav(
    wav_path,
    out_path,
    sampling_rate=16000
):
    # read & resample
    wav,sr = read_audio_16k(wav_path)
    wav_torch = torch.from_numpy(wav)
    # detect speech
    speech_timestamps = get_speech_timestamps(
        wav_torch,
        vad_model,
        sampling_rate=sampling_rate
    )

    # keep only speech
    speech_wav = collect_chunks(
        speech_timestamps,
        wav_torch
    )

    # save speech-only audio
    sf.write(out_path, speech_wav, sampling_rate)

    return speech_timestamps

