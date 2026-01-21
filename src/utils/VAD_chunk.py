import torch
import numpy as np
from rVADfast import rVADfast
import logging

logger = logging.getLogger(__name__)
def vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate):
    speech_ts = []
    start = None

    for label, t in zip(vad_labels, vad_timestamps):
        if label == 1 and start is None:
            start = int(t * sampling_rate)

        elif label == 0 and start is not None:
            end = int(t * sampling_rate)
            speech_ts.append({"start": start, "end": end})
            start = None

    if start is not None:
        speech_ts.append({
            "start": start,
            "end": int(vad_timestamps[-1] * sampling_rate)
        })

    return speech_ts

def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=30.0,
    max_pause_duration=0.6
):
    """
    wav: torch.Tensor (1D, 16kHz)
    returns: list of dicts with start/end in ORIGINAL time
    """
    logger.info("Using Silero VAD model")
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    speech_ts = get_speech_timestamps(wav,vad_model,sampling_rate=sampling_rate)
    #logger.info("Using rVAD model")
    #vad = rVADfast()
    #vad_labels, vad_timestamps = vad(wav, sampling_rate)
    #speech_ts = vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate)
    chunks = []

    chunk_start = None
    chunk_end = None

    for seg in speech_ts:
        seg_start = seg["start"] / sampling_rate
        seg_end = seg["end"] / sampling_rate

        if chunk_start is None:
            chunk_start = seg_start
            chunk_end = seg_end
            continue

        pause = seg_start - chunk_end

        # ❗ règle 1 : pause courte → on fusionne
        if pause <= max_pause_duration:
            # mais on respecte la durée max
            if seg_end - chunk_start <= max_chunk_duration:
                chunk_end = seg_end
            else:
                chunks.append({
                    "start": chunk_start,
                    "end": chunk_end
                })
                chunk_start = seg_start
                chunk_end = seg_end

        # ❗ règle 2 : pause longue → nouvelle unité
        else:
            chunks.append({
                "start": chunk_start,
                "end": chunk_end
            })
            chunk_start = seg_start
            chunk_end = seg_end

    # dernier chunk
    if chunk_start is not None:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end
        })

    return chunks


def extract_chunk_audio(wav, start, end, sr=16000):
    s = int(start * sr)
    e = int(end * sr)
    return wav[s:e].unsqueeze(0)   # (1, T)

def whisper_transcribe_chunks(
    asr_model,
    wav,
    chunks,
    sr=16000
):
    results = []

    for i, ch in enumerate(chunks):
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )

        wav_lens = torch.tensor([1.0])

        pred = asr_model.transcribe_batch(
            chunk_wav,
            wav_lens)
        hyp = " ".join(pred[0][0]).strip()
        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": hyp.strip()
        })

    return results

def ref_text_for_chunk(words, start, end):
    return " ".join(
        w for w, s, e in words
        if e > start and s < end
    )
