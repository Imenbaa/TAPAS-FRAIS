
import os
import soundfile as sf
from textgrid import TextGrid
import torch
from speechbrain.inference.ASR import WhisperASR
from apply_vad import *
from hyperpyyaml import load_hyperpyyaml
from normalise_text import normalization
from pathlib import Path
import logging
log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/whisper_tapas_verif_vad_chunk.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),  # remove if you want file-only
    ],
)
logger = logging.getLogger(__name__)
def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=30.0,
    max_pause_duration=0.6   # 👈 tolérance de pause (en secondes)
):
    """
    wav: torch.Tensor (1D, 16kHz)
    returns: list of dicts with start/end in ORIGINAL time
    """

    speech_ts = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate
    )

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

def load_words_from_textgrid(tg_path, tier_name="ORT-MAU"):
    tg = TextGrid()
    tg.read(tg_path)

    names = tg.getNames()
    if tier_name not in names:
        raise ValueError(
            f"Tier '{tier_name}' not found. Available: {names}"
        )

    tier_index = names.index(tier_name)
    tier = tg.tiers[tier_index]   # ✅ LA BONNE API CHEZ TOI

    words = []
    for interval in tier.intervals:
        if interval.mark.strip():
            words.append(
                (interval.mark.strip(),
                 interval.minTime,
                 interval.maxTime)
            )
    return words





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
def merge_transcriptions(results):
    results = sorted(results, key=lambda x: x["start"])
    return " ".join(r["text"] for r in results if r["text"])
def ref_text_for_chunk(words, start, end):
    return " ".join(
        w for w, s, e in words
        if e > start and s < end
    )

if __name__ == "__main__":
    wav_path = "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description"
    tg_path = "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans"
    asr_model = WhisperASR.from_hparams(
        source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",
        savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
    WERs= []
    for f in os.listdir(tg_path):
        wav_file = os.path.join(wav_path,f.split(".")[0] + ".wav")
        trans_file = os.path.join(tg_path,f)
        # load audio
        audio_np,sr = read_audio_16k(wav_file)

        wav = torch.from_numpy(audio_np)

        # VAD + chunking
        chunks = vad_chunk_with_timestamps(
            wav,
            sampling_rate=16000,
            max_chunk_duration=30.0
        )

        print("Number of chunks:", len(chunks))



        results = whisper_transcribe_chunks(
            asr_model,
            wav,
            chunks
        )

        full_hyp = merge_transcriptions(results)
        words = load_words_from_textgrid(
            trans_file,
            tier_name="ORT-MAU"
        )
        # Attach references
        for r in results:
            r["ref"] = ref_text_for_chunk(
                words, r["start"], r["end"]
            )
        for id,r in enumerate(results):
            print(f"[{r['start']:.2f}-{r['end']:.2f}]")
            print("REF:", r["ref"])
            print("HYP:", r["text"])
            if r["ref"].strip():
                wer_hparams["wer_stats"].clear()
                wer_hparams["wer_stats"].append(ids=list(range(len(r["ref"]))),
                                            predict=[normalization(r["text"])],
                                            target=[normalization(r["ref"])])

                stats = wer_hparams["wer_stats"].summarize()
                print(f'WER= {stats["WER"]},S = {stats["substitutions"]},D = {stats["deletions"]}, I = {stats["insertions"]}')
                logger.info(
                    "Chunk: %d | WER=%f | S=%d D=%d I=%d",
                    id,
                    stats["WER"],
                    stats["substitutions"],
                    stats["deletions"],
                    stats["insertions"],
                )
        logger.info("-" * 30)
        logger.info("-" * 30)
        ref_full = " ".join(
            r["ref"] for r in results if r["ref"].strip()
        )

        hyp_full = " ".join(
            r["text"] for r in results if r["text"].strip()
        )
        wer_hparams["wer_stats"].clear()
        wer_hparams["wer_stats"].append(ids=list(range(len(ref_full))),
                                        predict=[normalization(hyp_full)],
                                        target=[normalization(ref_full)])
        stats = wer_hparams["wer_stats"].summarize()
        logger.info(
            "File: %s | WER=%f | S=%d D=%d I=%d",
            f.split(".")[0],
            stats["WER"],
            stats["substitutions"],
            stats["deletions"],
            stats["insertions"],
        )
        logger.info("-" * 30)
        logger.info("-" * 30)
        WERs.append(stats["WER"])
    # Compute final WER
    global_stats = sum(WERs) / len(WERs)

    logger.info("===================== GLOBAL WER ======================")
    #logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", global_stats)
    logger.info("=============================================================")