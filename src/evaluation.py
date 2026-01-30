import argparse
import logging
import os
import csv
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import librosa
import torch
from utils.meta import get_audio_info
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
models = ["wav2vec","whisper-VAD-chunk","whisper-large","whisper-large-VAD-chunk","wav2vec2-VAD-chunk","conf_cv","conf_ester"]

parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")
parser.add_argument("--csv_path", type=str,required=True, help="The csv name")

args = parser.parse_args()

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)


def main(args):
    #-------------------------------- ASR models      -------------------------------
    #--------------------------------------------------------------------------------
    if args.model == "wav2vec" or args.model == "wav2vec2-VAD-chunk":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-medium" or args.model == "whisper-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-large" or args.model == "whisper-large-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device":"cuda"})

    WERs = []
    number_files = 0
    args.ref_trans = unicodedata.normalize("NFD", args.ref_trans)
    print(os.path.exists(args.ref_trans))
    if args.wav_data.endswith("PD") or args.wav_data.endswith("MSA"):

        tg_to_wav = {}
        for w in os.listdir(args.wav_data):
            for t in os.listdir(args.ref_trans):
                if w != '1HC-IAJC_éléments_extralinguisitiques-image.wav' and t !='1HC-IAJC_éléments_extralinguisitiques.txt':
                    if not t.startswith("._"):
                        if w.split("-")[1] in t:
                            tg_to_wav[t] = w
    else:
        tg_to_wav = list_files(args.ref_trans)
    print(tg_to_wav.keys(),tg_to_wav.values())
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "duration_sec", "samplerate", "channels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tg,wav in tg_to_wav.items():
            wav_file = os.path.join(args.wav_data, wav)
            trans_file = os.path.join(args.ref_trans, tg)
            if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid" and wav !="CCM-004773-01_L01.wav":
                number_files += 1
                logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
                info = get_audio_info(wav_file)
                writer.writerow({"filename": wav,**info})
                if "Rhapsodie" in args.ref_trans:
                    ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
                elif trans_file.endswith(".txt"):
                    ref_transcriptions = read_preprocess_transcription(trans_file)
                #elif "spont" in args.ref_trans:
                    #ref_transcriptions = extract_words_text(trans_file)
                else:
                    ref_transcriptions = get_textgrid_transcription_tapas(trans_file)

                # load audio
                audio_np, sr = read_audio_16k(wav_file)
                wav = torch.from_numpy(audio_np)
                # VAD + chunking
                chunks = vad_chunk_with_timestamps(wav)
                logging.info("Number of chunks: %d", len(chunks))

                if args.model == "whisper-VAD-chunk" or args.model == "wav2vec2-VAD-chunk" or args.model == "whisper-large-VAD-chunk":
                    results = whisper_transcribe_chunks(asr_model, args.model, wav, chunks)
                    if "Daoudi" not in args.ref_trans:

                        if "Rhapsodie" in args.ref_trans:
                            words = get_textgrid_transcription_rhap_chunk(trans_file)
                        else:
                            words = get_textgrid_transcription_chunk(trans_file)

                        ref_transcriptions,pred_transcriptions=wer_chunk(results,words)
                    else:
                        ref_transcriptions = remove_words(ref_transcriptions)
                        #ref_transcriptions = clean_french_disfluencies(remove_words(ref_transcriptions))
                        pred_transcriptions = " ".join(r["text"] for r in results if r["text"].strip())
                else:
                    pred_transcriptions = asr_model.transcribe_file(wav_file)
                    #if args.model == "whisper-medium" or args.model == "whisper-large":
                    pred_transcriptions = " ".join(seg.words for seg in pred_transcriptions)

                logger.info("-" * 30)
                logger.info("-" * 30)
                #print(f"Reference transcription: {ref_transcriptions}")
                #ref_transcriptions = remove_words(ref_transcriptions)
                print(normalization(pred_transcriptions))
                print(normalization(ref_transcriptions))
                # -------------------------------- WER per file   -------------------------------
                # --------------------------------------------------------------------------------
                WER = wer_segment(wav_file,ref_transcriptions,pred_transcriptions)
                WERs.append(WER)

            else:
                logging.warning(f"The file {wav_file} does not exist")

    # -------------------------------- Global WER     -------------------------------
    # --------------------------------------------------------------------------------
    logger.info("===================== GLOBAL WER ======================")
    logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", sum(WERs)/len(WERs))
    logger.info("=============================================================")

if __name__ == "__main__":
    main(args)