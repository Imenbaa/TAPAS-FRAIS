import argparse
import logging
import sys
import os
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from textgrid import TextGrid
from utils.read_transcription import *
from utils.normalise_text import normalization
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import librosa
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.apply_vad import *
from utils.resample16k import resample_dir

models = ["wav2vec","whisper-medium","wav2vec-benchmark"]
#datasets = ["TAPAS-FRAIS","CV","TYPALOC","Ester","Librispeech","Rhapsodie"]

#TAPAS-FRAIS_wav ="/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description", ref= "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid vérifié + 50 ans"


parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--wav_16k", type=str,required=True, help="folder or resampled data to 16k")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()
log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),  # remove if you want file-only
    ],
)
logger = logging.getLogger(__name__)
def main(args):
    #-------------------Resampling audio files if not 16K-------------------

    if not os.path.isdir(args.wav_16k):
        logging.info("The data is being resampled to 16000")
        resample_dir(in_dir=args.wav_data,out_dir=args.wav_16k)
    if len(os.listdir(args.wav_16k)) != 0:
        args.wav_data = args.wav_16k
    #-------------------------------- Model inference -------------------------------
    #--------------------------------------------------------------------------------
    if args.model == "wav2vec":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-medium":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "wav2vec-benchmark":
        asr_model = EncoderDecoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wav2vec2-fr-3k-large",savedir="/tmp/sb_cache", run_opts={"device":"cuda"})
    WERs = []
    tg_map={}
    number_files=0
    for f in os.listdir(args.ref_trans):
        print(f)
        if "Rhapsodie" in args.ref_trans:
            tg_map[f] = f.split("-")[0]+"-"+f.split("-")[1]+".TextGrid" #Rhap-D0005-Pro.TextGrid = Rhap-D0005.TextGrid
        else:
            tg_map[f] = f
        wav_file = os.path.join(args.wav_data, tg_map[f].split(".")[0]+".wav")
        trans_file = os.path.join(args.ref_trans, f)

        if os.path.exists(wav_file):
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(path=wav_file)} seconds")

            ref_transcriptions = get_textgrid_transcription_tapas(trans_file)
            if args.model == "whisper-medium":
                clean_wav = "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/tapas_vad/"+tg_map[f].split(".")[0]+"-VAD.wav"
                apply_vad_to_wav(wav_file, clean_wav)
                wav_file = clean_wav
            pred_transcriptions = asr_model.transcribe_file(wav_file)
            if args.model == "whisper-medium":
                pred_transcriptions =" ".join(seg.words for seg in pred_transcriptions)

            #print(pred_transcriptions)
            #print(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            wer_hparams["wer_stats"].clear()
            wer_hparams["wer_stats"].append(ids=list(range(len(ref_transcriptions))),predict=[normalization(pred_transcriptions)],target=[normalization(ref_transcriptions)])

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
            WERs.append(stats["WER"])
        else:
            logging.warning(f"The file {f.split('.')[0]+'.wav'} does not exist")
        #break

    # Compute final WER
    global_stats = sum(WERs)/len(WERs)

    logger.info("===================== GLOBAL WER ======================")
    logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", global_stats)
    logger.info("=============================================================")
if __name__ == "__main__":
    main(args)