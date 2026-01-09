import argparse
import logging
import sys
import os
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from textgrid import TextGrid
from utils.read_transcription import get_textgrid_transcription
from utils.normalise_text import normalization
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/wer_evaluation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),  # remove if you want file-only
    ],
)

wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")

models = ["wav2vec"]
#datasets = ["TAPAS-FRAIS","CV","TYPALOC","Ester","Librispeech","Rhapsodie"]

#TAPAS-FRAIS_wav ="/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description", ref= "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid vérifié + 50 ans"


parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")

args = parser.parse_args()
logger = logging.getLogger(__name__)

def main(args):
    #-------------------------------- Model inference -------------------------------
    #--------------------------------------------------------------------------------
    if args.model == "wav2vec":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/pretrained_models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/pretrained_models/asr-wav2vec2-commonvoice-fr")
        WERs = []
        for f in os.listdir(args.ref_trans):
            wer_hparams["wer_stats"].clear()
            ref_transcriptions = get_textgrid_transcription(os.path.join(args.ref_trans, f))
            pred_transcriptions = asr_model.transcribe_file(os.path.join(args.wav_data, f.split(".")[0]+".wav"))
            #wer_metric.append(f.split(".")[0],normalization(pred_transcriptions),normalization(ref_transcriptions))

            wer_hparams["wer_stats"].append(
                ids=list(range(len(ref_transcriptions))),
                predict=[normalization(pred_transcriptions)],
                target=[normalization(ref_transcriptions)])

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
        # Compute final WER
        global_stats = sum(WERs)/len(WERs)

        logger.info("===================== GLOBAL WER ======================")
        logger.info("WER: %.2f%%", global_stats)
        logger.info("=============================================================")
if __name__ == "__main__":
    main(args)