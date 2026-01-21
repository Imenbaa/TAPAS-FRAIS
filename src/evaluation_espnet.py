import argparse
import logging
import os

from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from espnet2.bin.asr_inference import Speech2Text
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment

parser = argparse.ArgumentParser(description='Process an audio file with ESPnet2 ASR.')
parser.add_argument('--wav_data', help='Path to the input WAV file or folder')
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument('--config', default='/vol/experiments3/rouas/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR_config.yaml', help='Path to the ASR config file (default: asr_config.yml)')
parser.add_argument('--model', default='/vol/experiments3/rouas/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR.pth', help='Path to the ASR model file (default: asr.pth)')
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()
log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/espnet/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)

def main(args):
    # -------------------------------- ASR models      -------------------------------
    # --------------------------------------------------------------------------------
    speech2text = Speech2Text(args.config, args.model,nbest=1,minlenratio=0.1,beam_size=40, device="cuda",dtype="float32")

    WERs = []
    number_files = 0
    tg_to_wav = list_files(args.ref_trans)
    for tg,wav in tg_to_wav.items():
        wav_file = os.path.join(args.wav_data, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            if "Rhapsodie" in args.ref_trans:
                ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
            else:
                ref_transcriptions = get_textgrid_transcription_tapas(trans_file)
            # load audio+apply silero VAD
            audio_np=apply_rvad_return_audio(wav_file)
            results = speech2text(speech=audio_np)
            nbests = [text for text, token, token_int, hyp in results]
            pred_transcriptions = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            ref_transcriptions = remove_words(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            # -------------------------------- WER per file   -------------------------------
            # --------------------------------------------------------------------------------
            WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions)
            WERs.append(WER)
        else:
            logging.warning(f"The file {wav_file} does not exist")
    # -------------------------------- Global WER     -------------------------------
    # --------------------------------------------------------------------------------
    logger.info("===================== GLOBAL WER ======================")
    logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", sum(WERs) / len(WERs))
    logger.info("=============================================================")
if __name__ == "__main__":
    main(args)