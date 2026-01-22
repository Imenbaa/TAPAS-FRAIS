import argparse
import logging
import os
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import librosa
import torch
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
models = ["wav2vec","whisper-medium","wav2vec-benchmark","whisper-VAD-chunk","whisper-large","whisper-large-VAD-chunk"]

parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)


def main(args):
    #-------------------Resampling audio files if not 16K-------------------

    """if not os.path.isdir(args.wav_16k):
        logging.info("The data is being resampled to 16000")
        resample_dir(in_dir=args.wav_data,out_dir=args.wav_16k)
    if len(os.listdir(args.wav_16k)) != 0:
        args.wav_data = args.wav_16k"""
    #-------------------------------- ASR models      -------------------------------
    #--------------------------------------------------------------------------------
    if args.model == "wav2vec":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-medium" or args.model == "whisper-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-large" or args.model == "whisper-large-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device":"cuda"})

    WERs = []
    number_files = 0
    tg_to_wav = list_files(args.ref_trans)
    print(tg_to_wav.keys(),len(tg_to_wav.values()))

    dict_pathologies={"SLA":[],"PARK|CEREB":[],"CTRL":[]}
    for tg,wav in tg_to_wav.items():
        wav_file = os.path.join(args.wav_data, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            if "Rhapsodie" in args.ref_trans:
                ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
            elif "spont" in args.ref_trans:
                ref_transcriptions = extract_words_text(trans_file)
            else:
                ref_transcriptions = get_textgrid_transcription_tapas(trans_file)

            if args.model == "whisper-medium-VAD":
                clean_wav = "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/tapas_vad/"+wav_file.split(".")[0]+"-VAD.wav"
                apply_vad_to_wav(wav_file, clean_wav)
                wav_file = clean_wav

            pred_transcriptions = asr_model.transcribe_file(wav_file)
            if args.model == "whisper-medium" or args.model == "whisper-large":
                pred_transcriptions =" ".join(seg.words for seg in pred_transcriptions)


            if args.model == "whisper-VAD-chunk" or args.model=="whisper-large-VAD-chunk":
                # load audio
                audio_np, sr = read_audio_16k(wav_file)
                wav = torch.from_numpy(audio_np)
                # VAD + chunking
                chunks = vad_chunk_with_timestamps(wav)
                logging.info("Number of chunks: %d", len(chunks))
                results = whisper_transcribe_chunks(asr_model, wav, chunks)
                if "Rhapsodie" in args.ref_trans:
                    words = get_textgrid_transcription_rhap_chunk(trans_file)
                else:
                    words = get_textgrid_transcription_chunk(trans_file)

                ref_transcriptions,pred_transcriptions=wer_chunk(results,words)

                logger.info("-" * 30)
                logger.info("-" * 30)
            print(f"Reference transcription: {ref_transcriptions}")
            ref_transcriptions = remove_words(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            # -------------------------------- WER per file   -------------------------------
            # --------------------------------------------------------------------------------
            WER = wer_segment(wav_file,ref_transcriptions,pred_transcriptions)
            if "spont" in args.ref_trans:
                if wav_file.split("/")[-1].startswith("AEX") or wav_file.split("/")[-1].startswith("BEX"):
                    dict_pathologies["CTRL"].append(WER)
                if wav_file.split("/")[-1].startswith("PHO"):
                    dict_pathologies["SLA"].append(WER)
                if wav_file.split("/")[-1].startswith("CCM"):
                    dict_pathologies["PARK|CEREB"].append(WER)
                logging.info(dict_pathologies)
            WERs.append(WER)

        else:
            logging.warning(f"The file {wav_file} does not exist")

    # -------------------------------- Global WER     -------------------------------
    # --------------------------------------------------------------------------------
    logger.info("===================== GLOBAL WER ======================")
    logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", sum(WERs)/len(WERs))
    logger.info("=============================================================")
    if "spont" in args.ref_trans:
        WER_path={}
        for k in dict_pathologies:
            WER_path[k]=sum(dict_pathologies[k])/len(dict_pathologies[k])
        logging.info(WER_path)
if __name__ == "__main__":
    main(args)