import subprocess
from pathlib import Path
import tempfile
import logging
import argparse
import logging
import os
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
import librosa
def run_asr(bash_script, wav_path, ctm_path):
    cmd = [
        bash_script,
        str(wav_path),
        str(ctm_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ASR failed for {wav_path}\n{result.stderr}"
        )
def ctm_to_text(ctm_path):
    words = []

    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                word = parts[4]
                if word != "<eps>":
                    words.append(word)

    return " ".join(words)

def transcribe_audio(
    wav_path,
    bash_script,
    work_dir=None
):
    """
    Transcrit un fichier audio WAV avec Kaldi (via recognizer.sh)
    et retourne le texte final.
    """

    wav_path = Path(wav_path)
    bash_script = Path(bash_script)

    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    if not bash_script.exists():
        raise FileNotFoundError(bash_script)

    # Dossier de travail pour le CTM
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    ctm_path = work_dir / f"{wav_path.stem}.ctm"

    # Appel du script bash
    cmd = [
        str(bash_script),
        str(wav_path),
        str(ctm_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ASR failed:\n{result.stderr}"
        )

    # CTM → texte
    words = []
    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                word = parts[4]
                if word != "<eps>":
                    words.append(word)

    text = " ".join(words)

    return text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcription ASR via Kaldi (recognizer.sh)")
    parser.add_argument("--wav_data",required=True,help="Chemin vers le dossier des WAV")
    parser.add_argument("--ref_trans", type=str, required=True, help="The reference transcription")
    parser.add_argument("--bash",default="/vol/experiments3/imbenamor/TAPAS-FRAIS/src/asr_FR_kaldi_hmm_tdnn.sh",help="Chemin vers le script recognizer.sh")
    parser.add_argument("--work-dir",default="/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/hmm_tdnn/",help="Dossier de travail pour les fichiers CTM (optionnel)")
    parser.add_argument("--log_file", type=str, required=True, help="The logfile name")

    args = parser.parse_args()
    log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/hmm_tdnn/" + args.log_file + ".log")
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    WERs = []
    number_files = 0
    tg_to_wav = list_files(args.ref_trans)
    for tg, wav in tg_to_wav.items():
        wav_file = os.path.join(args.wav_data, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg != "CCM-004773-01_L01.TextGrid" and wav != "CCM-004773-01_L01.wav":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            if "Rhapsodie" in args.ref_trans:
                ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
            else:
                ref_transcriptions = get_textgrid_transcription_tapas(trans_file)

            pred_transcriptions = transcribe_audio(wav_path=wav_file,bash_script=args.bash,work_dir=args.work_dir)
            ref_transcriptions = remove_words(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions)
            WERs.append(WER)
    logger.info("===================== GLOBAL WER ======================")
    logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", sum(WERs) / len(WERs))
    logger.info("=============================================================")


