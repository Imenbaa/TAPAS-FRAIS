
cd /vol/experiments3/imbenamor/TAPAS-FRAIS/src

#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_rhap_corrected
#python -m eval_rouas_models --pred_trans "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/transcription_tdnn_hmm/e2e_transcriptions_original" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_conformer_rouas

#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/HMM-TDNN/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_hmm_tdnn_rouas
#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/E2E-conformer-ester+vad/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_conf_rouas
#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/E2E-conformer-commonvoice/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_conf_cv_rouas

#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_vad_whisper_medium
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid non verifie" --log_file wer_tapas_nonverif_whisper_medium
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_rhap_whisper_medium
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid non verifie" --wav_16k /vol/experiments3/imbenamor/TAPAS-FRAIS/data/tapas16k-nonverif --log_file wer_tapas16k_nonverif_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_typaloc_PARK_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_typaloc_CEREB_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_typaloc_SLA_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_typaloc_CTR_wav2vec


#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_typaloc_PARK_whisper
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_typaloc_CEREB_whisper
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_typaloc_SLA_whisper
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_typaloc_CTR_whisper

#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_whisper_large_CEREB
#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_whisper_large_PARK
#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_whisper_large_SLA
#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_whisper_large_CTRL


python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid non verifie" --log_file wer_whisper-VAD-chunk_large-tapas-nonverif--rvad
#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_whisper_large-vad-chunk_rhap
