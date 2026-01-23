#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file CEREB_conf_rvad
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file CTRL_conf_rvad
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file PARK_conf_rvad

#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --vad silero --log_file CEREB_conf_silero
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --vad silero --log_file CTRL_conf_silero
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --vad silero --log_file PARK_conf_silero
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --vad rvad --log_file conf_tapas_rvad
#python -m evaluation_espnet --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --vad silero --log_file conf_tapas_silero

#python -m evaluation_espnet --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --vad rvad --log_file conf_rhap_rvad
#python -m evaluation_espnet --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --vad silero --log_file conf_rhap_silero
#python -m espnet_vad_rouas --i "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --vad rvad --log_file conf_rhap_rvad
#python -m espnet_vad_rouas --i "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --vad silero --log_file conf_rhap_silero

python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --vad rvad --log_file conf_tapas_rvad
python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --vad silero --log_file conf_tapas_silero

#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --vad silero --log_file CEREB_conf_silero
#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --vad rvad --log_file CEREB_conf_rvad

#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --vad silero --log_file CTRL_conf_silero
#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --vad rvad --log_file CTRL_conf_rvad

#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --vad silero --log_file PARK_conf_silero
#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --vad rvad --log_file PARK_conf_rvad

#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --vad rvad --log_file SLA_conf_rvad
#python -m espnet_vad_rouas --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --vad silero --log_file SLA_conf_silero

