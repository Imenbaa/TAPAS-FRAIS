## Requirements


### Bash

```
pip install -r src/requirements.txt
```


## TAPAS-FRAIS DATASET
- Belge
- descriptive speech
- Good quality audio
- Spontanous
- Word and phone transcription
- Manually Verified: people >= 50 years : 68 Tg
- Non verified: 49 Tg (47 found in folder)
- Duration per file: between 0.01 min and 4.99 min
- Duration of all dataset: 2h 48min 25s
- 128 wav file
- Sampling rate 44100
## Rhapsodie Dataset

## Models

- ##### asr-wav2vec2-commonvoice-fr( with CTC/Attention trained on CommonVoice French => WER=9.96) ==> [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr):
    - Use LeBenchmark/wav2vec2-FR-7K-large
    - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions (train.tsv) of CommonVoice (FR).
    - Acoustic model (wav2vec2.0 + CTC). A pretrained wav2vec 2.0 model (LeBenchmark/wav2vec2-FR-7K-large) is combined with two DNN layers and finetuned on CommonVoice FR. The obtained final acoustic representation is given to the CTC greedy decoder.
- ##### HMM-TDNN (trained on ester)

- ##### Conformer (trained on ester | Commonvoice)

- ##### Whisper-medium (finetuned on commonvoice-14.0)

## VAd+ chunking for whisper

Limits of applying whisper on data directly:
- Whisper is trained on short speech segments (30 sec), will split arbitrary
- Whisper may hallucinate during silent regions --> need of VAD
- Whisper drop uncertain speech --> incomplete transcriptions
- Whisper normalize and summarize transcription

==> VAD alone does change nothing, it even gave worser results than without

### Proposed solution:

- Applying VAD to decide where there is speech ==> Mark where speech is detected without removing silence or changing audio
- Speech segments produced by the VAD are grouped into chunks of maximum duration of 30 seconds. Consecutive speech segments separated by a pause ≤ 0.6 seconds are merged. Longer pauses create a new chunk.
With this method the original timestamp is preserved and we prevent over-segmentation caused by micro-pauses.
- Then, for each chunk, the audio is extracted directly from the original waveform using its timestamps and chunks are processed independently
- For models with VAD and with chunking, the audio files were resampled to 16KHz

| Datasets                       | Models                         | WER        |
|--------------------------------|--------------------------------|------------|
| TAPAS-FRAIS-verified           | asr-wav2vec2-commonvoice-fr    | **24.63%** |
| TAPAS-FRAIS-non-verified       | asr-wav2vec2-commonvoice-fr    | **21.14%** |
| TAPAS-FRAIS-verified           | whisper-medium                 | 59.81%     |
| TAPAS-FRAIS-non-verified       | whisper-medium                 | 62.17%     |
| TAPAS-FRAIS-verified           | whisper-medium-VAD             | 62.39%     |
| TAPAS-FRAIS-verified           | whisper-medium-VAD-chunk       | 31.13%     |
| TAPAS-FRAIS-non-verified       | whisper-medium-VAD-chunk       | 34.44%     |
| TAPAS-FRAIS-verified           | whisper-Large-VAD-chunk        | 23.41%     |
| TAPAS-FRAIS-non-verified       | whisper-Large-VAD-chunk        | 25.88%     |
| TAPAS-FRAIS-non-verified       | whisper-Large-VAD(rouas)-chunk | 36.20%     |
  ---------------------------------------------------------------------------

- Interpretation: With wav2vec the WER on non verified is always better than verified by human. But for whisper, the WER on verified is better than non verified, this could be explained by the fact that whisper has a closer transcriptions to human because it interprets speech.





| Datasets             | Models                  | WER Rouas (rVAD) | WER (with Normalization) |
|----------------------|-------------------------|------------------|--------------------------|
| TAPAS-FRAIS-verified | HMM-TDNN (ester)        | 30.9%            | 27.95%                   |
| TAPAS-FRAIS-verified | Conformer (ester)       | 34.7%            | 34.09%                   |
| TAPAS-FRAIS-verified | Conformer (commonvoice) | 29%              | 26.97%                   |
  ----------------------------------------------------------------------------



| Datasets                 | Models                      | WER        |
|--------------------------|-----------------------------|------------| 
| Rhapsodie                | asr-wav2vec2-commonvoice-fr | 36.99%     |
| Rhapsodie                | Whisper-medium              | 67.05%     |
| Rhapsodie                | Whisper-medium-VAD-chunk    | 50.79%     |
| Rhapsodie                | Hmm_TDNN (ester)            | 35.05%     |
| Rhapsodie                | Conformer (ester)           | **32.32%** |
| Rhapsodie                | Whisper-large-VAD-chunk     | 45.73%     |
  ------------------------------------------------------------------
Dans la version Rhapsodie corrected, il y a des fichiers wav qui ont été retirés. 

Rhapsodie(corrected) on asr-wav2vec2-commonvoice-fr WER=33.75%     


Dans typaloc CEREB il y a un fichier .mix.textgrid a changer to .TextGrid

| Datasets          | Models                       | WER        |
|-------------------|------------------------------|------------| 
| Typaloc (PARK-8)  | asr-wav2vec2-commonvoice-fr  | **36.14%** |
| Typaloc (PARK-8)  | Whisper-medium               | 65.59%     |
| Typaloc (PARK-8)  | whisper-medium-VAD-chunk     | 46.04%     |
| Typaloc (PARK-8)  | whisper-large                | 55.85%     |
| Typaloc (PARK-8)  | whisper-large-VAD-chunk      | 38.13%     |

| Datasets          | Models                       | WER        |
|-------------------|------------------------------|------------| 
| Typaloc (CEREB-7) | asr-wav2vec2-commonvoice-fr  | 38.89%     |
| Typaloc (CEREB-7) | Whisper-medium               | 65.30%     |
| Typaloc (CEREB-7) | whisper-medium-VAD-chunk     | 37.07%     |
| Typaloc (CEREB-7) | whisper-large                | 57.91%     |
| Typaloc (CEREB-7) | whisper-large-VAD-chunk      | **33.59%** |

| Datasets          | Models                       | WER        |
|-------------------|------------------------------|------------| 
| Typaloc (SLA-12)  | asr-wav2vec2-commonvoice-fr  | 62.79%     |
| Typaloc (SLA-12)  | Whisper-medium               | 65.08%     |
| Typaloc (SLA-12)  | whisper-medium-VAD-chunk     | 48.76%     |
| Typaloc (SLA-12)  | whisper-large                | 61.38%     |
| Typaloc (SLA-12)  | whisper-large-VAD-chunk      | **37.72%** |

| Datasets          | Models                       | WER        |
|-------------------|------------------------------|------------| 
| Typaloc (CTR-12)  | asr-wav2vec2-commonvoice-fr  | 18.64%     |
| Typaloc (CTR-12)  | Whisper-medium               | 60.18%     |
| Typaloc (CTR-12)  | whisper-medium-VAD-chunk     | 19.63%     |
| Typaloc (CTR-12)  | whisper-large                | 60.21%     |
| Typaloc (CTR-12)  | whisper-large-VAD-chunk      | **13.52%** |
--------------------------------------------------------------

### WER Analysis for datasets across ASR models
#### Rhapsodie dataset
<img src="figures/wer_rhap_plot.png" width="600">

#### TAPAS dataset
<img src="figures/wer_tapas_plot.png" width="600">

#### Typaloc dataset
<img src="figures/wer_typaloc_plot.png" width="600">
<img src="figures/wer_typaloc_heatmap.png" width="600">

Remarques: Typaloc CEREB: CCM-002710-01_L01.TextGrid la transcription est décalé au début, elle n'est pas fiable et elle contient le mot "respiration"
'a', 'jeimais', 'je', 'ne', 'jenase', 'pas', 'nonp', 'premiere', 'rencontre' ce n'est pas mis dans la trasncription ref mais il y a ces mots dans l'audio
il y a des mots bisarre comme: prov9, kutyrje,l9,s2l9pla,za,ze,z9z, [su=qui] [su],  (les) [su=villageois] viZ@Zwa [su],[su=lutins] detE [su], *syRl@plaSe*