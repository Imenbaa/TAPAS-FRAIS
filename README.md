## TAPAS-FRAIS DATASET
- Belge
- Good quality audio
- Spontanous
- Word and phone transcription
- Manually Verified: people >= 50 years : 68 TG
- Non verified: 49 TG
- Duration per file: between 0.01 min and 4.99 min
- Duration of all dataset: 2h 48min 25s
- 128 wav file

## Models

-asr-wav2vec2-commonvoice-fr: finetuned on commonvoice french => WER=9.96


| Datasets                 | Models                      | WER           |
|--------------------------|-----------------------------|---------------|
| TAPAS-FRAIS-verified     | asr-wav2vec2-commonvoice-fr | 21.14%        |
| TAPAS-FRAIS-non-verified | asr-wav2vec2-commonvoice-fr | 24.63%        |
