## Requirements


### Bash

```
pip install -r src/requirements.txt
```


## TAPAS-FRAIS DATASET
- Belge
- Good quality audio
- Spontanous
- Word and phone transcription
- Manually Verified: people >= 50 years : 68 Tg
- Non verified: 49 Tg
- Duration per file: between 0.01 min and 4.99 min
- Duration of all dataset: 2h 48min 25s
- 128 wav file
## Rhapsodie Dataset

## Models

- ### asr-wav2vec2-commonvoice-fr( finetuned on commonvoice french => WER=9.96) ==> [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr):
    - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions (train.tsv) of CommonVoice (FR).
    - Acoustic model (wav2vec2.0 + CTC). A pretrained wav2vec 2.0 model (LeBenchmark/wav2vec2-FR-7K-large) is combined with two DNN layers and finetuned on CommonVoice FR. The obtained final acoustic representation is given to the CTC greedy decoder.
- ### HMM-TDNN (trained on ester)
- ### Conformer (trained on ester)
- ### Conformer (trained on commonvoice)

| Datasets                 | Models                      | WER    |
|--------------------------|-----------------------------|--------|
| TAPAS-FRAIS-verified     | asr-wav2vec2-commonvoice-fr | 21.14% |
| TAPAS-FRAIS-non-verified | asr-wav2vec2-commonvoice-fr | 24.63% |
  ------------------------------------------------------------------
| Datasets             | Models                  | WER   | WER (normalized) |
|----------------------|-------------------------|-------|------------------|
| TAPAS-FRAIS-verified | HMM-TDNN (ester)        | 30.9% | 27.95%           |
| TAPAS-FRAIS-verified | Conformer (ester)       | 34.7% | 34.09%           |
| TAPAS-FRAIS-verified | Conformer (commonvoice) | 29%   | 26.97%           |
  ----------------------------------------------------------------------------



| Datasets                 | Models                      | WER    |
|--------------------------|-----------------------------|--------| 
| Rhapsodie                | asr-wav2vec2-commonvoice-fr | 36.99% |
| Rhapsodie(corrected)     | asr-wav2vec2-commonvoice-fr | 33.75% |
| Rhapsodie                | Hmm_TDNN (ester)            | 35.05% |
| Rhapsodie                | Conformer (ester)           | 32.32% |
  ------------------------------------------------------------------
Dans la version Rhapsodie corrected, il y a des fichiers wav qui ont été retirés.

