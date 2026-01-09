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

## Models

- ### asr-wav2vec2-commonvoice-fr( finetuned on commonvoice french => WER=9.96):
    - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions (train.tsv) of CommonVoice (FR).
    - Acoustic model (wav2vec2.0 + CTC). A pretrained wav2vec 2.0 model (LeBenchmark/wav2vec2-FR-7K-large) is combined with two DNN layers and finetuned on CommonVoice FR. The obtained final acoustic representation is given to the CTC greedy decoder.


| Datasets                 | Models                      | WER           |
|--------------------------|-----------------------------|---------------|
| TAPAS-FRAIS-verified     | asr-wav2vec2-commonvoice-fr | 21.14%        |
| TAPAS-FRAIS-non-verified | asr-wav2vec2-commonvoice-fr | 24.63%        |
