# Local ABSA V1 preflight report

## GPU

```text
GPU PREFLIGHT
-------------
CUDA available: false
Selected device: None
GPU: None
System GPU: NVIDIA GeForce RTX 4060 Laptop GPU
PyTorch: 2.13.0+cpu
PyTorch CUDA: None
Model execution device: None
GPU requirement: FAIL
Reason: CUDA GPU exists at the system level only if reported by nvidia-smi; this PyTorch build cannot use CUDA. Install a CUDA-enabled PyTorch build.
```

The RTX 4060 Laptop GPU is visible to the NVIDIA driver, but the active environment contains CPU-only PyTorch. Neural execution is therefore blocked until CUDA-enabled PyTorch is installed.

## Data and leakage

The repository contains 80 unique frozen human-labelled documents, 100 gold mentions, and 24 zero-aspect documents. These same 80 documents were used throughout OpenAI V1/V2/V3 development. No separate human-labelled training set or untouched evaluation set exists; training and testing on these records is prohibited.

## Recommended architecture

`zero_shot_multilingual_nli_ontology_entailment` using `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. It can condition on the frozen ontology definitions without fitting on the 80 gold labels. This makes an exploratory comparison possible while avoiding direct train/test overlap. Evidence spans are out of scope and will not be fabricated.

## Evaluation design

- Train: none.
- Threshold tuning: none for the first baseline; initial global threshold 0.70 is frozen before scoring.
- Evaluation: the 80 historically reused documents, labelled explicitly as exploratory development evidence, not confirmatory evidence.
- A genuinely unbiased comparison requires a new independently annotated holdout.

## Architecture assessment

- **setfit**: Not selected: no independent supervised training labels and sparse 20-label support.
- **xlmr_finetuning**: Not selected: 80 reused documents are insufficient for defensible 20-label fine-tuning and evaluation.
- **frozen_embeddings_linear**: Useful later if independent training labels become available; currently no clean train set.
- **zero_shot_multilingual_nli**: Selected first baseline: ontology-definition conditioning needs no gold training and permits an honest exploratory comparison on the reused benchmark.
- **generic_absa_libraries**: Not selected: aspect-term extraction formulations do not match the frozen category-level implicit-aspect task.

## Support

Aspect and language support tables are recorded in `local_absa_v1_preflight.json`. Rare aspects remain in the derived 20-column multi-label representation.

No neural model, paid API, OpenAI Batch, or full-corpus inference was run.
## CUDA repair and experiment completion (2026-08-21)

The project `.venv` was repaired with the official stable `torch==2.13.0+cu126` wheel; torchvision and torchaudio remained absent because the repository does not require them. `pip check` reported no broken requirements. PyTorch now reports CUDA 12.6 and one NVIDIA GeForce RTX 4060 Laptop GPU. The shared preflight passed a real 1024×1024 CUDA matrix multiplication with both input and output on `cuda:0`.

The pinned model `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` revision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` was downloaded to the local Hugging Face cache. A smoke forward pass verified model parameters, input IDs, attention mask and logits on `cuda:0`. No hosted inference endpoint was used.

The completed 80-document zero-shot evaluation used 1,600 ontology-entailment pairs, batch size 16, float16 autocast, and the fixed global threshold grid 0.50/0.60/0.70/0.80/0.90. The selected exploratory threshold is 0.50 because no threshold retained the minimum comparison recall of 0.64; 0.50 had the highest F1. It produced TP/FP/FN 23/85/77, precision/recall/F1 0.2130/0.2300/0.2212. Joint precision/recall/F1 was 0.1111/0.1200/0.1154. Matched sentiment accuracy/macro-F1 was 0.5217/0.3102 on 23 matched mentions. This is `LOCAL_NOT_CURRENTLY_VIABLE` relative to OpenAI V3 and remains reused-sample development evidence, not confirmatory evidence.
## Local experiment closure and production decision (2026-08-21)

The local experiment is formally `completed`, viability is `LOCAL_NOT_CURRENTLY_VIABLE`, and `local_selected_for_production=false`. All local outputs remain preserved as a research benchmark. Further zero-shot hypothesis tuning was deliberately stopped to avoid overfitting the reused 80-document development data. OpenAI V3 is selected for production with its known precision limitation; no V4, further local experiment, or additional manual review was performed. The repaired CUDA environment remains installed for potential future versioned research.
