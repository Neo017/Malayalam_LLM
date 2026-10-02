# Malayalam LLM

Open-source training recipes and evaluation tools for capable, responsible Malayalam language models.

Malayalam LLM is a community-first project for building language technology that understands and generates Malayalam (`മലയാളം`) while remaining useful in multilingual Indian workflows. The repository provides reproducible starter recipes for a small, approximately 1B-parameter causal language model:

- continued pretraining on clean Malayalam and Malayalam-English text;
- supervised instruction tuning for helpful, grounded responses;
- preference optimisation (DPO) from human or model-reviewed preference pairs;
- evaluation guidance for Malayalam generation, translation, reasoning and safety.

The project is designed for researchers, educators, developers and language-resource communities. It does not bundle a dataset or claim that a checkpoint is ready for production. Data provenance, licensing, personal-data removal, benchmark contamination and human review are part of the training process.

## Why Malayalam LLM?

Malayalam is spoken by tens of millions of people, yet high-quality open language-model resources remain limited compared with English. A Malayalam-focused model can improve search, education, translation, OCR post-correction, public-service access, speech and developer tooling—provided that it is evaluated by Malayalam speakers and released with transparent limitations.

## Repository status

This is a training and evaluation starter repository. The scripts are intentionally dataset-agnostic: supply a Hugging Face dataset identifier or a local dataset through command-line arguments. No private corpus, token, checkpoint or credential is embedded in the code.

## Quick start

```bash
git clone https://github.com/Neo017/Malayalam_LLM.git
cd Malayalam_LLM
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train]'
```

Authenticate only when a private or gated base model is required:

```bash
huggingface-cli login
```

### Continued pretraining

The text dataset must expose a text column. The script packs tokenized text into causal-language-model blocks.

```bash
python scripts/pretrain.py \
  --model MODEL_ID_OR_LOCAL_PATH \
  --dataset DATASET_ID_OR_LOCAL_PATH \
  --text-column text \
  --output-dir runs/mlm-pretrain
```

### Supervised fine-tuning

For instruction tuning, provide either a `text` column or a prompt/completion pair and select the columns explicitly.

```bash
python scripts/sft.py \
  --model runs/mlm-pretrain \
  --dataset DATASET_ID_OR_LOCAL_PATH \
  --text-column text \
  --output-dir runs/sft
```

### Preference optimisation / RLHF stage

The included recipe uses Direct Preference Optimisation (DPO), a stable preference-learning baseline for a small model. The preference dataset must contain `prompt`, `chosen`, and `rejected` columns. This stage does not require a separately trained reward model.

```bash
python scripts/rlhf_dpo.py \
  --model runs/sft \
  --dataset PREFERENCE_DATASET_ID_OR_LOCAL_PATH \
  --output-dir runs/dpo
```

Use `--lora` on the SFT or DPO command when GPU memory is limited. For a true online RLHF experiment, add a separately versioned reward model and compare against this DPO baseline; do not present DPO as PPO or as proof of human preference alignment.

## Recommended data contract

Training scripts do not select or download a dataset implicitly. The dataset owner should document:

| Stage | Required fields | Purpose |
|---|---|---|
| Pretraining | `text` | Malayalam or multilingual documents after filtering and deduplication |
| SFT | `text`, or `prompt` + `response` | Instruction-response examples with a documented format |
| Preference | `prompt`, `chosen`, `rejected` | Pairwise preference data with annotator/reviewer metadata kept separately |

Keep train, validation and test sources disjoint. Remove secrets and personal data, record source licences, preserve script and language labels, and inspect Malayalam Unicode normalisation. Deduplicate at document and near-duplicate levels before splitting.

## Evaluation

Report more than loss. At minimum, evaluate:

- Malayalam perplexity on a held-out, contamination-checked corpus;
- character and word correctness for spelling and OCR post-correction;
- translation with BLEU, chrF and COMET where appropriate;
- instruction-following and factuality with Malayalam-speaker review;
- refusal, privacy and prompt-injection behaviour;
- latency, memory use and long-context degradation.

Use the same held-out set across checkpoints, publish decoding settings, and report confidence intervals where possible. A lower validation loss is not automatically a better conversational model.

## Model scope

The starter configuration targets a roughly 1B causal decoder. Exact parameter count depends on the selected base architecture. This repository does not claim ownership of, or redistribute, a base model. Check the base model licence and its acceptable-use terms before training or publishing a derivative.

## Project layout

```text
Malayalam_LLM/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/1b.yaml
├── scripts/
│   ├── pretrain.py
│   ├── sft.py
│   └── rlhf_dpo.py
├── src/malayalam_llm/
│   └── __init__.py
└── .gitignore
```

## Responsible release checklist

Before publishing a checkpoint, include the base model and data licences, training compute, data mixture and filtering, known contamination, evaluation results, safety limitations, intended use, prohibited use and a reproducible hash of the configuration. Never commit API keys, Hugging Face tokens, raw personal data or unlicensed corpora.

## Contributing

Contributions are welcome in data documentation, Malayalam evaluation, Unicode tooling, training stability, inference efficiency and safety testing. Please open an issue before adding a new dataset or benchmark so licensing and contamination checks can be reviewed.

## Licence

The repository code is released under Apache-2.0. Models and datasets remain subject to their own licences and terms; see `LICENSE` and each model card.

