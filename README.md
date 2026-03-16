# CLAG: Adaptive Memory Organization via Agent-Driven Clustering for SLM Agents

This repository contains the **ACL submission (anonymous) code** for the paper:

**“CLAG: Adaptive Memory Organization via Agent-Driven Clustering for SLM Agents”**

CLAG organizes agent memory via **agent-driven clustering** and performs retrieval in a **two-stage** manner
(cluster selection → within-cluster retrieval). Please refer to the code for full implementation details.

---

## Requirements

- Python 3.9+ recommended
- Install dependencies (example):

> Note: Some NLTK resources may be downloaded at runtime (network required) if not present.

---

## Data Preparation

### HotpotQA / LoCoMo
- `HotpotQA` and `LoCoMo` are already provided under the `data/` directory in this package.

### BioASQ
BioASQ is **not included** in this package. Please download the datasets from:
- https://participants-area.bioasq.org/datasets/

Download:
- **training10b**
- **testdata set**

Then place the downloaded files under `data/` and run:
```bash
bash run_prepare_bioasq_all.sh
```

The script will generate processed files under `data/processed/` (e.g., chunked versions used by the experiments).

---

## Running CLAG Experiments

To run CLAG experiments/evaluation, execute **`test_CLAG.py`**.

Example (LoCoMo):
```bash
python3 test_CLAG.py \
  --dataset data/locomo10.json \
  --backend sglang \
  --model gpt-4o-mini
```

Common options (see `test_CLAG.py` for the full list):
- `--dataset`: path to the evaluation dataset JSON
- `--backend`: `openai` | `ollama` | `sglang` (default: `sglang`)
- `--model`: model name for the selected backend
- `--output`: output JSON path
- `--ratio`: evaluate a subset of the dataset (0–1)
- `--retrieve_k`: retrieval top-k (default: 10)

### Backend Notes
- **OpenAI backend**: set environment variable
```bash
export OPENAI_API_KEY="YOUR_KEY"
```
- **SGLang backend**: run an SGLang server (default `http://localhost:30000`).
You can configure via `--sglang_host` / `--sglang_port`.

---

## Code Pointers (Where to Look)
- `test_CLAG.py`: evaluation loop / CLI arguments / experiment entry point
- `CLAG_memory.py`: CLAG memory + clustering + (cluster→local) retrieval logic
- `prepare_bioasq.py`, `prepare_bioasq_gold_context.py`, `run_prepare_bioasq_all.sh`: BioASQ preprocessing

---

## Output
Runs may produce logs and artifacts under folders such as:
- `logs_CLAG/`
- `results_CLAG/`

---

## License
This code package is provided for **ACL anonymous submission** purposes.
