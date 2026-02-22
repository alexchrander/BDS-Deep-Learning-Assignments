# Green Patent Detection (PatentSBERTa)
### Multi-Agent System + HITL

Binary classifier for detecting green/sustainable technology patent claims, built using PatentSBERTa with a Multi-Agent System (MAS) debate pipeline and Human-in-the-Loop (HITL) workflow. This is Assignment 3, extending Assignment 2 by replacing the simple LLM labeling step with a three-agent debate system.

---

## Project Structure

```
├── Assignment3_Green_Patent_Detection_MAS.ipynb  # Main notebook
├── mas_label.py                                  # MAS inference script (Part C, HPC)
├── finetune.py                                   # Fine-tuning script (Part D, HPC)
├── slurm_mas.sh                                  # SLURM job for MAS labeling
├── slurm_finetune.sh                             # SLURM job for fine-tuning
├── pyproject.toml                                # Dependencies
├── csv/
│   ├── hitl_green_100.csv                        # 100 uncertain examples (reused from Assignment 2)
│   ├── mas_labeled.csv                           # MAS Judge output
│   └── mas_human_labeled.csv                     # Final gold labels after human review
├── embeddings/                                   # Frozen PatentSBERTa embeddings (not in git)
│   ├── X_train.npy                               # Embeddings for train_silver
│   ├── X_pool.npy                                # Embeddings for pool_unlabeled
│   └── X_eval.npy                                # Embeddings for eval_silver
├── parquet/                                      # Dataset splits (not in git)
│   ├── train_silver.parquet                      # Training split (70%)
│   ├── pool_unlabeled.parquet                    # Unlabeled pool (20%)
│   ├── eval_silver.parquet                       # Eval split (10%)
│   └── train_gold.parquet                        # Gold-enhanced training set
├── models/                                       # Fine-tuned model (not in git)
│   └── patentsberta-finetuned/
├── baseline_clf.pkl                              # Trained baseline classifier (not in git)
└── logs/                                         # SLURM job logs
```

> **Note:** The following are excluded from git via `.gitignore` due to file size. Parts A & B artifacts are reused directly from Assignment 2:
> - `embeddings/`
> - `parquet/`
> - `models/`
> - `baseline_clf.pkl`

---

## Setup

Install [uv](https://github.com/astral-sh/uv) and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

Create a `.env` file with your HuggingFace token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

---

## How to Run

The project is split between the notebook (local) and HPC scripts. Follow the parts in order.

### Parts A & B – Setup (Reused from Assignment 2)
Run the setup cells in the notebook to load the baseline classifier, embeddings, and the 100 high-risk claims. No recomputation needed — all artifacts are reused from Assignment 2.

### Part C – Multi-Agent System (MAS)

**Step 1 – MAS labeling on HPC:**
```bash
sbatch slurm_mas.sh
```
Runs `mas_label.py` which orchestrates a three-agent debate using LangGraph:
- **Advocate** (Mistral-7B-Instruct-v0.2) — argues FOR green classification
- **Skeptic** (Qwen2.5-7B-Instruct) — argues AGAINST green classification
- **Judge** (Meta-Llama-3-8B-Instruct) — weighs both arguments and produces final label

Output saved to `csv/mas_labeled.csv`.

**Step 2 – Human review (Notebook):**
Copy `mas_labeled.csv` back locally and run cells 9a–9f. An interactive widget shows the full debate (advocate + skeptic arguments) before you assign the final gold label. Output saved to `csv/mas_human_labeled.csv`.

### Part D – Fine-tune PatentSBERTa

**Step 1 – Merge gold labels (Notebook):**
Run the merge cell to create `parquet/train_gold.parquet` (train_silver + gold_100).

**Step 2 – Fine-tuning on HPC:**
```bash
sbatch slurm_finetune.sh
```
Reuses `finetune.py` from Assignment 2. Fine-tunes PatentSBERTa for 1 epoch and saves model to `models/patentsberta-finetuned`.

---

## Results

### Model Comparison (eval_silver)

| Model Version | Training Data Source | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Baseline (frozen) | Frozen Embeddings (No Fine-tuning) | 0.77 | 0.77 | 0.77 | 0.77 |
| Assignment 2 Model | Fine-tuned on Silver + Gold (Simple LLM) | 0.81 | 0.81 | 0.81 | 0.81 |
| Assignment 3 Model | Fine-tuned on Silver + Gold (MAS) | 0.81 | 0.81 | 0.81 | 0.81 |

### gold_100

| | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Assignment 2 Model | 0.57 | 0.67 | 0.52 | 0.62 |
| Assignment 3 Model | 0.52 | 0.52 | 0.52 | 0.52 |

Lower performance on gold_100 is expected — these were selected as the most uncertain examples by the baseline model.

---

## HITL Summary

- 100 uncertain examples labeled via three-agent MAS debate → human review
- MAS labeled 51 as not green, 47 as green (4% low confidence)
- 98/100 claims parsed successfully

### MAS vs Simple LLM Label Quality

| | Not Green | Green | Low Confidence |
|---|---|---|---|
| Assignment 2 (Mistral) | 95 | 5 | 72% |
| Assignment 3 (MAS) | 51 | 47 | 4% |

---

## HuggingFace

- Model: [alexchrander/patent-sberta-green-finetuned-mas](https://huggingface.co/alexchrander/patent-sberta-green-finetuned-mas)
- Dataset: [alexchrander/patents-green-mas-dataset](https://huggingface.co/datasets/alexchrander/patents-green-mas-dataset)