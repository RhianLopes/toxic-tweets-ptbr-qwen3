# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **OS:** Windows 11 — use Unix shell syntax in Bash (forward slashes, `/dev/null`, etc.)
- **Python:** 3.14, installed via python.org installer at `C:\Users\USER\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- **Package manager:** `uv` — use `uv run python` instead of `python` directly
- **WDAC restriction:** The uv-managed Python (`AppData\Roaming\uv\python\`) is blocked by Windows Application Control and cannot load DLLs (sqlite3, zmq, etc.). Always use the system Python 3.14 from `AppData\Local\Python\`.

## Common commands

```bash
# Execute a notebook (headless)
uv run python -m nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb --ExecutePreprocessor.timeout=120

# Start JupyterLab interactively
uv run jupyter lab

# Add a dependency
uv add <package>

# Sync dependencies after changing pyproject.toml
uv sync
```

> `uv run jupyter` (the .exe) is also blocked by WDAC. Always use `uv run python -m nbconvert` or `uv run python -m jupyter` instead.

## Project goal

Evaluate **Qwen3** (via Ollama, 100% local) on toxic content moderation in Brazilian Portuguese, using the **ToLD-Br** dataset. Three prompting strategies are compared: zero-shot, few-shot, and chain-of-thought. Primary metric: **F1-macro** (accuracy is misleading due to class imbalance).

## Dataset

- **Source:** `data/raw/toldBr_full.csv` — 21.000 tweets, 8 colunas (`text` + 6 toxicity scores + `label`). CSVs já incluídos no repositório.
- **Sample:** `data/sample/toldBr_sample_500.csv` — 500 tweets, stratified
- `label` is derived: the category with highest score ≥ 2, otherwise `not_toxic`
- Highly imbalanced: `not_toxic` = 80.65%, `racism` = 0.10%

## Status

| Notebook | Status | Output |
|---|---|---|
| `01_exploratory_analysis.ipynb` | Done | EDA do ToLD-Br |
| `02_sampling.ipynb` | Done | `data/sample/toldBr_sample_500.csv` |
| `03_zero_shot.ipynb` | Pending | `results/zero_shot_results.csv` |
| `04_few_shot.ipynb` | Pending | `results/few_shot_results.csv` |
| `05_chain_of_thought.ipynb` | Pending | `results/cot_results.csv` |
| `06_results_analysis.ipynb` | Pending | `results/metrics_summary.json` |

## Ollama / model

- Model: `qwen3:14b` (Q4 ~9 GB VRAM) or `qwen3:8b`
- API base: `http://localhost:11434`
- Endpoint: `POST /api/generate` with `{"model": "qwen3:14b", "prompt": "...", "stream": false}`

## Key findings so far

- `racism` has only 21 examples — proportional sampling yields 0 samples at N=500; handle explicitly if needed
- Tweet length is similar across categories (~87 chars average); `insult` slightly longer (~93)
- 361 duplicate tweets (1.72%) exist in the full dataset — already accounted for in the sample
