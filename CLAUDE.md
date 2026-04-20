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

Evaluate **Qwen3.5** (via Ollama, 100% local) on toxic content moderation in Brazilian Portuguese, using the **ToLD-Br** dataset. Two prompting strategies are compared: zero-shot and few-shot. Primary metric: **F1-macro** (accuracy is misleading due to class imbalance).

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
| `03_ollama_setup.ipynb` | Done | Setup e benchmark do Ollama |
| `04_zero_shot_v1_base.ipynb` | Done | `results/zero_shot_v1_base.csv` |
| `04_zero_shot_v2_descriptions.ipynb` | Done | `results/zero_shot_v2_descriptions.csv` |
| `04_zero_shot_v3_no_antibias.ipynb` | Done | `results/zero_shot_v3_no_antibias.csv` |
| `05_few_shot_v1_1ex.ipynb` | Done | `results/few_shot_v1_1ex.csv` |
| `05_few_shot_v2_2ex_antibias.ipynb` | Done | `results/few_shot_v2_2ex_antibias.csv` |
| `05_few_shot_v3_2ex.ipynb` | Done | `results/few_shot_v3_2ex.csv` |
| `06_results_analysis.ipynb` | Done | `results/metrics_summary.json` |
| `09_results_analysis_full.ipynb` | Done | `results/full/metrics_summary.json` |

## Variantes de prompting

### Zero-Shot (04)
- **v1 Base** — instrução mínima com lista de categorias
- **v2 Descriptions** — instrução com descrição textual de cada categoria
- **v3 No-Antibias** — v1 sem instrução de mitigação de viés

### Few-Shot (05)
- **v1 1-Example** — 1 exemplo por categoria no prompt
- **v2 2ex+Antibias** — 2 exemplos + instrução de mitigação de viés
- **v3 2-Examples** — 2 exemplos sem instrução de viés

## Ollama / model

- Model: `qwen3.5:9b` — instalado no WSL (Ubuntu 24.04), ~6.6 GB VRAM
- GPU: NVIDIA RTX 5070 (12 GB VRAM) — detectada pelo Ollama via WSL
- API base: `http://127.0.0.1:11434` (usar IP explícito, não `localhost` — evita resolução IPv6)
- Endpoint: `POST /api/generate` com `{"model": "qwen3.5:9b", "prompt": "...", "stream": false, "think": false}`
- **`think: false` é obrigatório** — sem isso o modelo entra em modo CoT e gera centenas de tokens antes de responder, tornando a classificação inviável (~3 min por tweet vs 0.22s)
- Velocidade medida: **126.9 tokens/s, 0.22s de latência** por classificação

## Ollama no WSL — configuração

O Ollama roda no WSL (Ubuntu 24.04), não no Windows nativo. Configurações necessárias:

1. **`~/.wslconfig`** (em `C:\Users\USER\`) — habilita rede espelhada para expor portas do WSL ao Windows:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. **`OLLAMA_HOST=0.0.0.0`** — necessário mesmo com mirrored networking para o Ollama aceitar conexões de fora do loopback WSL:
   ```
   /etc/systemd/system/ollama.service.d/override.conf
   [Service]
   Environment=OLLAMA_HOST=0.0.0.0
   ```

3. O serviço é gerenciado pelo systemd e inicia automaticamente com o WSL.

## Key findings

### Dataset
- `racism` has only 21 examples — proportional sampling yields 0 samples at N=500; handle explicitly if needed
- Tweet length is similar across categories (~87 chars average); `insult` slightly longer (~93)
- 361 duplicate tweets (1.72%) exist in the full dataset — already accounted for in the sample

### Resultados dos experimentos — dataset completo (20.813 tweets)

| Variante | F1-macro | F1-weighted |
|---|---|---|
| **FS-v2 2ex+Antibias** | **0.3173** | 0.7666 |
| ZS-v2 Descriptions | 0.2875 | 0.7240 |
| FS-v3 2-Examples | 0.2747 | 0.7659 |
| FS-v1 1-Example | 0.2706 | 0.7547 |
| ZS-v1 Base | 0.2238 | 0.7501 |
| ZS-v3 No-Antibias | 0.2206 | 0.7376 |

### Resultados da amostra de validação (500 tweets)

| Variante | F1-macro | F1-weighted |
|---|---|---|
| **FS-v1 1-Example** | **0.2994** | 0.7710 |
| FS-v2 2ex+Antibias | 0.2750 | 0.7712 |
| ZS-v2 Descriptions | 0.2673 | 0.7050 |
| FS-v3 2-Examples | 0.2606 | 0.7659 |
| ZS-v3 No-Antibias | 0.2516 | 0.7197 |
| ZS-v1 Base | 0.2347 | 0.7317 |

### Interpretações
- FS-v2 domina no full: era 2º no sample (+0.0423 F1-macro) — antibias + 2 exemplos venceu em escala
- Sample não discriminou variantes próximas: inversão FS-v2↔FS-v1 só ficou clara nos 20k tweets
- Few-shot supera zero-shot: 3 das 4 melhores variantes são few-shot
- ZS-v2 destaca-se entre zero-shot: descrições de categoria chegam ao 2º lugar geral
- Classes raras ganham suporte real: racism (21), xenophobia (31), misogyny (44) — saem do zero no full
- `not_toxic` estável em 0.82–0.87 F1 em todas as variantes no dataset completo
- `obscene` é a categoria tóxica melhor classificada (~0.22–0.39 F1 no full)
