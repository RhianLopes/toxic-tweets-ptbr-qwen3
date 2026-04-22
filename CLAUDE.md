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

Evaluate **Qwen3.5** (via Ollama, 100% local) on toxic content moderation in Brazilian Portuguese, using the **ToLD-Br** dataset. Strategies compared: zero-shot, few-shot, and RAG (BM25, dense, hybrid). Primary metric: **F1-macro** (accuracy is misleading due to class imbalance).

## Dataset

- **Source:** `data/raw/toldBr_full.csv` — 21.000 tweets, 8 colunas (`text` + 6 toxicity scores + `label`). CSVs já incluídos no repositório.
- **Sample:** `data/sample/toldBr_sample_500.csv` — 500 tweets, stratified
- `label` is derived: the category with highest score ≥ 2, otherwise `not_toxic`
- Highly imbalanced: `not_toxic` = 80.65%, `racism` = 0.10%

## Status

| Notebook / Script | Status | Output |
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
| `10_rag_data_split.ipynb` | Done | `data/full/toldBr_train.csv`, `data/full/toldBr_val.csv` |
| `scripts/10_rag_bm25_full.py` | Done | `results/full/rag_bm25_k3.csv` |
| `scripts/11_rag_vector_full.py` | Done | `results/full/rag_vector_k3.csv` |
| `scripts/12_rag_hybrid_qdrant_full.py` | Done | `results/full/rag_hybrid_qdrant_k3.csv` |
| `scripts/13_rag_diverse_bm25_full.py` | Done | `results/full/rag_diverse_bm25_k1.csv` |
| `scripts/14_rag_diverse_vector_full.py` | Done | `results/full/rag_diverse_vector_k1.csv` |
| `scripts/15_rag_diverse_hybrid_full.py` | Done | `results/full/rag_diverse_hybrid_k1.csv` |
| `16_rag_results_analysis.ipynb` | Done | Análise F1 das 6 variantes RAG |

## Variantes de prompting

### Zero-Shot (04)
- **v1 Base** — instrução mínima com lista de categorias
- **v2 Descriptions** — instrução com descrição textual de cada categoria
- **v3 No-Antibias** — v1 sem instrução de mitigação de viés

### Few-Shot (05)
- **v1 1-Example** — 1 exemplo por categoria no prompt
- **v2 2ex+Antibias** — 2 exemplos + instrução de mitigação de viés
- **v3 2-Examples** — 2 exemplos sem instrução de viés

### RAG (10-15) — split 80/20 estratificado (16.800 train / 4.200 val)

**K=3 global (top-3 mais similares):**
- **BM25** — top-3 via BM25 Okapi (`rank-bm25`)
- **Vector** — top-3 via embeddings MiniLM-L12 multilingual + cosine similarity
- **Hybrid Qdrant** — top-3 via dense (MiniLM) + sparse (TF-IDF) fundidos com RRF no Qdrant in-memory

**Diversidade forçada (top-1 por categoria = 7 exemplos):**
- **Diverse BM25** — BM25 dentro de cada categoria: melhor tweet de cada uma das 7 classes
- **Diverse Vector** — MiniLM top-1 por categoria: maior cosine similarity dentro de cada classe
- **Diverse Hybrid** — 7 Qdrant collections in-memory (uma por categoria), RRF sem filtros

Embeddings do corpus cacheados em `data/full/train_embeddings.npy`. TF-IDF usado no sparse: tweets curtos (~87 chars) eliminam a vantagem de normalização do BM25. 7 collections separadas contornam a limitação do Qdrant in-memory (sem suporte a payload index).

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

### Resultados RAG — val set 80/20 (4.200 tweets)

| # | Variante | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | **RAG-Hybrid Qdrant RRF (K=3)** | **0.2986** | 0.7771 |
| 2 | RAG-Diverse Vector MiniLM (1/classe) | 0.2890 | 0.7609 |
| 3 | RAG-BM25 (K=3) | 0.2874 | 0.7711 |
| 4 | RAG-Diverse Hybrid Qdrant RRF (1/classe) | 0.2847 | 0.7634 |
| 5 | RAG-Diverse BM25 (1/classe) | 0.2776 | 0.7591 |
| 6 | RAG-Vector MiniLM (K=3) | 0.2647 | 0.7689 |

- RAG-Hybrid K=3 lidera (0.2986): competitivo com FS-v2 (0.3173 no full) sem exemplos fixos
- Diversidade beneficia o dense: Diverse Vector (0.2890) > Vector K=3 (0.2647) — garante 1 exemplo/classe
- Diversidade não ajuda híbrido/BM25: K=3 global supera variantes diversas nesses métodos
- Val set tem classes raras muito pequenas (racism=4, xenophobia=6) — F1 dessas classes é ruidoso
