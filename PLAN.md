# toxic-tweets-ptbr-qwen3

> Classificação de comentários tóxicos em português brasileiro usando LLM local (Qwen3.5) via Ollama — sem API, sem custo, rodando na GPU.

---

## Objetivo

Avaliar a capacidade do **Qwen3.5:9b** em moderar conteúdo tóxico em português brasileiro, utilizando o dataset **ToLD-Br**, com duas abordagens de prompting:

| Abordagem | Descrição |
|---|---|
| **Zero-shot** | Classifica com descrição das categorias, sem exemplos |
| **Few-shot** | Classifica com 2 exemplos por categoria no prompt |

Tudo rodando **100% local** via Ollama no WSL, aproveitando a GPU (RTX 5070) para medir desempenho real (tokens/s, latência, custo zero vs. API paga).

---

## Dataset — ToLD-Br

**Repositório oficial:** [github.com/joaoaleite/ToLD-Br](https://github.com/joaoaleite/ToLD-Br)

- **~21.000 tweets** anotados manualmente em português brasileiro
- **Categorias:** `not_toxic`, `obscene`, `insult`, `homophobia`, `racism`, `misogyny`, `xenophobia`
- Dataset altamente desbalanceado: `not_toxic` = 80,65%
- Métrica principal: **F1-macro** (penaliza viés para classe majoritária)

---

## Estrutura do Projeto

```
toxic-tweets-ptbr-qwen3/
├── data/
│   ├── raw/                    # Dados originais do ToLD-Br
│   └── sample/                 # Amostra estratificada (500 tweets)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_sampling.ipynb
│   ├── 03_ollama_setup.ipynb
│   ├── 04_zero_shot.ipynb
│   ├── 05_few_shot.ipynb
│   └── 06_results_analysis.ipynb
├── results/
│   ├── zero_shot_results.csv
│   ├── few_shot_results.csv
│   ├── predictions_comparison.csv
│   ├── metrics_per_strategy.csv
│   └── metrics_summary.json
└── pyproject.toml
```

---

## Hardware de Referência

| Componente | Especificação |
|---|---|
| GPU | NVIDIA RTX 5070 (12 GB VRAM) |
| Modelo | Qwen3.5:9b (~6,6 GB VRAM) |
| Backend | Ollama (WSL Ubuntu 24.04) |
| OS | Windows 11 |

---

## Roadmap

- [x] Fase 1 — EDA do ToLD-Br
- [x] Fase 2 — Amostragem estratificada (500 tweets)
- [x] Fase 3 — Setup Ollama + Qwen3.5 no WSL
- [x] Fase 4 — Experimento Zero-shot
- [x] Fase 5 — Experimento Few-shot
- [x] Fase 6 — Análise comparativa de resultados
- [ ] Fase 7 — Artigo no Medium

---

## Artigo

**Título previsto:** *"Moderação de conteúdo em PT-BR com LLM local: testando o Qwen3.5 na GPU sem gastar um centavo com API"*

---

## Referências

- [ToLD-Br](https://github.com/joaoaleite/ToLD-Br)
- [Ollama](https://ollama.com)
- [Qwen3.5](https://ollama.com/library/qwen3.5)
