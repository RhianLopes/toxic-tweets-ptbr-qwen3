# toxic-tweets-ptbr-qwen3

Classificação de comentários tóxicos em português brasileiro usando **Qwen3** via Ollama — 100% local, sem API, rodando na GPU.

---

## Pré-requisitos

- Python 3.14 (instalado via [python.org](https://www.python.org/downloads/))
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)
- GPU com 12 GB+ de VRAM (testado na RTX 5070)

> **Windows:** o Python deve ser instalado pelo installer oficial do python.org. Instalações gerenciadas pelo `uv` (`AppData\Roaming\uv\python\`) são bloqueadas pela política de Controle de Aplicativo do Windows e impedem o kernel do Jupyter de iniciar.

---

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/toxic-tweets-ptbr-qwen3.git
cd toxic-tweets-ptbr-qwen3
```

### 2. Criar o ambiente e instalar dependências

```bash
uv venv --python "C:/Users/<SEU_USUARIO>/AppData/Local/Python/pythoncore-3.14-64/python.exe"
uv sync
```

### 3. Registrar o kernel do Jupyter

```bash
uv run python -m ipykernel install --user --name toxic-tweets --display-name "Python 3.14 (toxic-tweets)"
```

Abra qualquer notebook e selecione o kernel **Python 3.14 (toxic-tweets)**.

### 4. Instalar e configurar o Ollama

```bash
# Após instalar o Ollama, baixar o modelo
ollama pull qwen3:14b    # recomendado para 12 GB+ VRAM
# ou
ollama pull qwen3:8b     # versão mais leve
```

Verificar se está rodando:

```bash
ollama list
```

---

## Executar os notebooks

```bash
# Interativo (VS Code / JupyterLab)
uv run jupyter lab

# Ou executar em modo headless
uv run python -m nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb
```

---

## Estrutura do projeto

```
toxic-tweets-ptbr-qwen3/
├── data/
│   ├── raw/                          # Dataset original ToLD-Br
│   └── sample/                       # Amostra estratificada (500 tweets)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_sampling.ipynb
│   ├── 03_zero_shot.ipynb
│   ├── 04_few_shot.ipynb
│   ├── 05_chain_of_thought.ipynb
│   └── 06_results_analysis.ipynb
├── src/
│   ├── prompts.py
│   ├── classifier.py
│   ├── metrics.py
│   └── utils.py
├── results/
├── PLAN.md                           # Planejamento detalhado do projeto
└── pyproject.toml
```

---

## O que foi feito

### Fase 1 — Análise Exploratória (`01_exploratory_analysis.ipynb`)

Dataset: **ToLD-Br** — 21.000 tweets em português brasileiro, anotados manualmente com 6 categorias de toxicidade.

**Distribuição de labels:**

| Categoria   | N      | %      |
|-------------|--------|--------|
| not_toxic   | 16.937 | 80,65% |
| obscene     | 2.296  | 10,93% |
| insult      | 1.502  | 7,15%  |
| homophobia  | 169    | 0,80%  |
| misogyny    | 44     | 0,21%  |
| xenophobia  | 31     | 0,15%  |
| racism      | 21     | 0,10%  |

**Principais achados:**
- Dataset altamente desbalanceado: razão `not_toxic / toxic` de **4,2x**
- Zero nulos em todas as colunas
- 361 tweets duplicados (1,72%)
- Tweets de `insult` e `xenophobia` tendem a ser levemente mais longos (~93 chars vs. média geral de ~87)
- Accuracy não é métrica adequada — um classificador ingênuo que rotula tudo como `not_toxic` acertaria ~80%
- **Métrica principal dos experimentos: F1-macro**

### Fase 2 — Amostragem Estratificada (`02_sampling.ipynb`)

Gerada uma amostra de **500 tweets** preservando as proporções originais do dataset.

Saída: `data/sample/toldBr_sample_500.csv`

| Categoria   | Original | Amostra |
|-------------|----------|---------|
| not_toxic   | 80,65%   | 80,6%   |
| obscene     | 10,93%   | 11,0%   |
| insult      | 7,15%    | 7,2%    |
| homophobia  | 0,80%    | 0,8%    |
| misogyny    | 0,21%    | 0,2%    |
| xenophobia  | 0,15%    | 0,2%    |
| racism      | 0,10%    | —       |

> `racism` ficou com 0 amostras por ter apenas 21 exemplos no dataset original (0,1% × 500 = 0,5, arredondado para 0).

---

## Próximos passos

- [ ] Fase 3 — Setup Ollama + Qwen3 e teste de conexão
- [ ] Fase 4 — Experimento Zero-shot (`03_zero_shot.ipynb`)
- [ ] Fase 5 — Experimento Few-shot (`04_few_shot.ipynb`)
- [ ] Fase 6 — Experimento Chain-of-Thought (`05_chain_of_thought.ipynb`)
- [ ] Fase 7 — Análise comparativa (`06_results_analysis.ipynb`)
- [ ] Fase 8 — Artigo no Medium

---

## Referências

- [ToLD-Br — dataset original](https://github.com/joaoaleite/ToLD-Br) (os CSVs já estão em `data/raw/` neste repositório)
- [Ollama](https://ollama.com)
- [Qwen3 — Alibaba Cloud](https://qwenlm.github.io/blog/qwen3/)
