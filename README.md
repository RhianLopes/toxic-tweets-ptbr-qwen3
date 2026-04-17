# toxic-tweets-ptbr-qwen3

Classificação de comentários tóxicos em português brasileiro usando **Qwen3.5:9b** via Ollama — 100% local, sem API externa, rodando na GPU.

Dataset: **ToLD-Br** (21.000 tweets anotados). Estratégias avaliadas: 3 variantes de zero-shot e 3 de few-shot.

---

## Resultados

**Melhor variante: Few-Shot com 1 exemplo por categoria — F1-macro = 0.2994**

| # | Variante | F1-macro | F1-weighted | Accuracy |
|---|---|---|---|---|
| 1 | **FS-v1 1-Example** | **0.2994** | 0.7710 | 75.4% |
| 2 | FS-v2 2ex+Antibias | 0.2750 | 0.7712 | 76.0% |
| 3 | ZS-v2 Descriptions | 0.2673 | 0.7050 | 65.6% |
| 4 | FS-v3 2-Examples | 0.2606 | 0.7659 | 76.6% |
| 5 | ZS-v3 No-Antibias | 0.2516 | 0.7197 | 70.4% |
| 6 | ZS-v1 Base | 0.2347 | 0.7317 | 75.2% |

> **Métrica principal: F1-macro.** Accuracy é enganosa neste dataset — um classificador que rotula tudo como `not_toxic` acerta ~80% mas é inútil.

> **Escopo atual:** todos os experimentos foram realizados sobre a amostra estratificada de **500 tweets**. A avaliação sobre o dataset completo (21.000 tweets) ainda será feita.

### F1 por categoria — melhor variante (FS-v1)

| Categoria | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| not_toxic | 0.8670 | 0.8945 | 0.8412 | 403 |
| obscene | 0.4148 | 0.3500 | 0.5091 | 55 |
| homophobia | 0.5000 | 0.5000 | 0.5000 | 4 |
| insult | 0.3137 | 0.5333 | 0.2222 | 36 |
| misogyny | 0.0000 | — | — | 1 |
| racism | 0.0000 | — | — | 0 |
| xenophobia | 0.0000 | — | — | 1 |

### Conclusões

- **Few-shot > zero-shot** em F1-macro: os 3 primeiros lugares são todos few-shot
- **Mais exemplos ≠ melhor**: FS-v1 (1 exemplo) supera FS-v3 (2 exemplos) — prompts menores reduzem ambiguidade
- **Antibias prejudica**: adicionar instrução de mitigação de viés (FS-v2) reduziu F1-macro vs. FS-v1 simples
- **Classes raras zeradas**: misogyny, racism e xenophobia ficaram com F1=0 em todas as variantes — insuficientes na amostra de 500 tweets
- **Velocidade uniforme**: ~107 tokens/s em todas as variantes, independente do tamanho do prompt

---

## Pré-requisitos

- Python 3.14 (instalado via [python.org](https://www.python.org/downloads/))
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- WSL 2 com Ubuntu 24.04
- GPU com 12 GB+ de VRAM (testado na RTX 5070)

> **Windows:** o Python deve ser instalado pelo installer oficial do python.org. Instalações gerenciadas pelo `uv` (`AppData\Roaming\uv\python\`) são bloqueadas pela política de Controle de Aplicativo do Windows e impedem o kernel do Jupyter de iniciar.

---

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/RhianLopes/toxic-tweets-ptbr-qwen3.git
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

### 4. Instalar e configurar o Ollama (via WSL)

O Ollama roda no WSL para aproveitar a GPU via CUDA.

**4.1 — Habilitar rede espelhada no WSL**

Crie/edite `C:\Users\<SEU_USUARIO>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Reinicie o WSL:

```powershell
wsl --shutdown
```

**4.2 — Instalar o WSL Ubuntu 24.04**

```powershell
wsl --install Ubuntu-24.04
```

**4.3 — Instalar o Ollama no WSL**

```bash
sudo apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
```

**4.4 — Configurar Ollama para aceitar conexões do Windows**

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo sh -c 'echo "[Service]" > /etc/systemd/system/ollama.service.d/override.conf && echo "Environment=OLLAMA_HOST=0.0.0.0" >> /etc/systemd/system/ollama.service.d/override.conf'
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**4.5 — Baixar o modelo**

```bash
ollama pull qwen3.5:9b
```

Verificar se está rodando (do Windows):

```bash
curl http://127.0.0.1:11434
```

---

## Executar os notebooks

```bash
# Interativo
uv run python -m jupyter lab

# Headless (todos os notebooks de inferência levam ~2 min cada)
uv run python -m nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb --ExecutePreprocessor.timeout=600
```

---

## Estrutura do projeto

```
toxic-tweets-ptbr-qwen3/
├── data/
│   ├── raw/                                  # Dataset original ToLD-Br
│   └── sample/                               # Amostra estratificada (500 tweets)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_sampling.ipynb
│   ├── 03_ollama_setup.ipynb
│   ├── 04_zero_shot_v1_base.ipynb
│   ├── 04_zero_shot_v2_descriptions.ipynb
│   ├── 04_zero_shot_v3_no_antibias.ipynb
│   ├── 05_few_shot_v1_1ex.ipynb
│   ├── 05_few_shot_v2_2ex_antibias.ipynb
│   ├── 05_few_shot_v3_2ex.ipynb
│   └── 06_results_analysis.ipynb
├── results/
│   ├── zero_shot_v1_base.csv
│   ├── zero_shot_v2_descriptions.csv
│   ├── zero_shot_v3_no_antibias.csv
│   ├── few_shot_v1_1ex.csv
│   ├── few_shot_v2_2ex_antibias.csv
│   ├── few_shot_v3_2ex.csv
│   ├── predictions_comparison.csv
│   ├── metrics_per_strategy.csv
│   ├── metrics_summary.json
│   ├── ranking_f1_macro.png
│   ├── f1_por_categoria.png
│   └── confusion_matrices.png
└── pyproject.toml
```

---

## O que foi feito

### Fase 1 — Análise Exploratória (`01_exploratory_analysis.ipynb`)

Dataset: **ToLD-Br** — 21.000 tweets em português brasileiro, anotados manualmente com 6 categorias de toxicidade.

**Distribuição de labels:**

| Categoria | N | % |
|---|---|---|
| not_toxic | 16.937 | 80,65% |
| obscene | 2.296 | 10,93% |
| insult | 1.502 | 7,15% |
| homophobia | 169 | 0,80% |
| misogyny | 44 | 0,21% |
| xenophobia | 31 | 0,15% |
| racism | 21 | 0,10% |

**Principais achados:**
- Dataset altamente desbalanceado: razão `not_toxic / toxic` de **4,2x**
- Zero nulos em todas as colunas
- 361 tweets duplicados (1,72%)
- Tweets de `insult` e `xenophobia` tendem a ser levemente mais longos (~93 chars vs. média de ~87)
- Accuracy não é métrica adequada — um classificador ingênuo acertaria ~80% rotulando tudo como `not_toxic`
- **Métrica dos experimentos: F1-macro**

---

### Fase 2 — Amostragem Estratificada (`02_sampling.ipynb`)

Gerada uma amostra de **500 tweets** preservando as proporções originais do dataset.

Saída: `data/sample/toldBr_sample_500.csv`

| Categoria | Original | Amostra |
|---|---|---|
| not_toxic | 80,65% | 80,6% |
| obscene | 10,93% | 11,0% |
| insult | 7,15% | 7,2% |
| homophobia | 0,80% | 0,8% |
| misogyny | 0,21% | 0,2% |
| xenophobia | 0,15% | 0,2% |
| racism | 0,10% | — |

> `racism` ficou com 0 amostras — 0,1% × 500 = 0,5, arredondado para 0. Confirma a necessidade de oversampling explícito para classes muito raras.

---

### Fase 3 — Setup Ollama + Qwen3.5 (`03_ollama_setup.ipynb`)

Validação da infraestrutura de inferência local antes dos experimentos.

**Ambiente:**
- Ollama rodando no WSL (Ubuntu 24.04) com GPU via CUDA
- Modelo: `qwen3.5:9b` (~6,6 GB VRAM)
- GPU: NVIDIA RTX 5070 (12 GB VRAM)

**Benchmark de inferência (`think: false`):**

| Métrica | Valor |
|---|---|
| Tokens/s | 126,9 |
| Latência por tweet | 0,22s |
| Tokens gerados (classificação) | ~3 |

> Com `think: true` (padrão do Qwen3), o modelo entra em modo Chain-of-Thought e gera centenas de tokens antes de responder — inviável para 500 tweets. `think: false` é obrigatório.

---

### Fase 4 — Zero-Shot (`04_zero_shot_*.ipynb`)

Três variantes de prompt sem exemplos:

| Variante | Descrição | F1-macro |
|---|---|---|
| v1 Base | Lista de categorias + instrução mínima | 0.2347 |
| v2 Descriptions | Lista com descrição textual de cada categoria | 0.2673 |
| v3 No-Antibias | v1 sem instrução de mitigação de viés | 0.2516 |

**Achados:**
- Descrições das categorias (+0.03 F1 vs. base) ajudam o modelo a distinguir categorias próximas como `insult` e `obscene`
- A instrução de antibias da v1 é levemente benéfica: removê-la (v3) piora o desempenho
- Todas as variantes zeram nas classes raras (misogyny, racism, xenophobia)

---

### Fase 5 — Few-Shot (`05_few_shot_*.ipynb`)

Três variantes com exemplos no prompt:

| Variante | Descrição | F1-macro |
|---|---|---|
| v1 1-Example | 1 exemplo representativo por categoria | **0.2994** |
| v2 2ex+Antibias | 2 exemplos + instrução de mitigação de viés | 0.2750 |
| v3 2-Examples | 2 exemplos sem instrução de viés | 0.2606 |

**Achados:**
- Apenas 1 exemplo por categoria foi suficiente para superar todas as variantes zero-shot
- Dobrar os exemplos (v3) piorou o resultado em relação a v1 — prompts maiores podem introduzir ruído ou conflito com o estilo do tweet
- A instrução de antibias em v2 não compensou a penalidade de um prompt maior
- `insult` mostrou maior ganho com few-shot: F1 subiu de 0.087 (ZS-v1) para 0.314 (FS-v1)

---

### Fase 6 — Análise Comparativa (`06_results_analysis.ipynb`)

Consolidação e comparação das 6 variantes.

**Taxa de concordância entre variantes:**
- Few-shot × few-shot: ~82–84% de concordância entre si
- Zero-shot × few-shot: ~70–80%
- ZS-v2 diverge mais das demais (descrições criam distribuição de predições diferente)

**Gráficos gerados:**
- `ranking_f1_macro.png` — ranking horizontal por F1-macro
- `f1_por_categoria.png` — F1 por categoria em todas as variantes
- `confusion_matrices.png` — 6 matrizes de confusão lado a lado

---

## Referências

- [ToLD-Br — dataset original](https://github.com/joaoaleite/ToLD-Br)
- [Ollama](https://ollama.com)
- [Qwen3.5](https://ollama.com/library/qwen3.5)
