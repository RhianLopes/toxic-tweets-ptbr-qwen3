# toxic-tweets-ptbr-qwen3

Classificação de comentários tóxicos em português brasileiro usando **Qwen3.5:9b** via Ollama — 100% local, sem API externa, rodando na GPU.

Dataset: **ToLD-Br** (21.000 tweets anotados). Estratégias avaliadas: 3 variantes de zero-shot e 3 de few-shot.

---

## Resultados

### Variantes avaliadas

| Código | Tipo | O que muda no prompt |
|---|---|---|
| **ZS-v1 Base** | Zero-shot | Instrução mínima: lista das 7 categorias |
| **ZS-v2 Descriptions** | Zero-shot | Lista das categorias + descrição textual do que cada uma significa |
| **ZS-v3 No-Antibias** | Zero-shot | Igual à ZS-v1, mas sem a instrução de mitigação de viés |
| **FS-v1 1-Example** | Few-shot | 1 tweet de exemplo por categoria (7 exemplos no total) |
| **FS-v2 2ex+Antibias** | Few-shot | 2 tweets de exemplo por categoria + instrução de mitigação de viés |
| **FS-v3 2-Examples** | Few-shot | 2 tweets de exemplo por categoria, sem instrução de viés |

> **Zero-shot**: o modelo classifica sem ver exemplos — só com a instrução. **Few-shot**: o modelo vê exemplos reais de cada categoria antes de classificar.

---

### Dataset completo (20.813 tweets)

**Melhor variante: FS-v2 2ex+Antibias — F1-macro = 0.3173**

| # | Variante | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | **FS-v2 2ex+Antibias** | **0.3173** | 0.7666 |
| 2 | ZS-v2 Descriptions | 0.2875 | 0.7240 |
| 3 | FS-v3 2-Examples | 0.2747 | 0.7659 |
| 4 | FS-v1 1-Example | 0.2706 | 0.7547 |
| 5 | ZS-v1 Base | 0.2238 | 0.7501 |
| 6 | ZS-v3 No-Antibias | 0.2206 | 0.7376 |

> **Métrica principal: F1-macro.** Accuracy é enganosa neste dataset — um classificador que rotula tudo como `not_toxic` acerta ~80% mas é inútil.

### F1 por categoria — melhor variante no dataset completo (FS-v2)

| Categoria | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| not_toxic | 0.8722 | 0.8723 | 0.8720 | 16.783 |
| obscene | 0.3720 | 0.3526 | 0.3937 | 2.276 |
| insult | 0.2624 | 0.4303 | 0.1887 | 1.489 |
| homophobia | 0.4253 | 0.3717 | 0.4970 | 169 |
| misogyny | 0.0700 | 0.0423 | 0.2045 | 44 |
| xenophobia | 0.1167 | 0.0670 | 0.4516 | 31 |
| racism | 0.1023 | 0.0567 | 0.5238 | 21 |

### Amostra de validação (500 tweets)

| # | Variante | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | FS-v1 1-Example | 0.2994 | 0.7710 |
| 2 | FS-v2 2ex+Antibias | 0.2750 | 0.7712 |
| 3 | ZS-v2 Descriptions | 0.2673 | 0.7050 |
| 4 | FS-v3 2-Examples | 0.2606 | 0.7659 |
| 5 | ZS-v3 No-Antibias | 0.2516 | 0.7197 |
| 6 | ZS-v1 Base | 0.2347 | 0.7317 |

> A amostra foi usada para validação durante o desenvolvimento. FS-v2 era 2º no sample (+0.0423 F1-macro no full) — a inversão de ranking só ficou evidente no dataset completo.

### Conclusões

- **FS-v2 domina no dataset completo**: antibias + 2 exemplos mostrou vantagem real em escala — era 2º no sample, assume o 1º lugar com 20k tweets
- **Sample não discrimina variantes próximas**: FS-v2 e FS-v1 separadas por 0.02 F1 no sample; no full a diferença sobe para 0.047
- **Few-shot > zero-shot** em F1-macro: 3 das 4 melhores variantes são few-shot
- **ZS-v2 se destaca entre os zero-shot**: descrições de categoria chegam a 2º lugar geral — compensam parcialmente a ausência de exemplos
- **Classes raras ganham poder de avaliação**: no dataset completo, racism (21), xenophobia (31) e misogyny (44) saem do zero — FS-v2 atinge F1=0.10/0.12/0.07
- **Velocidade uniforme**: ~107–127 tokens/s em todas as variantes, independente do tamanho do prompt

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

## Executar os notebooks e scripts

```bash
# Interativo
uv run python -m jupyter lab

# Notebooks headless (análises)
uv run python -m nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb --ExecutePreprocessor.timeout=600

# Scripts de inferência no dataset completo (suportam retomada via checkpoint)
uv run python scripts/07_zero_shot_full_v1_base.py
uv run python scripts/07_zero_shot_full_v2_descriptions.py
uv run python scripts/07_zero_shot_full_v3_no_antibias.py
uv run python scripts/08_few_shot_full_v1_1ex.py
uv run python scripts/08_few_shot_full_v2_2ex_antibias.py
uv run python scripts/08_few_shot_full_v3_2ex.py
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
│   ├── 06_results_analysis.ipynb
│   └── 09_results_analysis_full.ipynb
├── scripts/
│   ├── 07_zero_shot_full_v1_base.py
│   ├── 07_zero_shot_full_v2_descriptions.py
│   ├── 07_zero_shot_full_v3_no_antibias.py
│   ├── 08_few_shot_full_v1_1ex.py
│   ├── 08_few_shot_full_v2_2ex_antibias.py
│   └── 08_few_shot_full_v3_2ex.py
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
│   ├── confusion_matrices.png
│   └── full/                                    # Dataset completo (20.813 tweets)
│       ├── zero_shot_v1_base.csv
│       ├── zero_shot_v2_descriptions.csv
│       ├── zero_shot_v3_no_antibias.csv
│       ├── few_shot_v1_1ex.csv
│       ├── few_shot_v2_2ex_antibias.csv
│       ├── few_shot_v3_2ex.csv
│       ├── metrics_summary.json
│       ├── metrics_per_strategy.csv
│       ├── ranking_f1_macro.png
│       ├── f1_por_categoria.png
│       └── confusion_matrices.png
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

### Fase 7 — Zero-Shot no dataset completo (`scripts/07_zero_shot_full_v*.py`)

Replicação das 3 variantes zero-shot sobre os 20.813 tweets do dataset completo. Scripts Python com checkpoint a cada 500 tweets para retomada automática em caso de interrupção.

### Fase 8 — Few-Shot no dataset completo (`scripts/08_few_shot_full_v*.py`)

Replicação das 3 variantes few-shot sobre o dataset completo. Mesma estrutura dos scripts da Fase 7.

### Fase 9 — Análise comparativa full (`09_results_analysis_full.ipynb`)

Consolidação e comparação das 6 variantes no dataset completo (~20.639 tweets válidos). Inclui comparação sample vs. full para cada variante e análise detalhada de classes raras.

**Ranking final — dataset completo:**

| # | Variante | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | **FS-v2 2ex+Antibias** | **0.3173** | 0.7666 |
| 2 | ZS-v2 Descriptions | 0.2875 | 0.7240 |
| 3 | FS-v3 2-Examples | 0.2747 | 0.7659 |
| 4 | FS-v1 1-Example | 0.2706 | 0.7547 |
| 5 | ZS-v1 Base | 0.2238 | 0.7501 |
| 6 | ZS-v3 No-Antibias | 0.2206 | 0.7376 |

**Comparação sample → full (Δ F1-macro):**

| Variante | Sample | Full | Δ |
|---|---|---|---|
| FS-v2 2ex+Antibias | 0.2750 | 0.3173 | +0.0423 |
| ZS-v2 Descriptions | 0.2673 | 0.2875 | +0.0202 |
| FS-v3 2-Examples | 0.2606 | 0.2747 | +0.0141 |
| FS-v1 1-Example | 0.2994 | 0.2706 | −0.0288 |
| ZS-v1 Base | 0.2347 | 0.2238 | −0.0109 |
| ZS-v3 No-Antibias | 0.2516 | 0.2206 | −0.0310 |

**Principais achados:**
- FS-v2 inverte ranking: era 2º no sample, assume 1º no full (+0.04 F1-macro)
- Sample estratificado foi representativo: magnitudes de Δ ≤ 0.03 para 5 das 6 variantes
- Classes raras ganham suporte real: racism (21), xenophobia (31) e misogyny (44) deixam de ser zeradas
- FS-v1 perde −0.029 F1-macro no full — mais sensível a tweets fora do padrão dos exemplos escolhidos
- `not_toxic` estável: F1 entre 0.82–0.87 em todas as variantes no dataset completo

---

## Referências

- [ToLD-Br — dataset original](https://github.com/joaoaleite/ToLD-Br)
- [Ollama](https://ollama.com)
- [Qwen3.5](https://ollama.com/library/qwen3.5)
