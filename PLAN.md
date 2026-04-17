# toxic-tweets-ptbr-qwen3

> Classificação de comentários tóxicos em português brasileiro usando LLM local (Qwen3) via Ollama — sem API, sem custo, rodando na sua GPU.

---

## 🎯 Objetivo

Avaliar a capacidade do **Qwen3** em moderar conteúdo tóxico em português brasileiro, utilizando o dataset **ToLD-Br**, com três abordagens de prompting:

| Abordagem | Descrição |
|---|---|
| **Zero-shot** | Classifica sem exemplos |
| **Few-shot** | Classifica com 3–5 exemplos no prompt |
| **Chain-of-Thought** | Raciocina antes de classificar |

Tudo rodando **100% local** via [Ollama](https://ollama.com), aproveitando a GPU para medir desempenho real (tokens/s, latência, custo zero vs. API paga).

---

## 📦 Dataset — ToLD-Br

**Repositório oficial:** [github.com/joaoaleite/ToLD-Br](https://github.com/joaoaleite/ToLD-Br)

O **ToLD-Br** (Toxic Language Dataset – Brazilian Portuguese) é um dos datasets mais completos para detecção de linguagem tóxica em PT-BR.

### Características

- **~21.000 tweets** coletados e anotados manualmente
- **Linguagem:** Português Brasileiro
- **Anotação:** crowdsourcing com múltiplos anotadores por tweet
- **Formato:** CSV com colunas de texto e labels

### Categorias de toxicidade

| Categoria | Descrição |
|---|---|
| `homophobia` | Discurso de ódio contra LGBTQIA+ |
| `racism` | Racismo e discriminação racial |
| `misogyny` | Misoginia e violência de gênero |
| `xenophobia` | Xenofobia e discriminação de origem |
| `obscene` | Linguagem obscena e vulgar |
| `insult` | Insultos e ofensas diretas |
| `other_toxic` | Outras formas de toxicidade |
| `not_toxic` | Comentário não tóxico |

### Estrutura dos arquivos

```
ToLD-Br/
├── data/
│   ├── toldBr_train.csv    # Treino
│   ├── toldBr_test.csv     # Teste
│   └── toldBr_val.csv      # Validação
└── README.md
```

---

## 🏗️ Estrutura do Projeto

```
toxic-tweets-ptbr-qwen3/
├── data/
│   ├── raw/                    # Dados originais do ToLD-Br
│   └── sample/                 # Amostra estratificada (500 tweets)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA do dataset
│   ├── 02_sampling.ipynb               # Criação da amostra
│   ├── 03_zero_shot.ipynb              # Experimento 1
│   ├── 04_few_shot.ipynb               # Experimento 2
│   ├── 05_chain_of_thought.ipynb       # Experimento 3
│   └── 06_results_analysis.ipynb       # Comparação e métricas
├── src/
│   ├── prompts.py              # Templates de prompts
│   ├── classifier.py           # Classe principal de classificação
│   ├── metrics.py              # Cálculo de F1, precision, recall
│   └── utils.py                # Funções auxiliares
├── results/
│   ├── zero_shot_results.csv
│   ├── few_shot_results.csv
│   ├── cot_results.csv
│   └── metrics_summary.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/toxic-tweets-ptbr-qwen3.git
cd toxic-tweets-ptbr-qwen3
```

### 2. Ambiente Python

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Baixar o dataset ToLD-Br

```bash
# Opção A — clonar o repo oficial dentro de data/raw/
git clone https://github.com/joaoaleite/ToLD-Br data/raw/ToLD-Br

# Opção B — download direto dos CSVs
mkdir -p data/raw
wget -P data/raw/ https://raw.githubusercontent.com/joaoaleite/ToLD-Br/main/data/toldBr_train.csv
wget -P data/raw/ https://raw.githubusercontent.com/joaoaleite/ToLD-Br/main/data/toldBr_test.csv
```

### 4. Instalar e configurar o Ollama

```bash
# Instalar Ollama (Windows/Mac/Linux)
# https://ollama.com/download

# Baixar o modelo Qwen3
ollama pull qwen3:14b        # Recomendado para GPU com 12GB+ VRAM
# ou
ollama pull qwen3:8b         # Versão mais leve

# Verificar se está rodando
ollama list
```

### 5. Testar a conexão com o modelo

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:14b",
        "prompt": "Olá, você está funcionando?",
        "stream": False
    }
)
print(response.json()["response"])
```

---

## 🧪 Experimentos

### Fase 1 — Exploração e Amostragem

**Notebook:** `01_exploratory_analysis.ipynb`

- Distribuição das categorias
- Comprimento médio dos tweets
- Exemplos por categoria
- Análise de balanceamento

**Notebook:** `02_sampling.ipynb`

```python
# Amostragem estratificada — 500 tweets
# Mantém proporção original das categorias
sample = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), int(500 * len(x) / len(df))), random_state=42)
)
```

---

### Fase 2 — Experimentos de Classificação

#### Experimento 1 — Zero-shot

```python
ZERO_SHOT_PROMPT = """
Você é um sistema de moderação de conteúdo para redes sociais em português brasileiro.

Classifique o comentário abaixo em UMA das categorias:
- NOT_TOXIC
- HOMOPHOBIA
- RACISM
- MISOGYNY
- XENOPHOBIA
- OBSCENE
- INSULT
- OTHER_TOXIC

Responda APENAS com o nome da categoria, sem explicações.

Comentário: {tweet}
Classificação:
"""
```

#### Experimento 2 — Few-shot

```python
FEW_SHOT_PROMPT = """
Você é um sistema de moderação de conteúdo para redes sociais em português brasileiro.

Aqui estão exemplos de classificação:

Comentário: "Que dia lindo hoje!"
Classificação: NOT_TOXIC

Comentário: "Esse [grupo] não merece direitos"
Classificação: HOMOPHOBIA

Comentário: "Vai tomar [palavrão], idiota"
Classificação: INSULT

Agora classifique o comentário abaixo em UMA das categorias:
NOT_TOXIC, HOMOPHOBIA, RACISM, MISOGYNY, XENOPHOBIA, OBSCENE, INSULT, OTHER_TOXIC

Responda APENAS com o nome da categoria.

Comentário: {tweet}
Classificação:
"""
```

#### Experimento 3 — Chain-of-Thought

```python
COT_PROMPT = """
Você é um sistema de moderação de conteúdo para redes sociais em português brasileiro.

Analise o comentário abaixo seguindo estes passos:
1. Identifique palavras ou expressões problemáticas
2. Avalie o contexto e a intenção
3. Determine se há algum grupo sendo atacado
4. Classifique em UMA categoria: NOT_TOXIC, HOMOPHOBIA, RACISM, MISOGYNY, XENOPHOBIA, OBSCENE, INSULT, OTHER_TOXIC

Formato de resposta:
Análise: [seu raciocínio]
Classificação: [CATEGORIA]

Comentário: {tweet}
"""
```

---

## 📊 Métricas Avaliadas

| Métrica | Descrição |
|---|---|
| **F1-score (macro)** | Média do F1 por categoria — penaliza desequilíbrio |
| **Precision** | Dos classificados como tóxicos, quantos realmente são |
| **Recall** | Dos tóxicos reais, quantos foram detectados |
| **Accuracy** | Acertos gerais |
| **Tokens/segundo** | Performance da GPU (diferencial do artigo) |
| **Latência média** | Tempo por tweet classificado |

---

## 🖥️ Hardware de Referência

| Componente | Especificação |
|---|---|
| GPU | NVIDIA RTX 5070 (12GB VRAM) |
| Modelo | Qwen3-14B (Q4 quantizado ~9GB) |
| Backend | Ollama |
| OS | Windows 11 |

> 💡 **Diferencial do artigo:** mostrar tokens/segundo na 5070 e comparar o custo estimado se o mesmo volume fosse processado via API (GPT-4o mini, Claude Haiku).

---

## 🗺️ Roadmap

- [ ] Fase 1 — Download e EDA do ToLD-Br
- [ ] Fase 2 — Amostragem estratificada (500 tweets)
- [ ] Fase 3 — Setup Ollama + Qwen3
- [ ] Fase 4 — Experimento Zero-shot
- [ ] Fase 5 — Experimento Few-shot
- [ ] Fase 6 — Experimento Chain-of-Thought
- [ ] Fase 7 — Análise comparativa de resultados
- [ ] Fase 8 — Artigo no Medium

---

## 📝 Artigo

Este projeto dará origem a um artigo no Medium documentando todo o processo, resultados e aprendizados.

**Título previsto:** *"Moderação de conteúdo em PT-BR com LLM local: testando o Qwen3 na GPU sem gastar um centavo com API"*

---

## 📚 Referências

- [ToLD-Br — Toxic Language Dataset for Brazilian Portuguese](https://github.com/joaoaleite/ToLD-Br)
- [Ollama — Run LLMs locally](https://ollama.com)
- [Qwen3 — Alibaba Cloud](https://qwenlm.github.io/blog/qwen3/)
- [HatEval 2019 Shared Task](http://hatespeech.di.unito.it/hateval.html)

---

## 📄 Licença

MIT
