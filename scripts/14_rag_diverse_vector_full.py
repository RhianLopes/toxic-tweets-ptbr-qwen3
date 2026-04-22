import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import polars as pl
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).parent.parent
OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
VALID_LABELS = {"NOT_TOXIC", "OBSCENE", "INSULT", "HOMOPHOBIA", "RACISM", "MISOGYNY", "XENOPHOBIA"}
CATEGORIES = ["not_toxic", "obscene", "insult", "homophobia", "racism", "misogyny", "xenophobia"]
OUTPUT = ROOT / "results" / "full" / "rag_diverse_vector_k1.csv"
EMBEDDINGS_CACHE = ROOT / "data" / "full" / "train_embeddings.npy"
CHECKPOINT_EVERY = 500
WORKERS = 2
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

# --- Corpus de retrieval ---
train_df = pl.read_csv(ROOT / "data" / "full" / "toldBr_train.csv")
train_texts = train_df["text"].to_list()
train_labels = train_df["label"].to_list()
print(f"Corpus: {len(train_texts)} tweets")

# --- Embeddings (com cache) ---
encoder = SentenceTransformer(EMBED_MODEL)

if EMBEDDINGS_CACHE.exists():
    print(f"Carregando embeddings do cache: {EMBEDDINGS_CACHE}")
    corpus_embs = np.load(EMBEDDINGS_CACHE)
else:
    print("Computando embeddings do corpus...")
    t_emb = time.time()
    corpus_embs = encoder.encode(
        train_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(EMBEDDINGS_CACHE, corpus_embs)
    print(f"Embeddings salvos | {time.time()-t_emb:.1f}s")

# --- Índices por categoria ---
corpus_by_cat = {cat: [] for cat in CATEGORIES}
for i, label in enumerate(train_labels):
    if label in corpus_by_cat:
        corpus_by_cat[label].append(i)

for cat, idxs in corpus_by_cat.items():
    print(f"  {cat}: {len(idxs)} tweets no corpus")

# --- Dataset de validação ---
val_df = pl.read_csv(ROOT / "data" / "full" / "toldBr_val.csv")
val_rows = list(val_df.iter_rows(named=True))
print(f"Validação: {len(val_rows)} tweets")

# Pré-computa embeddings do val
print("Computando embeddings do val...")
val_texts = [r["text"] for r in val_rows]
val_embs = encoder.encode(val_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)


def retrieve_diverse(q_emb: np.ndarray) -> list[tuple[str, str]]:
    examples = []
    for cat in CATEGORIES:
        indices = corpus_by_cat[cat]
        if not indices:
            continue
        sims = corpus_embs[indices] @ q_emb
        best = indices[int(np.argmax(sims))]
        examples.append((train_texts[best], train_labels[best]))
    return examples  # 7 exemplos, um por categoria


def build_prompt(tweet: str, examples: list[tuple[str, str]]) -> str:
    ex_block = "\n\n".join(
        f'Comentário: "{text}"\nClassificação: {label.upper()}'
        for text, label in examples
    )
    return (
        "Você é um sistema de moderação de conteúdo em português brasileiro.\n"
        "Classifique o comentário em UMA das categorias:\n"
        "NOT_TOXIC, OBSCENE, INSULT, HOMOPHOBIA, RACISM, MISOGYNY, XENOPHOBIA\n"
        "Responda APENAS com o nome da categoria.\n\n"
        f"Exemplos similares (um por categoria):\n{ex_block}\n\n"
        f"Comentário: {tweet}\n"
        "Classificação:"
    )


def parse_label(response: str) -> str:
    text = response.strip().upper().replace(" ", "_")
    for label in VALID_LABELS:
        if label in text:
            return label.lower()
    return "unknown"


# Pré-computa retrieval para todos os tweets do val
print("Executando retrieval diverso por categoria...")
t_ret = time.time()
all_examples = [retrieve_diverse(val_embs[i]) for i in range(len(val_rows))]
print(f"Retrieval concluído em {time.time()-t_ret:.1f}s")

if Path(OUTPUT).exists():
    done = pl.read_csv(OUTPUT)
    start_idx = len(done)
    resultados = done.to_dicts()
    print(f"Retomando do índice {start_idx}")
else:
    start_idx = 0
    resultados = []
    print("Início do zero")


def classify(args):
    i, row, examples = args
    payload = {
        "model": MODEL,
        "prompt": build_prompt(row["text"], examples),
        "stream": False,
        "think": False,
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
    data = r.json()
    predicao = parse_label(data["response"])
    tps = data["eval_count"] / (data["eval_duration"] / 1e9)
    return {
        "text": row["text"],
        "label": row["label"],
        "predicao": predicao,
        "resposta_raw": data["response"].strip(),
        "tokens_s": round(tps, 1),
    }


rows = [
    (i, val_rows[i], all_examples[i])
    for i in range(len(val_rows))
    if i >= start_idx
]
total = len(val_rows)
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    for batch_start in range(0, len(rows), CHECKPOINT_EVERY):
        batch = rows[batch_start : batch_start + CHECKPOINT_EVERY]
        batch_results = list(executor.map(classify, batch))
        resultados.extend(batch_results)
        pl.DataFrame(resultados).write_csv(OUTPUT)
        total_done = start_idx + batch_start + len(batch)
        elapsed = time.time() - t0
        processed_so_far = batch_start + len(batch)
        eta = (total - total_done) * (elapsed / processed_so_far)
        print(f"{total_done}/{total} | {elapsed/60:.1f}min | ETA {eta/60:.1f}min")

df = pl.DataFrame(resultados)
df.write_csv(OUTPUT)
print(f"\nConcluído em {(time.time()-t0)/60:.1f}min | UNKNOWN: {(df['predicao']=='unknown').sum()}")

y_true, y_pred = df["label"].to_list(), df["predicao"].to_list()
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"F1-macro: {f1:.4f}\n")
print(classification_report(y_true, y_pred, zero_division=0))
