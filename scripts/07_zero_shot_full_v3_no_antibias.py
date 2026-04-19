import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import polars as pl
import requests
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).parent.parent
OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
VALID_LABELS = {"NOT_TOXIC", "OBSCENE", "INSULT", "HOMOPHOBIA", "RACISM", "MISOGYNY", "XENOPHOBIA"}
OUTPUT = ROOT / "results" / "full" / "zero_shot_v3_no_antibias.csv"
CHECKPOINT_EVERY = 500
WORKERS = 2

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
df_full = pl.read_csv(ROOT / "data" / "full" / "toldBr_eval.csv")
print(f"Dataset: {len(df_full)} tweets")

if Path(OUTPUT).exists():
    done = pl.read_csv(OUTPUT)
    start_idx = len(done)
    resultados = done.to_dicts()
    print(f"Retomando do índice {start_idx}")
else:
    start_idx = 0
    resultados = []
    print("Início do zero")


def build_prompt(tweet: str) -> str:
    return (
        "Classifique o comentário abaixo em UMA das categorias:\n"
        "NOT_TOXIC, OBSCENE, INSULT, HOMOPHOBIA, RACISM, MISOGYNY, XENOPHOBIA\n"
        "Responda APENAS com o nome da categoria.\n\n"
        f"Comentário: {tweet}\n"
        "Classificação:"
    )


def parse_label(response: str) -> str:
    text = response.strip().upper().replace(" ", "_")
    for label in VALID_LABELS:
        if label in text:
            return label.lower()
    return "unknown"


def classify(args):
    i, row = args
    payload = {"model": MODEL, "prompt": build_prompt(row["text"]), "stream": False, "think": False}
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
    data = r.json()
    predicao = parse_label(data["response"])
    tps = data["eval_count"] / (data["eval_duration"] / 1e9)
    return {"text": row["text"], "label": row["label"], "predicao": predicao,
            "resposta_raw": data["response"].strip(), "tokens_s": round(tps, 1)}


rows = [(i, row) for i, row in enumerate(df_full.iter_rows(named=True)) if i >= start_idx]
total = len(df_full)
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    for batch_start in range(0, len(rows), CHECKPOINT_EVERY):
        batch = rows[batch_start:batch_start + CHECKPOINT_EVERY]
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
