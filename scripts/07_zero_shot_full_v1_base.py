import time
from pathlib import Path
import polars as pl
import requests
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).parent.parent
OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
VALID_LABELS = {"NOT_TOXIC", "OBSCENE", "INSULT", "HOMOPHOBIA", "RACISM", "MISOGYNY", "XENOPHOBIA"}
OUTPUT = ROOT / "results" / "full" / "zero_shot_v1_base.csv"
CHECKPOINT_EVERY = 500

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
        "Você é um sistema de moderação de conteúdo em português brasileiro.\n"
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


total = len(df_full)
t0 = time.time()

for i, row in enumerate(df_full.iter_rows(named=True)):
    if i < start_idx:
        continue
    payload = {"model": MODEL, "prompt": build_prompt(row["text"]), "stream": False, "think": False}
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
    data = r.json()
    predicao = parse_label(data["response"])
    tps = data["eval_count"] / (data["eval_duration"] / 1e9)
    resultados.append({"text": row["text"], "label": row["label"], "predicao": predicao,
                       "resposta_raw": data["response"].strip(), "tokens_s": round(tps, 1)})

    if (i + 1) % CHECKPOINT_EVERY == 0:
        pl.DataFrame(resultados).write_csv(OUTPUT)
        elapsed = time.time() - t0
        eta = (total - i - 1) * (elapsed / (i - start_idx + 1))
        print(f"{i+1}/{total} | {elapsed/60:.1f}min | ETA {eta/60:.1f}min")

df = pl.DataFrame(resultados)
df.write_csv(OUTPUT)
print(f"\nConcluído em {(time.time()-t0)/60:.1f}min | UNKNOWN: {(df['predicao']=='unknown').sum()}")

y_true, y_pred = df["label"].to_list(), df["predicao"].to_list()
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"F1-macro: {f1:.4f}\n")
print(classification_report(y_true, y_pred, zero_division=0))
