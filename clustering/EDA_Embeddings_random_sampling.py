#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding-based clustering + EDA for TruthSocial 10% sample (fixed paths).
"""

import os, re, emoji, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ==== Input and Output path ====
IN_PATH  = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial/truth_social_dataset_10pct/truths_10pct.tsv")
OUT_DIR  = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial/analysis")
MODEL    = "all-MiniLM-L6-v2"    
K        = 20                    
MAX_ROWS = 0                     

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Loading and data cleaning ====
TEXT_COLS = ("content","text","status","body","message")
def clean_text(s):
    if not isinstance(s,str): return ""
    s = re.sub(r"http\S+|@\w+|#\w+","",s)
    s = emoji.replace_emoji(s,"")
    s = re.sub(r"\s+"," ",s).strip()
    return s

def load_df(path,max_rows=0):
    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path,sep="\t",dtype=str,on_bad_lines="skip")
    if max_rows>0: df = df.head(max_rows)
    col = next((c for c in df.columns if c.lower() in TEXT_COLS),df.columns[0])
    df["text"] = df[col].fillna("")
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len()>10].reset_index(drop=True)
    print(f"[OK] {len(df)} rows after cleaning")
    return df

df = load_df(IN_PATH,MAX_ROWS)

# ==== Embedding ====
emb_path = OUT_DIR/"embeddings.npy"
if emb_path.exists():
    print("[INFO] Loading cached embeddings...")
    X = np.load(emb_path)
else:
    model = SentenceTransformer(MODEL)
    X = model.encode(df["clean"].tolist(),batch_size=64,show_progress_bar=True)
    np.save(emb_path,X)
print("Embeddings shape:",X.shape)

# ==== clustering ====
print(f"[INFO] Running KMeans k={K}")
km = KMeans(n_clusters=K,random_state=42,n_init=10)
labels = km.fit_predict(X)
df["cluster"]=labels

# ==== Top terms ====
EMOJI_WORDS = {
    "emoji","rolling_on_the_floor_laughing","joy","pray","heart","clap","fire",
    "eyes","amen","point_down","white_heart","thinking_face","100","boom"
}

DOMAIN_WORDS = {"truth","retruth","twitter","social","truth_social","rt","amp","like"}

PRONOUNS = {
    "you","your","yours","u","ur","we","our","ours","i","me","my","mine",
    "he","him","his","she","her","hers","they","them","their","theirs","it","its",
    "this","that","these","those","one","ones"
}

CUSTOM_STOPWORDS = EMOJI_WORDS | DOMAIN_WORDS | PRONOUNS

STOPWORDS = list(set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS)

TOKEN_PATTERN = r"(?u)\b(?!\d+\b)\w\w+\b"

vec = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=20,         
    max_df=0.5,         
    stop_words=STOPWORDS,
    token_pattern=TOKEN_PATTERN,
    lowercase=True,
)

X_tfidf = vec.fit_transform(df["clean"])
vocab = np.array(vec.get_feature_names_out())

rows = []
for c in sorted(df["cluster"].unique()):
    idx = np.where(df["cluster"] == c)[0]
    if len(idx) == 0:
        continue
    centroid = X_tfidf[idx].mean(axis=0).A1
    top_idx = centroid.argsort()[::-1][:15]
    top = vocab[top_idx]
    rows.append({"cluster": int(c), "size": len(idx), "top_terms": ", ".join(top)})

pd.DataFrame(rows).to_csv(OUT_DIR / "cluster_top_terms.csv", index=False)

examples=[]
for c in sorted(df["cluster"].unique()):
    sample=df[df["cluster"]==c]["clean"].sample(min(5,len(df[df["cluster"]==c])),random_state=42)
    for s in sample: examples.append({"cluster":int(c),"example":s})
pd.DataFrame(examples).to_csv(OUT_DIR/"cluster_examples.csv",index=False)

# ==== EDA graphs ====
## Post Length Histogram
plt.hist(df["clean"].str.len(),bins=50)
plt.title("Post length distribution")
plt.xlabel("chars"); plt.ylabel("count")
plt.tight_layout(); plt.savefig(OUT_DIR/"len_hist.png",dpi=160); plt.close()

# ---- pick & parse a timestamp column ----
TIME_COLS = ["timestamp", "time_scraped", "created_at", "createdAt", "time", "date"]
time_col = next((c for c in TIME_COLS if c in df.columns), None)

if time_col is not None:
    df["created_at_parsed"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
else:
    df["created_at_parsed"] = pd.NaT

# ---- Time tendency (if we have any valid timestamps) ----
if "created_at_parsed" in df.columns and df["created_at_parsed"].notna().any():
    s = df.dropna(subset=["created_at_parsed"]).copy()
    #UTC-aware Time
    s["day"] = s["created_at_parsed"].dt.date
    ts = s.groupby("day").size()

    plt.figure(figsize=(8,4))
    ts.plot()
    plt.title(f"Posts per day (source: {time_col})")
    plt.xlabel("Date"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_time_series.png", dpi=160)
    plt.close()
    print("[OK] fig_time_series.png")
else:
    print("[INFO] No usable timestamp; skipped fig_time_series.png")


## PCA
pca=PCA(n_components=2,random_state=42)
xy=pca.fit_transform(X)
plt.figure(figsize=(7,6))
plt.scatter(xy[:,0],xy[:,1],c=df["cluster"],s=6)
plt.title("PCA scatter (colored by cluster)")
plt.tight_layout(); plt.savefig(OUT_DIR/"fig_scatter_pca.png",dpi=160); plt.close()
print("[OK] fig_scatter_pca.png")

## UMAP
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2,random_state=42)
    xy2 = reducer.fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(xy2[:,0],xy2[:,1],c=df["cluster"],s=6)
    plt.title("UMAP scatter (colored by cluster)")
    plt.tight_layout(); plt.savefig(OUT_DIR/"fig_scatter_umap.png",dpi=160); plt.close()
    print("[OK] fig_scatter_umap.png")
else:
    print("[INFO] umap-learn not installed; skipped fig_scatter_umap.png")
df.to_csv(OUT_DIR/"clustered_posts.csv",index=False)
print(f"[DONE] Outputs saved in {OUT_DIR}")