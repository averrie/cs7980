#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Visualizations for: clustered_posts.csv, cluster_top_terms.csv, cluster_examples.csv
# Also builds an interactive 2D scatter (PCA) from embeddings.npy

import os, html, textwrap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib import font_manager

# ====== Optional backends ======
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

# ====== Paths ======
OUT_DIR = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial/analysis")
POSTS_CSV    = OUT_DIR / "clustered_posts.csv"
TERMS_CSV    = OUT_DIR / "cluster_top_terms.csv"
EXAMPLES_CSV = OUT_DIR / "cluster_examples.csv"
EMB_NPY      = OUT_DIR / "embeddings.npy"

OUT_VIZ = OUT_DIR / "viz"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ====== Fonts (TrueType for wordcloud) ======
DEJAVU_TTF = font_manager.findfont("DejaVu Sans")  # .ttf path
print("[INFO] Using TTF for wordcloud:", DEJAVU_TTF)

def truncate(s, n=120):
    s = "" if s is None else str(s)
    return (s[:n] + "…") if len(s) > n else s

# ====== Read CSVs (robust dtypes) ======
posts    = pd.read_csv(POSTS_CSV,    dtype=str, low_memory=False)
terms    = pd.read_csv(TERMS_CSV,    dtype=str, low_memory=False)
examples = pd.read_csv(EXAMPLES_CSV, dtype=str, low_memory=False)

# ====== 1) Cluster size bar chart ======
# ensure cluster numeric for plotting order
posts["cluster_num"] = pd.to_numeric(posts["cluster"], errors="coerce").fillna(-1).astype(int)
size = posts["cluster_num"].value_counts().sort_index()

plt.figure(figsize=(10, 4))
size.plot(kind="bar")
plt.title("Cluster sizes")
plt.xlabel("cluster"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(OUT_VIZ / "cluster_sizes_bar.png", dpi=160)
plt.close()

# ====== 2) Top-terms visuals ======
# explode top_terms into (cluster:int, term:str, weight:int)
def explode_terms(df):
    rows = []
    for _, r in df.iterrows():
        try:
            c = int(r["cluster"])
        except Exception:
            continue
        toks = [t.strip() for t in str(r.get("top_terms", "")).split(",") if t.strip()]
        L = len(toks)
        for rank, term in enumerate(toks, start=1):
            weight = L - rank + 1  # 15..1 之类的递减权重
            rows.append({"cluster": c, "term": term, "weight": weight})
    return pd.DataFrame(rows)

terms_exp = explode_terms(terms)

cluster_ids = (
    pd.to_numeric(terms["cluster"], errors="coerce")
      .dropna()
      .astype(int)
      .unique()
)
for c in sorted(cluster_ids):
    sub = terms_exp[terms_exp["cluster"] == c].sort_values("weight", ascending=False).head(15)
    if sub.empty:
        print(f"[WARN] cluster {c}: no terms to plot")
        continue
    plt.figure(figsize=(8, 5))
    plt.barh(sub["term"][::-1], sub["weight"][::-1])
    plt.title(f"Top terms (cluster {c})")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / f"top_terms_cluster_{c}.png", dpi=160)
    plt.close()

# Optional wordclouds per cluster
if HAS_WORDCLOUD:
    for c in sorted(cluster_ids):
        sub = terms_exp[terms_exp["cluster"] == c]
        if sub.empty:
            print(f"[WARN] cluster {c}: no terms for wordcloud; skipped")
            continue
        freqs = {}
        for _, r in sub.iterrows():
            term = str(r["term"]).strip()
            if not term:
                continue
            w = int(r["weight"]) if str(r["weight"]).isdigit() else 1
            freqs[term] = freqs.get(term, 0) + w
        if not freqs:
            print(f"[WARN] cluster {c}: empty freqs; skipped")
            continue
        try:
            wc = WordCloud(
                width=900, height=600, background_color="white", font_path=DEJAVU_TTF
            ).generate_from_frequencies(freqs)
            wc.to_file(str(OUT_VIZ / f"wordcloud_cluster_{c}.png"))
        except Exception as e:
            print(f"[WARN] wordcloud failed for cluster {c}: {e}")

# ====== 3) PCA scatter (static + interactive) from embeddings ======
if EMB_NPY.exists():
    X = np.load(EMB_NPY)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)

    clusters_num = posts["cluster_num"]  # already numeric
    
    text_col = "clean" if "clean" in posts.columns else posts.columns[0]
    scatter_df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "cluster": clusters_num.values,                  # numeric for matplotlib
        "cluster_str": posts["cluster"].astype(str),     # string for plotly legend
        "text": posts[text_col].map(lambda s: truncate(s, 160)).values
    })

    # static PNG
    plt.figure(figsize=(8, 7))
    plt.scatter(scatter_df["x"], scatter_df["y"],
                c=scatter_df["cluster"], s=5, cmap="tab20")
    plt.title("PCA scatter (colored by cluster)")
    plt.tight_layout()
    plt.savefig(OUT_VIZ / "pca_scatter.png", dpi=160)
    plt.close()

    # interactive HTML
    if HAS_PLOTLY:
        fig = px.scatter(
            scatter_df, x="x", y="y", color="cluster_str",
            hover_data={"text": True, "cluster_str": True, "x": False, "y": False},
            title="Embedding PCA (hover for text)"
        )
        fig.write_html(str(OUT_VIZ / "pca_scatter_interactive.html"), include_plotlyjs="cdn")
else:
    print(f"[INFO] Embeddings not found at {EMB_NPY}; skipped PCA plots")

# ====== 4) Simple HTML report of examples ======
html_parts = [
    "<html><head><meta charset='utf-8'>",
    "<style>body{font-family:Arial;max-width:900px;margin:20px auto;} ",
    ".cl{margin-bottom:24px;} .ex{margin:6px 0;padding:8px;background:#fafafa;border-radius:8px;border:1px solid #eee;}",
    "</style></head><body>"
]
html_parts.append("<h1>Cluster Examples</h1>")
for c in sorted(pd.to_numeric(examples["cluster"], errors="coerce").dropna().astype(int).unique()):
    sub = examples[examples["cluster"].astype(str) == str(c)]["example"].tolist()
    html_parts.append(f"<div class='cl'><h2>Cluster {c}</h2>")
    for s in sub:
        html_parts.append(f"<div class='ex'>{html.escape(truncate(s, 500))}</div>")
    html_parts.append("</div>")
html_parts.append("</body></html>")
(OUT_VIZ / "cluster_examples.html").write_text("\n".join(html_parts), encoding="utf-8")

print("[OK] Wrote visualizations to:", OUT_VIZ)
if HAS_PLOTLY:
    print(" - pca_scatter_interactive.html (interactive)")
if HAS_WORDCLOUD:
    print(" - wordcloud_cluster_*.png")
