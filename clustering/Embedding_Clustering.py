#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding-based clustering + EDA for TruthSocial + Bluesky (comparative version).
"""

import os, re, emoji, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ==== Input and Output path ====
TRUTH_PATH   = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial/truth_social_dataset_10pct/truths_10pct.tsv")
BLUESKY_PATH = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/bluesky/dataset/bluesky_80k.tsv")
OUT_DIR  = Path("/Users/yukkihsu/Desktop/Assignment/cs7980/embedding")
MODEL    = "all-MiniLM-L6-v2"
MAX_ROWS = 0           
BALANCE_PER_SOURCE = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== Loading and data cleaning ====
TEXT_COLS = ("content", "text", "status", "body", "message")

def clean_text(s):
    """Clean text: remove URLs, mentions, hashtags, and emojis"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|@\w+|#\w+", "", s)  # Remove URLs, @mentions, #hashtags
    s = emoji.replace_emoji(s, "")          # Remove emojis
    s = re.sub(r"\s+", " ", s).strip()      # Normalize whitespace
    return s

def load_df(path, source_name, max_rows=0):
    """Load and preprocess dataset for a given platform (TruthSocial / Bluesky)."""
    print(f"[INFO] Loading {source_name} from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip")

    if max_rows > 0:
        df = df.head(max_rows)

    print(f"[{source_name}] [1] Initial rows: {len(df)}")

    # Find text column
    col = next((c for c in df.columns if c.lower() in TEXT_COLS), df.columns[0])
    df["text"] = df[col].fillna("")

    # Clean text
    df["clean"] = df["text"].map(clean_text)

    # Word count
    df["word_count"] = df["clean"].str.split().str.len()

    # Filter: at least 20 characters AND at least 5 words
    df = df[
        (df["clean"].str.len() >= 20) &
        (df["word_count"] >= 5)
    ].copy()
    print(f"[{source_name}] [2] After length/word filter: {len(df)}")

    # Remove duplicates (per platform)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean"], keep="first")
    print(f"[{source_name}] [3] After deduplication: {len(df)} (removed {before_dedup - len(df)} duplicates)")

    # Add source/platform label
    df["source"] = source_name

    df = df.reset_index(drop=True)
    print(f"[{source_name}] [OK] Final dataset: {len(df)} rows\n")

    return df

# ---- Load both platforms ----
truth_df   = load_df(TRUTH_PATH,   "TruthSocial", max_rows=MAX_ROWS)
bluesky_df = load_df(BLUESKY_PATH, "Bluesky",     max_rows=MAX_ROWS)

# ---- Optional balancing ----
if BALANCE_PER_SOURCE > 0:
    truth_df   = truth_df.sample(min(BALANCE_PER_SOURCE, len(truth_df)), random_state=42)
    bluesky_df = bluesky_df.sample(min(BALANCE_PER_SOURCE, len(bluesky_df)), random_state=42)
    print(f"[INFO] After balancing: TruthSocial={len(truth_df)}, Bluesky={len(bluesky_df)}\n")

# ---- Merge both datasets ----
df = pd.concat([truth_df, bluesky_df], ignore_index=True)
print(f"[INFO] Merged dataset size: {len(df)} (TruthSocial={len(truth_df)}, Bluesky={len(bluesky_df)})\n")

# ==== Embedding ====
emb_path = OUT_DIR / "embeddings_both.npy"
if emb_path.exists():
    print("[INFO] Loading cached embeddings...")
    X = np.load(emb_path)
    # Verify embeddings match current dataframe
    if X.shape[0] != len(df):
        print(f"[WARNING] Cached embeddings shape {X.shape} doesn't match df length {len(df)}")
        print("[INFO] Regenerating embeddings...")
        model = SentenceTransformer(MODEL)
        X = model.encode(df["clean"].tolist(), batch_size=64, show_progress_bar=True)
        np.save(emb_path, X)
else:
    print("[INFO] Generating embeddings...")
    model = SentenceTransformer(MODEL)
    X = model.encode(df["clean"].tolist(), batch_size=64, show_progress_bar=True)
    np.save(emb_path, X)

print(f"Embeddings shape: {X.shape}\n")

# ==== Multi-metric K evaluation ====
print("="*60)
print("EVALUATING OPTIMAL K WITH MULTIPLE METRICS (MERGED DATA)")
print("="*60)

Ks = range(5, 31)
silhouette_scores = []
inertias = []
calinski_scores = []

for k in tqdm(Ks, desc="Testing K values"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Silhouette Score
    sil_score = silhouette_score(X, labels, sample_size=5000 if len(X) > 5000 else None)
    silhouette_scores.append(sil_score)

    # Inertia (for Elbow method)
    inertias.append(km.inertia_)

    # Calinski-Harabasz Score
    cal_score = calinski_harabasz_score(X, labels)
    calinski_scores.append(cal_score)

    print(f"K={k:2d} | Silhouette={sil_score:.4f} | Inertia={km.inertia_:,.0f} | Calinski-Harabasz={cal_score:.2f}")

# Save metrics
metrics_df = pd.DataFrame({
    'K': list(Ks),
    'Silhouette': silhouette_scores,
    'Inertia': inertias,
    'Calinski_Harabasz': calinski_scores
})
metrics_df.to_csv(OUT_DIR / "clustering_metrics_both.csv", index=False)
print(f"\n[OK] Metrics saved to clustering_metrics_both.csv")

# Plot metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Silhouette Score
axes[0].plot(Ks, silhouette_scores, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].set_title('Silhouette Score vs K', fontsize=14)
axes[0].grid(True, alpha=0.3)
best_sil_k = Ks[np.argmax(silhouette_scores)]
axes[0].axvline(best_sil_k, color='r', linestyle='--', alpha=0.5, label=f'Best K={best_sil_k}')
axes[0].legend()

# Elbow Method (Inertia)
axes[1].plot(Ks, inertias, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Inertia', fontsize=12)
axes[1].set_title('Elbow Method (Inertia vs K)', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Calinski-Harabasz Score (higher is better)
axes[2].plot(Ks, calinski_scores, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=12)
axes[2].set_title('Calinski-Harabasz Score vs K', fontsize=14)
axes[2].grid(True, alpha=0.3)
best_cal_k = Ks[np.argmax(calinski_scores)]
axes[2].axvline(best_cal_k, color='r', linestyle='--', alpha=0.5, label=f'Best K={best_cal_k}')
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "clustering_metrics_both.png", dpi=160)
plt.close()
print(f"[OK] clustering_metrics_both.png saved")

# Auto-detect elbow point (using second derivative)
inertia_diff2 = np.diff(inertias, 2)
elbow_k = Ks[np.argmax(inertia_diff2) + 2]

print("\n" + "="*60)
print("METRIC SUMMARY (MERGED DATA)")
print("="*60)
print(f"Highest Silhouette Score:      K = {best_sil_k} (score = {max(silhouette_scores):.4f})")
print(f"Highest Calinski-Harabasz:     K = {best_cal_k} (score = {max(calinski_scores):.2f})")
print(f"Elbow point (auto-detected):   K = {elbow_k}")
print(f"\nRecommended K range: [{elbow_k-2}, {best_sil_k+3}]")
print("="*60 + "\n")

# ==== Compare candidate K values ====
print("="*60)
print("COMPARING CANDIDATE K VALUES (MERGED)")
print("="*60)

candidate_Ks = [10, 14, 18, 22]
print(f"Testing K values: {candidate_Ks}\n")

# Prepare TF-IDF for term extraction
EMOJI_WORDS = {
    "emoji","rolling_on_the_floor_laughing","joy","pray","heart","clap","fire",
    "eyes","amen","point_down","white_heart","thinking_face","100","boom",
    "steam_locomotive","dart","face_vomiting","raised_hands","blue_heart",
    "smiling_face_with_3_hearts","hearts","heart_eyes","sunglasses","grin","rage",
    "face_with_symbols_on_mouth","scream","bangbang","latin_cross","eagle"
}

DOMAIN_WORDS = {"truth","retruth","twitter","social","truth_social","rt","amp","like","follow","followers"}

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

for K_test in candidate_Ks:
    print(f"\n{'='*50}")
    print(f"K = {K_test}")
    print(f"{'='*50}")

    km_test = KMeans(n_clusters=K_test, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(X)

    # Cluster size statistics
    unique, counts = np.unique(labels_test, return_counts=True)
    print(f"\nCluster size distribution:")
    print(f"  Min: {counts.min():,} | Max: {counts.max():,} | Mean: {counts.mean():.0f} | Std: {counts.std():.0f}")

    # Statistics on the cluster×source structure under this K platform
    tmp = df.copy()
    tmp["cluster_tmp"] = labels_test
    print("\nCluster × Source (counts):")
    print(pd.crosstab(tmp["cluster_tmp"], tmp["source"]))

    # Show top 5 terms per cluster
    print(f"\nTop 5 terms per cluster:")
    for c in sorted(unique):
        idx = np.where(labels_test == c)[0]
        if len(idx) == 0:
            continue

        centroid = X_tfidf[idx].mean(axis=0).A1
        top_idx = centroid.argsort()[::-1][:5]
        top_terms = ", ".join(vocab[top_idx])

        print(f"  C{c:2d} (n={len(idx):5,}): {top_terms}")

print("\n" + "="*60)
print("Please review the above comparisons and choose your K value.")
print("="*60 + "\n")

# ==== Final clustering with chosen K ====
K = 18 
print(f"[INFO] Running final clustering with K = {K}")

km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels

# Final cluster statistics (overall)
print(f"\nFinal cluster distribution (ALL):")
cluster_counts = df["cluster"].value_counts().sort_index()
for c, count in cluster_counts.items():
    pct = 100 * count / len(df)
    print(f"  Cluster {c:2d}: {count:6,} posts ({pct:5.2f}%)")

# Final cluster × source comparison
print("\nFinal Cluster × Source (counts):")
cluster_source_ct = pd.crosstab(df["cluster"], df["source"])
print(cluster_source_ct)

cluster_source_ct.to_csv(OUT_DIR / "cluster_source_crosstab.csv")
print("[OK] cluster_source_crosstab.csv saved")

# ===== Extract top terms for final K (overall) =====
print(f"\n[INFO] Extracting top terms for K={K} (overall)...")

rows = []
for c in sorted(df["cluster"].unique()):
    idx = np.where(df["cluster"] == c)[0]
    if len(idx) == 0:
        continue

    centroid = X_tfidf[idx].mean(axis=0).A1
    top_idx = centroid.argsort()[::-1][:15]
    top = vocab[top_idx]
    rows.append({
        "cluster": int(c),
        "size": len(idx),
        "top_terms": ", ".join(top)
    })

pd.DataFrame(rows).to_csv(OUT_DIR / "cluster_top_terms_both.csv", index=False)
print(f"[OK] cluster_top_terms_both.csv saved")

# ===== Extract example posts (with source info) =====
examples = []
for c in sorted(df["cluster"].unique()):
    cluster_df = df[df["cluster"] == c]
    sample_size = min(5, len(cluster_df))
    sample = cluster_df.sample(sample_size, random_state=42)[["clean", "source"]]
    for _, row in sample.iterrows():
        examples.append({
            "cluster": int(c),
            "source": row["source"],
            "example": row["clean"]
        })

pd.DataFrame(examples).to_csv(OUT_DIR / "cluster_examples_both.csv", index=False)
print(f"[OK] cluster_examples_both.csv saved")

# ==== EDA visualizations (merged) ====
print(f"\n[INFO] Generating visualizations...")

# Post length distribution
plt.figure(figsize=(10, 6))
plt.hist(df["clean"].str.len(), bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df["clean"].str.len().mean(), color='r', linestyle='--',
            label=f'Mean: {df["clean"].str.len().mean():.0f}')
plt.axvline(df["clean"].str.len().median(), color='g', linestyle='--',
            label=f'Median: {df["clean"].str.len().median():.0f}')
plt.title("Post Length Distribution (merged, after cleaning)", fontsize=14)
plt.xlabel("Characters", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "len_hist_both.png", dpi=160)
plt.close()
print(f"[OK] len_hist_both.png")

# Word count distribution
plt.figure(figsize=(10, 6))
plt.hist(df["word_count"], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df["word_count"].mean(), color='r', linestyle='--',
            label=f'Mean: {df["word_count"].mean():.1f}')
plt.axvline(df["word_count"].median(), color='g', linestyle='--',
            label=f'Median: {df["word_count"].median():.1f}')
plt.title("Word Count Distribution (merged)", fontsize=14)
plt.xlabel("Words", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "word_count_hist_both.png", dpi=160)
plt.close()
print(f"[OK] word_count_hist_both.png")

# Cluster size bar chart (overall)
plt.figure(figsize=(12, 6))
cluster_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title(f"Cluster Size Distribution (merged, K={K})", fontsize=14)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Number of Posts", fontsize=12)
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_sizes_both.png", dpi=160)
plt.close()
print(f"[OK] cluster_sizes_both.png")

# Cluster size by source (stacked bar)
cluster_source_prop = cluster_source_ct.div(cluster_source_ct.sum(axis=1), axis=0)
cluster_source_prop.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title(f"Cluster Composition by Source (proportion, K={K})", fontsize=14)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Proportion", fontsize=12)
plt.legend(title="Source")
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_source_stacked_both.png", dpi=160)
plt.close()
print(f"[OK] cluster_source_stacked_both.png")

def rgb_density_overlay(xy, source, out_path, bins=400, margin=0.02, gamma=0.5):
    """
    xy: (N,2) 2D coords (PCA or UMAP)
    source: pd.Series with values 'TruthSocial' or 'Bluesky'
    out_path: Path to save figure
    bins: image resolution
    margin: plot padding around data
    gamma: gamma correction for better visual balance (0.5 ~ sqrt)
    """
    mask_t = (source == "TruthSocial").values
    mask_b = (source == "Bluesky").values

    x, y = xy[:,0], xy[:,1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx, dy = xmax - xmin, ymax - ymin
    xmin, xmax = xmin - dx*margin, xmax + dx*margin
    ymin, ymax = ymin - dy*margin, ymax + dy*margin
    extent = [xmin, xmax, ymin, ymax]

    # 2D histogram density (computed separately for both platforms)
    H_r, xedges, yedges = np.histogram2d(x[mask_t], y[mask_t], bins=bins, range=[[xmin, xmax],[ymin, ymax]])
    H_b, _,      _      = np.histogram2d(x[mask_b], y[mask_b], bins=bins, range=[[xmin, xmax],[ymin, ymax]])

    # Normalize + gamma correction to prevent large clusters from suppressing smaller ones
    def normalize(H):
        H = H.astype(float)
        if H.max() > 0:
            H /= H.max()
        # gamma < 1 enhances visibility in low-density regions
        return np.power(H, gamma)

    R = normalize(H_r).T  # imshow uses (row,col) = (y,x), so transpose
    B = normalize(H_b).T
    G = np.zeros_like(R)  # green channel unused
    RGB = np.dstack([R, G, B])

    plt.figure(figsize=(10, 8))
    plt.imshow(RGB, origin='lower', extent=extent, aspect='auto')
    # Simple legend using colored patches to represent platforms
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color=(1,0,0,0.7), label=f"TruthSocial (n={(mask_t).sum():,})"),
        Patch(color=(0,0,1,0.7), label=f"Bluesky (n={(mask_b).sum():,})"),
        Patch(color=(1,0,1,0.7), label="Overlap (purple)")
    ], loc='upper right', frameon=True)

    plt.title("Platform density (purple = overlap)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] saved {out_path}")

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

def _truncate(s, n=200):
    s = "" if s is None else str(s)
    return (s[:n] + "…") if len(s) > n else s

def build_interactive_scatter(xy, df, title, out_html):
    """
    xy: (N,2) 2D coords (PCA or UMAP)
    df: DataFrame with columns ['clean','source','cluster']
    title: plot title
    out_html: output path
    """
    if not HAS_PLOTLY:
        print("[INFO] Plotly not installed; skip interactive scatter.")
        return

    scatter_df = pd.DataFrame({
        "x": xy[:,0],
        "y": xy[:,1],
        "cluster": df["cluster"].astype(int),
        "source": df["source"].astype(str),
        "text": df["clean"].map(lambda s: _truncate(s, 220))
    })

    fig = px.scatter(
        scatter_df, x="x", y="y",
        color="cluster",  # initially color by cluster
        hover_data={"text": True, "cluster": True, "source": True, "x": False, "y": False},
        title=title,
        render_mode="webgl" 
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        hoverlabel=dict(namelength=-1),
        margin=dict(l=30, r=30, t=60, b=30)
    )

    # Construct an alternative view colored by source (using same coordinates)
    fig_by_source = px.scatter(
        scatter_df, x="x", y="y",
        color="source",
        hover_data={"text": True, "cluster": True, "source": True, "x": False, "y": False},
        render_mode="webgl"
    )
    fig_by_source.update_traces(marker=dict(size=4, opacity=0.7))

    # Add the second figure’s traces and toggle visibility through dropdown
    for tr in fig_by_source.data:
        fig.add_trace(tr)

    n_cluster_traces = len(fig.data) - len(fig_by_source.data)
    n_source_traces  = len(fig_by_source.data)

    # Visibility states: 0 = color by cluster; 1 = color by source
    vis_cluster = [True]*n_cluster_traces + [False]*n_source_traces
    vis_source  = [False]*n_cluster_traces + [True]*n_source_traces

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            x=1.0, xanchor="right", y=1.15, yanchor="top",
            buttons=[
                dict(label="Color: cluster", method="update",
                     args=[{"visible": vis_cluster},
                           {"title": title + " — colored by cluster"}]),
                dict(label="Color: source", method="update",
                     args=[{"visible": vis_source},
                           {"title": title + " — colored by source"}]),
            ]
        )]
    )

    fig.update_traces(selected_marker=dict(size=6, opacity=1.0),
                      unselected_marker=dict(opacity=0.25))
    fig.update_layout(dragmode="lasso")  

    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[OK] interactive scatter written to {out_html}")

# === PCA ===
pca = PCA(n_components=2, random_state=42)
xy_pca = pca.fit_transform(X)
rgb_density_overlay(xy_pca, df["source"], OUT_DIR / "pca_rgb_density_overlay.png")
build_interactive_scatter(
    xy_pca, df,
    title=f"Embedding PCA (N={len(df):,})",
    out_html=OUT_DIR/"interactive_pca.html"
)

# === UMAP ===
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    xy_umap = reducer.fit_transform(X)
    rgb_density_overlay(xy_umap, df["source"], OUT_DIR / "umap_rgb_density_overlay.png")
    build_interactive_scatter(
        xy_umap, df,
        title=f"Embedding UMAP (N={len(df):,})",
        out_html=OUT_DIR/"interactive_umap.html"
    )

# Save final clustered dataset
df.to_csv(OUT_DIR / "clustered_posts_both.csv", index=False)
print(f"[OK] clustered_posts_both.csv saved")

print("\n" + "="*60)
print(f"ALL DONE! Outputs saved in {OUT_DIR}")
print("="*60)
print(f"\nGenerated files (main ones):")
print(f"  - clustering_metrics_both.csv / .png")
print(f"  - cluster_top_terms_both.csv")
print(f"  - cluster_examples_both.csv")
print(f"  - cluster_source_crosstab.csv")
print(f"  - clustered_posts_both.csv")
print(f"  - len_hist_both.png, word_count_hist_both.png, cluster_sizes_both.png")
print(f"  - cluster_source_stacked_both.png")
print(f"  - fig_scatter_pca_both.png, fig_scatter_umap_both.png")
