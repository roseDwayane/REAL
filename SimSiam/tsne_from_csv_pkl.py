#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE visualization for EEG features stored as PKL files, matched via an info CSV.
- Robust filename normalization to align CSV fif names with feature PKLs
- Optional SelfAttention preprocessing (silently skipped if unavailable)
- Optional PCA before t-SNE for stability
- Sampling controls per-file and globally
- EC/EO aware markers if available in CSV (`EC-EO` column) or inferred from filename prefix
"""
import argparse
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except Exception:
    TSNE = None
    PCA = None

# Optional SelfAttention (graceful fallback)
def _identity(x): return x
try:
    from SimSiamLib.EEGLab_Transform import SelfAttention as _SA
    def maybe_self_attention(arr, T: float):
        sa = _SA(T=T)
        try:
            return sa(arr)
        except Exception:
            return arr
except Exception:
    def maybe_self_attention(arr, T: float):
        # Fallback: no-op if library unavailable
        return arr

def norm_key_strip(name_or_path: str) -> str:
    """
    Normalize file/path to a comparable key:
    - Keep only basename
    - Strip extensions: .fif, .fif.gz, .pkl
    - Strip trailing -epo/_epo/-raw/_raw (+ anything following)
    - Strip trial suffixes like _pre_trial001, -trial2, _trial003, etc.
    - Collapse -, _ to single underscore, remove spaces, lowercase
    """
    base = Path(str(name_or_path)).name
    base = re.sub(r'\.fif(\.gz)?$', '', base, flags=re.I)
    base = re.sub(r'\.pkl$', '', base, flags=re.I)
    base = re.sub(r'[-_](epo|raw)(?:[-_].*)?$', '', base, flags=re.I)
    base = re.sub(r'([-_]pre)?[-_]?trial\d+$', '', base, flags=re.I)
    # If trial appears with other suffixes after it
    base = re.sub(r'([-_]pre)?[-_]?trial\d+([-_].*)?$', '', base, flags=re.I)
    base = base.strip()
    base = re.sub(r'\s+', '', base)
    base = re.sub(r'[-_]+' , '_', base)
    return base.lower()

def build_pkl_index(features_dir: str):
    """Walk features_dir for .pkl files and return mapping key -> file path"""
    idx = {}
    for root, _, files in os.walk(features_dir):
        for f in files:
            if f.lower().endswith('.pkl'):
                p = os.path.join(root, f)
                idx[norm_key_strip(p)] = p
    return idx

def infer_eye_from_filename(path: str):
    name = Path(path).name.upper()
    if name.startswith('EC'):
        return 'EC'
    if name.startswith('EO'):
        return 'EO'
    return 'UNK'

def clamp_perplexity(perplexity: float, n_samples: int) -> float:
    # TSNE requires perplexity < n_samples and > 1
    if n_samples <= 5:
        return max(1.0, min(2.0, perplexity))
    upper = max(2.0, (n_samples - 1) / 3.0 - 1e-6)
    return float(max(2.0, min(perplexity, upper)))

def load_all_features(info_csv: str,
                      features_dir: str,
                      max_per_file: int = 200,
                      max_total: int = 0,
                      use_self_attention: bool = False,
                      sa_T: float = 1.0,
                      seed: int = 42):
    """
    Returns (X, y, eyes, hit_rows, miss_rows)
    - X: (N, D) float32
    - y: (N,) int64 labels
    - eyes: (N,) str EC/EO/UNK
    - hit_rows: list of (csv_index, file, matched_pkl_path, n_used)
    - miss_rows: list of (csv_index, file)
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(info_csv)
    # Column normalization
    df.columns = df.columns.str.replace('\ufeff','', regex=False).str.strip()

    if 'file' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have at least 'file' and 'label' columns.")

    pkl_idx = build_pkl_index(features_dir)
    X_list, y_list, eyes_list = [], [], []
    hit_rows, miss_rows = [], []

    for i, row in df.iterrows():
        fpath = str(row['file']).strip()
        key = norm_key_strip(fpath)
        pkl_path = pkl_idx.get(key)
        if pkl_path is None:
            miss_rows.append((i, fpath))
            continue

        # Load features (expects shape (n_samples, D))
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                feat = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load PKL: {pkl_path} => {e}")

        feat = np.asarray(feat)
        if feat.ndim == 1:  # list of vectors
            try:
                feat = np.stack(feat, axis=0)
            except Exception:
                feat = np.atleast_2d(feat)

        if use_self_attention:
            try:
                feat = maybe_self_attention(feat, T=sa_T)
            except Exception:
                pass

        n = feat.shape[0]
        take = np.arange(n)
        if max_per_file and max_per_file > 0 and n > max_per_file:
            take = rng.choice(n, size=max_per_file, replace=False)
        feat = feat[take]

        # labels & EC/EO
        label = int(row['label'])
        eye = row['EC-EO'] if 'EC-EO' in df.columns else infer_eye_from_filename(fpath)
        y_list.append(np.full((feat.shape[0],), label, dtype=np.int64))
        eyes_list.append(np.full((feat.shape[0],), str(eye), dtype=object))

        X_list.append(feat.astype(np.float32, copy=False))
        hit_rows.append((i, fpath, pkl_path, feat.shape[0]))

    if not X_list:
        raise RuntimeError("No features matched between CSV and PKL dir. Check naming and paths.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    eyes = np.concatenate(eyes_list, axis=0)

    # Optional global downsample
    if max_total and max_total > 0 and X.shape[0] > max_total:
        sel = np.random.default_rng(seed).choice(X.shape[0], size=max_total, replace=False)
        X, y, eyes = X[sel], y[sel], eyes[sel]

    return X, y, eyes, hit_rows, miss_rows

def run_tsne(X: np.ndarray,
             y: np.ndarray,
             eyes: np.ndarray,
             outdir: str,
             title_stub: str,
             seed: int = 42,
             perplexity: float = 30.0,
             pca_dims: int = 50):
    if TSNE is None:
        raise ImportError("scikit-learn not found. Please install scikit-learn to use t-SNE.")
    os.makedirs(outdir, exist_ok=True)

    n = X.shape[0]
    perp = clamp_perplexity(perplexity, n)

    # PCA (optional but recommended)
    X2 = X
    if pca_dims and pca_dims > 0 and PCA is not None and X.shape[1] > pca_dims:
        pca = PCA(n_components=pca_dims, random_state=seed, svd_solver='auto')
        X2 = pca.fit_transform(X)

    tsne = TSNE(n_components=2,
                learning_rate='auto',
                init='pca',
                perplexity=perp,
                n_iter=2000,
                n_iter_without_progress=500,
                random_state=seed,
                verbose=1)
    emb = tsne.fit_transform(X2)

    # Save embedding
    npz_path = os.path.join(outdir, f"{title_stub}_tsne_p{perp:.1f}_pca{pca_dims}.npz")
    np.savez(npz_path, emb=emb, y=y, eyes=eyes)
    print(f"Saved embedding to: {npz_path}")

    # Plot
    fig = plt.figure(figsize=(6, 5), dpi=140)
    ax = fig.add_subplot(111)

    # Color map for labels (categorical)
    labels = sorted(set(y.tolist()))
    colors = plt.cm.get_cmap('tab10', max(10, len(labels)))
    markers = {'EC': 'o', 'EO': '^', 'UNK': 'x'}

    for lab in labels:
        for eye, m in markers.items():
            mask = (y == lab) & (eyes == eye)
            if np.any(mask):
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, marker=m, alpha=0.7,
                           label=f"{eye}_{lab}", c=[colors(lab)])

    # Legends (avoid duplicates)
    handles, labels_txt = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_txt, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8, frameon=True)

    ax.set_title(f"{title_stub}  (N={n}, perp={perp:.1f}, pca={pca_dims})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    png_path = os.path.join(outdir, f"{title_stub}_tsne_p{perp:.1f}_pca{pca_dims}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"Saved figure to: {png_path}")
    return png_path, npz_path

def main():
    ap = argparse.ArgumentParser(description="t-SNE visualization from CSV (file,label[,EC-EO]) + PKL feature dir")
    ap.add_argument("--info-csv", required=True, help="Path to info CSV with at least 'file' and 'label' columns.")
    ap.add_argument("--features-dir", required=True, help="Directory containing per-file .pkl features.")
    ap.add_argument("--outdir", default="./Fig_t_SNE", help="Output directory for PNG/NPZ files.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (auto-clamped if too large).")
    ap.add_argument("--pca-dims", type=int, default=50, help="PCA dims before t-SNE (0 to disable)." )
    ap.add_argument("--max-per-file", type=int, default=200, help="Max samples to draw per PKL file (<=0 to keep all)." )
    ap.add_argument("--max-total", type=int, default=0, help="Global cap on total samples (0 to disable)." )
    ap.add_argument("--self-attention", action="store_true", help="Apply SelfAttention(T) if library available." )
    ap.add_argument("--sa-T", type=float, default=1.0, help="Temperature for SelfAttention if used." )
    args = ap.parse_args()

    # Load data
    X, y, eyes, hits, misses = load_all_features(
        info_csv=args.info_csv,
        features_dir=args.features_dir,
        max_per_file=args.max_per_file,
        max_total=args.max_total,
        use_self_attention=args.self_attention,
        sa_T=args.sa_T,
        seed=args.seed,
    )

    # Report matching stats
    report = {
        "matched_files": len(hits),
        "missing_files": len(misses),
        "total_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
    }
    # Save a small JSON report
    os.makedirs(args.outdir, exist_ok=True)
    rep_path = os.path.join(args.outdir, "tsne_input_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    title_stub = Path(args.features_dir).name
    png_path, npz_path = run_tsne(
        X, y, eyes,
        outdir=args.outdir,
        title_stub=title_stub,
        seed=args.seed,
        perplexity=args.perplexity,
        pca_dims=args.pca_dims
    )

    print("=== Done ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Figure: {png_path}")
    print(f"Embedding: {npz_path}")

if __name__ == "__main__":
    main()
