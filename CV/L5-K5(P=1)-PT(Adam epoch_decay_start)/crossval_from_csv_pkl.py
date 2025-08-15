#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Cross-validation on EEG features from CSV (info) + PKL files.
# - Splits at FILE level with StratifiedKFold
# - Trains a small PyTorch MLP on per-epoch/per-segment features (samples from PKL)
# - Aggregates per-sample predictions back to FILE by averaging positive probs
# - Saves fold predictions and an overall summary CSV
#
# Usage:
#   python crossval_from_csv_pkl.py --info_csv /path/to/info.csv --features_dir /path/to/features --out_dir ./cv_out
#
# Notes:
# - info.csv must have at least: 'file', 'label' columns; optional 'Black_List' is ignored
# - PKL files are matched to rows in info.csv by a tolerant filename key (ignores suffixes & case)
# - PKL contents can be shape (N, D) or (N, ...); the last dimensions are flattened to D

import argparse
import os
import re
from pathlib import Path
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

def norm_key_strip(name_or_path: str) -> str:
    base = Path(str(name_or_path)).name
    base = re.sub(r'\.fif(\.gz)?$', '', base, flags=re.I)
    base = re.sub(r'\.pkl$', '', base, flags=re.I)
    base = re.sub(r'[-_](epo|raw)(?:[-_].*)?$', '', base, flags=re.I)
    base = re.sub(r'([-_]pre)?[-_]?trial\d+$', '', base, flags=re.I)
    base = re.sub(r'([-_]pre)?[-_]?trial\d+([-_].*)?$', '', base, flags=re.I)
    base = base.strip()
    base = re.sub(r'\s+', '', base)
    base = re.sub(r'[-_]+', '_', base)
    return base.lower()

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def index_pkls(features_dir: str):
    idx = {}
    for root, _, files in os.walk(features_dir):
        for f in files:
            if f.lower().endswith('.pkl'):
                p = os.path.join(root, f)
                idx[norm_key_strip(p)] = p
    return idx

def load_info(info_csv: str) -> pd.DataFrame:
    df = pd.read_csv(info_csv)
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    for col in ['file', 'label']:
        if col not in df.columns:
            raise ValueError(f"info_csv missing required column: '{col}'")
    df['label'] = df['label'].astype(int)
    return df

def load_x_y_from_info(df_info: pd.DataFrame, pkl_index, feature_dim: int = None):
    X_list, y_list, file_ids = [], [], []
    used_files = []
    miss = []
    for file_idx, (_, row) in enumerate(df_info.iterrows()):
        key = norm_key_strip(str(row['file']).strip())
        pkl_path = pkl_index.get(key)
        if pkl_path is None:
            miss.append(key)
            continue
        import pickle
        with open(pkl_path, 'rb') as f:
            arr = pickle.load(f)
        x = np.asarray(arr)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        if feature_dim is not None:
            if x.shape[1] > feature_dim:
                x = x[:, :feature_dim]
            elif x.shape[1] < feature_dim:
                pad = np.zeros((x.shape[0], feature_dim - x.shape[1]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
        y = np.full((x.shape[0],), int(row['label']), dtype=np.int64)
        X_list.append(x.astype(np.float32))
        y_list.append(y)
        file_ids += [file_idx] * x.shape[0]
        used_files.append(str(row['file']).strip())
    if not X_list:
        raise ValueError("No PKL matched the CSV. Check features_dir & filenames. Missing (first 10 keys): " + str(miss[:10]))
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return X, y, file_ids, used_files

class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_fold(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight_decay=1e-4, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0
    bad = 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = logits.argmax(1)
                vcorrect += int((pred == yb).sum().item())
                vtotal += int(yb.numel())
        val_acc = vcorrect / max(vtotal,1)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    df_info_all = load_info(args.info_csv)
    if args.keyword:
        kws = tuple([k.strip().upper() for k in args.keyword.split(',') if k.strip()])
        if len(kws) > 0:
            mask = df_info_all['file'].astype(str).str.upper().str.startswith(kws)
            df_info_all = df_info_all[mask].reset_index(drop=True)
    pkl_map = index_pkls(args.features_dir)
    X, y, file_ids, used_files = load_x_y_from_info(df_info_all, pkl_map, feature_dim=args.feature_dim)
    file_to_indices = {}
    for i, fid in enumerate(file_ids):
        file_to_indices.setdefault(fid, []).append(i)
    n_files = len(used_files)
    if n_files < args.folds:
        raise ValueError(f"Not enough files ({n_files}) for {args.folds}-fold CV.")
    y_file = []
    for fid in range(n_files):
        y_file.append(int(y[file_to_indices[fid][0]]))
    y_file = np.array(y_file, dtype=np.int64)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_rows = []
    fold_reports = []
    for fold, (train_files, test_files) in enumerate(skf.split(np.arange(n_files), y_file), 1):
        train_idx = np.concatenate([file_to_indices[fid] for fid in train_files])
        test_idx  = np.concatenate([file_to_indices[fid] for fid in test_files])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed+fold)
        y_train_samples = y[train_idx]
        (tr_samp_idx, va_samp_idx), = sss.split(train_idx, y_train_samples)
        tr_idx = train_idx[tr_samp_idx]
        va_idx = train_idx[va_samp_idx]
        X_tr = torch.from_numpy(X[tr_idx]); y_tr = torch.from_numpy(y[tr_idx])
        X_va = torch.from_numpy(X[va_idx]); y_va = torch.from_numpy(y[va_idx])
        X_te = torch.from_numpy(X[test_idx]); y_te = torch.from_numpy(y[test_idx])
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        model = MLP(in_dim=X.shape[1], n_classes=2, p_drop=args.dropout).to(device)
        model, best_val = train_fold(model, tr_loader, va_loader, device,
                                     epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)
        probs_te = predict_proba(model, te_loader, device)
        test_file_ids = [file_ids[i] for i in test_idx]
        file_pred_rows = []
        for fid in sorted(set(test_file_ids)):
            import numpy as _np
            samp_positions = _np.where(_np.array(test_file_ids) == fid)[0]
            p1 = float(probs_te[samp_positions, 1].mean())
            y_true_file = y_file[fid]
            y_pred_file = int(p1 >= 0.5)
            file_pred_rows.append({
                "fold": fold,
                "file_index": int(fid),
                "file": used_files[fid],
                "y_true": int(y_true_file),
                "prob_1": p1,
                "y_pred": y_pred_file,
            })
        df_fold = pd.DataFrame(file_pred_rows).sort_values("file_index").reset_index(drop=True)
        df_fold.to_csv(os.path.join(args.out_dir, f"fold_{fold}_file_preds.csv"), index=False)
        y_true_fold = df_fold["y_true"].to_numpy()
        y_pred_fold = df_fold["y_pred"].to_numpy()
        rpt = classification_report(y_true_fold, y_pred_fold, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_fold, y_pred_fold, labels=[0,1])
        tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        fold_reports.append({
            "fold": fold,
            "n_files_test": int(len(df_fold)),
            "accuracy": float(rpt["accuracy"]),
            "precision_pos": float(rpt["1"]["precision"]),
            "recall_pos": float(rpt["1"]["recall"]),
            "f1_pos": float(rpt["1"]["f1-score"]),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
        all_rows.append(df_fold)
    df_all = pd.concat(all_rows, axis=0).reset_index(drop=True)
    df_all.to_csv(os.path.join(args.out_dir, "all_folds_file_preds.csv"), index=False)
    df_rep = pd.DataFrame(fold_reports)
    mean_row = df_rep.drop(columns=["fold","n_files_test"]).mean(numeric_only=True).to_dict()
    std_row = df_rep.drop(columns=["fold","n_files_test"]).std(numeric_only=True).to_dict()
    mean_row.update({"fold": "mean", "n_files_test": float(df_rep["n_files_test"].mean())})
    std_row.update({"fold": "std", "n_files_test": float(df_rep["n_files_test"].std())})
    df_rep = pd.concat([df_rep, pd.DataFrame([mean_row, std_row])], axis=0, ignore_index=True)
    df_rep.to_csv(os.path.join(args.out_dir, "summary_metrics.csv"), index=False)
    report = {
        "accuracy_mean": float(df_rep.loc[df_rep["fold"]=="mean","accuracy"].values[0]),
        "precision_pos_mean": float(df_rep.loc[df_rep["fold"]=="mean","precision_pos"].values[0]),
        "recall_pos_mean": float(df_rep.loc[df_rep["fold"]=="mean","recall_pos"].values[0]),
        "f1_pos_mean": float(df_rep.loc[df_rep["fold"]=="mean","f1_pos"].values[0]),
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Done. Outputs saved to:", args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--info_csv", type=str, required=True, help="Path to info CSV with columns: file,label[,Black_List]")
    p.add_argument("--features_dir", type=str, required=True, help="Directory containing PKL feature files")
    p.add_argument("--out_dir", type=str, default="./cv_out")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--feature_dim", type=int, default=None, help="Force features to this dimension (pad/truncate)")
    p.add_argument("--keyword", type=str, default="", help="Optional CSV filename prefix filter, e.g. 'EC,EO,DC'")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()
    main(args)
