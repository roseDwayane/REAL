#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEGLAB .set -> MNE .fif converter
- 連續(raw) .set：輸出一個 .fif
- 分段(epochs/trials) .set：每個 trial 另存一個檔案：{stem}_trial###-epo.fif

用法：
  python eeglab_to_fif.py "C:\\data\\subject01.set"
  python eeglab_to_fif.py "C:\\data\\eeglab_projects" -o "C:\\out" -r --ref avg --montage auto
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Dict

import numpy as np
import mne


def _guess_and_set_montage(info_owner, montage: Optional[str]) -> None:
    """
    info_owner: 可以是 Raw 或 Epochs；兩者皆支援 .set_montage()
    """
    if montage is None:
        return

    if montage == "auto":
        has_dig = info_owner.info.get("dig") is not None and len(info_owner.info["dig"]) > 0
        if has_dig:
            return
        try:
            std = mne.channels.make_standard_montage("standard_1005")
            info_owner.set_montage(std, match_case=False, on_missing="ignore", verbose="error")
            print("[info] Applied 'standard_1005' montage (auto).")
        except Exception as e:
            print(f"[warn] Auto montage failed to apply: {e}")
        return

    try:
        std = mne.channels.make_standard_montage(montage)
        info_owner.set_montage(std, match_case=False, on_missing="ignore")
        print(f"[info] Applied montage: {montage}")
    except Exception as e:
        print(f"[warn] Failed to apply montage '{montage}': {e}")


def _maybe_set_reference(info_owner, ref: Optional[str]) -> None:
    if ref is None:
        return
    if ref.lower() in {"avg", "average"}:
        info_owner.set_eeg_reference("average", projection=False)
        print("[info] Set EEG reference to average.")
        return
    ref_chs = [ch.strip() for ch in ref.split(",") if ch.strip()]
    if not ref_chs:
        return
    info_owner.set_eeg_reference(ref_channels=ref_chs, projection=False)
    print(f"[info] Set EEG reference to: {ref_chs}")


def _add_stim_from_annotations(raw: mne.io.BaseRaw, event_id: Optional[Dict[str, int]] = None) -> None:
    if raw.annotations is None or len(raw.annotations) == 0:
        print("[info] No annotations found; skip STIM channel creation.")
        return
    events, _event_id = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if events.size == 0:
        print("[info] No events parsed from annotations; skip STIM channel creation.")
        return
    n_times = raw.n_times
    sfreq = raw.info["sfreq"]
    first_samp = raw.first_samp
    stim_data = np.zeros((1, n_times), dtype=np.int16)
    for samp, _, code in events:
        idx = int(samp - first_samp)
        if 0 <= idx < n_times:
            stim_data[0, idx] = int(code)
    stim_info = mne.create_info(ch_names=["STI 014"], sfreq=sfreq, ch_types=["stim"])
    stim_raw = mne.io.RawArray(stim_data, stim_info, verbose=False)
    raw.add_channels([stim_raw], force_update_info=True)
    print("[info] Added STIM channel 'STI 014' from annotations.")


def convert_raw(set_path: Path, out_dir: Path, montage: Optional[str], ref: Optional[str],
                make_stim: bool, overwrite: bool) -> Optional[Path]:
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="warning")
    except Exception as e:
        print(f"[error] read_raw_eeglab failed: {e}")
        return None

    _guess_and_set_montage(raw, montage)
    _maybe_set_reference(raw, ref)

    if make_stim:
        _add_stim_from_annotations(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (set_path.stem + ".fif")
    try:
        raw.save(str(out_path), overwrite=overwrite)
        print(f"[ok] Saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"[error] Fail to save {out_path}: {e}")
        return None


def convert_epochs_split(set_path: Path, out_dir: Path, montage: Optional[str], ref: Optional[str],
                         overwrite: bool) -> int:
    """
    將 epochs .set 讀入並拆成單一 trial 檔案：{stem}_trial###-epo.fif
    回傳成功數量
    """
    try:
        epochs = mne.io.read_epochs_eeglab(str(set_path), verbose="warning")
    except Exception as e:
        print(f"[error] read_epochs_eeglab failed: {e}")
        return 0

    # 套用 montage / 參考
    _guess_and_set_montage(epochs, montage)
    _maybe_set_reference(epochs, ref)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(epochs)
    width = max(3, len(str(n)))
    ok = 0
    for i in range(n):
        # 只保留第 i 個 trial
        ep_i = epochs[i]  # slicing 回傳新的 Epochs，僅含一個 epoch
        out_path = out_dir / (f"{set_path.stem}_trial{str(i+1).zfill(width)}-epo.fif")
        try:
            ep_i.save(str(out_path), overwrite=overwrite)
            ok += 1
            print(f"[ok] Saved: {out_path}")
        except Exception as e:
            print(f"[error] Fail to save {out_path}: {e}")
    return ok


def convert_one(set_path: Path, out_dir: Path, montage: Optional[str], ref: Optional[str],
                make_stim: bool, overwrite: bool) -> Optional[Path]:
    """
    嘗試當作 raw 讀取；若失敗且為 epochs .set，改以 epochs 讀入並逐 trial 輸出。
    若為 epochs，回傳 None（因為輸出多個檔案），但視為成功（印出總成功數）。
    """
    # 先試 raw
    try:
        return convert_raw(set_path, out_dir, montage, ref, make_stim, overwrite)
    except Exception:
        pass

    # 若 raw 讀取失敗，嘗試 epochs 模式（拆檔）
    print("[info] Fall back to epochs mode (split per trial).")
    cnt = convert_epochs_split(set_path, out_dir, montage, ref, overwrite)
    if cnt > 0:
        # 回傳 None 但標示成功
        return None
    else:
        return None


def iter_set_files(root: Path, recursive: bool = False):
    if root.is_file():
        return [root]
    if recursive:
        return sorted(root.rglob("*.set")) + sorted(root.rglob("*.SET"))
    else:
        return sorted(root.glob("*.set")) + sorted(root.glob("*.SET"))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EEGLAB .set to MNE .fif 。若為 epochs 檔，會自動拆成每 trial 一個 -epo.fif"
    )
    p.add_argument("input", help="輸入 .set 檔或資料夾路徑")
    p.add_argument("-o", "--out", default=None, help="輸出資料夾 (預設為輸入所在資料夾)")
    p.add_argument("-r", "--recursive", action="store_true", help="資料夾模式：遞迴尋找 .set")
    p.add_argument("--montage", default=None,
                   help="電極座標：None(不處理) / auto(自動嘗試 standard_1005) / 或明確名稱如 standard_1020, standard_1005")
    p.add_argument("--ref", default=None,
                   help="參考：None(不變更) / avg(平均參考) / 指定通道，例如 'CZ' 或 'M1,M2'")
    p.add_argument("--stim", action="store_true", help="(僅 raw 有效) 由 annotations 生成 'STI 014' STIM channel")
    p.add_argument("--overwrite", action="store_true", help="若輸出已存在則覆蓋")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    in_path = Path(args.input).expanduser().resolve()

    out_dir = Path(args.out).expanduser().resolve() if args.out else ((in_path.parent if in_path.is_file() else in_path).resolve())

    set_files = iter_set_files(in_path, recursive=args.recursive)
    if not set_files:
        print("[error] 找不到任何 .set 檔案")
        return 2

    print(f"[info] Found {len(set_files)} file(s).")
    ok, fail = 0, 0
    for f in set_files:
        # 先嘗試 raw；若為 epochs 將自動 fallback 並逐 trial 輸出
        try:
            res = mne.io.read_raw_eeglab(str(f), preload=True, verbose="error")
            # 能讀成 raw，交給 raw 轉檔
            res = convert_raw(f, out_dir, args.montage, args.ref, args.stim, args.overwrite)
            if res is None:
                fail += 1
            else:
                ok += 1
        except Exception as e:
            # raw 不行，走 epochs 模式
            print(f"[info] {f.name} looks like epochs (.set with trials). Split per trial...")
            cnt = convert_epochs_split(f, out_dir, args.montage, args.ref, args.overwrite)
            if cnt > 0:
                ok += 1  # 視為該檔成功處理
            else:
                fail += 1

    print(f"[done] success: {ok}, failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
