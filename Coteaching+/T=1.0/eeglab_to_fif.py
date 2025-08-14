#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEGLAB .set -> MNE .fif converter

Usage examples:
  # 單檔轉換
  python eeglab_to_fif.py "C:\data\subject01.set"

  # 指定輸出資料夾、平均參考、嘗試套用標準電極座標
  python eeglab_to_fif.py "C:\data\subject01.set" -o "C:\out" --ref avg --montage standard_1005

  # 將整個資料夾內的 .set 皆轉換 (遞迴)
  python eeglab_to_fif.py "C:\data\eeglab_projects" -o "C:\out" -r

Notes:
- 需先安裝: pip install mne numpy
- .set 搭配的 .fdt 檔案需放在同一資料夾 (MNE 會自動讀取)
- 事件(Event)會保存在 MNE 的 Annotations 中；若需要 STIM channel，可加 --stim 生成 "STI 014"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Dict

import numpy as np
import mne


def _guess_and_set_montage(raw: mne.io.BaseRaw, montage: Optional[str]) -> None:
    """
    Auto / explicit montage handling.
    - If montage is None: do nothing (沿用 EEGLAB 檔內之座標/不處理)
    - If montage is 'auto': 若檔內沒有座標，嘗試 standard_1005；若 channel 名稱對不上會忽略
    - If montage is explicit (e.g. 'standard_1020', 'standard_1005', 'biosemi64' ...): 強制嘗試套用
    """
    if montage is None:
        return

    if montage == "auto":
        # 若已有 dig/monatge 就跳過
        has_dig = raw.info.get("dig") is not None and len(raw.info["dig"]) > 0
        if has_dig:
            return
        try:
            std = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(std, match_case=False, on_missing="ignore", verbose="error")
            print("[info] Applied 'standard_1005' montage (auto).")
        except Exception as e:
            print(f"[warn] Auto montage failed to apply: {e}")
        return

    # 明確指定 montage 名稱
    try:
        std = mne.channels.make_standard_montage(montage)
        raw.set_montage(std, match_case=False, on_missing="ignore")
        print(f"[info] Applied montage: {montage}")
    except Exception as e:
        print(f"[warn] Failed to apply montage '{montage}': {e}")


def _maybe_set_reference(raw: mne.io.BaseRaw, ref: Optional[str]) -> None:
    """
    參考電極設定：
      - None: 不變更 (沿用 EEGLAB)
      - 'avg' / 'average': 設定平均參考
      - 其他字串: 視為單一或多個通道名稱 (以逗號分隔) 作為參考
    """
    if ref is None:
        return

    if ref.lower() in {"avg", "average"}:
        raw.set_eeg_reference("average", projection=False)
        print("[info] Set EEG reference to average.")
        return

    # 多個通道用逗號切
    ref_chs = [ch.strip() for ch in ref.split(",") if ch.strip()]
    if not ref_chs:
        return
    raw.set_eeg_reference(ref_channels=ref_chs, projection=False)
    print(f"[info] Set EEG reference to: {ref_chs}")


def _add_stim_from_annotations(raw: mne.io.BaseRaw, event_id: Optional[Dict[str, int]] = None) -> None:
    """
    將 Annotations 轉成 STIM channel 'STI 014'（在事件發生樣本點寫入事件代碼）。
    注意：同一時間若有多個事件，後者會覆蓋前者的代碼。
    """
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


def convert_one(
    set_path: Path,
    out_dir: Path,
    montage: Optional[str] = None,
    ref: Optional[str] = None,
    make_stim: bool = False,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    轉換單一 .set 檔到 .fif。成功則回傳輸出路徑。
    """
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="warning")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return None
    except Exception as e:
        print(f"[error] Fail to read {set_path}: {e}")
        return None

    # 座標與參考設定
    _guess_and_set_montage(raw, montage)
    _maybe_set_reference(raw, ref)

    # 可選：從註解建立 STIM 通道
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


def iter_set_files(root: Path, recursive: bool = False) -> Sequence[Path]:
    """列舉資料夾內所有 .set 檔案"""
    if root.is_file():
        return [root]
    if recursive:
        return sorted(root.rglob("*.set")) + sorted(root.rglob("*.SET"))
    else:
        return sorted(root.glob("*.set")) + sorted(root.glob("*.SET"))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert EEGLAB .set to MNE .fif (保留 annotations / 可選 STIM channel)"
    )
    p.add_argument("input", help="輸入 .set 檔或資料夾路徑")
    p.add_argument("-o", "--out", default=None, help="輸出資料夾 (預設為輸入所在資料夾)")
    p.add_argument("-r", "--recursive", action="store_true", help="資料夾模式：遞迴尋找 .set")
    p.add_argument("--montage", default=None,
                   help="電極座標：None(不處理) / auto(自動嘗試 standard_1005) / 或明確名稱如 standard_1020, standard_1005")
    p.add_argument("--ref", default=None,
                   help="參考：None(不變更) / avg(平均參考) / 指定通道，例如 'CZ' 或 'M1,M2'")
    p.add_argument("--stim", action="store_true", help="由 annotations 生成 'STI 014' STIM channel")
    p.add_argument("--overwrite", action="store_true", help="若輸出已存在則覆蓋")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    in_path = Path(args.input).expanduser().resolve()

    # 決定輸出資料夾
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (in_path.parent if in_path.is_file() else in_path).resolve()

    set_files = iter_set_files(in_path, recursive=args.recursive)
    if not set_files:
        print("[error] 找不到任何 .set 檔案")
        return 2

    print(f"[info] Found {len(set_files)} file(s).")
    ok, fail = 0, 0
    for f in set_files:
        res = convert_one(
            set_path=f,
            out_dir=out_dir,
            montage=args.montage,
            ref=args.ref,
            make_stim=args.stim,
            overwrite=args.overwrite,
        )
        if res is None:
            fail += 1
        else:
            ok += 1

    print(f"[done] success: {ok}, failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
