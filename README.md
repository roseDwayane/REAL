# REAL: Robust EEG Analysis and Labeling for Chronic Migraine Patients Amidst Label Noises

## 0) 建環境（建議 Conda，含 GPU）

```bash
# 建立與啟用環境（Python 3.10 或 3.11 皆可）
conda create -n eeglab-ct python=3.10 -y
conda activate eeglab-ct

# 安裝 PyTorch（依你電腦的 CUDA 版本調整；此例 CUDA 12.1）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 其他套件
pip install -r requirements.txt
# 若你完全不會呼叫 limit_gpu_memory()，無需裝 TensorFlow
# 若需要，才：pip install "tensorflow==2.15.*"
```

確認 GPU：

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 1) 準備資料（.fif）與目錄

你的主程式會用 `mne.io.Raw(file_path)` 讀 `.fif`，並假定**每個檔名以前綴 `keyword='EC'` 過濾**。資料結構建議如下：

```
D:\EEG\mne_fif_root\
 ├─ CTL\        # 類別 0
 │   ├─ EC_001.fif
 │   └─ EC_002.fif
 └─ NOT_CTL\    # 類別 1
     ├─ EC_101.fif
     └─ EC_102.fif
```

如果你目前是 EEGLAB `.set` 檔，先用我給的工具轉：

```bash
python convert_eeglab_to_fif.py
```

（打開檔案把 `src_dir`、`dst_dir` 改成你的路徑；如需通道重排，取消註解並依你的通道順序修改。）

> 注意：你的模型假定 30 通道，順序為
> `['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T3','C3','CZ','C4','T4','TP7','CP3','CPZ','CP4','TP8','T5','P3','PZ','P4','T6','O1','OZ','O2']`
> 若你的資料不同，請在轉檔或前處理時重排到這個順序。

## 2) 準備 SimSiam 特徵（.pkl）

這份程式是「**用已抽好的特徵做 Co-teaching 訓練**」。對每個 `.fif`，`features_path` 下面要有**同名 `.pkl`**，內容是一個 `numpy.ndarray` 形狀約為 `[N, feature_dim]`（預設 `feature_dim=512`），例如：

```
D:\EEG\features\simsiam\
 ├─ EC_001.fif.pkl     # 內含 shape=(N1, 512)
 ├─ EC_002.fif.pkl     # 內含 shape=(N2, 512)
 ├─ EC_101.fif.pkl
 └─ EC_102.fif.pkl
```

如果你的特徵不是 512 維，請把 `eeg_config['feature_dim']` 改成一致。

## 3) 建立 `cfg6.py`

把我給你的 [cfg6\_template.py](sandbox:/mnt/data/cfg6_template.py) 下載、改路徑後，存成你專案的 `cfg6.py`。**這是關鍵**：主程式開頭 `from cfg6 import *` 會到這裡抓到：

* `base_path`、`class_paths`、`class_names`、`class_labels`
* `features_path`（特徵 `.pkl` 所在處）
* `epochs_path`（中繼 CSV 輸出位置）
* `keyword`（檔名前綴過濾用；若不要過濾設成 `''`）
* `rate_col='Rate'`（畫圖用的欄位名，避免 NameError）

> 注意：你貼的主程式底部**又寫了一次** `features_path = '../../SimSiam(Imb2)/Features/...'`。
> 請**刪掉或改掉**那一行，避免覆蓋掉 `cfg6.py` 的設定，否則會找不到你的 `.pkl`。

## 4) 檢查 Python 專案匯入路徑

程式會匯入很多自家模組（例如 `EEGLab.*`、`Co_teachingLab.*`、`utils.*`）。請確認：

* 這些資料夾在你的專案內，且 **`PYTHONPATH`** 能找到
* 或者你用相對路徑的匯入（維持你現有專案結構）
  若看到 `ModuleNotFoundError: No module named 'EEGLab'`，就是路徑沒對。把專案根目錄加入 `PYTHONPATH` 或在 IDE 設定專案根。

## 5) 按主程式流程跑

主程式 `__main__` 區塊做的事（我用白話解釋）：

1. **建立物件與記錄器**

   * `eeg = EEGLabLOSOFeature()`
   * `eeg.register_logging()` → 會把 `eeg_config` 的超參等資訊寫到 `*.log`

2. **蒐集資料清單，寫入 `<code>_<classes>_<keyword>_info.csv`**

   * `eeg.collect_dataset_info_with_black_list()`
     會掃 `base_path / class_paths[c]` 下的 `.fif`，過濾 `keyword` 開頭的檔名，記錄 `n_times` 等資訊。

3. **產生 epochs 組合清單**

   * `eeg.make_epochs_sn_df()`
     用 `duration=1.5`、`stride=1.0`（依 250 Hz 設計）把每個檔要抽哪些 epochs 編號寫到 `*_epochs.csv`。

4. **載入特徵到記憶體（合併成一個大陣列與標籤）**

   * `eeg.load_feature_data_to_array()`
     從 `features_path` 讀每個 `.pkl`，串成 `dataset_array`（Tensor）與 `dataset_labels`（Tensor）。
     若開啟 `eeg_config['data_transform']=True`，會做 `SelfAttention(T)`；
     若 `data_normalize=True`，會做 L2-normalize。

5. **設定訓練函式**

   * `eeg.fit_method = eeg.train_co_teaching_plus_adam`
     （也可改成 `train_co_teaching_plus` 用 SGD 與餘弦學習率）

6. **跑 5 組（group=0..4）：每組都訓練 + 預測 + 存檔**

   * `eeg.make_epochs_sn_df()`（再產一次保險）
   * `eeg.evaluate_self_feature()`

     * 內部會把 White List 全部當 train/val，訓練雙網路（Co-teaching++），最後用 **最後 10 個 epoch（190–199）** 的權重平均機率做推論，並把每個 epoch 的 `Pred/Prob/SC` 等欄位寫回 `*_epochs.csv`
   * `eeg.save_epochs_info_as(g)` → 會把 `*_epochs.csv` 另存成 `*_epochs_g.csv`

7. **投票 + 總結**

   * `eeg.vote_and_save()`：把上面 5 次輸出的 `*_epochs_0..4.csv` 投票，產生 Black List，並把結果寫回 `*_info.csv`，另存 `*_BL.json`
   * `eeg.summary()`：整理 `evaluate.csv`，輸出平均 Acc/TPR/Recall 與常見錯誤樣本 ID

> 訓練細節：
>
> * 兩個同構網路 `model1/model2`（`EEGNet_PT452_FC1`），初期 `init_epoch=5` 做 warm-up 的 Co-teaching，之後切換 Co-teaching++。
> * `epochs=200`，**只有第 190–199 epoch** 的權重會被存成 `..._A_*.pt / ..._B_*.pt` 並拿來做最終平均投票推論。
> * 批次大小 `batch_size=128`，若記憶體不夠就降（例如 64/32）。

## 6) 關鍵設定（`eeg_config` 可改）

```python
eeg_config = {
  'duration': 1.5, 'stride': 1.0, 'CF': 1,               # epoch 切片
  'feature_dim': 512, 'fc_dim': 64,                      # 特徵維度/FC維度
  'data_normalize': False, 'data_transform': True, 'T':1.0,
  'epochs': 200, 'batch_size': 128, 'lr_init': 1e-3,     # Adam 預設
  'epoch_decay_start': 80, 'weight_decay': 1e-4,
  'net': EEGNet_PT452_FC1,
  'dataset_train': DatasetCoTeachingArrayBalance,
  'dataset_predict': DatasetCoTeachingArray,
  'init_epoch': 5, 'forget_rate': 0.2, 'num_gradual': 10, 'exponent': 1,
  'loss_fn': loss_coteaching_plus_m5,
  'Note': 'Co-teaching Plus for qualify with SimSiam feature.'
}
```

* **`feature_dim` 一定要跟你 `.pkl` 的第二維一致**。不然 FC 尺寸會對不上。
* 降低 `batch_size` 可以避免 CUDA OOM。
* 如果你不想做 `SelfAttention`，把 `data_transform=False`。

## 7) 執行

確保你在專案根（能 `import EEGLab.*` 等），執行：

```bash
python your_main_script.py
```

終端你會看到：

* `cuda:0` 或 `cpu`（取決於你的 GPU 狀態）
* 每個 epoch 的 Acc/Loss 摘要
  輸出產物會包含：
* `*.log`（訓練過程）
* `*_info.csv`、`*_epochs.csv`、`*_epochs_0..4.csv`
* `*_evaluate.csv`（彙總）
* 模型權重 `..._A_190.pt` \~ `..._A_199.pt` 及 `..._B_*.pt`
* 評估圖（放在 `self.log.fig_path`；名稱包含 `Testing.png`、`Black_List.png` 等）

---

# 常見錯誤排查（直接對症下藥）

1. **`FileNotFoundError: ... .pkl`**

* `features_path` 要與 `.fif` 檔名對上（`EC_001.fif` ↔ `EC_001.fif.pkl`）。
* 檢查主程式底部是否有把 `features_path` 覆蓋掉（請刪或改那一行）。

2. **`ValueError: shapes ...`（特徵維度不合）**

* 把 `eeg_config['feature_dim']` 改成你的 `.pkl` 向量長度。

3. **`len(raw.info['ch_names']) < 30` 被跳過**

* 你的資料通道數不足或名稱對不上。請轉檔時重排/補齊到固定的 30 通道並確保命名一致。

4. **`ModuleNotFoundError: EEGLab / Co_teachingLab / utils ...`**

* 設定 `PYTHONPATH` 指到你的專案根，或在 IDE 把專案根設為 Source Root。

5. **`CUDA out of memory`**

* 降低 `batch_size`（128 → 64 → 32），或先用 CPU 版確認流程。
* 若同機裝了 TensorFlow 也吃 GPU，避免同時載入；你若不呼叫 `limit_gpu_memory()` 就根本不用裝 TF。

6. **`NameError: rate_col`**

* 在 `cfg6.py` 設定 `rate_col = 'Rate'`（我已在範本幫你放好）。

7. **什麼都顯示 CPU**

* 你的 PyTorch 可能不是 GPU 版，或 CUDA 驅動版本不符。請用 conda 指令再裝一次（見步驟 0）。

---

# 最小測試策略（先確定流程通）

* 先放 **各 1–2 個 `.fif`** 到 `CTL` 與 `NOT_CTL`。
* 為這 4 個 `.fif` 準備最小 `.pkl` 檔（就算每個只有 20 筆向量也行），`feature_dim=512`。
* 把 `epochs` 暫時降到 **5**、`batch_size` 設 **32**，看流程能否從頭到尾產生 `*_epochs_0..4.csv` 與圖檔。
* 成功後再恢復正式參數。