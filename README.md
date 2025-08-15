# REAL: Robust EEG Analysis and Labeling for Chronic Migraine Patients Amid Label Noise

## 0) Environment Setup (Conda, GPU recommended)

`# Create & activate environment (Python 3.10 or 3.11) conda create -n eeglab-ct python=3.10 -y conda activate eeglab-ct  # Install PyTorch (adjust CUDA version to your system; example uses CUDA 12.1) conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y  # Other dependencies pip install -r requirements.txt # If you will never call limit_gpu_memory(), you don't need TensorFlow. # Otherwise, only then install:  pip install "tensorflow==2.15.*"`

Verify GPU:

`import torch print(torch.cuda.is_available(), torch.cuda.get_device_name(0))`

---

## 1) Data Preparation (.fif) and Directory Layout

The main script reads `.fif` via `mne.io.Raw(file_path)` and **optionally filters files by a filename prefix** (default `keyword='EC'`). A recommended layout:

`D:\EEG\mne_fif_root\  ├─ CTL\        # class 0  │   ├─ EC_001.fif  │   └─ EC_002.fif  └─ NOT_CTL\    # class 1      ├─ EC_101.fif      └─ EC_102.fif`

If your current data are EEGLAB `.set`, convert them first:

`python convert_eeglab_to_fif.py`

Open the script and update `src_dir` / `dst_dir`. If channel reordering is required, uncomment the relevant section and edit the target order accordingly.

> **Assumption (30 channels).**  
> The model assumes 30 channels in this fixed order:
> 
> `['FP1','FP2',  'F7','F3','FZ','F4','F8',  'FT7','FC3','FCZ','FC4','FT8',  'T3','C3','CZ','C4','T4',  'TP7','CP3','CPZ','CP4','TP8',  'T5','P3','PZ','P4','T6',  'O1','OZ','O2']`
> 
> If your data differ, please reorder during conversion/preprocessing.

---

## 2) Prepare SimSiam Features (.pkl)

This repository **trains Co-teaching on pre-extracted features**. For each `.fif` file, place a **same-name `.pkl`** under `features_path`. Each `.pkl` should store a `numpy.ndarray` of shape `[N, feature_dim]` (default `feature_dim=512`), e.g.:

`D:\EEG\features\simsiam\  ├─ EC_001.fif.pkl     # contains shape=(N1, 512)  ├─ EC_002.fif.pkl     # contains shape=(N2, 512)  ├─ EC_101.fif.pkl  └─ EC_102.fif.pkl`

If your features are not 512-dimensional, set `eeg_config['feature_dim']` accordingly.

---

## 3) Create `cfg6.py`

Download the provided `cfg6_template.py`, update the paths, and save it as `cfg6.py` at the project root.  
The main script (`from cfg6 import *`) expects:

- `base_path`, `class_paths`, `class_names`, `class_labels`
    
- `features_path` (directory containing the `.pkl` features)
    
- `epochs_path` (where intermediate CSVs will be written)
    
- `keyword` (filename prefix filter; set `''` to disable)
    
- `rate_col = 'Rate'` (used when plotting to avoid `NameError`)
    

> **Important.** The main script **also** sets  
> `features_path = '../../SimSiam(Imb2)/Features/...'` near the bottom.  
> **Remove or change that line**, otherwise it overrides `cfg6.py` and the code may fail to locate your `.pkl` files.

---

## 4) Python Import Paths

The project imports several local packages (e.g., `EEGLab.*`, `Co_teachingLib.*`, `utils.*`). Ensure that:

- These folders exist in your project, and **`PYTHONPATH`** includes the project root; or
    
- Your IDE marks the repository root as a **Source Root**.
    

If you see `ModuleNotFoundError: No module named 'EEGLab'`, the import path is not set correctly.

---

## 5) End-to-End Workflow (What the main script does)

1. **Instantiate and configure logging**
    
    - `eeg = EEGLabLOSOFeaturePytorch()`
        
    - `eeg.register_logging()` → logs key hyperparameters from `eeg_config` into `*.log`.
        
2. **Scan data and write `<code>_<classes>_<keyword>_info.csv`**
    
    - `eeg.collect_dataset_info_with_black_list()`  
        Recursively scans `base_path / class_paths[c]` for `.fif`, optionally filters by `keyword` prefix, records `n_times`, etc., and flags items in the black list.
        
3. **Enumerate epoch groups**
    
    - `eeg.make_epochs_sn_df()`  
        Using `duration=1.5`, `stride=1.0` (assuming 250 Hz), writes which epochs to extract per file into `*_epochs.csv`.
        
4. **Load features into memory (single large array + labels)**
    
    - `eeg.load_feature_data_to_array()`  
        Reads every `.pkl` from `features_path`, concatenates into `dataset_array` and `dataset_labels` (torch tensors).  
        If `eeg_config['data_transform']=True`, applies `SelfAttention(T)`; if `data_normalize=True`, applies L2-normalization.
        
5. **Choose the training routine**
    
    - `eeg.fit_method = eeg.train_pytorch_adam` (or another provided trainer).
        
6. **Run multiple groups (e.g., `group = 0..9`): train → predict → save**
    
    - For each `g`:
        
        - `eeg.group = g`
            
        - **Modeling & cross-validation**:  
            `eeg.evaluate_lpso_with_k_fold_feature(n_splits_fold=5, random_state=..., n_splits_test_set=5, random_state_test=..., P=1)`  
            Trains with K-fold on the subset, predicts on the held-out folds (and later the blacklist).
            
        - **Aggregate to file-level “rate”** and plot:  
            `eeg.evaluate_rate_of_fif(g)`
            
        - **Versioned CSV snapshot**:  
            `eeg.save_epochs_info_as(g)`
            
        - **(Optional) Additional summaries**:  
            `eeg.evaluate_rate_of_fif_n(g)`  
            `eeg.evaluate_total_rate_of_fif(range(g+1))`  
            `eeg.evaluate_total_rate_of_fif_bl(range(g+1))`
            
7. **Voting & summary (if using the voting pipeline)**
    
    - A typical pipeline may also compute a final vote across groups, generate a Black List (BL), and write consolidated metrics (`*_evaluate.csv`) and figures to `self.log.fig_path`.
        

> **Training specifics (typical defaults):**
> 
> - Backbone: `EEGNet_PT452_FC1`
>     
> - `epochs = 200`; models from the final 10 epochs (190–199) are saved and used for ensemble prediction.
>     
> - Default `batch_size = 1024` for feature-level training (reduce if OOM).
>     

---

## 6) Key Configuration (`eeg_config`)

`eeg_config = {   'duration': 1.5, 'stride': 1.0, 'CF': 1,               # epoch slicing   'feature_dim': 512,                                     # must match your .pkl embedding size   'data_normalize': False, 'data_transform': True, 'T': 1.0,   'epochs': 200, 'batch_size': 1024, 'lr_init': 1e-3,     # Adam defaults   'epoch_decay_start': 80, 'weight_decay': 1e-4,   'net': EEGNet_PT452_FC1, 'l2_weight': 1.0e-4,   'dataset': DatasetCoTeachingArray,   'drop_bad_model': True, 'val_acc_th': 0.85, 'val_loss_th': 2.0 }`

**Notes**

- **`feature_dim` must equal the second dimension of your `.pkl` arrays.** Otherwise, the FC layer dimensions will mismatch.
    
- Reduce `batch_size` if you encounter CUDA OOM.
    
- Disable `SelfAttention` by setting `data_transform=False`.
    

---

## 7) Execution

From the project root (where `import EEGLab.*` etc. works), run:

`python your_main_script.py`

You should observe:

- Device printout (`cuda:0` or `cpu`)
    
- Per-epoch training/validation summaries  
    Artifacts include:
    
- `*.log` (training trace)
    
- `*_info.csv`, `*_epochs.csv`, `*_epochs_0..N.csv`
    
- `*_evaluate.csv` (aggregated metrics)
    
- Model weights (e.g., `..._190.pt`–`..._199.pt`)
    
- Figures (e.g., `Testing.png`, `Black_List.png`) under `self.log.fig_path`
    

---

## Troubleshooting

1. **`FileNotFoundError: ... .pkl`**
    
    - Ensure `features_path` matches `.fif` basenames (`EC_001.fif` ↔ `EC_001.fif.pkl`).
        
    - Check the main script for a stray `features_path = '.../SimSiam(Imb2)/Features/...'` at the bottom—**remove or update** it to avoid overriding `cfg6.py`.
        
2. **`ValueError: shapes ...` (dimension mismatch)**
    
    - Set `eeg_config['feature_dim']` to the actual vector length stored in `.pkl`.
        
3. **`len(raw.info['ch_names']) < 30` and files are skipped**
    
    - Your data have fewer channels or a different ordering. Reorder/complete to the fixed 30-channel template and ensure consistent naming.
        
4. **`ModuleNotFoundError: EEGLab / Co_teachingLib / utils ...`**
    
    - Add the repository root to `PYTHONPATH`, or mark the root as **Source Root** in your IDE.
        
5. **`CUDA out of memory`**
    
    - Lower `batch_size` (e.g., 1024 → 512 → 256 → 128).
        
    - Avoid loading TensorFlow unless necessary (it may reserve GPU memory). If installed, do not import or initialize it while training with PyTorch.
        
6. **`NameError: rate_col`**
    
    - Define `rate_col = 'Rate'` in `cfg6.py` (the template already includes this).
        
7. **GPU not detected**
    
    - Your PyTorch build or drivers may not match the system CUDA version. Reinstall PyTorch using the appropriate CUDA toolkit as shown in Step 0.
        

---

## Minimal Sanity Test (recommended)

- Place **1–2 `.fif` files per class** under `CTL` and `NOT_CTL`.
    
- Create minimal `.pkl` features for those files (even ~20 vectors each is fine), set `feature_dim=512` (or match your actual size).
    
- Temporarily reduce `epochs` to **5** and `batch_size` to **32** to validate the pipeline end-to-end (generation of `*_epochs_0..N.csv` and figures).
    
- Once confirmed, restore full training parameters.