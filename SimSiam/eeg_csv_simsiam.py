
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG SimSiam pipeline (CSV-based)
--------------------------------
- Each CSV file is **one trial** with shape **30 x 1024** (or 1024 x 30; will be transposed).
- Folder layout:
    base_path/
      CTL/
        L_1_4_1.csv
        L_1_4_2.csv
        ...
      CM/
        L_1_8_1.csv
        ...
- The script scans CSVs, builds info CSVs, trains a SimSiam encoder, and extracts per-file features.
- No MNE dependency.

Tested on CPU with a tiny demo dataset.
"""

import os, re, math, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config (edit to your paths)
# -----------------------------

#base_path = "./demo_data"   # <- change this to your dataset root
base_path = 'C:/Users/user/pythonproject/REAL/dataset/MI_data'
class_names = ['CTL', 'CM']
class_paths = {'CTL': 'CTL', 'CM': 'CM'}
class_labels = {'CTL': 0, 'CM': 1}

eeg_config = {
    'duration': 1.5,     # seconds
    'stride': 1.0,       # seconds
    'CF': 1,             # channel factor (kept for compatibility)
    'kern_length': 125,  # temporal kernel length
    'data_normalize': False,
    'lr_init': 0.2,      # initial LR
    'weight_decay': 1.0e-4,
    'epochs': 200,         # set larger for real training
    'batch_size': 16,
    'accum_iter': 2,
    'dim': 128,          # encoder feature dim
    'pred_dim': 64,      # predictor hidden dim
    'over_sampling': False,
    'Note': 'CSV SimSiam for EEG'
}

# Optional subset filtering; set to ['L_'] to match your naming prefix, or [] to disable
keyword = []  # e.g., ['L_']


# -----------------------------
# Utilities
# -----------------------------

def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def first_int_or(index_like, default_val):
    try:
        return int(index_like[0])
    except Exception:
        return int(default_val)

def trial_stem(file_name):
    """
    Remove trailing _<index>.csv to find siblings of the same block, e.g. L_1_4_1.csv -> L_1_4_
    """
    m = re.match(r'^(.*?_)\\d+\\.csv$', file_name)
    if m:
        return m.group(1)
    return file_name[: max(2, min(30, len(file_name)))]

CSV_FILE_RE = re.compile(r'^[LR]_\d+_\d+_\d+\.csv$', re.IGNORECASE)

def read_csv_trial(file_path: str) -> np.ndarray:
    """
    Read one trial CSV; return (30,1024) float32. Auto transpose if (1024,30).
    """
    arr = pd.read_csv(file_path, header=None).to_numpy()
    if arr.shape == (30, 1024):
        pass
    elif arr.shape == (1024, 30):
        arr = arr.T
    else:
        raise ValueError(f'CSV shape must be 30x1024, got {arr.shape} at {file_path}')
    return arr.astype('float32')


# -----------------------------
# EEG augmentations for contrastive learning
# -----------------------------

class EEGAugment:
    def __init__(self, samples, jitter_std=0.01, scale_low=0.9, scale_high=1.1, dropout_p=0.05):
        self.samples = samples
        self.jitter_std = jitter_std
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.dropout_p = dropout_p

    def random_crop(self, x):
        # x: (C, T_total), crop to (C, samples)
        T_total = x.shape[1]
        if self.samples >= T_total:
            start = 0
        else:
            start = np.random.randint(0, T_total - self.samples + 1)
        return x[:, start:start+self.samples]

    def jitter(self, x):
        return x + np.random.normal(0.0, self.jitter_std, size=x.shape).astype('float32')

    def scale(self, x):
        s = np.random.uniform(self.scale_low, self.scale_high)
        return x * s

    def dropout(self, x):
        mask = (np.random.rand(*x.shape) > self.dropout_p).astype('float32')
        return x * mask

    def __call__(self, x):
        # x: (C, T_total)
        y = self.random_crop(x)
        y = self.jitter(y)
        y = self.scale(y)
        y = self.dropout(y)
        return y


# -----------------------------
# Dataset for SimSiam (returns two augmented views)
# -----------------------------

class DatasetSimSiamCouple(torch.utils.data.Dataset):
    def __init__(self, df_info_all, dataset_dict, samples, channels=1, is_test=False):
        self.df = df_info_all.reset_index(drop=True)
        self.data = dataset_dict
        self.samples = samples
        self.channels = channels
        self.is_test = is_test
        self.aug = EEGAugment(samples=samples)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # x_all: (1, C, T_total)
        x_all = self.data[idx].numpy()  # (1,C,T)
        x = x_all[0]                    # (C,T)
        v1 = self.aug(x)
        v2 = self.aug(x)
        # to (1,C,samples)
        v1 = v1[None, :, :]
        v2 = v2[None, :, :]
        return (torch.from_numpy(v1), torch.from_numpy(v2)), 0


# -----------------------------
# Minimal EEGNet-like backbone
# Input: (B, 1, C, T)
# Output: vector (dim)
# -----------------------------

class EEGNetBackbone(nn.Module):
    def __init__(self, nb_classes=128, kern_length=125, fc_dim=128, samples=376, channels=30):
        super().__init__()
        self.channels = channels
        # Temporal conv
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, kern_length), padding=(0, kern_length//2), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
        )
        # Depthwise spatial conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        # Pool + projection
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * samples, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, nb_classes)   # final feature dim
        )

    def forward(self, x):
        # x: (B,1,C,T)
        h = self.temporal_conv(x)
        h = self.spatial_conv(h)          # (B,32,1,T)
        h = self.pool(h)                  # (B,32,1,T')
        h = torch.squeeze(h, dim=2)       # (B,32,T')
        h = self.flatten(h)               # (B, 32*T')
        z = self.fc(h)                    # (B, dim)
        return z


# -----------------------------
# SimSiam wrapper
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimSiam(nn.Module):
    def __init__(self, encoder, dim, pred_dim):
        super().__init__()
        self.encoder = encoder                # returns dim
        self.projector = MLP(dim, dim, dim)   # simple projector
        self.predictor = MLP(dim, pred_dim, dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


# -----------------------------
# Main trainer class
# -----------------------------

class EEGLabSimSiam:
    def __init__(self):
        self.duration = eeg_config['duration']
        self.stride = eeg_config['stride']
        self.samples = int(self.duration * 250.0) + 1   # consistent with your code
        self.eeg_channels = ['FP1','FP2','F7','F3','FZ','F4','F8',
                             'FT7','FC3','FCZ','FC4','FT8',
                             'T3','C3','CZ','C4','T4',
                             'TP7','CP3','CPZ','CP4','TP8',
                             'T5','P3','PZ','P4','T6',
                             'O1','OZ','O2']
        self.eeg_channel_num = len(self.eeg_channels)
        class_all = '-'.join(class_names)
        self.bone_file_name = 'csvsimsiam_' + class_all
        self.info_file   = self.bone_file_name + '_info.csv'
        self.info_all_file = self.bone_file_name + '_info_all.csv'
        self.log_file = self.bone_file_name + '_log.csv'
        self.state_dict_file = self.bone_file_name + '.pt'
        self.features_path = './Features/%s_%s_%s' % ('csvsimsiam',
                                                      str(eeg_config['duration']),
                                                      str(eeg_config['stride']))
        mkdir(self.features_path)
        self.dataset_dict = {}

    # ---------- scanning ----------
    def collect_dataset_info_all(self):
        df = {'file': [], 'eeg_name': [], 'EC-EO': [], 'label': [],
              'n_times': [], 'std': [], 'std2': [], 'std3': []}
        for eeg_name in class_names:
            root_dir = os.path.join(base_path, class_paths[eeg_name])
            for r, _, files in os.walk(root_dir):
                for f in files:
                    if not f.lower().endswith('.csv'): 
                        continue
                    if len(keyword) > 0 and f[:2] not in keyword: 
                        continue
                    if not CSV_FILE_RE.match(f): 
                        continue
                    fp = os.path.join(r, f)
                    data = read_csv_trial(fp)
                    if data.shape[0] < self.eeg_channel_num:
                        continue
                    xn = data
                    df['std'].append(float(np.sqrt(np.mean(np.square(xn)))))
                    cstd = np.std(xn, axis=1); df['std2'].append(float(cstd.mean()))
                    df['std3'].append(float(np.std(xn)))
                    df['n_times'].append(int(xn.shape[1]))
                    df['EC-EO'].append(f[:2])
                    df['file'].append(fp)
                    df['eeg_name'].append(eeg_name)
                    df['label'].append(class_labels[eeg_name])
        pd.DataFrame(df).to_csv(self.info_all_file, index=False)
        # also write a copy to info_file for convenience
        pd.DataFrame(df).to_csv(self.info_file, index=False)

    # ---------- load to memory ----------
    def load_csv_data_to_mem_all(self, normalize=True):
        self.dataset_dict.clear()
        df = pd.read_csv(self.info_all_file)
        for idx in tqdm(range(len(df)), desc='Load CSV to mem'):
            fp = str(df.iloc[idx]['file']).strip()
            raw = read_csv_trial(fp)                      # (30,1024)
            if normalize:
                c_std = np.std(raw, axis=1, keepdims=True)
                xn_std = c_std.mean()
                x_all = (raw / (xn_std + 1e-8))[None, :, :].astype('float32')
            else:
                x_all = (raw[None, :, :] * 1000).astype('float32')
            self.dataset_dict[idx] = torch.from_numpy(x_all)
        return self.dataset_dict

    # ---------- dataloader ----------
    def build_train_loader(self, batch_size=64):
        df_all = pd.read_csv(self.info_all_file)
        dict_data = self.load_csv_data_to_mem_all(normalize=eeg_config['data_normalize'])
        dataset = DatasetSimSiamCouple(df_all, dict_data, samples=self.samples, channels=1, is_test=False)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=True)
        return loader

    # ---------- training ----------
    def train_torch(self, epochs=200, batch_size=256, accum_iter=4, device='cpu'):
        encoder = EEGNetBackbone(nb_classes=eeg_config['dim'],
                                 kern_length=eeg_config['kern_length'],
                                 fc_dim=eeg_config['dim'],
                                 samples=self.samples,
                                 channels=self.eeg_channel_num)
        model = SimSiam(encoder, eeg_config['dim'], eeg_config['pred_dim']).to(device)
        criterion = nn.CosineSimilarity(dim=1).to(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=eeg_config['lr_init'],
                                    momentum=0.9,
                                    weight_decay=eeg_config['weight_decay'])
        train_loader = self.build_train_loader(batch_size=batch_size)

        hist = []
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            running = 0.0
            n = 0
            for bidx, (images, _) in enumerate(train_loader):
                x1 = images[0].to(device, non_blocking=True).float()
                x2 = images[1].to(device, non_blocking=True).float()
                p1, p2, z1, z2 = model(x1, x2)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                (loss/accum_iter).backward()
                if (bidx + 1) % accum_iter == 0 or (bidx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                running += float(loss.detach().cpu()) * x1.size(0)
                n += x1.size(0)
            ep_loss = running / max(1, n)
            hist.append({'epoch': epoch+1, 'loss': ep_loss})
            print(f'Epoch {epoch+1}/{epochs} - loss: {ep_loss:.4f}')
            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(model.state_dict(), self.state_dict_file)

        pd.DataFrame(hist).to_csv(self.log_file, index=False)
        return self.state_dict_file

    # ---------- prediction / feature extraction ----------
    def build_predict_model(self, device='cpu'):
        encoder = EEGNetBackbone(nb_classes=eeg_config['dim'],
                                 kern_length=eeg_config['kern_length'],
                                 fc_dim=eeg_config['dim'],
                                 samples=self.samples,
                                 channels=self.eeg_channel_num)
        model = SimSiam(encoder, eeg_config['dim'], eeg_config['pred_dim'])
        if os.path.exists(self.state_dict_file):
            state_dict = torch.load(self.state_dict_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        features_model = model.encoder
        for p in features_model.parameters():
            p.requires_grad = False
        # remove final linear if you want raw features pre-projection (we already output dim features)
        return features_model.to(device).eval()

    def file_to_epochs_array(self, file_path, coef=1.0):
        # For CSV: return (N,1,C,samples)
        data = read_csv_trial(file_path)   # (30,1024)
        if eeg_config['data_normalize']:
            c_std = np.std(data, axis=1, keepdims=True)
            data = data / (c_std.mean() + 1e-8) * 1.0e-2
        x_all = data[None, :, :]           # (1,30,1024)
        stride = int(self.stride * 250.0 / max(coef, 1e-8))
        T = x_all.shape[-1]
        samples = self.samples
        n_epochs = max(0, (T - samples) // stride + 1)
        if n_epochs == 0:
            start = max(0, (T - samples)//2)
            return x_all[:, None, :, start:start+samples].astype('float32')
        offsets = np.arange(n_epochs) * stride
        ep_list = [x_all[:, :, off: off+samples] for off in offsets]
        return np.array(ep_list, dtype='float32')  # (N,1,30,samples)

    def predict_torch(self, device='cpu'):
        model = self.build_predict_model(device=device)
        df_all = pd.read_csv(self.info_all_file)
        for idx in tqdm(range(len(df_all)), desc='Extract'):
            fp = str(df_all.iloc[idx]['file']).strip()
            X = self.file_to_epochs_array(fp, 1.0)  # (N,1,30,T)
            with torch.no_grad():
                feats = []
                for i in range(0, X.shape[0], 16):
                    xb = torch.from_numpy(X[i:i+16]).to(device)
                    f = model(xb).cpu().numpy()
                    feats.append(f)
                feats = np.concatenate(feats, axis=0)
            base = os.path.basename(fp)
            out = os.path.join(self.features_path, base + '.pkl')
            with open(out, 'wb') as f:
                pickle.dump(feats, f)

# -----------------------------
# Running as script
# -----------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    eeg = EEGLabSimSiam()
    eeg.collect_dataset_info_all()
    model_path = eeg.train_torch(epochs=eeg_config['epochs'],
                                 batch_size=eeg_config['batch_size'],
                                 accum_iter=eeg_config['accum_iter'],
                                 device=device)
    print('Saved model to:', model_path)
    eeg.predict_torch(device=device)
    print('Done. Features at:', eeg.features_path)

if __name__ == '__main__':
    main()
