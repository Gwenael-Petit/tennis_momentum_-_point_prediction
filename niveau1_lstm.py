"""
NIVEAU 1 - LSTM Baseline (version corrigée)
============================================
Corrections vs original :
  [FIX 1] Split sur match_id avant build_sequences (évite le leakage)
  [FIX 2] Scaler fitté sur train uniquement
  [FIX 3] Sauvegarde dans Drive
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
BASE_DIR    = "/content/drive/MyDrive/tennis_project"  # ← modifier si besoin
CSV_PATH    = f"{BASE_DIR}/USD.txt"

SEQ_LEN     = 10
BATCH_SIZE  = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
EPOCHS      = 20
LR          = 1e-3
DROPOUT     = 0.3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "p1_score", "p2_score",
    "p1_games_won", "p2_games_won",
    "p1_sets", "p2_sets",
    "p1_serve", "p2_serve",
    "p1_ace", "p2_ace",
    "p1_winner", "p2_winner",
    "p1_double_fault", "p2_double_fault",
    "p1_unf_err", "p2_unf_err",
    "p1_distance_run", "p2_distance_run",
    "p1_points_diff", "p2_points_diff",
    "p1_game_diff", "p2_game_diff",
    "p1_set_diff",
    "p1_serve_speed", "p2_serve_speed",
]
TARGET_COL = "Y"
SCORE_MAP  = {'0':0,'15':1,'30':2,'40':3,'AD':4, 0:0,15:1,30:2,40:3}


# ──────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    return df.reset_index(drop=True)


def build_sequences(df):
    X, y = [], []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        feat  = match[FEATURE_COLS].values
        tgt   = match[TARGET_COL].values
        for i in range(SEQ_LEN, len(match)):
            X.append(feat[i-SEQ_LEN:i])
            y.append(tgt[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class TennisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ──────────────────────────────────────────────
# MODÈLE
# ──────────────────────────────────────────────
class LSTMPointPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))


# ──────────────────────────────────────────────
# ENTRAÎNEMENT
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total, preds, labels = 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total += criterion(logits, yb).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return total/len(loader), accuracy_score(labels, preds)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
print("📂 Chargement des données...")
df = load_and_prepare(CSV_PATH)
print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

# [FIX 1] Split sur match_id
match_ids = df["match_id"].unique()
np.random.seed(42); np.random.shuffle(match_ids)
split_idx = int(0.8 * len(match_ids))
train_ids = set(match_ids[:split_idx])
val_ids   = set(match_ids[split_idx:])

df_train = df[df["match_id"].isin(train_ids)].copy().reset_index(drop=True)
df_val   = df[df["match_id"].isin(val_ids)].copy().reset_index(drop=True)
print(f"   Train : {len(df_train)} points ({len(train_ids)} matchs)")
print(f"   Val   : {len(df_val)} points ({len(val_ids)} matchs)")

# [FIX 2] Scaler fitté sur train uniquement
scaler = StandardScaler()
df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS].fillna(0))
df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS].fillna(0))

print("🔨 Construction des séquences...")
X_tr, y_tr = build_sequences(df_train)
X_vl, y_vl = build_sequences(df_val)
print(f"   Train : {X_tr.shape}  Val : {X_vl.shape}")

train_loader = DataLoader(TennisDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TennisDataset(X_vl, y_vl), BATCH_SIZE)

model     = LSTMPointPredictor(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"\n🚀 Entraînement sur {DEVICE}  |  "
      f"{sum(p.numel() for p in model.parameters()):,} params\n")

best_acc = 0
for epoch in range(1, EPOCHS+1):
    train_loss         = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc  = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        # [FIX 3] Sauvegarde dans Drive
        torch.save(model.state_dict(), f"{BASE_DIR}/best_lstm.pt")

print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
print(f"   Checkpoint : {BASE_DIR}/best_lstm.pt")