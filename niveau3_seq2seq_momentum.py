"""
NIVEAU 3 - Seq2Seq + Momentum (version corrigée)
=================================================
Corrections vs original :
  [FIX 1] Split sur match_id avant build_sequences
  [FIX 2] Scaler fitté sur train uniquement
  [FIX 3] Sauvegarde dans Drive
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
BASE_DIR        = "/content/tennis_project"
CSV_PATH        = f"{BASE_DIR}/USD.txt"

SEQ_LEN         = 10
PRED_LEN        = 5
MOM_WINDOW      = 8
MOM_DECAY       = 0.85
BATCH_SIZE      = 64
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
EPOCHS          = 20
LR              = 1e-3
DROPOUT         = 0.3
LAMBDA_MOMENTUM = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "p1_score","p2_score","p1_games_won","p2_games_won",
    "p1_sets","p2_sets","p1_serve","p2_serve",
    "p1_ace","p2_ace","p1_winner","p2_winner",
    "p1_double_fault","p2_double_fault","p1_unf_err","p2_unf_err",
    "p1_distance_run","p2_distance_run",
    "p1_points_diff","p2_points_diff","p1_game_diff","p2_game_diff",
    "p1_set_diff","p1_serve_speed","p2_serve_speed",
]
TARGET_COL = "Y"
SCORE_MAP  = {'0':0,'15':1,'30':2,'40':3,'AD':4, 0:0,15:1,30:2,40:3}


# ──────────────────────────────────────────────
# MOMENTUM
# ──────────────────────────────────────────────
def compute_momentum(results, window, decay):
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay**k for k in range(window)])
    weights /= weights.sum()
    mom = np.zeros(len(signed))
    for t in range(len(signed)):
        s = max(0, t-window+1)
        w = weights[-(t-s+1):]
        mom[t] = np.dot(signed[s:t+1], w)
    return mom.astype(np.float32)


# ──────────────────────────────────────────────
# DONNÉES
# ──────────────────────────────────────────────
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    mom_list = []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        mom_list.extend(compute_momentum(match[TARGET_COL].values,
                                         MOM_WINDOW, MOM_DECAY).tolist())
    df["momentum"] = mom_list
    return df.reset_index(drop=True)


def build_sequences(df):
    Xf, Xm, yp, ym = [], [], [], []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        feat  = match[FEATURE_COLS].values
        mom   = match["momentum"].values
        tgt   = match[TARGET_COL].values
        for i in range(len(match)-SEQ_LEN-PRED_LEN+1):
            Xf.append(feat[i:i+SEQ_LEN])
            Xm.append(mom[i:i+SEQ_LEN])
            yp.append(tgt[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
            ym.append(mom[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
    return (np.array(Xf, dtype=np.float32),
            np.array(Xm, dtype=np.float32),
            np.array(yp, dtype=np.int64),
            np.array(ym, dtype=np.float32))


class TennisDataset(Dataset):
    def __init__(self, Xf, Xm, yp, ym):
        self.Xf = torch.tensor(Xf)
        self.Xm = torch.tensor(Xm).unsqueeze(-1)
        self.yp = torch.tensor(yp)
        self.ym = torch.tensor(ym)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i): return self.Xf[i], self.Xm[i], self.yp[i], self.ym[i]


# ──────────────────────────────────────────────
# MODÈLE
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, feat, hidden, layers, drop):
        super().__init__()
        self.lstm = nn.LSTM(feat+1, hidden, layers, batch_first=True,
                             dropout=drop if layers>1 else 0)
    def forward(self, xf, xm):
        _, (h,c) = self.lstm(torch.cat([xf,xm],dim=-1)); return h, c

class PointDecoder(nn.Module):
    def __init__(self, hidden, layers, drop, nc=2):
        super().__init__()
        self.lstm = nn.LSTM(nc, hidden, layers, batch_first=True,
                             dropout=drop if layers>1 else 0)
        self.fc = nn.Linear(hidden, nc); self.nc = nc
    def forward(self, h, c, pl, targets=None, tf=0.5):
        B   = h.size(1)
        x_t = torch.zeros(B,1,self.nc,device=h.device); out=[]
        for t in range(pl):
            o,(h,c) = self.lstm(x_t,(h,c))
            lg = self.fc(o.squeeze(1)); out.append(lg.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf:
                x_t = F.one_hot(targets[:,t],self.nc).float().unsqueeze(1)
            else:
                x_t = F.one_hot(lg.argmax(1),self.nc).float().unsqueeze(1)
        return torch.cat(out,dim=1)

class MomentumDecoder(nn.Module):
    def __init__(self, hidden, layers, drop):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                             dropout=drop if layers>1 else 0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, h, c, pl, targets=None, tf=0.5):
        B   = h.size(1)
        x_t = torch.zeros(B,1,1,device=h.device); out=[]
        for t in range(pl):
            o,(h,c) = self.lstm(x_t,(h,c))
            pm = self.fc(o.squeeze(1)); out.append(pm.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf:
                x_t = targets[:,t].unsqueeze(1).unsqueeze(1)
            else:
                x_t = pm.unsqueeze(1)
        return torch.cat(out,dim=1).squeeze(-1)

class Seq2SeqWithMomentum(nn.Module):
    def __init__(self, feat, hidden, layers, drop):
        super().__init__()
        self.enc  = Encoder(feat, hidden, layers, drop)
        self.pdec = PointDecoder(hidden, layers, drop)
        self.mdec = MomentumDecoder(hidden, layers, drop)
    def forward(self, xf, xm, pl, yp=None, ym=None, tf=0.5):
        h,c = self.enc(xf,xm)
        return self.pdec(h.clone(),c.clone(),pl,yp,tf), \
               self.mdec(h.clone(),c.clone(),pl,ym,tf)


# ──────────────────────────────────────────────
# ENTRAÎNEMENT
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, ce, mse):
    model.train(); total=0
    for xf,xm,yp,ym in loader:
        xf,xm,yp,ym = xf.to(DEVICE),xm.to(DEVICE),yp.to(DEVICE),ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mp = model(xf,xm,PRED_LEN,yp,ym,tf=0.5)
        loss = ce(logits.view(-1,2),yp.view(-1)) + LAMBDA_MOMENTUM*mse(mp,ym)
        loss.backward(); optimizer.step(); total+=loss.item()
    return total/len(loader)

def evaluate(model, loader, ce, mse):
    model.eval(); total,preds,labels,mp_all,mt_all = 0,[],[],[],[]
    with torch.no_grad():
        for xf,xm,yp,ym in loader:
            xf,xm,yp,ym = xf.to(DEVICE),xm.to(DEVICE),yp.to(DEVICE),ym.to(DEVICE)
            logits,mp = model(xf,xm,PRED_LEN,tf=0.0)
            total += (ce(logits.view(-1,2),yp.view(-1))+LAMBDA_MOMENTUM*mse(mp,ym)).item()
            preds.append(logits.argmax(-1).cpu().numpy())    # (B, PRED_LEN)
            labels.append(yp.cpu().numpy())
            mp_all.extend(mp.cpu().numpy().flatten())
            mt_all.extend(ym.cpu().numpy().flatten())
    preds  = np.concatenate(preds)    # (N, PRED_LEN)
    labels = np.concatenate(labels)
    acc_t1  = accuracy_score(labels[:,0], preds[:,0])   # t+1 uniquement
    acc_all = accuracy_score(labels.flatten(), preds.flatten())
    return (total/len(loader), acc_t1, acc_all,
            mean_squared_error(mt_all, mp_all))


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
print("📂 Chargement + calcul momentum...")
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
print(f"   Train : {len(df_train)} points  Val : {len(df_val)} points")

# [FIX 2] Scaler sur train uniquement
scaler = StandardScaler()
df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS].fillna(0))
df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS].fillna(0))

mom_scaler = StandardScaler()
df_train[["momentum"]] = mom_scaler.fit_transform(df_train[["momentum"]])
df_val[["momentum"]]   = mom_scaler.transform(df_val[["momentum"]])

print("🔨 Construction des séquences...")
Xf_tr,Xm_tr,yp_tr,ym_tr = build_sequences(df_train)
Xf_vl,Xm_vl,yp_vl,ym_vl = build_sequences(df_val)
print(f"   Train : {Xf_tr.shape}  Val : {Xf_vl.shape}")

train_loader = DataLoader(TennisDataset(Xf_tr,Xm_tr,yp_tr,ym_tr), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TennisDataset(Xf_vl,Xm_vl,yp_vl,ym_vl), BATCH_SIZE)

model     = Seq2SeqWithMomentum(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ce_loss   = nn.CrossEntropyLoss()
mse_loss  = nn.MSELoss()

print(f"\n🚀 Entraînement sur {DEVICE}  |  "
      f"{sum(p.numel() for p in model.parameters()):,} params\n")

best_acc = 0
for epoch in range(1, EPOCHS+1):
    train_loss                    = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss)
    val_loss,val_acc,val_acc_all,m_mse = evaluate(model, val_loader, ce_loss, mse_loss)
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | "
          f"acc(t+1)={val_acc:.4f} | acc(all)={val_acc_all:.4f} | mom_MSE={m_mse:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{BASE_DIR}/best_seq2seq_momentum.pt")

print(f"\n✅ Meilleure acc (t+1) : {best_acc:.4f}")
print(f"   Checkpoint : {BASE_DIR}/best_seq2seq_momentum.pt")