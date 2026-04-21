"""
NIVEAU 4 - HydraNet : Architecture Multi-têtes (état de l'art)
==============================================================
Inspiré de l'architecture HydraNet (2025), on remplace l'encoder LSTM
simple par un encoder multi-scale avec attention, et on ajoute :
  - une variable latente de momentum (espace continu)
  - 2 têtes de décodage (points + momentum)
  - mécanisme d'attention cross-séquence

Architecture :
  ┌─────────────────────────────────────────┐
  │  Multi-Scale Encoder (LSTM + Attention) │
  │  → ConvBlock (court terme)              │
  │  → LSTMBlock (moyen terme)              │
  │  → Self-Attention (long terme)          │
  └──────────────────┬──────────────────────┘
                     │ context vector
            ┌────────┴────────┐
            ▼                 ▼
   ┌─────────────────┐  ┌──────────────────┐
   │ Head 1          │  │ Head 2           │
   │ Seq2Seq Points  │  │ Seq2Seq Momentum │
   │ (classification)│  │ (régression)     │
   └─────────────────┘  └──────────────────┘

Référence : https://github.com/ReyJerry/HydraNet
"""

import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN         = 10
PRED_LEN        = 5
MOM_WINDOW      = 8
MOM_DECAY       = 0.85
BATCH_SIZE      = 64
D_MODEL         = 128      # dimension interne du modèle
N_HEADS         = 4        # têtes d'attention
NUM_LSTM_LAYERS = 2
CONV_KERNEL     = 3        # noyau convolutif court-terme
EPOCHS          = 30
LR              = 5e-4
DROPOUT         = 0.2
LAMBDA_MOM      = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

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

# Mapping des scores tennis → numérique
SCORE_MAP = {
    '0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
    0: 0, 15: 1, 30: 2, 40: 3  # au cas où certaines valeurs sont déjà int
}

# ─────────────────────────────────────────────
# 2. DONNÉES (identique niveau 3)
# ─────────────────────────────────────────────
def compute_momentum(results, window, decay):
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()
    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start = max(0, t - window + 1)
        w     = weights[-(t - start + 1):]
        momentum[t] = np.dot(signed[start : t + 1], w)
    return momentum.astype(np.float32)


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, sep=",")

    # Encoder les scores tennis
    df['p1_score'] = df['p1_score'].map(SCORE_MAP)
    df['p2_score'] = df['p2_score'].map(SCORE_MAP)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    mom_list = []
    for _, group in df.groupby("match_id"):
        group = group.sort_values("point_no")
        mom_list.extend(compute_momentum(group[TARGET_COL].values, MOM_WINDOW, MOM_DECAY).tolist())
    df["momentum"] = mom_list
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df, scaler


def build_sequences(df):
    X_feat, X_mom, y_points, y_mom = [], [], [], []
    for _, group in df.groupby("match_id"):
        group    = group.sort_values("point_no").reset_index(drop=True)
        features = group[FEATURE_COLS].values
        points   = group[TARGET_COL].values
        moms     = group["momentum"].values
        for i in range(len(group) - SEQ_LEN - PRED_LEN + 1):
            X_feat.append(features[i : i + SEQ_LEN])
            X_mom.append(moms[i : i + SEQ_LEN])
            y_points.append(points[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
            y_mom.append(moms[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
    return (np.array(X_feat,   dtype=np.float32),
            np.array(X_mom,    dtype=np.float32),
            np.array(y_points, dtype=np.int64),
            np.array(y_mom,    dtype=np.float32))


class TennisDataset(Dataset):
    def __init__(self, Xf, Xm, yp, ym):
        self.Xf = torch.tensor(Xf)
        self.Xm = torch.tensor(Xm).unsqueeze(-1)
        self.yp = torch.tensor(yp)
        self.ym = torch.tensor(ym)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i): return self.Xf[i], self.Xm[i], self.yp[i], self.ym[i]


# ─────────────────────────────────────────────
# 3. BLOCS DE L'ENCODER MULTI-SCALE
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Capture les patterns courts (quelques points consécutifs)."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv  = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm  = nn.LayerNorm(out_channels)
        self.act   = nn.GELU()

    def forward(self, x):
        # x : (B, T, C)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, T, out_channels)
        return self.act(self.norm(out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HydraEncoder(nn.Module):
    """
    Encoder multi-scale :
      1. Projection vers d_model
      2. ConvBlock (court terme)
      3. LSTM  (moyen terme)
      4. Self-Attention (long terme / dépendances globales)
    Fusion des 3 représentations → vecteur de contexte
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        # Projection d'entrée
        self.proj = nn.Linear(feat_size + 1, d_model)   # +1 pour momentum

        # Court terme
        self.conv_block = ConvBlock(d_model, d_model, conv_k)

        # Moyen terme
        self.lstm = nn.LSTM(d_model, d_model, num_lstm_layers,
                            batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0)

        # Long terme
        self.pos_enc   = PositionalEncoding(d_model)
        encoder_layer  = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                                    dropout, batch_first=True,
                                                    norm_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion des 3 branches
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_feat, x_mom):
        # Concaténation features + momentum
        x = torch.cat([x_feat, x_mom], dim=-1)   # (B, T, F+1)
        x = self.proj(x)                          # (B, T, d_model)

        # Branche 1 : ConvBlock
        conv_out = self.conv_block(x)             # (B, T, d_model)

        # Branche 2 : LSTM
        lstm_out, (h, c) = self.lstm(x)           # (B, T, d_model)

        # Branche 3 : Attention
        attn_out = self.attention(self.pos_enc(x))  # (B, T, d_model)

        # Fusion des représentations au dernier pas de temps
        last_conv = conv_out[:, -1]     # (B, d_model)
        last_lstm = lstm_out[:, -1]
        last_attn = attn_out[:, -1]

        fused = torch.cat([last_conv, last_lstm, last_attn], dim=-1)  # (B, 3*d_model)
        context = self.dropout(self.fusion(fused))                     # (B, d_model)

        # On retourne aussi h, c du LSTM pour initialiser les décodeurs
        return context, h, c


# ─────────────────────────────────────────────
# 4. DÉCODEURS (similaires au niveau 3, améliorés avec attention)
# ─────────────────────────────────────────────
class HydraPointHead(nn.Module):
    """Tête de prédiction des points avec attention sur le contexte."""
    def __init__(self, d_model, num_layers, dropout, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(num_classes + d_model, d_model, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc   = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

    def forward(self, context, h, c, pred_len, targets=None, tf_ratio=0.5):
        B = h.size(1)
        x_t   = torch.zeros(B, 1, self.num_classes, device=h.device)
        ctx_t = context.unsqueeze(1)   # (B, 1, d_model)
        outputs = []
        for t in range(pred_len):
            inp = torch.cat([x_t, ctx_t], dim=-1)   # (B, 1, num_classes + d_model)
            out, (h, c) = self.lstm(inp, (h, c))
            logit = self.fc(out.squeeze(1))          # (B, num_classes)
            outputs.append(logit.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf_ratio:
                x_t = F.one_hot(targets[:, t], self.num_classes).float().unsqueeze(1)
            else:
                x_t = F.one_hot(logit.argmax(1), self.num_classes).float().unsqueeze(1)
        return torch.cat(outputs, dim=1)   # (B, pred_len, 2)


class HydraMomentumHead(nn.Module):
    """Tête de prédiction du momentum avec contexte."""
    def __init__(self, d_model, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(1 + d_model, d_model, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, context, h, c, pred_len, targets=None, tf_ratio=0.5):
        B = h.size(1)
        x_t   = torch.zeros(B, 1, 1, device=h.device)
        ctx_t = context.unsqueeze(1)
        outputs = []
        for t in range(pred_len):
            inp = torch.cat([x_t, ctx_t], dim=-1)   # (B, 1, 1 + d_model)
            out, (h, c) = self.lstm(inp, (h, c))
            pred_m = self.fc(out.squeeze(1))         # (B, 1)
            outputs.append(pred_m.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf_ratio:
                x_t = targets[:, t].unsqueeze(1).unsqueeze(1)
            else:
                x_t = pred_m.unsqueeze(1)
        return torch.cat(outputs, dim=1).squeeze(-1)   # (B, pred_len)


# ─────────────────────────────────────────────
# 5. MODÈLE COMPLET HYDRANET
# ─────────────────────────────────────────────
class HydraNet(nn.Module):
    """
    Architecture complète :
      HydraEncoder → context vector
        ├─ HydraPointHead    (prédiction points)
        └─ HydraMomentumHead (prédiction momentum)
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        self.encoder       = HydraEncoder(feat_size, d_model, n_heads,
                                          num_lstm_layers, conv_k, dropout)
        self.point_head    = HydraPointHead(d_model, num_lstm_layers, dropout)
        self.momentum_head = HydraMomentumHead(d_model, num_lstm_layers, dropout)

    def forward(self, x_feat, x_mom, pred_len,
                y_points=None, y_mom=None, tf_ratio=0.5):
        context, h, c = self.encoder(x_feat, x_mom)

        point_logits = self.point_head(context, h.clone(), c.clone(),
                                       pred_len, y_points, tf_ratio)
        mom_preds    = self.momentum_head(context, h.clone(), c.clone(),
                                          pred_len, y_mom, tf_ratio)
        return point_logits, mom_preds


# ─────────────────────────────────────────────
# 6. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
def train(model, loader, optimizer, ce_loss, mse_loss, scheduler=None):
    model.train()
    total_loss = 0
    for Xf, Xm, yp, ym in loader:
        Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mom_preds = model(Xf, Xm, PRED_LEN, yp, ym, tf_ratio=0.5)
        loss = ce_loss(logits.view(-1, 2), yp.view(-1)) + LAMBDA_MOM * mse_loss(mom_preds, ym)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    if scheduler: scheduler.step()
    return total_loss / len(loader)


def evaluate(model, loader, ce_loss, mse_loss):
    model.eval()
    total_loss, preds_all, labels_all, mom_pred_all, mom_true_all = 0, [], [], [], []
    with torch.no_grad():
        for Xf, Xm, yp, ym in loader:
            Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
            logits, mom_preds = model(Xf, Xm, PRED_LEN, tf_ratio=0.0)
            total_loss += (ce_loss(logits.view(-1, 2), yp.view(-1)) +
                           LAMBDA_MOM * mse_loss(mom_preds, ym)).item()
            preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
            labels_all.extend(yp.cpu().numpy().flatten())
            mom_pred_all.extend(mom_preds.cpu().numpy().flatten())
            mom_true_all.extend(ym.cpu().numpy().flatten())

    acc     = accuracy_score(labels_all, preds_all)
    mom_mse = mean_squared_error(mom_true_all, mom_pred_all)
    return total_loss / len(loader), acc, mom_mse


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + calcul momentum...")
    df, scaler = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    print("🔨 Construction des séquences...")
    X_feat, X_mom, y_points, y_mom = build_sequences(df)
    print(f"   X_feat: {X_feat.shape}  |  y_points: {y_points.shape}")

    split = int(0.8 * len(X_feat))
    train_ds = TennisDataset(X_feat[:split], X_mom[:split], y_points[:split], y_mom[:split])
    val_ds   = TennisDataset(X_feat[split:], X_mom[split:], y_points[split:], y_mom[split:])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

    model = HydraNet(
        feat_size=len(FEATURE_COLS),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_lstm_layers=NUM_LSTM_LAYERS,
        conv_k=CONV_KERNEL,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 HydraNet sur {DEVICE}  |  {n_params:,} paramètres\n")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, ce_loss, mse_loss, scheduler)
        val_loss, val_acc, mom_mse = evaluate(model, val_loader, ce_loss, mse_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
              f"mom_MSE={mom_mse:.5f} | lr={lr_now:.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "config": {
                            "feat_size": len(FEATURE_COLS),
                            "d_model": D_MODEL,
                            "n_heads": N_HEADS,
                            "num_lstm_layers": NUM_LSTM_LAYERS,
                            "conv_k": CONV_KERNEL,
                            "dropout": DROPOUT,
                        }},
                       "best_hydranet.pt")
            # Scaler sauvegardé séparément (évite les conflits PyTorch/sklearn)
            joblib.dump(scaler, "scaler.pkl")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("   Checkpoint sauvegardé dans best_hydranet.pt")

    # ─── Exemple d'inférence ──────────────────────────────────────────────
    print("\n🔍 Exemple d'inférence sur les 3 premiers échantillons du val set...")
    model.load_state_dict(torch.load("best_hydranet.pt", weights_only=False)["model"])
    model.eval()
    sample_Xf = torch.tensor(X_feat[split : split + 3]).to(DEVICE)
    sample_Xm = torch.tensor(X_mom[split : split + 3]).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        logits, mom_preds = model(sample_Xf, sample_Xm, PRED_LEN, tf_ratio=0.0)
    points_pred = logits.argmax(-1).cpu().numpy()
    mom_pred    = mom_preds.cpu().numpy()
    print(f"   Points prédits   : {points_pred}")
    print(f"   Points réels     : {y_points[split : split + 3]}")
    print(f"   Momentum prédit  : {np.round(mom_pred, 3)}")
    print(f"   Momentum réel    : {np.round(y_mom[split : split + 3], 3)}")
