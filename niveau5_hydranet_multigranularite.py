"""
NIVEAU 5 v5 - HydraNet Multi-Granularité (architecture originale stabilisée)
=============================================================================

Diagnostic final après v1→v4 :
  - L'architecture ORIGINALE (GranularityAttention simple avec feat_emb.mean)
    faisait bien varier les GranW, mais l'accuracy BAISSAIT à chaque époche.
  - Les v2/v3/v4 ont corrigé l'accuracy mais cassé les GranW en rendant
    l'attention trop symétrique.

v5 = code original + 3 fixes ciblés sur les causes de baisse d'accuracy :

  [FIX 1] Split sur match_id (pas sur les séquences)
    Le split original `int(0.8 * len(X_feat))` coupait au milieu des matchs
    → des séquences du même match se retrouvaient dans train ET val
    → leakage massif → val_acc artificiellement haute au début, puis chute.
    Fix : split sur les match_id avant build_sequences, comme en v2/v3/v4.

  [FIX 2] Scaler fitté sur train uniquement
    Le StandardScaler original était fitté sur tout le dataset (train+val)
    → leakage de distribution → comportement instable en val.
    Fix : fit sur df_train, transform sur df_val.

  [FIX 3] Teacher forcing décroissant 0.9 → 0.1
    Le TF fixe à 0.5 forçait le modèle à dépendre de la vraie cible pendant
    l'inférence → écart train/val grandissant → accuracy val décroissante.
    Fix : TF décroît linéairement sur les epochs.

  NE PAS TOUCHER : GranularityAttention originale (feat_emb.mean + T=2.0)
    C'est cette architecture simple qui faisait varier les GranW.
    Les versions GRU/Bahdanau étaient trop symétriques à l'init.
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
BATCH_SIZE      = 64
D_MODEL         = 128
N_HEADS         = 4
NUM_LSTM_LAYERS = 2
CONV_KERNEL     = 3
EPOCHS          = 30
LR              = 1e-4
DROPOUT         = 0.2
LAMBDA_MOM      = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

MOM_PT_WINDOW   = 8
MOM_PT_DECAY    = 0.85
MOM_GAME_WINDOW = 6
MOM_GAME_DECAY  = 0.75
MOM_SET_WINDOW  = 4
MOM_SET_DECAY   = 0.60

SCORE_MAP = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
              0: 0,  15: 1,  30: 2,  40: 3}

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


# ─────────────────────────────────────────────
# 2. CALCUL DU MOMENTUM (code original inchangé)
# ─────────────────────────────────────────────
def compute_momentum_points(results: np.ndarray, window: int, decay: float) -> np.ndarray:
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()
    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start        = max(0, t - window + 1)
        w            = weights[-(t - start + 1):]
        momentum[t]  = np.dot(signed[start : t + 1], w)
    return momentum.astype(np.float32)


def compute_momentum_games(df_match: pd.DataFrame, window: int, decay: float) -> np.ndarray:
    momentum = np.zeros(len(df_match))
    game_results = {}
    for (s, g), grp in df_match.groupby(["set_no", "game_no"]):
        victor = grp["game_victor"].dropna()
        if len(victor) > 0:
            last = victor.iloc[-1]
            game_results[(s, g)] = 1.0 if last == 1 else -1.0

    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()

    for idx, row in df_match.iterrows():
        s, g = row["set_no"], row["game_no"]
        past_games = [(s, gg) for gg in range(1, g) if (s, gg) in game_results]
        for ss in range(1, s):
            max_g = df_match[df_match["set_no"] == ss]["game_no"].max()
            if pd.notna(max_g):
                for gg in range(1, int(max_g) + 1):
                    if (ss, gg) in game_results:
                        past_games.append((ss, gg))
        recent = past_games[-window:]
        if len(recent) == 0:
            momentum[idx] = 0.0
        else:
            vals = np.array([game_results[k] for k in recent])
            w    = weights[-len(vals):]
            momentum[idx] = np.dot(vals, w)
    return momentum.astype(np.float32)


def compute_momentum_sets(df_match: pd.DataFrame, window: int, decay: float) -> np.ndarray:
    momentum = np.zeros(len(df_match))
    set_results = {}
    for s, grp in df_match.groupby("set_no"):
        victor = grp["set_victor"].dropna()
        if len(victor) > 0:
            last = victor.iloc[-1]
            set_results[s] = 1.0 if last == 1 else -1.0

    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()

    for idx, row in df_match.iterrows():
        s = row["set_no"]
        past_sets = [ss for ss in range(1, s) if ss in set_results]
        recent    = past_sets[-window:]
        if len(recent) == 0:
            momentum[idx] = 0.0
        else:
            vals = np.array([set_results[ss] for ss in recent])
            w    = weights[-len(vals):]
            momentum[idx] = np.dot(vals, w)
    return momentum.astype(np.float32)


# ─────────────────────────────────────────────
# 3. CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Charge le CSV et calcule les 3 momentum bruts.
    [FIX 2] La normalisation est déplacée après le split train/val.
    """
    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    mom_pt_list, mom_game_list, mom_set_list = [], [], []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        mom_pt_list.extend(
            compute_momentum_points(match[TARGET_COL].values,
                                    MOM_PT_WINDOW, MOM_PT_DECAY).tolist())
        mom_game_list.extend(
            compute_momentum_games(match, MOM_GAME_WINDOW, MOM_GAME_DECAY).tolist())
        mom_set_list.extend(
            compute_momentum_sets(match, MOM_SET_WINDOW, MOM_SET_DECAY).tolist())

    df["mom_point"] = mom_pt_list
    df["mom_game"]  = mom_game_list
    df["mom_set"]   = mom_set_list
    return df


def fit_and_apply_scalers(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    [FIX 2] Fit sur train uniquement.
    On normalise features ET momentum séparément mais tous deux sur train only.
    On conserve la co-normalisation des momentum (comme l'original) car c'est
    ce qui permettait aux GranW de varier — la température T=2.0 compensait
    le fait que les 3 canaux avaient std≈1.
    """
    feat_scaler = StandardScaler()
    df_train[FEATURE_COLS] = feat_scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = feat_scaler.transform(df_val[FEATURE_COLS])

    mom_scaler = StandardScaler()
    df_train[["mom_point", "mom_game", "mom_set"]] = mom_scaler.fit_transform(
        df_train[["mom_point", "mom_game", "mom_set"]])
    df_val[["mom_point", "mom_game", "mom_set"]] = mom_scaler.transform(
        df_val[["mom_point", "mom_game", "mom_set"]])

    print(f"   mom_point — mean={df_train['mom_point'].mean():.3f}  std={df_train['mom_point'].std():.3f}")
    print(f"   mom_game  — mean={df_train['mom_game'].mean():.3f}  std={df_train['mom_game'].std():.3f}")
    print(f"   mom_set   — mean={df_train['mom_set'].mean():.3f}  std={df_train['mom_set'].std():.3f}")

    return df_train, df_val, feat_scaler, mom_scaler


# ─────────────────────────────────────────────
# 4. CONSTRUCTION DES SÉQUENCES
# ─────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    X_feat, X_mom, y_points, y_mom = [], [], [], []
    for _, match in df.groupby("match_id"):
        match    = match.sort_values("point_no").reset_index(drop=True)
        features = match[FEATURE_COLS].values
        points   = match[TARGET_COL].values
        moms     = match[["mom_point", "mom_game", "mom_set"]].values

        for i in range(len(match) - SEQ_LEN - PRED_LEN + 1):
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
        self.Xm = torch.tensor(Xm)
        self.yp = torch.tensor(yp)
        self.ym = torch.tensor(ym)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i): return self.Xf[i], self.Xm[i], self.yp[i], self.ym[i]


# ─────────────────────────────────────────────
# 5. ARCHITECTURE (originale, inchangée)
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size-1)//2)
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x.transpose(1,2)).transpose(1,2)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GranularityAttention(nn.Module):
    """
    Architecture ORIGINALE — conservée telle quelle car c'est elle
    qui faisait varier les GranW.

    Pourquoi elle fonctionne mieux que les versions GRU/Bahdanau :
      - feat_emb.mean(dim=1) produit un contexte différent pour chaque batch,
        ce qui crée de la variance dans les scores d'attention.
      - La température fixe T=2.0 empêche le softmax de coller à 0.33
        sans forcer la spécialisation.
      - Le réseau attn_net (Linear→Tanh→Linear) a des poids initialisés
        aléatoirement de façon asymétrique entre les 3 sorties → brise
        la symétrie dès l'epoch 1.
    """
    def __init__(self, d_model, n_gran=3):
        super().__init__()
        self.n_gran   = n_gran
        self.proj     = nn.Linear(1, d_model)
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, n_gran),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_mom, context):
        """
        x_mom   : (B, T, 3)
        context : (B, d_model) — feat_emb.mean(dim=1) depuis l'encoder
        """
        TEMPERATURE  = 2.0
        attn_weights = torch.softmax(self.attn_net(context) / TEMPERATURE, dim=-1)  # (B, 3)
        weighted     = (x_mom * attn_weights.unsqueeze(1)).sum(dim=-1, keepdim=True) # (B, T, 1)
        out          = self.proj(weighted)       # (B, T, d_model)
        return self.out_proj(out), attn_weights


class HydraEncoderMG(nn.Module):
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        self.feat_proj  = nn.Linear(feat_size, d_model)
        self.gran_attn  = GranularityAttention(d_model, n_gran=3)
        self.fusion_in  = nn.Linear(d_model * 2, d_model)

        self.conv_block = ConvBlock(d_model, d_model, conv_k)
        self.lstm       = nn.LSTM(d_model, d_model, num_lstm_layers, batch_first=True,
                                   dropout=dropout if num_lstm_layers > 1 else 0)
        self.pos_enc    = PositionalEncoding(d_model)
        enc_layer       = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                                      dropout, batch_first=True, norm_first=True)
        self.attention  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fusion_out = nn.Linear(d_model * 3, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x_feat, x_mom):
        feat_emb = self.feat_proj(x_feat)             # (B, T, d_model)
        context  = feat_emb.mean(dim=1)               # (B, d_model) — original conservé
        mom_emb, gran_weights = self.gran_attn(x_mom, context)

        x = self.fusion_in(torch.cat([feat_emb, mom_emb], dim=-1))

        conv_out         = self.conv_block(x)
        lstm_out, (h, c) = self.lstm(x)
        attn_out         = self.attention(self.pos_enc(x))

        fused   = torch.cat([conv_out[:, -1], lstm_out[:, -1], attn_out[:, -1]], dim=-1)
        context = self.dropout(self.fusion_out(fused))
        return context, h, c, gran_weights


class HydraPointHead(nn.Module):
    def __init__(self, d_model, num_layers, dropout, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(num_classes + d_model, d_model, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.fc   = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

    def forward(self, context, h, c, pred_len, targets=None, tf_ratio=0.5):
        B   = h.size(1)
        x_t = torch.zeros(B, 1, self.num_classes, device=h.device)
        ctx = context.unsqueeze(1)
        outputs = []
        for t in range(pred_len):
            out, (h, c) = self.lstm(torch.cat([x_t, ctx], dim=-1), (h, c))
            logit = self.fc(out.squeeze(1))
            outputs.append(logit.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf_ratio:
                x_t = F.one_hot(targets[:, t], self.num_classes).float().unsqueeze(1)
            else:
                x_t = F.one_hot(logit.argmax(1), self.num_classes).float().unsqueeze(1)
        return torch.cat(outputs, dim=1)


class HydraMomentumHead(nn.Module):
    def __init__(self, d_model, num_layers, dropout, n_gran=3):
        super().__init__()
        self.n_gran = n_gran
        self.lstm   = nn.LSTM(n_gran + d_model, d_model, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc     = nn.Linear(d_model, n_gran)

    def forward(self, context, h, c, pred_len, targets=None, tf_ratio=0.5):
        B   = h.size(1)
        x_t = torch.zeros(B, 1, self.n_gran, device=h.device)
        ctx = context.unsqueeze(1)
        outputs = []
        for t in range(pred_len):
            out, (h, c) = self.lstm(torch.cat([x_t, ctx], dim=-1), (h, c))
            pred_m = self.fc(out.squeeze(1))
            outputs.append(pred_m.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf_ratio:
                x_t = targets[:, t, :].unsqueeze(1)
            else:
                x_t = pred_m.unsqueeze(1)
        return torch.cat(outputs, dim=1)


class HydraNetMG(nn.Module):
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        self.encoder       = HydraEncoderMG(feat_size, d_model, n_heads,
                                             num_lstm_layers, conv_k, dropout)
        self.point_head    = HydraPointHead(d_model, num_lstm_layers, dropout)
        self.momentum_head = HydraMomentumHead(d_model, num_lstm_layers, dropout, n_gran=3)

    def forward(self, x_feat, x_mom, pred_len,
                y_points=None, y_mom=None, tf_ratio=0.5):
        context, h, c, gran_weights = self.encoder(x_feat, x_mom)
        point_logits = self.point_head(context, h.clone(), c.clone(),
                                       pred_len, y_points, tf_ratio)
        mom_preds    = self.momentum_head(context, h.clone(), c.clone(),
                                          pred_len, y_mom, tf_ratio)
        return point_logits, mom_preds, gran_weights


# ─────────────────────────────────────────────
# 6. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
def get_tf_ratio(epoch: int, total: int, start=0.9, end=0.1) -> float:
    """[FIX 3] Teacher forcing décroissant linéairement."""
    return start + (end - start) * (epoch - 1) / max(total - 1, 1)


def train_epoch(model, loader, optimizer, ce_loss, mse_loss, tf_ratio: float):
    model.train()
    total_loss = 0
    for Xf, Xm, yp, ym in loader:
        Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mom_preds, gran_w = model(Xf, Xm, PRED_LEN, yp, ym, tf_ratio=tf_ratio)
        loss_pts = ce_loss(logits.view(-1, 2), yp.view(-1))
        loss_mom = mse_loss(mom_preds, ym)
        entropy  = -(gran_w * torch.log(gran_w + 1e-8)).sum(dim=-1).mean()
        loss = loss_pts + LAMBDA_MOM * loss_mom - 0.01 * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, ce_loss, mse_loss):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []
    mp_all, mt_all        = [], []
    gran_weights_all      = []

    with torch.no_grad():
        for Xf, Xm, yp, ym in loader:
            Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
            logits, mom_preds, gran_w = model(Xf, Xm, PRED_LEN, tf_ratio=0.0)
            loss = (ce_loss(logits.view(-1, 2), yp.view(-1))
                    + LAMBDA_MOM * mse_loss(mom_preds, ym))
            total_loss += loss.item()
            preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
            labels_all.extend(yp.cpu().numpy().flatten())
            mp_all.append(mom_preds.cpu().numpy())
            mt_all.append(ym.cpu().numpy())
            gran_weights_all.append(gran_w.cpu().numpy())

    acc       = accuracy_score(labels_all, preds_all)
    mp_all    = np.concatenate(mp_all, axis=0)
    mt_all    = np.concatenate(mt_all, axis=0)
    mse_pt    = mean_squared_error(mt_all[:,:,0].flatten(), mp_all[:,:,0].flatten())
    mse_game  = mean_squared_error(mt_all[:,:,1].flatten(), mp_all[:,:,1].flatten())
    mse_set   = mean_squared_error(mt_all[:,:,2].flatten(), mp_all[:,:,2].flatten())
    gran_mean = np.concatenate(gran_weights_all, axis=0).mean(axis=0)

    return total_loss / len(loader), acc, mse_pt, mse_game, mse_set, gran_mean


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + calcul momentum multi-granularité (v5)...")
    df = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    # [FIX 1] Split sur match_id AVANT build_sequences et normalisation
    match_ids = df["match_id"].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)
    split_idx = int(0.8 * len(match_ids))
    train_ids = set(match_ids[:split_idx])
    val_ids   = set(match_ids[split_idx:])

    df_train = df[df["match_id"].isin(train_ids)].copy().reset_index(drop=True)
    df_val   = df[df["match_id"].isin(val_ids)].copy().reset_index(drop=True)
    print(f"   Train : {len(df_train)} points ({len(train_ids)} matchs)")
    print(f"   Val   : {len(df_val)} points ({len(val_ids)} matchs)")

    # [FIX 2] Normalisation après split, fit sur train only
    print("\n🔧 Normalisation (fit sur train uniquement) [FIX 2]...")
    df_train, df_val, feat_scaler, mom_scaler = fit_and_apply_scalers(df_train, df_val)

    print("\n🔨 Construction des séquences [FIX 1]...")
    X_feat_tr, X_mom_tr, y_pts_tr, y_mom_tr = build_sequences(df_train)
    X_feat_vl, X_mom_vl, y_pts_vl, y_mom_vl = build_sequences(df_val)
    print(f"   Train — X_feat={X_feat_tr.shape}  X_mom={X_mom_tr.shape}")
    print(f"   Val   — X_feat={X_feat_vl.shape}  X_mom={X_mom_vl.shape}")

    train_ds     = TennisDataset(X_feat_tr, X_mom_tr, y_pts_tr, y_mom_tr)
    val_ds       = TennisDataset(X_feat_vl, X_mom_vl, y_pts_vl, y_mom_vl)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=(DEVICE=="cuda"))

    model     = HydraNetMG(len(FEATURE_COLS), D_MODEL, N_HEADS,
                            NUM_LSTM_LAYERS, CONV_KERNEL, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 HydraNet MG v5 sur {DEVICE}  |  {n_params:,} paramètres")
    print(f"   Architecture originale + [FIX 1] split match_id "
          f"+ [FIX 2] scaler train-only + [FIX 3] TF décroissant\n")

    best_acc = 0
    history  = {"train_loss": [], "val_loss": [], "val_acc": [], "gran_w": []}

    for epoch in range(1, EPOCHS + 1):
        tf         = get_tf_ratio(epoch, EPOCHS)   # [FIX 3]
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss, tf)
        val_loss, val_acc, mse_pt, mse_game, mse_set, gran_w = evaluate(
            model, val_loader, ce_loss, mse_loss)
        scheduler.step()

        # Enregistrement de l'historique pour les visualisations
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["gran_w"].append(gran_w.tolist())
        history.setdefault("mse_pt",   []).append(float(mse_pt))
        history.setdefault("mse_game", []).append(float(mse_game))
        history.setdefault("mse_set",  []).append(float(mse_set))

        print(f"Epoch {epoch:02d}/{EPOCHS} | tf={tf:.2f} | "
              f"loss={train_loss:.4f} | acc={val_acc:.4f} | "
              f"MSE[pt={mse_pt:.4f} game={mse_game:.4f} set={mse_set:.4f}] | "
              f"GranW[pt={gran_w[0]:.2f} game={gran_w[1]:.2f} set={gran_w[2]:.2f}]")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "feat_size": len(FEATURE_COLS), "d_model": D_MODEL,
                    "n_heads": N_HEADS, "num_lstm_layers": NUM_LSTM_LAYERS,
                    "conv_k": CONV_KERNEL, "dropout": DROPOUT,
                },
            }, "best_hydranet_mg_v5.pt")
            joblib.dump(feat_scaler, "feat_scaler_mg_v5.pkl")
            joblib.dump(mom_scaler,  "mom_scaler_mg_v5.pkl")

    # Sauvegarde de l'historique pour visualize_hydranet.py
    joblib.dump(history, "history.pkl")
    print("   history.pkl sauvegardé → utilisable par visualize_hydranet.py")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("\n📊 Poids de granularité moyens (dernière epoch val) :")
    print(f"   Point : {gran_w[0]:.3f}  |  Jeu : {gran_w[1]:.3f}  |  Set : {gran_w[2]:.3f}")
    print("\n   Interprétation :")
    print("   → La granularité dominante est celle que le modèle juge la plus")
    print("     informative en moyenne sur les matchs de validation.")
    print("   → Des poids variables entre matchs (visible en mode debug) indiquent")
    print("     que le contexte influence bien le choix de granularité.")
    print("\n   Checkpoints : best_hydranet_mg_v5.pt, feat_scaler_mg_v5.pkl, mom_scaler_mg_v5.pkl")