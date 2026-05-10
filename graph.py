"""
visualize_hydranet.py — Visualisations complètes du modèle HydraNet MG
=======================================================================
Génère 8 graphiques séparés :
  1. Courbes d'apprentissage (loss train/val + accuracy val)
  2. Évolution des GranW par epoch
  3. Distribution finale des GranW
  4. Matrice de confusion
  5. Accuracy par position dans la séquence (point 1→5)
  6. Accuracy par contexte score (deuce, break point, normal)
  7. Accuracy par momentum set (favori vs outsider)
  8. Calibration (reliability diagram)

Usage :
  python visualize_hydranet.py --csv USD.txt --checkpoint best_hydranet_mg_v5.pt
  python visualize_hydranet.py --csv USD.txt --checkpoint best_hydranet_mg_v6.pt --v6

  Les graphiques sont sauvegardés dans ./figures/
"""

import os
import argparse
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              mean_squared_error)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# STYLE GLOBAL
# ─────────────────────────────────────────────
COLORS  = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5"]
GRAY    = "#CCCCCC"
DARK    = "#222222"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   GRAY,
    "axes.grid":        True,
    "grid.color":       "#EEEEEE",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.color":      DARK,
    "ytick.color":      DARK,
})

# ─────────────────────────────────────────────
# CONFIG (doit correspondre au modèle entraîné)
# ─────────────────────────────────────────────
SEQ_LEN         = 10
PRED_LEN        = 5
BATCH_SIZE      = 256
D_MODEL         = 128
N_HEADS         = 4
NUM_LSTM_LAYERS = 2
CONV_KERNEL     = 3
DROPOUT         = 0.2
LAMBDA_MOM      = 0.5
MOM_PT_WINDOW   = 8;  MOM_PT_DECAY   = 0.85
MOM_GAME_WINDOW = 6;  MOM_GAME_DECAY = 0.75
MOM_SET_WINDOW  = 4;  MOM_SET_DECAY  = 0.60
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_MAP = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
              0: 0,  15: 1,  30: 2,  40: 3}

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


# ─────────────────────────────────────────────
# MOMENTUM
# ─────────────────────────────────────────────
def compute_momentum_points(results, window, decay):
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay**k for k in range(window)]); weights /= weights.sum()
    mom = np.zeros(len(signed))
    for t in range(len(signed)):
        s = max(0, t - window + 1)
        w = weights[-(t-s+1):]
        mom[t] = np.dot(signed[s:t+1], w)
    return mom.astype(np.float32)

def compute_momentum_games(df_match, window, decay):
    mom = np.zeros(len(df_match))
    game_res = {}
    for (s,g), grp in df_match.groupby(["set_no","game_no"]):
        v = grp["game_victor"].dropna()
        if len(v): game_res[(s,g)] = 1.0 if v.iloc[-1]==1 else -1.0
    weights = np.array([decay**k for k in range(window)]); weights /= weights.sum()
    for idx, row in df_match.iterrows():
        s,g = row["set_no"], row["game_no"]
        past = [(s,gg) for gg in range(1,g) if (s,gg) in game_res]
        for ss in range(1,s):
            mx = df_match[df_match["set_no"]==ss]["game_no"].max()
            if pd.notna(mx):
                for gg in range(1,int(mx)+1):
                    if (ss,gg) in game_res: past.append((ss,gg))
        recent = past[-window:]
        if not recent: mom[idx] = 0.0
        else:
            vals = np.array([game_res[k] for k in recent])
            w = weights[-len(vals):]
            mom[idx] = np.dot(vals,w)
    return mom.astype(np.float32)

def compute_momentum_sets(df_match, window, decay):
    mom = np.zeros(len(df_match))
    set_res = {}
    for s, grp in df_match.groupby("set_no"):
        v = grp["set_victor"].dropna()
        if len(v): set_res[s] = 1.0 if v.iloc[-1]==1 else -1.0
    weights = np.array([decay**k for k in range(window)]); weights /= weights.sum()
    for idx, row in df_match.iterrows():
        s = row["set_no"]
        past = [ss for ss in range(1,s) if ss in set_res]
        recent = past[-window:]
        if not recent: mom[idx] = 0.0
        else:
            vals = np.array([set_res[ss] for ss in recent])
            w = weights[-len(vals):]
            mom[idx] = np.dot(vals,w)
    return mom.astype(np.float32)


# ─────────────────────────────────────────────
# CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────────
def load_val_data(csv_path, feat_scaler_path, mom_scaler_path):
    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    mom_pt, mom_gm, mom_st = [], [], []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        mom_pt.extend(compute_momentum_points(match[TARGET_COL].values,
                      MOM_PT_WINDOW, MOM_PT_DECAY).tolist())
        mom_gm.extend(compute_momentum_games(match, MOM_GAME_WINDOW,
                      MOM_GAME_DECAY).tolist())
        mom_st.extend(compute_momentum_sets(match, MOM_SET_WINDOW,
                      MOM_SET_DECAY).tolist())
    df["mom_point"] = mom_pt
    df["mom_game"]  = mom_gm
    df["mom_set"]   = mom_st

    # Split identique à l'entraînement (même seed)
    match_ids = df["match_id"].unique()
    np.random.seed(42); np.random.shuffle(match_ids)
    val_ids = set(match_ids[int(0.8*len(match_ids)):])
    df_val  = df[df["match_id"].isin(val_ids)].copy().reset_index(drop=True)

    feat_scaler = joblib.load(feat_scaler_path)
    mom_scaler  = joblib.load(mom_scaler_path)
    df_val[FEATURE_COLS] = feat_scaler.transform(df_val[FEATURE_COLS].fillna(0))
    df_val[["mom_point","mom_game","mom_set"]] = mom_scaler.transform(
        df_val[["mom_point","mom_game","mom_set"]])

    return df_val


def build_sequences_with_meta(df):
    """Construit les séquences + métadonnées pour les analyses contextuelles."""
    X_feat, X_mom, y_pts, y_mom = [], [], [], []
    # Métadonnées par séquence
    meta_score_ctx   = []   # 0=normal, 1=deuce, 2=break_point
    meta_mom_set_bin = []   # -1/0/+1 momentum set catégorisé
    meta_match_id    = []

    for mid, match in df.groupby("match_id"):
        match    = match.sort_values("point_no").reset_index(drop=True)
        features = match[FEATURE_COLS].fillna(0).values
        points   = match[TARGET_COL].values
        moms     = match[["mom_point","mom_game","mom_set"]].values
        s1       = match["p1_score"].fillna(0).astype(int).values
        s2       = match["p2_score"].fillna(0).astype(int).values
        mom_set_raw = match["mom_set"].values

        for i in range(len(match) - SEQ_LEN - PRED_LEN + 1):
            X_feat.append(features[i:i+SEQ_LEN])
            X_mom.append(moms[i:i+SEQ_LEN])
            y_pts.append(points[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
            y_mom.append(moms[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
            meta_match_id.append(mid)

            # Contexte score au dernier pas de la fenêtre
            t = i + SEQ_LEN - 1
            if s1[t] >= 3 and s2[t] >= 3:
                meta_score_ctx.append(1)   # deuce
            elif (s1[t] >= 3 and s2[t] < s1[t]) or (s2[t] >= 3 and s1[t] < s2[t]):
                meta_score_ctx.append(2)   # break point
            else:
                meta_score_ctx.append(0)   # normal

            # Momentum set : catégorisé
            ms = mom_set_raw[t]
            if ms > 0.1:   meta_mom_set_bin.append(1)
            elif ms < -0.1: meta_mom_set_bin.append(-1)
            else:           meta_mom_set_bin.append(0)

    return (np.array(X_feat, dtype=np.float32),
            np.array(X_mom,  dtype=np.float32),
            np.array(y_pts,  dtype=np.int64),
            np.array(y_mom,  dtype=np.float32),
            np.array(meta_score_ctx),
            np.array(meta_mom_set_bin))


# ─────────────────────────────────────────────
# ARCHITECTURE (identique v5)
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=(ks-1)//2)
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x.transpose(1,2)).transpose(1,2)))

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=500):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

class GranularityAttention(nn.Module):
    def __init__(self, d_model, n_gran=3):
        super().__init__()
        self.proj     = nn.Linear(1, d_model)
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.Tanh(),
            nn.Linear(d_model//2, n_gran))
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x_mom, context):
        w = torch.softmax(self.attn_net(context)/2.0, dim=-1)
        weighted = (x_mom * w.unsqueeze(1)).sum(dim=-1, keepdim=True)
        return self.out_proj(self.proj(weighted)), w

class HydraEncoderMG(nn.Module):
    def __init__(self, feat_size, d, n_heads, n_lstm, ck, drop):
        super().__init__()
        self.feat_proj  = nn.Linear(feat_size, d)
        self.gran_attn  = GranularityAttention(d)
        self.fusion_in  = nn.Linear(d*2, d)
        self.conv_block = ConvBlock(d, d, ck)
        self.lstm       = nn.LSTM(d, d, n_lstm, batch_first=True,
                                   dropout=drop if n_lstm>1 else 0)
        self.pos_enc    = PositionalEncoding(d)
        enc             = nn.TransformerEncoderLayer(d, n_heads, d*2, drop,
                                                      batch_first=True, norm_first=True)
        self.attention  = nn.TransformerEncoder(enc, num_layers=2)
        self.fusion_out = nn.Linear(d*3, d)
        self.drop       = nn.Dropout(drop)
    def forward(self, xf, xm):
        fe = self.feat_proj(xf)
        me, gw = self.gran_attn(xm, fe.mean(dim=1))
        x = self.fusion_in(torch.cat([fe, me], dim=-1))
        co = self.conv_block(x)
        lo, (h,c) = self.lstm(x)
        ao = self.attention(self.pos_enc(x))
        fused = torch.cat([co[:,-1], lo[:,-1], ao[:,-1]], dim=-1)
        return self.drop(self.fusion_out(fused)), h, c, gw

class HydraPointHead(nn.Module):
    def __init__(self, d, n_layers, drop, nc=2):
        super().__init__()
        self.lstm = nn.LSTM(nc+d, d, n_layers, batch_first=True,
                             dropout=drop if n_layers>1 else 0)
        self.fc   = nn.Linear(d, nc); self.nc = nc
    def forward(self, ctx, h, c, pl, targets=None, tf=0.0):
        B   = h.size(1)
        x_t = torch.zeros(B, 1, self.nc, device=h.device)
        out = []
        for t in range(pl):
            o,(h,c) = self.lstm(torch.cat([x_t, ctx.unsqueeze(1)], -1), (h,c))
            lg = self.fc(o.squeeze(1)); out.append(lg.unsqueeze(1))
            x_t = F.one_hot(lg.argmax(1), self.nc).float().unsqueeze(1)
        return torch.cat(out, dim=1)

class HydraMomentumHead(nn.Module):
    def __init__(self, d, n_layers, drop, ng=3):
        super().__init__()
        self.ng   = ng
        self.lstm = nn.LSTM(ng+d, d, n_layers, batch_first=True,
                             dropout=drop if n_layers>1 else 0)
        self.fc   = nn.Linear(d, ng)
    def forward(self, ctx, h, c, pl, targets=None, tf=0.0):
        B   = h.size(1)
        x_t = torch.zeros(B, 1, self.ng, device=h.device)
        out = []
        for t in range(pl):
            o,(h,c) = self.lstm(torch.cat([x_t, ctx.unsqueeze(1)], -1), (h,c))
            pm = self.fc(o.squeeze(1)); out.append(pm.unsqueeze(1))
            x_t = pm.unsqueeze(1)
        return torch.cat(out, dim=1)

class HydraNetMG(nn.Module):
    def __init__(self, feat_size, d, n_heads, n_lstm, ck, drop):
        super().__init__()
        self.encoder  = HydraEncoderMG(feat_size, d, n_heads, n_lstm, ck, drop)
        self.pt_head  = HydraPointHead(d, n_lstm, drop)
        self.mom_head = HydraMomentumHead(d, n_lstm, drop)
    def forward(self, xf, xm, pl):
        ctx, h, c, gw = self.encoder(xf, xm)
        return self.pt_head(ctx, h.clone(), c.clone(), pl), \
               self.mom_head(ctx, h.clone(), c.clone(), pl), gw


class TennisDataset(Dataset):
    def __init__(self, Xf, Xm, yp, ym):
        self.Xf = torch.tensor(Xf); self.Xm = torch.tensor(Xm)
        self.yp = torch.tensor(yp); self.ym = torch.tensor(ym)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i): return self.Xf[i], self.Xm[i], self.yp[i], self.ym[i]


# ─────────────────────────────────────────────
# INFÉRENCE COMPLÈTE
# ─────────────────────────────────────────────
def run_inference(model, Xf, Xm, yp, ym):
    """Retourne preds, probs, gran_weights, losses par epoch (à passer en dehors)."""
    ds     = TennisDataset(Xf, Xm, yp, ym)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    all_gran, all_mom_pred, all_mom_true = [], [], []

    with torch.no_grad():
        for xf, xm, yp_b, ym_b in loader:
            xf, xm = xf.to(DEVICE), xm.to(DEVICE)
            logits, mom_pred, gw = model(xf, xm, PRED_LEN)
            probs = torch.softmax(logits, dim=-1)[:,:,1]   # prob classe 1
            all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yp_b.numpy().tolist())
            all_gran.extend(gw.cpu().numpy().tolist())
            all_mom_pred.extend(mom_pred.cpu().numpy().tolist())
            all_mom_true.extend(ym_b.numpy().tolist())

    return (np.array(all_preds),       # (N, PRED_LEN)
            np.array(all_probs),       # (N, PRED_LEN)
            np.array(all_labels),      # (N, PRED_LEN)
            np.array(all_gran),        # (N, 3)
            np.array(all_mom_pred),    # (N, PRED_LEN, 3)
            np.array(all_mom_true))    # (N, PRED_LEN, 3)


# ─────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────

def plot_learning_curves(history: dict, save_path="figures/01_learning_curves.png"):
    """
    Graphique 1 : courbes d'apprentissage.
    history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history["train_loss"], color=COLORS[0], lw=2, label="Train loss")
    ax1.plot(epochs, history["val_loss"],   color=COLORS[1], lw=2, label="Val loss",
             linestyle="--")
    ax1.set_xlabel("Époque"); ax1.set_ylabel("Loss (CE + MSE momentum)")
    ax1.set_title("Courbes de perte"); ax1.legend()

    ax2.plot(epochs, history["val_acc"], color=COLORS[2], lw=2.5)
    ax2.axhline(max(history["val_acc"]), color=COLORS[1], linestyle=":",
                lw=1.5, label=f"Best = {max(history['val_acc']):.4f}")
    ax2.set_xlabel("Époque"); ax2.set_ylabel("Accuracy (val)")
    ax2.set_title("Accuracy de validation"); ax2.legend()
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    fig.suptitle("Courbes d'apprentissage — HydraNet MG", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_granw_evolution(history: dict, save_path="figures/02_granw_evolution.png"):
    """
    Graphique 2 : évolution des 3 poids GranularityAttention par epoch.
    history["gran_w"] = [(pt, game, set), ...] une entrée par epoch
    """
    gw   = np.array(history["gran_w"])   # (E, 3)
    ep   = range(1, len(gw)+1)
    labels = ["Point (court terme)", "Jeu (moyen terme)", "Set (long terme)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, lbl) in enumerate(zip(COLORS, labels)):
        ax.plot(ep, gw[:,i], color=col, lw=2.5, label=lbl, marker="o",
                markersize=3)
    ax.axhline(1/3, color=GRAY, linestyle=":", lw=1.5, label="Uniforme (1/3)")
    ax.set_xlabel("Époque"); ax.set_ylabel("Poids d'attention (softmax)")
    ax.set_title("Évolution des poids de granularité — GranularityAttention")
    ax.legend(); ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Annotation de la valeur finale
    for i, col in enumerate(COLORS[:3]):
        ax.annotate(f"{gw[-1,i]:.2f}", xy=(len(gw), gw[-1,i]),
                    xytext=(5,0), textcoords="offset points",
                    color=col, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_granw_distribution(gran_weights, save_path="figures/03_granw_distribution.png"):
    """
    Graphique 3 : distribution des poids GranW sur toute la val set.
    gran_weights : (N, 3)
    """
    labels = ["Point\n(court terme)", "Jeu\n(moyen terme)", "Set\n(long terme)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for i, (ax, lbl, col) in enumerate(zip(axes, labels, COLORS)):
        vals = gran_weights[:, i]
        ax.hist(vals, bins=40, color=col, alpha=0.85, edgecolor="white")
        ax.axvline(vals.mean(), color=DARK, linestyle="--", lw=2,
                   label=f"Moyenne : {vals.mean():.3f}")
        ax.set_title(lbl); ax.set_xlabel("Poids d'attention")
        ax.set_ylabel("Nombre de séquences" if i == 0 else "")
        ax.legend()

    fig.suptitle("Distribution des poids GranularityAttention\n"
                 "(sur l'ensemble de validation)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_confusion_matrix(preds, labels, save_path="figures/04_confusion_matrix.png"):
    """
    Graphique 4 : matrice de confusion (sur tous les points prédits).
    """
    p_flat = preds.flatten()
    l_flat = labels.flatten()
    cm = confusion_matrix(l_flat, p_flat)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    tick_labels = ["J2 gagne\nle point", "J1 gagne\nle point"]
    ax.set_xticks([0,1]); ax.set_xticklabels(tick_labels)
    ax.set_yticks([0,1]); ax.set_yticklabels(tick_labels, rotation=45)
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Réalité")
    ax.set_title("Matrice de confusion (normalisée)")

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i,j] > 0.6 else DARK
            ax.text(j, i, f"{cm_norm[i,j]:.2%}\n({cm[i,j]:,})",
                    ha="center", va="center", color=color, fontsize=11)

    acc = accuracy_score(l_flat, p_flat)
    ax.set_xlabel(f"Prédiction  —  Accuracy globale : {acc:.2%}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_accuracy_by_position(preds, labels,
                               save_path="figures/05_accuracy_by_position.png"):
    """
    Graphique 5 : accuracy selon la position du point prédit (1→PRED_LEN).
    Montre si le modèle est meilleur sur le point immédiat que sur le 5ème.
    """
    accs = [accuracy_score(labels[:,t], preds[:,t])
            for t in range(preds.shape[1])]
    positions = [f"t+{t+1}" for t in range(len(accs))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(positions, accs, color=COLORS[0], alpha=0.85, edgecolor="white",
                  width=0.55)
    ax.axhline(np.mean(accs), color=COLORS[1], linestyle="--", lw=2,
               label=f"Moyenne : {np.mean(accs):.2%}")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLORS[0])

    ax.set_xlabel("Position du point prédit dans la séquence future")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy par position de prédiction\n"
                 "(t+1 = prochain point, t+5 = 5ème point futur)")
    ax.legend()
    ax.set_ylim(min(accs) - 0.02, max(accs) + 0.025)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_accuracy_by_score_context(preds, labels, score_ctx,
                                    save_path="figures/06_accuracy_by_score_context.png"):
    """
    Graphique 6 : accuracy selon le contexte score au moment de la prédiction.
    score_ctx : 0=normal, 1=deuce, 2=break point
    """
    ctx_labels = {0: "Normal", 1: "Deuce", 2: "Break point"}
    ctx_colors = [COLORS[0], COLORS[2], COLORS[1]]

    accs, counts = [], []
    for k in [0, 1, 2]:
        mask = score_ctx == k
        if mask.sum() == 0:
            accs.append(0); counts.append(0); continue
        a = accuracy_score(labels[mask].flatten(), preds[mask].flatten())
        accs.append(a); counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([ctx_labels[k] for k in [0,1,2]], accs,
                  color=ctx_colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(accuracy_score(labels.flatten(), preds.flatten()),
               color=GRAY, linestyle=":", lw=2, label="Accuracy globale")

    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{acc:.2%}\n(n={count:,})", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Contexte score"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy par contexte score\n"
                 "(situation au dernier pas de la fenêtre d'entrée)")
    ax.legend()
    ax.set_ylim(min(accs)-0.03, max(accs)+0.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_accuracy_by_momentum_set(preds, labels, mom_set_bin,
                                   save_path="figures/07_accuracy_by_momentum_set.png"):
    """
    Graphique 7 : accuracy selon le momentum set (J1 favori / équilibré / outsider).
    """
    bin_labels = {-1: "J1 outsider\n(mom set < −0.1)",
                   0: "Équilibré\n(|mom set| ≤ 0.1)",
                   1: "J1 favori\n(mom set > +0.1)"}
    bin_colors = [COLORS[1], COLORS[4], COLORS[0]]

    accs, counts = [], []
    for k in [-1, 0, 1]:
        mask = mom_set_bin == k
        if mask.sum() == 0:
            accs.append(0); counts.append(0); continue
        a = accuracy_score(labels[mask].flatten(), preds[mask].flatten())
        accs.append(a); counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([bin_labels[k] for k in [-1,0,1]], accs,
                  color=bin_colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(accuracy_score(labels.flatten(), preds.flatten()),
               color=GRAY, linestyle=":", lw=2, label="Accuracy globale")

    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{acc:.2%}\n(n={count:,})", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Situation momentum set"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy selon le momentum de set\n"
                 "(le modèle est-il meilleur quand la situation est claire ?)")
    ax.legend()
    ax.set_ylim(min(accs)-0.03, max(accs)+0.04)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_calibration(probs, labels, n_bins=10,
                     save_path="figures/08_calibration.png"):
    """
    Graphique 8 : reliability diagram (calibration du modèle).
    Répond à : "quand le modèle dit 70% de confiance, a-t-il raison 70% du temps ?"
    """
    p_flat = probs.flatten()
    l_flat = labels.flatten().astype(float)

    bin_edges = np.linspace(0, 1, n_bins+1)
    bin_acc, bin_conf, bin_count = [], [], []

    for i in range(n_bins):
        mask = (p_flat >= bin_edges[i]) & (p_flat < bin_edges[i+1])
        if mask.sum() == 0: continue
        bin_acc.append(l_flat[mask].mean())
        bin_conf.append(p_flat[mask].mean())
        bin_count.append(mask.sum())

    bin_acc   = np.array(bin_acc)
    bin_conf  = np.array(bin_conf)
    bin_count = np.array(bin_count)

    # Expected Calibration Error
    ece = np.sum(bin_count * np.abs(bin_acc - bin_conf)) / bin_count.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1], [0,1], "k--", lw=1.5, label="Calibration parfaite")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.6, color=COLORS[0],
           edgecolor="white", label="Modèle (accuracy réelle)")
    ax.scatter(bin_conf, bin_acc, color=COLORS[0], zorder=5, s=50)

    ax.set_xlabel("Confiance prédite (probabilité classe 1)")
    ax.set_ylabel("Accuracy réelle")
    ax.set_title(f"Diagramme de calibration\nECE = {ece:.4f}")
    ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_momentum_mse_evolution(history: dict,
                                save_path="figures/09_momentum_mse_evolution.png"):
    """
    Graphique 9 : évolution des MSE de prédiction momentum par epoch.
    Montre quelle granularité est la plus difficile à prédire,
    et si le modèle progresse sur chacune.
    history doit contenir : mse_pt, mse_game, mse_set
    """
    if "mse_pt" not in history:
        print("  ⚠ mse_pt absent de history — graphique 9 ignoré.")
        return

    epochs = range(1, len(history["mse_pt"]) + 1)
    labels = ["Point (court terme)", "Jeu (moyen terme)", "Set (long terme)"]
    keys   = ["mse_pt", "mse_game", "mse_set"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, lbl, col in zip(keys, labels, COLORS):
        ax.plot(epochs, history[key], color=col, lw=2.5, label=lbl,
                marker="o", markersize=3)

    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE (prédiction momentum)")
    ax.set_title("Évolution de l'erreur de prédiction par granularité de momentum\n"
                 "(plus bas = le modèle prédit mieux la dynamique future)")
    ax.legend()

    # Annotation valeurs finales
    for key, col in zip(keys, COLORS):
        last = history[key][-1]
        ax.annotate(f"{last:.4f}", xy=(len(epochs), last),
                    xytext=(5, 0), textcoords="offset points",
                    color=col, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_momentum_pred_vs_true(mom_pred, mom_true, n_samples=2000,
                                save_path="figures/10_momentum_pred_vs_true.png"):
    """
    Graphique 10 : scatter prédiction vs réalité pour les 3 granularités.
    Chaque point = une prédiction de momentum à t+1.
    Une diagonale parfaite = prédiction parfaite.

    mom_pred, mom_true : (N, PRED_LEN, 3)
    On prend t+1 (premier pas de prédiction) pour plus de lisibilité.
    """
    gran_names = ["Point (court terme)", "Jeu (moyen terme)", "Set (long terme)"]
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))

    # Sous-échantillonnage pour la lisibilité
    idx = np.random.choice(len(mom_pred), min(n_samples, len(mom_pred)), replace=False)

    for i, (ax, name, col) in enumerate(zip(axes, gran_names, COLORS)):
        true = mom_true[idx, 0, i]   # t+1
        pred = mom_pred[idx, 0, i]

        mse  = mean_squared_error(mom_true[:, 0, i], mom_pred[:, 0, i])
        corr = np.corrcoef(mom_true[:, 0, i], mom_pred[:, 0, i])[0, 1]

        ax.scatter(true, pred, alpha=0.15, s=8, color=col, rasterized=True)

        # Ligne diagonale (prédiction parfaite)
        lim = max(abs(true).max(), abs(pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="Parfait")

        # Droite de régression
        m, b = np.polyfit(true, pred, 1)
        x_line = np.linspace(-lim, lim, 100)
        ax.plot(x_line, m*x_line + b, color=DARK, lw=1.5, linestyle="-",
                label=f"Régression (r={corr:.2f})")

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Momentum réel (t+1)")
        ax.set_ylabel("Momentum prédit (t+1)" if i == 0 else "")
        ax.set_title(f"{name}\nMSE={mse:.4f}  r={corr:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Prédiction vs Réalité du momentum (t+1)\n"
                 f"({n_samples} séquences échantillonnées)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def plot_momentum_residuals(mom_pred, mom_true,
                             save_path="figures/11_momentum_residuals.png"):
    """
    Graphique 11 : distribution des résidus (erreurs) de prédiction momentum.
    residual = pred - true
    Un résidu centré en 0 avec faible variance = bonne calibration du momentum.
    Montre si le modèle a un biais systématique (sur- ou sous-estimation).

    mom_pred, mom_true : (N, PRED_LEN, 3)
    On agrège tous les pas de prédiction (t+1 → t+PRED_LEN).
    """
    gran_names = ["Point\n(court terme)", "Jeu\n(moyen terme)", "Set\n(long terme)"]
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))

    for i, (ax, name, col) in enumerate(zip(axes, gran_names, COLORS)):
        residuals = (mom_pred[:, :, i] - mom_true[:, :, i]).flatten()
        bias      = residuals.mean()
        std_res   = residuals.std()

        ax.hist(residuals, bins=60, color=col, alpha=0.8, edgecolor="white",
                density=True)
        ax.axvline(0,    color="black",   lw=2,   linestyle="--", label="0 (idéal)")
        ax.axvline(bias, color=COLORS[1], lw=2,   linestyle="-",
                   label=f"Biais : {bias:+.4f}")

        # Courbe gaussienne de référence
        x = np.linspace(residuals.min(), residuals.max(), 200)
        from scipy.stats import norm as scipy_norm
        ax.plot(x, scipy_norm.pdf(x, bias, std_res),
                color=DARK, lw=2, linestyle=":", label=f"σ={std_res:.3f}")

        ax.set_xlabel("Résidu (prédit − réel)")
        ax.set_ylabel("Densité" if i == 0 else "")
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution des résidus de prédiction momentum\n"
                 "(tous les pas t+1→t+5 agrégés)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True,  help="Chemin vers le CSV de données")
    parser.add_argument("--checkpoint", required=True,  help="Chemin vers le .pt du modèle")
    parser.add_argument("--feat-scaler", default="feat_scaler_mg_v5.pkl")
    parser.add_argument("--mom-scaler",  default="mom_scaler_mg_v5.pkl")
    parser.add_argument("--history",    default=None,
                        help="Chemin vers history.pkl (si sauvegardé pendant l'entraînement)")
    args = parser.parse_args()

    print("📂 Chargement des données de validation...")
    df_val = load_val_data(args.csv, args.feat_scaler, args.mom_scaler)
    print(f"   {len(df_val)} points de validation")

    print("🔨 Construction des séquences avec métadonnées...")
    Xf, Xm, yp, ym, score_ctx, mom_set_bin = build_sequences_with_meta(df_val)
    print(f"   {len(Xf)} séquences")

    print("🚀 Chargement du modèle...")
    ckpt   = torch.load(args.checkpoint, map_location=DEVICE)
    cfg    = ckpt["config"]
    model  = HydraNetMG(cfg["feat_size"], cfg["d_model"], cfg["n_heads"],
                        cfg["num_lstm_layers"], cfg["conv_k"], cfg["dropout"]).to(DEVICE)
    state_dict = ckpt["model"]

    # Renommage des clés pour correspondre à l'architecture actuelle
    key_mapping = {
        "point_head": "pt_head",
        "momentum_head": "mom_head",
    }

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_name, new_name in key_mapping.items():
            if k.startswith(old_name):
                new_key = k.replace(old_name, new_name, 1)
                break
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {n_params:,} paramètres chargés")

    print("\n⚙️  Inférence sur la validation...")
    preds, probs, labels, gran_w, mom_pred, mom_true = run_inference(model, Xf, Xm, yp, ym)
    acc_global = accuracy_score(labels.flatten(), preds.flatten())
    print(f"   Accuracy globale : {acc_global:.4f}")

    print("\n📊 Génération des graphiques...")

    # Graphique 1 : courbes d'apprentissage (nécessite history)
    if args.history and os.path.exists(args.history):
        history = joblib.load(args.history)
        plot_learning_curves(history)
        plot_granw_evolution(history)
    else:
        print("  ⚠ history.pkl non fourni — graphiques 1 et 2 ignorés.")
        print("    Ajouter dans la boucle d'entraînement v5 :")
        print("      history['train_loss'].append(train_loss)")
        print("      history['val_loss'].append(val_loss)")
        print("      history['val_acc'].append(val_acc)")
        print("      history['gran_w'].append(gran_w.tolist())")
        print("    Et sauvegarder : joblib.dump(history, 'history.pkl')")

    # Graphique 3 : distribution GranW
    plot_granw_distribution(gran_w)

    # Graphique 4 : matrice de confusion
    plot_confusion_matrix(preds, labels)

    # Graphique 5 : accuracy par position
    plot_accuracy_by_position(preds, labels)

    # Graphique 6 : accuracy par contexte score
    plot_accuracy_by_score_context(preds, labels, score_ctx)

    # Graphique 7 : accuracy par momentum set
    plot_accuracy_by_momentum_set(preds, labels, mom_set_bin)

    # Graphique 8 : calibration
    plot_calibration(probs, labels)

    # ── Graphiques momentum ────────────────────────────────────────────────
    print("\n📈 Graphiques de performance momentum...")

    # Graphique 9 : MSE momentum par epoch (nécessite history)
    if args.history and os.path.exists(args.history):
        history = joblib.load(args.history)
        plot_momentum_mse_evolution(history)

    # Graphique 10 : scatter prédiction vs réalité
    plot_momentum_pred_vs_true(mom_pred, mom_true)

    # Graphique 11 : distribution des résidus
    plot_momentum_residuals(mom_pred, mom_true)

    print(f"\n✅ Tous les graphiques sauvegardés dans ./figures/")
    print(f"   Accuracy globale : {acc_global:.4f}")
    print(f"\n   Graphiques générés :")
    print(f"   01_learning_curves        — Loss + accuracy par epoch")
    print(f"   02_granw_evolution        — Évolution des poids GranW")
    print(f"   03_granw_distribution     — Distribution des poids sur la val")
    print(f"   04_confusion_matrix       — Matrice de confusion")
    print(f"   05_accuracy_by_position   — Accuracy t+1 → t+5")
    print(f"   06_accuracy_by_score_ctx  — Normal / Deuce / Break point")
    print(f"   07_accuracy_by_mom_set    — Favori / Équilibré / Outsider")
    print(f"   08_calibration            — Reliability diagram + ECE")
    print(f"   09_momentum_mse_evolution — MSE momentum par epoch et granularité")
    print(f"   10_momentum_pred_vs_true  — Scatter prédit vs réel (t+1)")
    print(f"   11_momentum_residuals     — Distribution des résidus momentum")
    print(f"\n💡 Pour activer les graphiques 1 et 2, ajouter dans v5/v6 :")
    print("""
    history = {"train_loss":[], "val_loss":[], "val_acc":[], "gran_w":[]}
    # Dans la boucle d'entraînement :
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["gran_w"].append(gran_w.tolist())
    # Après la boucle :
    joblib.dump(history, "history.pkl")
    """)