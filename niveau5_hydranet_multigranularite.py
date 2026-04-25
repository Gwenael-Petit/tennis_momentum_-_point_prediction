"""
NIVEAU 5 - Momentum Multi-Granularité (inspiré HydraNet 2025)
==============================================================
Innovation clé : le momentum est calculé à 3 niveaux simultanément,
inspiré du vrai HydraNet qui modélise l'adversité à plusieurs échelles.

  ┌─────────────────────────────────────────────────────────┐
  │  Momentum Point  (fenêtre courte, décroissance rapide)  │
  │  Momentum Jeu    (agrégé par jeu dans le set courant)   │
  │  Momentum Set    (agrégé par set dans le match)         │
  └─────────────────────────────────────────────────────────┘
            ↓ concaténés avec les features
  ┌──────────────────────────────────────────────────────┐
  │  HydraEncoder Multi-Scale                           │
  │  ├─ ConvBlock  (patterns locaux)                    │
  │  ├─ LSTM       (dépendances temporelles)            │
  │  └─ Attention  (dépendances globales)               │
  └──────────────────────────────────────────────────────┘
            ↓ context vector partagé
       ┌────┴────┐
       ▼         ▼
  Head Points  Head Momentum
  (seq2seq     (seq2seq
  classif.)    régression)

Pourquoi c'est innovant ?
  - La littérature existante utilise un momentum à granularité unique
  - Ici, l'encoder reçoit TROIS signaux de momentum simultanément :
    court terme (point), moyen terme (jeu), long terme (set)
  - Cela permet à l'encoder d'apprendre QUELLE granularité est
    la plus prédictive selon le contexte du match
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
LAMBDA_MOM      = 0.5       # poids loss momentum vs points
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paramètres momentum par granularité ──────
# Point : réactif, fenêtre courte
MOM_PT_WINDOW   = 8
MOM_PT_DECAY    = 0.85
# Jeu : moins réactif
MOM_GAME_WINDOW = 6
MOM_GAME_DECAY  = 0.75
# Set : inertiel, fenêtre large
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
# 2. CALCUL DU MOMENTUM MULTI-GRANULARITÉ
# ─────────────────────────────────────────────
def compute_momentum_points(results: np.ndarray, window: int, decay: float) -> np.ndarray:
    """
    Momentum au niveau POINT.
    Fenêtre glissante sur les résultats bruts (±1).
    Signal le plus réactif : capte les séries de points consécutifs.
    """
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()
    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start = max(0, t - window + 1)
        w     = weights[-(t - start + 1):]
        momentum[t] = np.dot(signed[start : t + 1], w)
    return momentum.astype(np.float32)


def compute_momentum_games(df_match: pd.DataFrame, window: int, decay: float) -> np.ndarray:
    """
    Momentum au niveau JEU.
    Pour chaque point, on regarde les résultats des derniers 'window' jeux
    dans le set courant et on propage la valeur à tous les points du jeu.

    Résultat d'un jeu : +1 si J1 a gagné le jeu, -1 sinon (via game_victor).
    Signal moyen terme : capte les séries de jeux gagnés.
    """
    momentum = np.zeros(len(df_match))

    # On construit d'abord le résultat de chaque jeu (set_no, game_no)
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
        # Jeux passés dans le même set, avant le jeu courant
        past_games = [(s, gg) for gg in range(1, g) if (s, gg) in game_results]
        # + jeux des sets précédents (jusqu'à window jeux au total)
        for ss in range(1, s):
            max_g = df_match[df_match["set_no"] == ss]["game_no"].max()
            if pd.notna(max_g):
                for gg in range(1, int(max_g) + 1):
                    if (ss, gg) in game_results:
                        past_games.append((ss, gg))

        # On garde les 'window' derniers jeux
        recent = past_games[-window:]
        if len(recent) == 0:
            momentum[idx] = 0.0
        else:
            vals = np.array([game_results[k] for k in recent])
            w    = weights[-len(vals):]
            momentum[idx] = np.dot(vals, w)

    return momentum.astype(np.float32)


def compute_momentum_sets(df_match: pd.DataFrame, window: int, decay: float) -> np.ndarray:
    """
    Momentum au niveau SET.
    Pour chaque point, on regarde les résultats des derniers 'window' sets.
    Signal lent : capte l'avantage psychologique sur le match entier.
    """
    momentum = np.zeros(len(df_match))

    # Résultat de chaque set
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
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, sep=",")

    # Encoder les scores tennis
    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    # Calcul des 3 niveaux de momentum par match
    mom_pt_list, mom_game_list, mom_set_list = [], [], []

    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)

        mom_pt   = compute_momentum_points(match[TARGET_COL].values,
                                           MOM_PT_WINDOW, MOM_PT_DECAY)
        mom_game = compute_momentum_games(match, MOM_GAME_WINDOW, MOM_GAME_DECAY)
        mom_set  = compute_momentum_sets(match, MOM_SET_WINDOW, MOM_SET_DECAY)

        mom_pt_list.extend(mom_pt.tolist())
        mom_game_list.extend(mom_game.tolist())
        mom_set_list.extend(mom_set.tolist())

    df["mom_point"] = mom_pt_list
    df["mom_game"]  = mom_game_list
    df["mom_set"]   = mom_set_list

    # Normalisation des features brutes
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    print(f"   mom_point — mean={df['mom_point'].mean():.3f}  std={df['mom_point'].std():.3f}")
    print(f"   mom_game  — mean={df['mom_game'].mean():.3f}  std={df['mom_game'].std():.3f}")
    print(f"   mom_set   — mean={df['mom_set'].mean():.3f}  std={df['mom_set'].std():.3f}")

    return df, scaler


# ─────────────────────────────────────────────
# 4. CONSTRUCTION DES SÉQUENCES
# ─────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    """
    X_feat  : (N, SEQ_LEN, F)      features brutes
    X_mom   : (N, SEQ_LEN, 3)      momentum [point, jeu, set] par pas de temps
    y_points: (N, PRED_LEN)        qui gagne chaque point futur
    y_mom   : (N, PRED_LEN, 3)     momentum [point, jeu, set] futur
    """
    X_feat, X_mom, y_points, y_mom = [], [], [], []

    for _, match in df.groupby("match_id"):
        match    = match.sort_values("point_no").reset_index(drop=True)
        features = match[FEATURE_COLS].values                             # (T, F)
        points   = match[TARGET_COL].values                               # (T,)
        moms     = match[["mom_point", "mom_game", "mom_set"]].values     # (T, 3)

        for i in range(len(match) - SEQ_LEN - PRED_LEN + 1):
            X_feat.append(features[i : i + SEQ_LEN])
            X_mom.append(moms[i : i + SEQ_LEN])                             # (SEQ_LEN, 3)
            y_points.append(points[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
            y_mom.append(moms[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])        # (PRED_LEN, 3)

    return (np.array(X_feat,   dtype=np.float32),   # (N, SEQ_LEN, F)
            np.array(X_mom,    dtype=np.float32),   # (N, SEQ_LEN, 3)
            np.array(y_points, dtype=np.int64),     # (N, PRED_LEN)
            np.array(y_mom,    dtype=np.float32))   # (N, PRED_LEN, 3)


class TennisDataset(Dataset):
    def __init__(self, Xf, Xm, yp, ym):
        self.Xf = torch.tensor(Xf)   # (N, SEQ_LEN, F)
        self.Xm = torch.tensor(Xm)   # (N, SEQ_LEN, 3)
        self.yp = torch.tensor(yp)   # (N, PRED_LEN)
        self.ym = torch.tensor(ym)   # (N, PRED_LEN, 3)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i): return self.Xf[i], self.Xm[i], self.yp[i], self.ym[i]


# ─────────────────────────────────────────────
# 5. ARCHITECTURE
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Capture les patterns locaux (court terme)."""
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
    Module d'attention sur les 3 granularités de momentum.

    Correction v2 : les poids sont calculés depuis les 3 signaux de momentum
    eux-mêmes (self-attention temporelle), combinés avec le contexte features.
    Temperature scaling pour éviter le collapse vers une seule granularité.

    Le modèle décide lui-même : "pour CE contexte de match,
    est-ce le momentum de point, de jeu ou de set qui est
    le plus informatif ?"
    """
    TEMPERATURE = 1.5   # > 1 → distribution plus uniforme, évite le collapse

    def __init__(self, d_model, n_gran=3):
        super().__init__()
        self.n_gran  = n_gran
        # Projection de chaque granularité individuellement vers d_model/4
        self.gran_proj = nn.Linear(1, d_model // 4)
        # Réseau d'attention : prend [mean(mom_pt), mean(mom_jeu), mean(mom_set)]
        # concaténé avec contexte features réduit → poids sur les 3 granularités
        self.attn_net = nn.Sequential(
            nn.Linear(n_gran + d_model // 4, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, n_gran),
        )
        # Projection finale vers d_model
        self.out_proj = nn.Linear(n_gran, d_model)
        self.ctx_proj = nn.Linear(d_model, d_model // 4)

    def forward(self, x_mom, context):
        """
        x_mom   : (B, T, 3) — momentum [point, jeu, set] à chaque pas
        context : (B, d_model) — vecteur de contexte issu des features
        Retourne : (B, T, d_model), attn_weights (B, 3)
        """
        # Signal de chaque granularité résumé sur T : (B, 3)
        mom_summary = x_mom.mean(dim=1)           # moyenne temporelle des 3 mom

        # Contexte features réduit : (B, d_model//4)
        ctx_reduced = self.ctx_proj(context)       # (B, d_model//4)

        # Concaténation : signaux momentum + contexte features → scores
        scores = self.attn_net(
            torch.cat([mom_summary, ctx_reduced], dim=-1)
        )                                          # (B, 3)

        # Temperature scaling pour éviter le collapse
        attn_weights = torch.softmax(scores / self.TEMPERATURE, dim=-1)  # (B, 3)

        # Pondération des 3 granularités à chaque pas de temps
        # x_mom : (B, T, 3) * attn_weights (B, 1, 3) → (B, T, 3)
        weighted = x_mom * attn_weights.unsqueeze(1)   # (B, T, 3)

        # Projection vers d_model
        out = self.out_proj(weighted)              # (B, T, d_model)
        return out, attn_weights


class HydraEncoderMG(nn.Module):
    """
    Encoder Multi-Granularité.

    Différence vs niveau 4 :
      - L'entrée momentum est (B, T, 3) au lieu de (B, T, 1)
      - Un module GranularityAttention apprend à fusionner les 3 niveaux
      - La représentation du momentum est apprise, pas juste concaténée
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        # Projection features brutes
        self.feat_proj  = nn.Linear(feat_size, d_model)
        # Module d'attention sur les granularités
        self.gran_attn  = GranularityAttention(d_model, n_gran=3)
        # Fusion features + momentum pondéré → d_model
        self.fusion_in  = nn.Linear(d_model * 2, d_model)

        # Branches multi-scale
        self.conv_block = ConvBlock(d_model, d_model, conv_k)
        self.lstm       = nn.LSTM(d_model, d_model, num_lstm_layers, batch_first=True,
                                   dropout=dropout if num_lstm_layers > 1 else 0)
        self.pos_enc    = PositionalEncoding(d_model)
        enc_layer       = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                                      dropout, batch_first=True, norm_first=True)
        self.attention  = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Fusion finale des 3 branches
        self.fusion_out = nn.Linear(d_model * 3, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x_feat, x_mom):
        """
        x_feat : (B, T, F)
        x_mom  : (B, T, 3)
        """
        # 1. Projection des features
        feat_emb = self.feat_proj(x_feat)   # (B, T, d_model)

        # 2. Attention sur les 3 granularités de momentum
        #    Le contexte utilisé est la moyenne temporelle des features
        context  = feat_emb.mean(dim=1)     # (B, d_model)
        mom_emb, gran_weights = self.gran_attn(x_mom, context)  # (B, T, d_model)

        # 3. Fusion features + momentum
        x = self.fusion_in(torch.cat([feat_emb, mom_emb], dim=-1))  # (B, T, d_model)

        # 4. Branches multi-scale
        conv_out             = self.conv_block(x)
        lstm_out, (h, c)     = self.lstm(x)
        attn_out             = self.attention(self.pos_enc(x))

        # 5. Fusion des derniers états des 3 branches
        fused   = torch.cat([conv_out[:, -1], lstm_out[:, -1], attn_out[:, -1]], dim=-1)
        context = self.dropout(self.fusion_out(fused))   # (B, d_model)

        return context, h, c, gran_weights   # gran_weights pour analyse/visualisation


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
        return torch.cat(outputs, dim=1)   # (B, PRED_LEN, 2)


class HydraMomentumHead(nn.Module):
    """
    Prédit les 3 granularités de momentum simultanément.
    Output : (B, PRED_LEN, 3) — [mom_point, mom_game, mom_set] futurs
    """
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
            pred_m = self.fc(out.squeeze(1))   # (B, 3)
            outputs.append(pred_m.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < tf_ratio:
                x_t = targets[:, t, :].unsqueeze(1)   # (B, 1, 3)
            else:
                x_t = pred_m.unsqueeze(1)
        return torch.cat(outputs, dim=1)   # (B, PRED_LEN, 3)


class HydraNetMG(nn.Module):
    """
    HydraNet Multi-Granularité — architecture complète.
    """
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
def train_epoch(model, loader, optimizer, ce_loss, mse_loss):
    model.train()
    total_loss = 0
    for Xf, Xm, yp, ym in loader:
        Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mom_preds, gran_w = model(Xf, Xm, PRED_LEN, yp, ym, tf_ratio=0.5)
        # Loss points (classification) + loss momentum (régression sur 3 granularités)
        loss_pts = ce_loss(logits.view(-1, 2), yp.view(-1))
        loss_mom = mse_loss(mom_preds, ym)
        # Entropy reg : pénalise les distributions trop concentrées
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
    mp_all, mt_all = [], []
    gran_weights_all = []

    with torch.no_grad():
        for Xf, Xm, yp, ym in loader:
            Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
            logits, mom_preds, gran_w = model(Xf, Xm, PRED_LEN, tf_ratio=0.0)

            loss = ce_loss(logits.view(-1, 2), yp.view(-1)) + LAMBDA_MOM * mse_loss(mom_preds, ym)
            total_loss += loss.item()

            preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
            labels_all.extend(yp.cpu().numpy().flatten())
            mp_all.append(mom_preds.cpu().numpy())
            mt_all.append(ym.cpu().numpy())
            gran_weights_all.append(gran_w.cpu().numpy())

    acc      = accuracy_score(labels_all, preds_all)
    mp_all   = np.concatenate(mp_all, axis=0)
    mt_all   = np.concatenate(mt_all, axis=0)
    mse_pt   = mean_squared_error(mt_all[:,:,0].flatten(), mp_all[:,:,0].flatten())
    mse_game = mean_squared_error(mt_all[:,:,1].flatten(), mp_all[:,:,1].flatten())
    mse_set  = mean_squared_error(mt_all[:,:,2].flatten(), mp_all[:,:,2].flatten())

    # Poids d'attention moyens sur la validation (intéressant pour le mémoire !)
    gran_mean = np.concatenate(gran_weights_all, axis=0).mean(axis=0)

    return (total_loss / len(loader), acc,
            mse_pt, mse_game, mse_set,
            gran_mean)


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + calcul momentum multi-granularité...")
    df, scaler = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    print("\n🔨 Construction des séquences...")
    X_feat, X_mom, y_points, y_mom = build_sequences(df)
    print(f"   X_feat : {X_feat.shape}")
    print(f"   X_mom  : {X_mom.shape}   ← 3 granularités par pas de temps")
    print(f"   y_mom  : {y_mom.shape}   ← 3 granularités à prédire")

    split    = int(0.8 * len(X_feat))
    train_ds = TennisDataset(X_feat[:split], X_mom[:split], y_points[:split], y_mom[:split])
    val_ds   = TennisDataset(X_feat[split:], X_mom[split:], y_points[split:], y_mom[split:])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=(DEVICE=="cuda"))

    model     = HydraNetMG(len(FEATURE_COLS), D_MODEL, N_HEADS,
                            NUM_LSTM_LAYERS, CONV_KERNEL, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 HydraNet Multi-Granularité sur {DEVICE}  |  {n_params:,} paramètres\n")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss)
        val_loss, val_acc, mse_pt, mse_game, mse_set, gran_w = evaluate(
            model, val_loader, ce_loss, mse_loss)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"loss={train_loss:.4f} | acc={val_acc:.4f} | "
              f"MSE[pt={mse_pt:.4f} game={mse_game:.4f} set={mse_set:.4f}] | "
              f"GranW[pt={gran_w[0]:.2f} game={gran_w[1]:.2f} set={gran_w[2]:.2f}]")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "config": {
                "feat_size": len(FEATURE_COLS), "d_model": D_MODEL,
                "n_heads": N_HEADS, "num_lstm_layers": NUM_LSTM_LAYERS,
                "conv_k": CONV_KERNEL, "dropout": DROPOUT,
            }}, "best_hydranet_mg.pt")
            joblib.dump(scaler, "scaler_mg.pkl")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("\n📊 Poids de granularité moyens sur la validation :")
    print(f"   Point : {gran_w[0]:.3f}  |  Jeu : {gran_w[1]:.3f}  |  Set : {gran_w[2]:.3f}")
    print("   → La granularité avec le poids le plus élevé est celle que le modèle")
    print("     juge la plus informative pour prédire les points futurs.")
    print("\n   Checkpoint sauvegardé dans best_hydranet_mg.pt")