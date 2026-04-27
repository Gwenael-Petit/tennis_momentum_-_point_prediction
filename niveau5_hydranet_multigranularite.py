"""
NIVEAU 5 v4 - HydraNet à 3 Signaux Hétérogènes
================================================
Rupture fondamentale avec v1/v2/v3 :

Les 3 versions précédentes utilisaient 3 fenêtres glissantes sur le même
signal binaire Y (victoire du point). Après normalisation, les 3 canaux
deviennent quasi-identiques → l'attention n'a rien à discriminer → 0.33.

v4 remplace les 3 granularités temporelles par 3 SIGNAUX DE NATURE DIFFÉRENTE :

  ┌──────────────────────────────────────────────────────────────────────┐
  │  [S1] mom_psycho  : momentum classique sur les points (haute fréq.)  │
  │       → "Est-il dans une bonne série de points ?"                    │
  │       Dynamique : fluctue point par point, très bruité               │
  │                                                                      │
  │  [S2] mom_tactical: efficacité sur les moments clés (basse fréq.)    │
  │       → "Gagne-t-il quand ça compte ? (break pts, deuce, AD)"        │
  │       Dynamique : événementielle, longues plages à 0 entre les       │
  │                   points importants → structurellement différent      │
  │                                                                      │
  │  [S3] mom_physical: tendance physique (monotone lente)               │
  │       → "Sa forme physique se dégrade-t-elle ?"                      │
  │       Dynamique : tendance lente sur serve_speed + unf_err +         │
  │                   distance_run → quasi-monotone intra-set            │
  └──────────────────────────────────────────────────────────────────────┘

Ces 3 signaux ont des DYNAMIQUES TEMPORELLES structurellement différentes.
Les GRU dédiés de GranularityAttention peuvent enfin les distinguer.

L'attention apprend alors : dans ce contexte de match,
  - un bris de dynamique physique → S3 dominant
  - un joueur qui perd tous ses break points → S2 dominant
  - une série de 5 points consécutifs → S1 dominant

Conservé de v3 : GRU indépendants, skip connection, teacher forcing
décroissant, temperature apprise, split match_id, scaler train-only.
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
CONV_KERNEL     = 5
EPOCHS          = 30
LR              = 1e-4
DROPOUT         = 0.2
LAMBDA_MOM      = 0.3        # réduit car les 3 signaux sont + difficiles à prédire
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Paramètres S1 (psychologique)
MOM_PSY_WINDOW  = 8
MOM_PSY_DECAY   = 0.85

# Paramètres S2 (tactique) — fenêtre en nombre de POINTS IMPORTANTS
MOM_TAC_WINDOW  = 6          # 6 derniers points "importants"
MOM_TAC_DECAY   = 0.80

# Paramètres S3 (physique) — fenêtre glissante sur les stats physiques
MOM_PHY_WINDOW  = 12         # plus longue : tendance lente

MOM_CLIP        = 3.0        # clip au lieu de normalisation (préserve le contraste)

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
N_GRAN     = 3
GRU_DIM    = D_MODEL // 4


# ─────────────────────────────────────────────
# 2. CALCUL DES 3 SIGNAUX HÉTÉROGÈNES
# ─────────────────────────────────────────────

def compute_signal_psycho(results: np.ndarray, window: int, decay: float) -> np.ndarray:
    """
    S1 — Momentum psychologique.
    Fenêtre glissante pondérée sur les victoires/défaites de points.
    Signal : haute fréquence, très bruité, fluctue point par point.
    Inchangé vs v1 : c'est le signal de référence.
    """
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()
    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start        = max(0, t - window + 1)
        w            = weights[-(t - start + 1):]
        momentum[t]  = np.dot(signed[start : t + 1], w)
    return momentum.astype(np.float32)


def _is_pressure_point(s1: int, s2: int) -> bool:
    """
    Retourne True si le score courant est un "moment clé" :
    break point, deuce, AD, ou score serré en fin de jeu (40-30, 30-40).
    s1, s2 : scores encodés (0=0, 1=15, 2=30, 3=40, 4=AD)
    """
    # Deuce / Avantage
    if s1 >= 3 and s2 >= 3:
        return True
    # Break point pour le relanceur (serveur à 40, relanceur à 30 ou 40)
    if s1 == 3 and s2 >= 2:
        return True
    if s2 == 3 and s1 >= 2:
        return True
    return False


def compute_signal_tactical(df_match: pd.DataFrame, window: int, decay: float) -> np.ndarray:
    """
    S2 — Momentum tactique.
    Ne s'actualise QUE sur les points importants (break pts, deuce, AD).
    Entre deux points importants, la valeur reste constante (plateau).

    Dynamique temporelle : événementielle, avec de longues plages plates
    entre les moments clés. Structurellement différent de S1 (qui varie
    à chaque point) et de S3 (qui est monotone).

    +1 si J1 gagne le point important, -1 sinon.
    Pondération exponentielle sur les `window` derniers points importants.
    """
    p1_scores = df_match["p1_score"].values
    p2_scores = df_match["p2_score"].values
    results   = df_match[TARGET_COL].values

    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()

    key_results  = []   # historique des résultats sur points importants
    momentum     = np.zeros(len(df_match))
    last_mom     = 0.0

    for t in range(len(df_match)):
        s1 = int(p1_scores[t]) if not np.isnan(p1_scores[t]) else 0
        s2 = int(p2_scores[t]) if not np.isnan(p2_scores[t]) else 0

        if _is_pressure_point(s1, s2):
            key_results.append(1.0 if results[t] == 1 else -1.0)
            recent   = key_results[-window:]
            w        = weights[-len(recent):]
            last_mom = float(np.dot(recent, w))

        momentum[t] = last_mom

    return momentum.astype(np.float32)


def compute_signal_physical(df_match: pd.DataFrame, window: int) -> np.ndarray:
    """
    S3 — Momentum physique.
    Capture la TENDANCE de dégradation/amélioration physique relative de J1
    par rapport à J2 sur une fenêtre glissante.

    Composantes normalisées et combinées :
      - serve_speed_diff  = p1_serve_speed - p2_serve_speed
        (vitesse de service : diminue avec la fatigue)
      - unf_err_diff      = p2_unf_err - p1_unf_err
        (erreurs non forcées : augmentent avec la fatigue — signe inversé)
      - distance_diff     = p1_distance_run - p2_distance_run
        (distance courue : signal ambigu — peut indiquer défense forcée)

    On calcule la TENDANCE (slope) de la somme pondérée sur la fenêtre,
    pas juste la moyenne : ce qui compte c'est si ça se dégrade ou s'améliore.

    Dynamique : quasi-monotone intra-set, basse fréquence.
    """
    serve_diff = (df_match["p1_serve_speed"].fillna(0).values
                  - df_match["p2_serve_speed"].fillna(0).values)
    err_diff   = (df_match["p2_unf_err"].fillna(0).values
                  - df_match["p1_unf_err"].fillna(0).values)
    dist_diff  = (df_match["p1_distance_run"].fillna(0).values
                  - df_match["p2_distance_run"].fillna(0).values)

    # Normalisation locale (évite les problèmes d'échelle entre les 3 composantes)
    def _norm(x):
        std = x.std()
        return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else np.zeros_like(x)

    composite  = _norm(serve_diff) + _norm(err_diff) + _norm(dist_diff)

    # Tendance (slope) sur fenêtre glissante via régression linéaire simple
    momentum = np.zeros(len(composite))
    for t in range(len(composite)):
        start = max(0, t - window + 1)
        seg   = composite[start : t + 1]
        if len(seg) < 3:
            momentum[t] = 0.0
        else:
            # Slope normalisée de la régression linéaire
            x     = np.arange(len(seg), dtype=np.float32)
            x    -= x.mean()
            denom = (x * x).sum()
            if denom > 1e-8:
                momentum[t] = float((x * seg).sum() / denom)
            else:
                momentum[t] = 0.0

    return momentum.astype(np.float32)


# ─────────────────────────────────────────────
# 3. CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Charge le CSV et calcule les 3 signaux hétérogènes BRUTS (clippés).
    La normalisation des features est faite après le split train/val.
    Les signaux momentum sont clippés à [-MOM_CLIP, MOM_CLIP] sans normalisation.
    """
    df = pd.read_csv(csv_path, sep=",", low_memory=False)

    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    s1_list, s2_list, s3_list = [], [], []
    for _, match in df.groupby("match_id"):
        match = match.sort_values("point_no").reset_index(drop=True)
        s1_list.extend(
            compute_signal_psycho(match[TARGET_COL].values,
                                  MOM_PSY_WINDOW, MOM_PSY_DECAY).tolist())
        s2_list.extend(
            compute_signal_tactical(match, MOM_TAC_WINDOW, MOM_TAC_DECAY).tolist())
        s3_list.extend(
            compute_signal_physical(match, MOM_PHY_WINDOW).tolist())

    df["mom_psycho"]   = np.clip(s1_list, -MOM_CLIP, MOM_CLIP)
    df["mom_tactical"] = np.clip(s2_list, -MOM_CLIP, MOM_CLIP)
    df["mom_physical"] = np.clip(s3_list, -MOM_CLIP, MOM_CLIP)

    print(f"   mom_psycho   — mean={df['mom_psycho'].mean():.3f}  "
          f"std={df['mom_psycho'].std():.3f}  "
          f"(fréq. élevée : varie chaque point)")
    print(f"   mom_tactical — mean={df['mom_tactical'].mean():.3f}  "
          f"std={df['mom_tactical'].std():.3f}  "
          f"(événementiel : plateaux entre moments clés)")
    pct_nonzero = (df["mom_tactical"] != 0).mean() * 100
    print(f"                  {pct_nonzero:.1f}% de points non-nuls (moments clés)")
    print(f"   mom_physical — mean={df['mom_physical'].mean():.3f}  "
          f"std={df['mom_physical'].std():.3f}  "
          f"(tendance lente : slope physique)")

    return df


MOM_COLS = ["mom_psycho", "mom_tactical", "mom_physical"]


def fit_and_apply_scalers(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Fit sur train uniquement, applique sur train+val."""
    feat_scaler = StandardScaler()
    df_train[FEATURE_COLS] = feat_scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = feat_scaler.transform(df_val[FEATURE_COLS])
    # Momentum : PAS de normalisation, contraste naturel préservé
    return df_train, df_val, feat_scaler


# ─────────────────────────────────────────────
# 4. CONSTRUCTION DES SÉQUENCES
# ─────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    X_feat, X_mom, y_points, y_mom = [], [], [], []
    for _, match in df.groupby("match_id"):
        match    = match.sort_values("point_no").reset_index(drop=True)
        features = match[FEATURE_COLS].values
        points   = match[TARGET_COL].values
        moms     = match[MOM_COLS].values

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
# 5. ARCHITECTURE
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
    Attention sur les 3 signaux hétérogènes avec GRU indépendants.

    Chaque signal a son GRU dédié qui encode sa dynamique temporelle propre :
      - GRU_psycho  encode les fluctuations rapides point-à-point
      - GRU_tactical encode les plateaux et sauts événementiels
      - GRU_physical encode la tendance monotone lente

    Ces 3 dynamiques sont structurellement différentes → les GRU produisent
    des représentations réellement distinctes → l'attention peut discriminer.

    Query = contexte LSTM principal (dépend du contexte du match).
    Score = compatibilité additive Bahdanau entre chaque représentation et query.
    """
    def __init__(self, d_model, n_gran=N_GRAN, gru_dim=GRU_DIM):
        super().__init__()
        self.n_gran  = n_gran
        self.gru_dim = gru_dim

        self.gran_grus = nn.ModuleList([
            nn.GRU(1, gru_dim, num_layers=1, batch_first=True)
            for _ in range(n_gran)
        ])

        self.query_proj  = nn.Linear(d_model, gru_dim)
        self.score_proj  = nn.Linear(gru_dim, 1)
        self.out_proj    = nn.Linear(gru_dim, d_model)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x_mom, context):
        """
        x_mom   : (B, T, 3)    — 3 signaux hétérogènes sur la fenêtre
        context : (B, d_model) — contexte GRU pré-calculé (query)
        """
        B, T, _ = x_mom.shape

        # Encoder chaque signal avec son GRU dédié
        gran_reps = []
        for i, gru in enumerate(self.gran_grus):
            _, h = gru(x_mom[:, :, i:i+1])       # h : (1, B, gru_dim)
            gran_reps.append(h.squeeze(0))         # (B, gru_dim)
        gran_reps = torch.stack(gran_reps, dim=1)  # (B, 3, gru_dim)

        # Query depuis le contexte
        query = self.query_proj(context).unsqueeze(1)      # (B, 1, gru_dim)

        # Score additif Bahdanau
        combined = torch.tanh(gran_reps + query)           # (B, 3, gru_dim)
        scores   = self.score_proj(combined).squeeze(-1)   # (B, 3)

        temp         = self.log_temperature.exp().clamp(min=0.1)
        attn_weights = torch.softmax(scores / temp, dim=-1)  # (B, 3)

        # Représentation pondérée → projection → broadcast sur T
        weighted_rep = (gran_reps * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, gru_dim)
        out = self.out_proj(weighted_rep).unsqueeze(1).expand(-1, T, -1)    # (B, T, d_model)

        return out, attn_weights


class HydraEncoderMG(nn.Module):
    """
    Encoder avec GRU pré-contexte + GranularityAttention + skip connection.
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout,
                 n_gran=N_GRAN, gru_dim=GRU_DIM):
        super().__init__()
        self.feat_proj  = nn.Linear(feat_size, d_model)

        # GRU léger pour pré-contexte (query de l'attention)
        self.pre_gru    = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

        self.gran_attn  = GranularityAttention(d_model, n_gran=n_gran, gru_dim=gru_dim)

        # Skip connection : feat_emb (d) + mom_emb (d) + mom bruts (3)
        self.fusion_in  = nn.Linear(d_model + d_model + n_gran, d_model)

        # Branches multi-scale
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
        feat_emb = self.feat_proj(x_feat)                  # (B, T, d_model)

        # Pré-contexte via GRU
        _, h_pre    = self.pre_gru(feat_emb)
        pre_context = h_pre.squeeze(0)                     # (B, d_model)

        # Attention sur les 3 signaux hétérogènes
        mom_emb, gran_weights = self.gran_attn(x_mom, pre_context)  # (B, T, d_model)

        # Fusion avec skip connection des signaux bruts
        x = self.fusion_in(torch.cat([feat_emb, mom_emb, x_mom], dim=-1))  # (B, T, d_model)

        # Branches multi-scale
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
    def __init__(self, d_model, num_layers, dropout, n_gran=N_GRAN):
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
        self.momentum_head = HydraMomentumHead(d_model, num_lstm_layers, dropout)

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
    return start + (end - start) * (epoch - 1) / max(total - 1, 1)


def train_epoch(model, loader, optimizer, ce_loss, mse_loss, tf_ratio):
    model.train()
    total_loss = 0
    for Xf, Xm, yp, ym in loader:
        Xf, Xm, yp, ym = Xf.to(DEVICE), Xm.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mom_preds, gran_w = model(Xf, Xm, PRED_LEN, yp, ym, tf_ratio=tf_ratio)
        loss_pts = ce_loss(logits.view(-1, 2), yp.view(-1))
        loss_mom = mse_loss(mom_preds, ym)
        # Régularisation entropie : encourage la diversification des poids
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

    acc      = accuracy_score(labels_all, preds_all)
    mp_all   = np.concatenate(mp_all, axis=0)
    mt_all   = np.concatenate(mt_all, axis=0)
    mse_psy  = mean_squared_error(mt_all[:,:,0].flatten(), mp_all[:,:,0].flatten())
    mse_tac  = mean_squared_error(mt_all[:,:,1].flatten(), mp_all[:,:,1].flatten())
    mse_phy  = mean_squared_error(mt_all[:,:,2].flatten(), mp_all[:,:,2].flatten())
    gran_mean = np.concatenate(gran_weights_all, axis=0).mean(axis=0)

    return total_loss / len(loader), acc, mse_psy, mse_tac, mse_phy, gran_mean


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + calcul des 3 signaux hétérogènes (v4)...")
    df = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    # Split sur match_id avant normalisation
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

    print("\n🔧 Normalisation features (train only) — signaux momentum bruts conservés...")
    df_train, df_val, feat_scaler = fit_and_apply_scalers(df_train, df_val)

    print("\n🔨 Construction des séquences...")
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
    print(f"\n🚀 HydraNet MG v4 sur {DEVICE}  |  {n_params:,} paramètres")
    print(f"   Signaux : [S1] psycho (haute fréq.)  "
          f"[S2] tactical (événementiel)  [S3] physical (tendance lente)\n")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tf         = get_tf_ratio(epoch, EPOCHS)
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss, tf)
        val_loss, val_acc, mse_psy, mse_tac, mse_phy, gran_w = evaluate(
            model, val_loader, ce_loss, mse_loss)
        scheduler.step()

        temp = model.encoder.gran_attn.log_temperature.exp().item()

        print(f"Epoch {epoch:02d}/{EPOCHS} | tf={tf:.2f} | "
              f"loss={train_loss:.4f} | acc={val_acc:.4f} | "
              f"MSE[psy={mse_psy:.4f} tac={mse_tac:.4f} phy={mse_phy:.4f}] | "
              f"GranW[psy={gran_w[0]:.2f} tac={gran_w[1]:.2f} phy={gran_w[2]:.2f}] | "
              f"T={temp:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "feat_size": len(FEATURE_COLS), "d_model": D_MODEL,
                    "n_heads": N_HEADS, "num_lstm_layers": NUM_LSTM_LAYERS,
                    "conv_k": CONV_KERNEL, "dropout": DROPOUT,
                },
            }, "best_hydranet_mg_v4.pt")
            joblib.dump(feat_scaler, "feat_scaler_mg_v4.pkl")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("\n📊 Poids de granularité moyens (dernière epoch val) :")
    print(f"   Psycho   : {gran_w[0]:.3f}")
    print(f"   Tactical : {gran_w[1]:.3f}")
    print(f"   Physical : {gran_w[2]:.3f}")
    print(f"   Temperature finale : {model.encoder.gran_attn.log_temperature.exp().item():.3f}")
    print("\n   Interprétation :")
    print("   → Si psy >> autres : le run de points courant est dominant")
    print("   → Si tac >> autres : les moments clés (break pts, deuce) sont dominants")
    print("   → Si phy >> autres : la forme physique relative est dominante")
    print("\n   Checkpoint : best_hydranet_mg_v4.pt")