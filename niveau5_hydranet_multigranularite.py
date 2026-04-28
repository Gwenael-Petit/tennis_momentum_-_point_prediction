"""
NIVEAU 5 v6 - HydraNet Multi-Granularité (v5 + tous les leviers d'accuracy)
============================================================================
Base : v5 (architecture originale + 3 fixes de stabilité)

3 leviers d'accuracy combinés :

  [LEV 1] Features d'interaction tennis (≈+0.5-1% accuracy attendu)
    Le modèle v5 devait apprendre seul que p1_score=3 ET p2_score=3 = deuce.
    On lui donne directement ces features explicites :
      - is_deuce           : score >= 40-40
      - is_break_point     : break point pour le relanceur
      - score_pressure     : p1_score * p2_score (interaction continue)
      - serve_dominance    : ace - double_fault par joueur
      - winner_ratio       : winners / (winners + unf_err) par joueur
      - fatigue_proxy      : distance_run * point_no_in_game (accumulation)
      - momentum_x_score   : mom_point * p1_score (interaction momentum×score)
    → 10 features supplémentaires, total 35 features temporelles

  [LEV 2] SEQ_LEN 10→20 (≈+0.3-0.7% accuracy attendu)
    Avec SEQ_LEN=10, le modèle voyait ≈1.5 jeux.
    Avec SEQ_LEN=20, il voit ≈3 jeux → peut capter les vraies séries de jeux
    et les retournements de situation au niveau du set.
    → Plus de séquences valides par match (window plus large)

  [LEV 3] Contexte match statique injecté dans le context vector (≈+0.5-1%)
    Le modèle ne savait pas sur quelle surface on joue, ni les rankings
    relatifs des joueurs. Ces infos sont constantes par match et prédisent
    structurellement le style de jeu et les probabilités de victoire.
    Features statiques (par match, constantes sur toute la séquence) :
      - surface encodée (hard/clay/grass/carpet → embedding 4D)
      - ranking_diff = p1_rank - p2_rank (normalisé)
      - ranking_ratio = p1_rank / (p1_rank + p2_rank)
      - best_of (3 ou 5 sets)
    → Injectées via un StaticContextNet → vecteur concaténé au context final

  Architecture :
    HydraEncoderMG (inchangée v5)
      ↓ context (B, d_model)
    StaticContextNet(static_feats) → static_ctx (B, d_model//4)
    concat([context, static_ctx]) → Linear → final_context (B, d_model)
      ↓
    HydraPointHead + HydraMomentumHead (inchangés)
"""

import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN         = 20          # [LEV 2] 10 → 20
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

# Features temporelles de base (identiques à v5)
BASE_FEATURE_COLS = [
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

# [LEV 1] Features d'interaction calculées dans load_and_prepare
INTERACTION_COLS = [
    "is_deuce",
    "is_break_point_p1",   # break point pour J1 (J2 sert)
    "is_break_point_p2",   # break point pour J2 (J1 sert)
    "score_pressure",      # p1_score * p2_score
    "p1_serve_dominance",  # ace - double_fault
    "p2_serve_dominance",
    "p1_winner_ratio",     # winners / (winners + unf_err + 1)
    "p2_winner_ratio",
    "mom_x_score",         # mom_point * p1_score (interaction momentum×score)
    "score_advantage",     # |p1_score - p2_score| (tension du point)
]

FEATURE_COLS = BASE_FEATURE_COLS + INTERACTION_COLS  # 35 features au total

# [LEV 3] Features statiques du match
STATIC_COLS = [
    "surface_enc",          # surface encodée en entier (→ embedding)
    "ranking_diff",         # p1_rank - p2_rank normalisé
    "ranking_ratio",        # p1_rank / (p1_rank + p2_rank)
    "best_of",              # 3 ou 5 sets
]
N_SURFACES  = 5            # hard, clay, grass, carpet, unknown

TARGET_COL  = "Y"


# ─────────────────────────────────────────────
# 2. CALCUL DU MOMENTUM (identique v5)
# ─────────────────────────────────────────────
def compute_momentum_points(results, window, decay):
    signed  = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()
    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start       = max(0, t - window + 1)
        w           = weights[-(t - start + 1):]
        momentum[t] = np.dot(signed[start:t+1], w)
    return momentum.astype(np.float32)


def compute_momentum_games(df_match, window, decay):
    momentum     = np.zeros(len(df_match))
    game_results = {}
    for (s, g), grp in df_match.groupby(["set_no", "game_no"]):
        victor = grp["game_victor"].dropna()
        if len(victor) > 0:
            game_results[(s, g)] = 1.0 if victor.iloc[-1] == 1 else -1.0

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
        if not recent:
            momentum[idx] = 0.0
        else:
            vals = np.array([game_results[k] for k in recent])
            w    = weights[-len(vals):]
            momentum[idx] = np.dot(vals, w)
    return momentum.astype(np.float32)


def compute_momentum_sets(df_match, window, decay):
    momentum    = np.zeros(len(df_match))
    set_results = {}
    for s, grp in df_match.groupby("set_no"):
        victor = grp["set_victor"].dropna()
        if len(victor) > 0:
            set_results[s] = 1.0 if victor.iloc[-1] == 1 else -1.0

    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()

    for idx, row in df_match.iterrows():
        s         = row["set_no"]
        past_sets = [ss for ss in range(1, s) if ss in set_results]
        recent    = past_sets[-window:]
        if not recent:
            momentum[idx] = 0.0
        else:
            vals = np.array([set_results[ss] for ss in recent])
            w    = weights[-len(vals):]
            momentum[idx] = np.dot(vals, w)
    return momentum.astype(np.float32)


# ─────────────────────────────────────────────
# 3. FEATURES D'INTERACTION [LEV 1]
# ─────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [LEV 1] Ajoute 10 features d'interaction explicites.
    Appelé AVANT la normalisation (sur les valeurs brutes).
    """
    s1 = df["p1_score"].fillna(0).astype(int)
    s2 = df["p2_score"].fillna(0).astype(int)

    # Deuce : les deux joueurs à 40+ (encodé ≥ 3)
    df["is_deuce"] = ((s1 >= 3) & (s2 >= 3)).astype(np.float32)

    # Break point pour J1 : J2 sert (p2_serve=1), J1 à 40 ou AD, J2 derrière
    df["is_break_point_p1"] = (
        (df["p2_serve"].fillna(0) == 1) & (s1 >= 3) & (s2 < s1)
    ).astype(np.float32)

    # Break point pour J2 : J1 sert (p1_serve=1), J2 à 40 ou AD, J1 derrière
    df["is_break_point_p2"] = (
        (df["p1_serve"].fillna(0) == 1) & (s2 >= 3) & (s1 < s2)
    ).astype(np.float32)

    # Interaction continue score (tension du jeu)
    df["score_pressure"]  = (s1 * s2).astype(np.float32)
    df["score_advantage"] = (s1 - s2).abs().astype(np.float32)

    # Dominance au service
    df["p1_serve_dominance"] = (
        df["p1_ace"].fillna(0) - df["p1_double_fault"].fillna(0)
    ).astype(np.float32)
    df["p2_serve_dominance"] = (
        df["p2_ace"].fillna(0) - df["p2_double_fault"].fillna(0)
    ).astype(np.float32)

    # Ratio winner (efficacité offensive)
    df["p1_winner_ratio"] = (
        df["p1_winner"].fillna(0) /
        (df["p1_winner"].fillna(0) + df["p1_unf_err"].fillna(0) + 1)
    ).astype(np.float32)
    df["p2_winner_ratio"] = (
        df["p2_winner"].fillna(0) /
        (df["p2_winner"].fillna(0) + df["p2_unf_err"].fillna(0) + 1)
    ).astype(np.float32)

    # Interaction momentum × score (sera calculée après le momentum)
    # Initialisée à 0, remplie dans load_and_prepare après le calcul momentum
    df["mom_x_score"] = 0.0

    return df


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [LEV 3] Ajoute les features statiques du match (constantes par match_id).
    surface → encodée ordinalement (0=unknown, 1=hard, 2=clay, 3=grass, 4=carpet)
    ranking_diff, ranking_ratio → normalisées plus bas
    best_of → 3 ou 5, normalisé
    """
    surface_map = {"hard": 1, "clay": 2, "grass": 3, "carpet": 4,
                   "Hard": 1, "Clay": 2, "Grass": 3, "Carpet": 4}

    if "surface" in df.columns:
        df["surface_enc"] = df["surface"].map(surface_map).fillna(0).astype(np.float32)
    else:
        df["surface_enc"] = 0.0

    if "p1_rank" in df.columns and "p2_rank" in df.columns:
        r1 = df["p1_rank"].fillna(200).astype(float)
        r2 = df["p2_rank"].fillna(200).astype(float)
        df["ranking_diff"]  = (r1 - r2).astype(np.float32)
        df["ranking_ratio"] = (r1 / (r1 + r2 + 1e-8)).astype(np.float32)
    else:
        df["ranking_diff"]  = 0.0
        df["ranking_ratio"] = 0.5

    if "best_of" in df.columns:
        df["best_of"] = df["best_of"].fillna(3).astype(np.float32) / 5.0
    else:
        df["best_of"] = 0.6   # 3/5 par défaut

    return df


# ─────────────────────────────────────────────
# 4. CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=",", low_memory=False)

    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)

    # [LEV 1] Features d'interaction (avant dropna pour avoir les scores)
    df = add_interaction_features(df)
    # [LEV 3] Features statiques
    df = add_static_features(df)

    df = df.dropna(subset=BASE_FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    # Calcul momentum par match
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

    # [LEV 1] Interaction momentum × score (maintenant que le momentum est calculé)
    df["mom_x_score"] = (df["mom_point"] * df["p1_score"].fillna(0)).astype(np.float32)

    print(f"   Features temporelles : {len(FEATURE_COLS)} "
          f"(base={len(BASE_FEATURE_COLS)} + interaction={len(INTERACTION_COLS)})")
    print(f"   Features statiques   : {len(STATIC_COLS)}")
    print(f"   mom_point — mean={df['mom_point'].mean():.3f}  std={df['mom_point'].std():.3f}")
    print(f"   mom_game  — mean={df['mom_game'].mean():.3f}  std={df['mom_game'].std():.3f}")
    print(f"   mom_set   — mean={df['mom_set'].mean():.3f}  std={df['mom_set'].std():.3f}")

    return df


def fit_and_apply_scalers(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Fit sur train uniquement (FIX 2 de v5, conservé)."""
    feat_scaler = StandardScaler()
    df_train[FEATURE_COLS] = feat_scaler.fit_transform(
        df_train[FEATURE_COLS].fillna(0))
    df_val[FEATURE_COLS]   = feat_scaler.transform(
        df_val[FEATURE_COLS].fillna(0))

    mom_scaler = StandardScaler()
    df_train[["mom_point", "mom_game", "mom_set"]] = mom_scaler.fit_transform(
        df_train[["mom_point", "mom_game", "mom_set"]])
    df_val[["mom_point", "mom_game", "mom_set"]] = mom_scaler.transform(
        df_val[["mom_point", "mom_game", "mom_set"]])

    # [LEV 3] Normalisation features statiques (sur train)
    static_scaler = StandardScaler()
    df_train[["ranking_diff", "ranking_ratio", "best_of"]] = \
        static_scaler.fit_transform(
            df_train[["ranking_diff", "ranking_ratio", "best_of"]].fillna(0))
    df_val[["ranking_diff", "ranking_ratio", "best_of"]] = \
        static_scaler.transform(
            df_val[["ranking_diff", "ranking_ratio", "best_of"]].fillna(0))
    # surface_enc reste entier (→ embedding dans le modèle)

    return df_train, df_val, feat_scaler, mom_scaler, static_scaler


# ─────────────────────────────────────────────
# 5. CONSTRUCTION DES SÉQUENCES
# ─────────────────────────────────────────────
def build_sequences(df: pd.DataFrame):
    """
    Retourne aussi X_static : (N, 4) — features statiques par séquence.
    La surface est gardée comme entier pour l'embedding.
    """
    X_feat, X_mom, X_static, y_points, y_mom = [], [], [], [], []

    for _, match in df.groupby("match_id"):
        match    = match.sort_values("point_no").reset_index(drop=True)
        features = match[FEATURE_COLS].fillna(0).values          # (T, 35)
        points   = match[TARGET_COL].values                       # (T,)
        moms     = match[["mom_point", "mom_game", "mom_set"]].values  # (T, 3)
        # Features statiques : identiques pour toute la séquence
        static   = match[STATIC_COLS].fillna(0).values            # (T, 4)

        for i in range(len(match) - SEQ_LEN - PRED_LEN + 1):
            X_feat.append(features[i : i + SEQ_LEN])
            X_mom.append(moms[i : i + SEQ_LEN])
            X_static.append(static[i])                            # (4,) — constante
            y_points.append(points[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
            y_mom.append(moms[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])

    return (np.array(X_feat,    dtype=np.float32),
            np.array(X_mom,     dtype=np.float32),
            np.array(X_static,  dtype=np.float32),
            np.array(y_points,  dtype=np.int64),
            np.array(y_mom,     dtype=np.float32))


class TennisDataset(Dataset):
    def __init__(self, Xf, Xm, Xs, yp, ym):
        self.Xf = torch.tensor(Xf)
        self.Xm = torch.tensor(Xm)
        self.Xs = torch.tensor(Xs)
        self.yp = torch.tensor(yp)
        self.ym = torch.tensor(ym)
    def __len__(self): return len(self.yp)
    def __getitem__(self, i):
        return self.Xf[i], self.Xm[i], self.Xs[i], self.yp[i], self.ym[i]


# ─────────────────────────────────────────────
# 6. ARCHITECTURE
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
    """Architecture originale conservée (fait varier les GranW)."""
    def __init__(self, d_model, n_gran=3):
        super().__init__()
        self.proj     = nn.Linear(1, d_model)
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, n_gran),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_mom, context):
        TEMPERATURE  = 2.0
        attn_weights = torch.softmax(self.attn_net(context) / TEMPERATURE, dim=-1)
        weighted     = (x_mom * attn_weights.unsqueeze(1)).sum(dim=-1, keepdim=True)
        out          = self.proj(weighted)
        return self.out_proj(out), attn_weights


class StaticContextNet(nn.Module):
    """
    [LEV 3] Encode les features statiques du match en un vecteur de contexte.
    La surface est encodée via un embedding appris, les autres features
    sont projetées depuis leur valeur normalisée.
    """
    def __init__(self, n_surfaces, d_out):
        super().__init__()
        self.surface_emb = nn.Embedding(n_surfaces, 8)
        # surface_emb(8) + ranking_diff(1) + ranking_ratio(1) + best_of(1) = 11
        self.net = nn.Sequential(
            nn.Linear(11, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x_static):
        """
        x_static : (B, 4) — [surface_enc, ranking_diff, ranking_ratio, best_of]
        """
        surface_idx  = x_static[:, 0].long().clamp(0, N_SURFACES - 1)
        surface_emb  = self.surface_emb(surface_idx)          # (B, 8)
        other        = x_static[:, 1:]                        # (B, 3)
        combined     = torch.cat([surface_emb, other], dim=-1) # (B, 11)
        return self.net(combined)                              # (B, d_out)


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
        feat_emb = self.feat_proj(x_feat)
        context  = feat_emb.mean(dim=1)
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
    """
    HydraNet v6 : encoder temporel + contexte statique match [LEV 3].

    Le context vector final est enrichi par StaticContextNet avant
    d'être passé aux têtes de prédiction.
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout,
                 n_surfaces=N_SURFACES):
        super().__init__()
        static_dim = d_model // 4   # 32

        self.encoder       = HydraEncoderMG(feat_size, d_model, n_heads,
                                             num_lstm_layers, conv_k, dropout)
        self.static_net    = StaticContextNet(n_surfaces, static_dim)  # [LEV 3]
        # Fusion context temporel + context statique
        self.ctx_fusion    = nn.Sequential(
            nn.Linear(d_model + static_dim, d_model),
            nn.ReLU(),
        )
        self.point_head    = HydraPointHead(d_model, num_lstm_layers, dropout)
        self.momentum_head = HydraMomentumHead(d_model, num_lstm_layers, dropout)

    def forward(self, x_feat, x_mom, x_static, pred_len,
                y_points=None, y_mom=None, tf_ratio=0.5):
        # Encoder temporel
        temp_ctx, h, c, gran_weights = self.encoder(x_feat, x_mom)

        # Contexte statique [LEV 3]
        static_ctx = self.static_net(x_static)                    # (B, static_dim)

        # Fusion : contexte enrichi
        final_ctx  = self.ctx_fusion(
            torch.cat([temp_ctx, static_ctx], dim=-1))             # (B, d_model)

        point_logits = self.point_head(final_ctx, h.clone(), c.clone(),
                                       pred_len, y_points, tf_ratio)
        mom_preds    = self.momentum_head(final_ctx, h.clone(), c.clone(),
                                          pred_len, y_mom, tf_ratio)
        return point_logits, mom_preds, gran_weights


# ─────────────────────────────────────────────
# 7. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
def get_tf_ratio(epoch, total, start=0.9, end=0.1):
    return start + (end - start) * (epoch - 1) / max(total - 1, 1)


def train_epoch(model, loader, optimizer, ce_loss, mse_loss, tf_ratio):
    model.train()
    total_loss = 0
    for Xf, Xm, Xs, yp, ym in loader:
        Xf  = Xf.to(DEVICE)
        Xm  = Xm.to(DEVICE)
        Xs  = Xs.to(DEVICE)
        yp  = yp.to(DEVICE)
        ym  = ym.to(DEVICE)
        optimizer.zero_grad()
        logits, mom_preds, gran_w = model(Xf, Xm, Xs, PRED_LEN, yp, ym, tf_ratio)
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
        for Xf, Xm, Xs, yp, ym in loader:
            Xf, Xm, Xs = Xf.to(DEVICE), Xm.to(DEVICE), Xs.to(DEVICE)
            yp, ym      = yp.to(DEVICE), ym.to(DEVICE)
            logits, mom_preds, gran_w = model(Xf, Xm, Xs, PRED_LEN, tf_ratio=0.0)
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
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + features d'interaction + momentum (v6)...")
    df = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    # Split sur match_id avant normalisation (FIX 1 de v5)
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

    print("\n🔧 Normalisation (train only)...")
    df_train, df_val, feat_scaler, mom_scaler, static_scaler = \
        fit_and_apply_scalers(df_train, df_val)

    print(f"\n🔨 Construction des séquences (SEQ_LEN={SEQ_LEN}) [LEV 2]...")
    X_feat_tr, X_mom_tr, X_static_tr, y_pts_tr, y_mom_tr = build_sequences(df_train)
    X_feat_vl, X_mom_vl, X_static_vl, y_pts_vl, y_mom_vl = build_sequences(df_val)
    print(f"   Train — X_feat={X_feat_tr.shape}  X_mom={X_mom_tr.shape}  "
          f"X_static={X_static_tr.shape}")
    print(f"   Val   — X_feat={X_feat_vl.shape}  X_mom={X_mom_vl.shape}  "
          f"X_static={X_static_vl.shape}")

    train_ds = TennisDataset(X_feat_tr, X_mom_tr, X_static_tr, y_pts_tr, y_mom_tr)
    val_ds   = TennisDataset(X_feat_vl, X_mom_vl, X_static_vl, y_pts_vl, y_mom_vl)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=(DEVICE=="cuda"))

    model = HydraNetMG(
        feat_size=len(FEATURE_COLS),
        d_model=D_MODEL, n_heads=N_HEADS,
        num_lstm_layers=NUM_LSTM_LAYERS,
        conv_k=CONV_KERNEL, dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 HydraNet MG v6 sur {DEVICE}  |  {n_params:,} paramètres")
    print(f"   [LEV 1] {len(FEATURE_COLS)} features temporelles "
          f"(+{len(INTERACTION_COLS)} interactions)")
    print(f"   [LEV 2] SEQ_LEN={SEQ_LEN}")
    print(f"   [LEV 3] Contexte statique match (surface + rankings + best_of)\n")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tf         = get_tf_ratio(epoch, EPOCHS)
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss, tf)
        val_loss, val_acc, mse_pt, mse_game, mse_set, gran_w = evaluate(
            model, val_loader, ce_loss, mse_loss)
        scheduler.step()

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
            }, "best_hydranet_mg_v6.pt")
            joblib.dump(feat_scaler,   "feat_scaler_mg_v6.pkl")
            joblib.dump(mom_scaler,    "mom_scaler_mg_v6.pkl")
            joblib.dump(static_scaler, "static_scaler_mg_v6.pkl")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("\n📊 Poids de granularité moyens (dernière epoch val) :")
    print(f"   Point : {gran_w[0]:.3f}  |  Jeu : {gran_w[1]:.3f}  |  Set : {gran_w[2]:.3f}")
    print("\n   Note : si ranking_diff/surface_enc sont à 0 (colonnes absentes du CSV),")
    print("   le LEV 3 n'apportera pas de gain — vérifier les noms de colonnes.")
    print("\n   Checkpoints : best_hydranet_mg_v6.pt + 3 scalers")