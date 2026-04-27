"""
NIVEAU 5 v2 - Momentum Multi-Granularité (HydraNet corrigé)
============================================================
Corrections apportées vs v1 :

  [FIX 1] GranW figés à 0.33
    → GranularityAttention reçoit maintenant le vrai contexte LSTM (pré-run)
      au lieu de la moyenne des features, ce qui lui donne un signal
      discriminant dépendant du contexte du match.
    → Les 3 momentum ne sont plus co-normalisés : chacun garde son contraste
      naturel (mom_pt ≈ ±1 réactif, mom_game plus lissé, mom_set très plat),
      ce qui crée un signal exploitable par l'attention.

  [FIX 2] Data leakage du StandardScaler
    → Le scaler des features ET les mom_scalers sont fittés uniquement sur
      le train set, puis appliqués sur val/test.
    → Le split est fait sur les match_id (pas sur les séquences) pour éviter
      le leakage entre séquences chevauchantes.

  [FIX 3] Bug ordre temporel dans compute_momentum_games
    → Les jeux des sets précédents étaient ajoutés après ceux du set courant,
      cassant l'ordre chronologique. Corrigé avec une liste triée (s, g).

  [FIX 4] Teacher forcing décroissant
    → tf_ratio passe de 0.9 (epoch 1) à 0.1 (epoch EPOCHS) linéairement,
      ce qui stabilise l'entraînement et améliore la généralisation.

  [FIX 5] Temperature apprise
    → La temperature du softmax de GranularityAttention est un paramètre
      appris (initialisé à 1.0, clampé > 0.1), permettant au modèle de
      contrôler lui-même son niveau de spécialisation.

  [FIX 6] Conv kernel agrandi 3→5
    → Capture des patterns sur une fenêtre locale plus large.
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
CONV_KERNEL     = 5       # [FIX 6] kernel agrandi 3→5
EPOCHS          = 30
LR              = 1e-4
DROPOUT         = 0.2
LAMBDA_MOM      = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paramètres momentum par granularité ──────
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
    Momentum au niveau JEU — version corrigée [FIX 3].

    v1 bug : les jeux des sets précédents étaient ajoutés après ceux du
    set courant → liste non triée → [-window:] prenait les mauvais jeux.

    Fix : on construit une liste de (set_no, game_no) triée chronologiquement,
    puis on filtre les jeux strictement antérieurs au jeu courant.
    """
    momentum = np.zeros(len(df_match))

    # Résultat de chaque jeu : +1 / -1
    game_results = {}
    for (s, g), grp in df_match.groupby(["set_no", "game_no"]):
        victor = grp["game_victor"].dropna()
        if len(victor) > 0:
            last = victor.iloc[-1]
            game_results[(s, g)] = 1.0 if last == 1 else -1.0

    # Liste ordonnée chronologiquement [FIX 3]
    all_games_sorted = sorted(game_results.keys())

    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()

    for idx, row in df_match.iterrows():
        s, g = int(row["set_no"]), int(row["game_no"])
        # Jeux strictement antérieurs au jeu courant (ordre chrono garanti)
        past_games = [(ss, gg) for (ss, gg) in all_games_sorted
                      if (ss < s) or (ss == s and gg < g)]
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
    Signal lent : avantage psychologique sur le match entier.
    """
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
        s = int(row["set_no"])
        past_sets = [ss for ss in sorted(set_results.keys()) if ss < s]
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
    Retourne le DataFrame avec features + momentum BRUTS (non normalisés).
    [FIX 2] La normalisation est déplacée après le split train/val.
    """
    df = pd.read_csv(csv_path, sep=",", low_memory=False)

    df["p1_score"] = df["p1_score"].map(SCORE_MAP)
    df["p2_score"] = df["p2_score"].map(SCORE_MAP)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    df = df.reset_index(drop=True)

    # Calcul des 3 niveaux de momentum (valeurs brutes, pas normalisées)
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

    print(f"   mom_point (brut) — mean={df['mom_point'].mean():.3f}  std={df['mom_point'].std():.3f}")
    print(f"   mom_game  (brut) — mean={df['mom_game'].mean():.3f}  std={df['mom_game'].std():.3f}")
    print(f"   mom_set   (brut) — mean={df['mom_set'].mean():.3f}  std={df['mom_set'].std():.3f}")
    return df


def fit_and_apply_scalers(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    [FIX 2] Fit sur train uniquement, applique sur train + val.

    [FIX 1] Les 3 momentum sont normalisés SÉPARÉMENT (pas ensemble).
    Cela préserve leur contraste naturel d'échelle/variance, signal clé
    pour GranularityAttention.
    """
    feat_scaler = StandardScaler()
    df_train[FEATURE_COLS] = feat_scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = feat_scaler.transform(df_val[FEATURE_COLS])

    mom_scalers = {}
    for col in ["mom_point", "mom_game", "mom_set"]:
        sc = StandardScaler()
        df_train[[col]] = sc.fit_transform(df_train[[col]])
        df_val[[col]]   = sc.transform(df_val[[col]])
        mom_scalers[col] = sc

    return df_train, df_val, feat_scaler, mom_scalers


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
# 5. ARCHITECTURE
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Capture les patterns locaux."""
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
    Module d'attention sur les 3 granularités — version corrigée [FIX 1 + FIX 5].

    Signal d'entrée : stats momentum (mean+std par granularité = 6 valeurs)
                    + contexte LSTM pré-calculé (d_model valeurs).
    Cela permet d'apprendre "dans CE contexte de match, quelle granularité compte".

    La temperature est un paramètre appris (log-paramétrisé → toujours > 0) [FIX 5].
    """
    def __init__(self, d_model, n_gran=3):
        super().__init__()
        self.n_gran = n_gran
        # [FIX 1] Entrée enrichie : stats (6) + contexte LSTM (d_model)
        self.attn_net = nn.Sequential(
            nn.Linear(n_gran * 2 + d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_gran),
        )
        self.out_proj = nn.Linear(n_gran, d_model)
        # [FIX 5] Temperature apprise, initialisée à 1.0
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x_mom, context):
        """
        x_mom   : (B, T, 3)    — 3 granularités sur la fenêtre temporelle
        context : (B, d_model) — contexte LSTM pré-calculé [FIX 1]
        """
        mom_mean = x_mom.mean(dim=1)           # (B, 3)
        mom_std  = x_mom.std(dim=1) + 1e-6    # (B, 3)

        # Signal enrichi avec contexte LSTM [FIX 1]
        stats        = torch.cat([mom_mean, mom_std, context], dim=-1)  # (B, 6+d_model)
        scores       = self.attn_net(stats)                              # (B, 3)

        # Temperature apprise, clampée > 0.1 [FIX 5]
        temperature  = self.log_temperature.exp().clamp(min=0.1)
        attn_weights = torch.softmax(scores / temperature, dim=-1)       # (B, 3)

        weighted = x_mom * attn_weights.unsqueeze(1)    # (B, T, 3)
        out      = self.out_proj(weighted)               # (B, T, d_model)
        return out, attn_weights


class HydraEncoderMG(nn.Module):
    """
    Encoder Multi-Granularité — version corrigée [FIX 1].

    Modification clé : un LSTM léger (1 layer) tourne en premier sur les
    features projetées pour produire un contexte réel (h[-1]).
    Ce contexte est passé à GranularityAttention au lieu de la moyenne.
    """
    def __init__(self, feat_size, d_model, n_heads, num_lstm_layers, conv_k, dropout):
        super().__init__()
        self.feat_proj  = nn.Linear(feat_size, d_model)

        # [FIX 1] LSTM léger pour pré-contexte
        self.pre_lstm   = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

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
        # 1. Projection des features
        feat_emb = self.feat_proj(x_feat)                      # (B, T, d_model)

        # 2. Pré-run LSTM → contexte discriminant [FIX 1]
        _, (h_pre, _) = self.pre_lstm(feat_emb)
        pre_context   = h_pre[-1]                               # (B, d_model)

        # 3. Attention granularité avec vrai contexte [FIX 1]
        mom_emb, gran_weights = self.gran_attn(x_mom, pre_context)  # (B, T, d_model)

        # 4. Fusion features + momentum pondéré
        x = self.fusion_in(torch.cat([feat_emb, mom_emb], dim=-1))  # (B, T, d_model)

        # 5. Branches multi-scale
        conv_out         = self.conv_block(x)
        lstm_out, (h, c) = self.lstm(x)
        attn_out         = self.attention(self.pos_enc(x))

        # 6. Fusion des derniers états
        fused   = torch.cat([conv_out[:, -1], lstm_out[:, -1], attn_out[:, -1]], dim=-1)
        context = self.dropout(self.fusion_out(fused))          # (B, d_model)

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
def get_tf_ratio(epoch: int, total: int, start: float = 0.9, end: float = 0.1) -> float:
    """[FIX 4] Teacher forcing décroissant linéairement de start→end."""
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

    print("📂 Chargement + calcul momentum multi-granularité (v2)...")
    df = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points  |  {len(df['match_id'].unique())} matchs")

    # ── [FIX 2] Split sur match_id avant toute normalisation ──────────────
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

    print("\n🔧 Normalisation (fit sur train uniquement) [FIX 2]...")
    df_train, df_val, feat_scaler, mom_scalers = fit_and_apply_scalers(df_train, df_val)

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
    print(f"\n🚀 HydraNet MG v2 sur {DEVICE}  |  {n_params:,} paramètres\n")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tf         = get_tf_ratio(epoch, EPOCHS)        # [FIX 4]
        train_loss = train_epoch(model, train_loader, optimizer, ce_loss, mse_loss, tf)
        val_loss, val_acc, mse_pt, mse_game, mse_set, gran_w = evaluate(
            model, val_loader, ce_loss, mse_loss)
        scheduler.step()

        temp = model.encoder.gran_attn.log_temperature.exp().item()  # [FIX 5]

        print(f"Epoch {epoch:02d}/{EPOCHS} | tf={tf:.2f} | "
              f"loss={train_loss:.4f} | acc={val_acc:.4f} | "
              f"MSE[pt={mse_pt:.4f} game={mse_game:.4f} set={mse_set:.4f}] | "
              f"GranW[pt={gran_w[0]:.2f} game={gran_w[1]:.2f} set={gran_w[2]:.2f}] | "
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
            }, "best_hydranet_mg_v2.pt")
            joblib.dump(feat_scaler,  "feat_scaler_mg_v2.pkl")
            joblib.dump(mom_scalers,  "mom_scalers_mg_v2.pkl")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("\n📊 Poids de granularité moyens (dernière epoch) :")
    print(f"   Point : {gran_w[0]:.3f}  |  Jeu : {gran_w[1]:.3f}  |  Set : {gran_w[2]:.3f}")
    print(f"   Temperature finale : {model.encoder.gran_attn.log_temperature.exp().item():.3f}")
    print("\n   → Si T < 1 : le modèle a appris à se spécialiser (distribution piquée)")
    print("   → Si T > 1 : le modèle préfère l'uniformité (distribution plate)")
    print("\n   Checkpoint sauvegardé dans best_hydranet_mg_v2.pt")