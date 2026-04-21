"""
NIVEAU 3 - Modélisation du Momentum
=====================================
On enrichit le niveau 2 avec :
  1. Une feature "momentum" calculée depuis les données brutes
  2. Un décodeur dédié à la prédiction du momentum futur (variable latente)

Définition du momentum utilisée ici :
  momentum_t = Σ w^k * result_{t-k}   pour k=0..W-1
  où result = +1 si j1 gagne, -1 si j2 gagne,
  et w est un facteur de décroissance exponentielle (les points récents comptent plus).

  momentum_t ∈ [-1, 1] (normalisé)
  → momentum > 0  : j1 en momentum positif
  → momentum < 0  : j2 en momentum positif

L'objectif est de prédire à la fois :
  - la séquence de points futurs (tête 1)
  - la séquence de momentum futur (tête 2)  ← NOUVEAU
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN        = 10
PRED_LEN       = 5
MOM_WINDOW     = 8      # fenêtre pour le calcul du momentum
MOM_DECAY      = 0.85   # facteur de décroissance exponentielle
BATCH_SIZE     = 64
HIDDEN_SIZE    = 128
NUM_LAYERS     = 2
EPOCHS         = 20
LR             = 1e-3
DROPOUT        = 0.3
LAMBDA_MOMENTUM = 0.5   # poids de la loss momentum vs points
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

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
# 2. CALCUL DU MOMENTUM
# ─────────────────────────────────────────────
def compute_momentum(results: np.ndarray, window: int, decay: float) -> np.ndarray:
    """
    results : tableau binaire (0 ou 1) représentant qui gagne chaque point.
    Retourne un tableau de momentum en [-1, 1].
    """
    # On convertit en ±1
    signed = np.where(results == 1, 1.0, -1.0)
    weights = np.array([decay ** k for k in range(window)])
    weights /= weights.sum()   # normalisation

    momentum = np.zeros(len(signed))
    for t in range(len(signed)):
        start = max(0, t - window + 1)
        w     = weights[-(t - start + 1):]
        momentum[t] = np.dot(signed[start : t + 1], w)

    return momentum.astype(np.float32)


# ─────────────────────────────────────────────
# 3. CHARGEMENT & SÉQUENCES
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, sep=",")

    # Encoder les scores tennis
    df['p1_score'] = df['p1_score'].map(SCORE_MAP)
    df['p2_score'] = df['p2_score'].map(SCORE_MAP)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)

    # Calcul du momentum par match
    mom_list = []
    for _, group in df.groupby("match_id"):
        group = group.sort_values("point_no")
        mom   = compute_momentum(group[TARGET_COL].values, MOM_WINDOW, MOM_DECAY)
        mom_list.extend(mom.tolist())
    df["momentum"] = mom_list

    # Normalisation features
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    return df, scaler


def build_sequences(df):
    X_feat, X_mom, y_points, y_mom = [], [], [], []

    for _, group in df.groupby("match_id"):
        group    = group.sort_values("point_no").reset_index(drop=True)
        features = group[FEATURE_COLS].values     # (T, F)
        points   = group[TARGET_COL].values       # (T,)
        moms     = group["momentum"].values       # (T,)

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
    def __init__(self, X_feat, X_mom, y_points, y_mom):
        self.X_feat   = torch.tensor(X_feat)
        self.X_mom    = torch.tensor(X_mom).unsqueeze(-1)   # (N, T, 1)
        self.y_points = torch.tensor(y_points)
        self.y_mom    = torch.tensor(y_mom)

    def __len__(self): return len(self.y_points)

    def __getitem__(self, idx):
        return self.X_feat[idx], self.X_mom[idx], self.y_points[idx], self.y_mom[idx]


# ─────────────────────────────────────────────
# 4. MODÈLE : ENCODER + 2 DÉCODEURS
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """Encode features + momentum en un vecteur de contexte."""
    def __init__(self, feat_size, hidden_size, num_layers, dropout):
        super().__init__()
        # On concatène features et momentum → feat_size + 1
        self.lstm = nn.LSTM(feat_size + 1, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

    def forward(self, x_feat, x_mom):
        x = torch.cat([x_feat, x_mom], dim=-1)   # (B, T, F+1)
        _, (h, c) = self.lstm(x)
        return h, c


class PointDecoder(nn.Module):
    """Décode → séquence de points (classification binaire)."""
    def __init__(self, hidden_size, num_layers, dropout, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(num_classes, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, h, c, pred_len, targets=None, teacher_forcing=0.5):
        B = h.size(1)
        x_t = torch.zeros(B, 1, self.num_classes, device=h.device)
        outputs = []
        for t in range(pred_len):
            out, (h, c) = self.lstm(x_t, (h, c))
            logit = self.fc(out.squeeze(1))
            outputs.append(logit.unsqueeze(1))
            if targets is not None and torch.rand(1).item() < teacher_forcing:
                x_t = nn.functional.one_hot(targets[:, t], self.num_classes).float().unsqueeze(1)
            else:
                x_t = nn.functional.one_hot(logit.argmax(1), self.num_classes).float().unsqueeze(1)
        return torch.cat(outputs, dim=1)   # (B, pred_len, 2)


class MomentumDecoder(nn.Module):
    """Décode → séquence de momentum (régression)."""
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h, c, pred_len, targets=None, teacher_forcing=0.5):
        B = h.size(1)
        x_t = torch.zeros(B, 1, 1, device=h.device)
        outputs = []
        for t in range(pred_len):
            out, (h, c) = self.lstm(x_t, (h, c))
            pred_mom = self.fc(out.squeeze(1))     # (B, 1)
            outputs.append(pred_mom.unsqueeze(1))  # (B, 1, 1)
            if targets is not None and torch.rand(1).item() < teacher_forcing:
                x_t = targets[:, t].unsqueeze(1).unsqueeze(1)   # (B, 1, 1)
            else:
                x_t = pred_mom.unsqueeze(1)
        return torch.cat(outputs, dim=1).squeeze(-1)   # (B, pred_len)


class Seq2SeqWithMomentum(nn.Module):
    def __init__(self, feat_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder          = Encoder(feat_size, hidden_size, num_layers, dropout)
        self.point_decoder    = PointDecoder(hidden_size, num_layers, dropout)
        self.momentum_decoder = MomentumDecoder(hidden_size, num_layers, dropout)

    def forward(self, x_feat, x_mom, pred_len,
                y_points=None, y_mom=None, teacher_forcing=0.5):
        h, c = self.encoder(x_feat, x_mom)
        # Chaque décodeur part du MÊME état caché (on clone pour ne pas les coupler)
        points_logits = self.point_decoder(h.clone(), c.clone(), pred_len,
                                           y_points, teacher_forcing)
        mom_preds     = self.momentum_decoder(h.clone(), c.clone(), pred_len,
                                              y_mom, teacher_forcing)
        return points_logits, mom_preds


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
def train(model, loader, optimizer, ce_loss, mse_loss):
    model.train()
    total_loss = 0
    for X_feat, X_mom, y_points, y_mom in loader:
        X_feat, X_mom = X_feat.to(DEVICE), X_mom.to(DEVICE)
        y_points, y_mom = y_points.to(DEVICE), y_mom.to(DEVICE)
        optimizer.zero_grad()

        logits, mom_preds = model(X_feat, X_mom, PRED_LEN,
                                  y_points, y_mom, teacher_forcing=0.5)

        loss_pts = ce_loss(logits.view(-1, 2), y_points.view(-1))
        loss_mom = mse_loss(mom_preds, y_mom)
        loss     = loss_pts + LAMBDA_MOMENTUM * loss_mom
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, ce_loss, mse_loss):
    model.eval()
    total_loss, all_preds, all_labels, all_mom_pred, all_mom_true = 0, [], [], [], []
    with torch.no_grad():
        for X_feat, X_mom, y_points, y_mom in loader:
            X_feat, X_mom = X_feat.to(DEVICE), X_mom.to(DEVICE)
            y_points, y_mom = y_points.to(DEVICE), y_mom.to(DEVICE)
            logits, mom_preds = model(X_feat, X_mom, PRED_LEN, teacher_forcing=0.0)

            loss = ce_loss(logits.view(-1, 2), y_points.view(-1)) + \
                   LAMBDA_MOMENTUM * mse_loss(mom_preds, y_mom)
            total_loss += loss.item()

            all_preds.extend(logits.argmax(-1).cpu().numpy().flatten())
            all_labels.extend(y_points.cpu().numpy().flatten())
            all_mom_pred.extend(mom_preds.cpu().numpy().flatten())
            all_mom_true.extend(y_mom.cpu().numpy().flatten())

    acc     = accuracy_score(all_labels, all_preds)
    mom_mse = mean_squared_error(all_mom_true, all_mom_pred)
    return total_loss / len(loader), acc, mom_mse


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement + calcul momentum...")
    df, scaler = load_and_prepare(CSV_PATH)
    print(f"   Momentum moyen : {df['momentum'].mean():.4f}  |  std : {df['momentum'].std():.4f}")

    print("🔨 Construction des séquences...")
    X_feat, X_mom, y_points, y_mom = build_sequences(df)
    print(f"   X_feat: {X_feat.shape}  |  y_points: {y_points.shape}")

    split = int(0.8 * len(X_feat))
    train_ds = TennisDataset(X_feat[:split], X_mom[:split], y_points[:split], y_mom[:split])
    val_ds   = TennisDataset(X_feat[split:], X_mom[split:], y_points[split:], y_mom[split:])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE)

    model     = Seq2SeqWithMomentum(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    print(f"\n🚀 Entraînement sur {DEVICE}  |  {sum(p.numel() for p in model.parameters()):,} params")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, ce_loss, mse_loss)
        val_loss, val_acc, mom_mse = evaluate(model, val_loader, ce_loss, mse_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | mom_MSE={mom_mse:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_seq2seq_momentum.pt")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
