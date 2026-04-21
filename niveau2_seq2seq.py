"""
NIVEAU 2 - Seq2Seq : Prédiction d'une séquence de points futurs
===============================================================
On ne prédit plus UN point, mais une SÉQUENCE de K points à venir.
C'est la première brique vers la modélisation du momentum.

Architecture :
    Encoder LSTM  →  context vector  →  Decoder LSTM
    Input : séquence passée (seq_len points)
    Output : séquence future (pred_len points)

Note : ce fichier réutilise les fonctions de niveau1_lstm.py.
       Lance d'abord niveau1_lstm.py pour t'assurer que tout fonctionne.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN     = 10    # longueur de la séquence d'entrée (passé)
PRED_LEN    = 5     # longueur de la séquence à prédire (futur)
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
TARGET_COL = "Y"   # 1 si joueur 1 gagne le point

# Mapping des scores tennis → numérique
SCORE_MAP = {
    '0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
    0: 0, 15: 1, 30: 2, 40: 3  # au cas où certaines valeurs sont déjà int
}

# ─────────────────────────────────────────────
# 2. DONNÉES : séquences (X → Y_seq)
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, sep=",")

    # Encoder les scores tennis
    df['p1_score'] = df['p1_score'].map(SCORE_MAP)
    df['p2_score'] = df['p2_score'].map(SCORE_MAP)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df, scaler


def build_sequences(df, seq_len, pred_len):
    """
    Pour chaque match, on glisse une fenêtre :
      X : points [i, i+seq_len[      → features (seq_len, F)
      y : labels [i+seq_len, i+seq_len+pred_len[  → cibles (pred_len,)
    """
    X_list, y_list = [], []

    for _, group in df.groupby("match_id"):
        group    = group.sort_values("point_no").reset_index(drop=True)
        features = group[FEATURE_COLS].values
        targets  = group[TARGET_COL].values

        for i in range(len(group) - seq_len - pred_len + 1):
            X_list.append(features[i : i + seq_len])
            y_list.append(targets[i + seq_len : i + seq_len + pred_len])

    X = np.array(X_list, dtype=np.float32)   # (N, seq_len, F)
    y = np.array(y_list, dtype=np.int64)     # (N, pred_len)
    return X, y


class TennisSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 3. MODÈLE SEQ2SEQ
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        # x : (B, seq_len, input_size)
        _, (h, c) = self.lstm(x)
        return h, c   # état caché final : (num_layers, B, hidden)


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_classes=2):
        super().__init__()
        # À chaque pas, l'entrée du décodeur est le one-hot du point prédit
        self.lstm = nn.LSTM(num_classes, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward_step(self, x_t, h, c):
        """Un pas de décodage. x_t : (B, 1, num_classes)"""
        out, (h, c) = self.lstm(x_t, (h, c))
        logit = self.fc(out.squeeze(1))   # (B, num_classes)
        return logit, h, c

    def forward(self, h, c, pred_len, targets=None, teacher_forcing_ratio=0.5):
        """
        Décodage auto-régressif avec teacher forcing optionnel.
        targets : (B, pred_len) — labels ground truth pour teacher forcing
        Retourne logits : (B, pred_len, num_classes)
        """
        B = h.size(1)
        # Token de démarrage : vecteur nul
        x_t = torch.zeros(B, 1, self.num_classes, device=h.device)
        outputs = []

        for t in range(pred_len):
            logit, h, c = self.forward_step(x_t, h, c)  # (B, num_classes)
            outputs.append(logit.unsqueeze(1))            # → (B, 1, num_classes)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing : on donne le vrai label
                true_label = targets[:, t]                     # (B,)
                x_t = nn.functional.one_hot(true_label, self.num_classes).float().unsqueeze(1)
            else:
                # Inférence : on donne la prédiction précédente
                pred = logit.argmax(dim=1)
                x_t = nn.functional.one_hot(pred, self.num_classes).float().unsqueeze(1)

        return torch.cat(outputs, dim=1)   # (B, pred_len, num_classes)


class Seq2SeqPoints(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, dropout)

    def forward(self, x, pred_len, targets=None, teacher_forcing_ratio=0.5):
        h, c = self.encoder(x)
        return self.decoder(h, c, pred_len, targets, teacher_forcing_ratio)


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()

        # logits : (B, pred_len, 2)
        logits = model(X_batch, PRED_LEN, targets=y_batch, teacher_forcing_ratio=0.5)

        # On calcule la loss sur tous les pas
        # logits reshape → (B*pred_len, 2), y → (B*pred_len,)
        loss = criterion(logits.view(-1, 2), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch, PRED_LEN, teacher_forcing_ratio=0.0)
            loss   = criterion(logits.view(-1, 2), y_batch.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement des données...")
    df, scaler = load_and_prepare(CSV_PATH)

    print("🔨 Construction des séquences...")
    X, y = build_sequences(df, SEQ_LEN, PRED_LEN)
    print(f"   X: {X.shape}  |  y: {y.shape}")

    split = int(0.8 * len(X))
    train_loader = DataLoader(TennisSeqDataset(X[:split], y[:split]), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TennisSeqDataset(X[split:], y[split:]), BATCH_SIZE)

    model     = Seq2SeqPoints(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Entraînement sur {DEVICE}  |  {sum(p.numel() for p in model.parameters()):,} params")

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_seq2seq.pt")

    print(f"\n✅ Meilleure val_acc : {best_acc:.4f}")
    print("   Modèle sauvegardé dans best_seq2seq.pt")
