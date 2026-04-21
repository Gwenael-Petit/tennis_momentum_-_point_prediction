"""
NIVEAU 1 - LSTM Baseline
========================
Objectif : Prédire le prochain point (Y) à partir des N points précédents.
C'est le point de départ classique, souvent utilisé dans la littérature.

Input  : séquence de longueur SEQ_LEN de features par point
Output : classe binaire (qui gagne le prochain point : joueur 1 ou 2)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN     = 10        # nb de points passés utilisés comme contexte
BATCH_SIZE  = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
EPOCHS      = 20
LR          = 1e-3
DROPOUT     = 0.3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Features utilisées (ajuste selon ce qui est pertinent pour toi)
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
TARGET_COL = "Y"   # 1 si joueur 1 gagne le point, 0 sinon

# Mapping des scores tennis → numérique
SCORE_MAP = {
    '0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4,
    0: 0, 15: 1, 30: 2, 40: 3  # au cas où certaines valeurs sont déjà int
}


# ─────────────────────────────────────────────
# 2. CHARGEMENT & PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, sep=",")

    # Encoder les scores tennis
    df['p1_score'] = df['p1_score'].map(SCORE_MAP)
    df['p2_score'] = df['p2_score'].map(SCORE_MAP)

    # Nettoyage basique
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Encodage de la cible : 1 = joueur 1 gagne, 0 = joueur 2 gagne
    df[TARGET_COL] = (df[TARGET_COL] == 1).astype(int)

    # Normalisation des features
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    return df, scaler


def build_sequences(df: pd.DataFrame, seq_len: int):
    """
    Pour chaque match (match_id), on construit des fenêtres glissantes
    de taille seq_len → on prédit le point suivant.
    """
    X_list, y_list = [], []

    for match_id, group in df.groupby("match_id"):
        group = group.sort_values("point_no").reset_index(drop=True)
        features = group[FEATURE_COLS].values   # (T, F)
        targets  = group[TARGET_COL].values     # (T,)

        for i in range(seq_len, len(group)):
            X_list.append(features[i - seq_len : i])   # (seq_len, F)
            y_list.append(targets[i])                  # scalaire

    X = np.array(X_list, dtype=np.float32)   # (N, seq_len, F)
    y = np.array(y_list, dtype=np.int64)     # (N,)
    return X, y


# ─────────────────────────────────────────────
# 3. DATASET PYTORCH
# ─────────────────────────────────────────────
class TennisPointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 4. MODÈLE LSTM
# ─────────────────────────────────────────────
class LSTMPointPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)   # 2 classes : j1 gagne / j2 gagne

    def forward(self, x):
        # x : (B, seq_len, input_size)
        out, _ = self.lstm(x)          # (B, seq_len, hidden)
        out = self.dropout(out[:, -1]) # on prend le dernier état caché
        return self.fc(out)            # (B, 2)


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
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
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

    print("📂 Chargement des données...")
    df, scaler = load_and_prepare(CSV_PATH)
    print(f"   {len(df)} points chargés, {len(df['match_id'].unique())} matchs")

    print("🔨 Construction des séquences...")
    X, y = build_sequences(df, SEQ_LEN)
    print(f"   {len(X)} séquences créées, shape X: {X.shape}")

    # Split train/val (80/20) — sans mélanger les matchs
    split = int(0.8 * len(X))
    train_dataset = TennisPointDataset(X[:split], y[:split])
    val_dataset   = TennisPointDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    # Modèle
    INPUT_SIZE = len(FEATURE_COLS)
    model = LSTMPointPredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Entraînement sur {DEVICE}...")
    print(f"   Modèle : {sum(p.numel() for p in model.parameters()):,} paramètres\n")

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lstm.pt")

    print(f"\n✅ Meilleure val_acc : {best_val_acc:.4f}")
    print("   Modèle sauvegardé dans best_lstm.pt")
