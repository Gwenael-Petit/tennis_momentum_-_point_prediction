"""
Exploration du dataset — Statistiques pour le diapo
====================================================
Usage : python explore_dataset.py usopen.csv
"""

import sys
import pandas as pd
import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "USD.txt"

df = pd.read_csv(CSV_PATH, sep=",")

print("=" * 55)
print("  STATISTIQUES DU DATASET")
print("=" * 55)

# ── Volume brut ──────────────────────────────────────────
n_matches = df["match_id"].nunique()
n_points  = len(df)

print(f"\n📦 VOLUME")
print(f"   Matchs  : {n_matches}")
print(f"   Points  : {n_points:,}")
print(f"   Moyenne points/match : {n_points/n_matches:.0f}")

# ── Sets ─────────────────────────────────────────────────
if "set_no" in df.columns:
    sets_per_match = df.groupby("match_id")["set_no"].nunique()
    total_sets     = df.groupby("match_id")["set_no"].nunique().sum()
    print(f"\n🎾 SETS")
    print(f"   Total sets    : {total_sets}")
    print(f"   Moy/match     : {sets_per_match.mean():.1f}")
    print(f"   Min/Max       : {sets_per_match.min()} / {sets_per_match.max()}")

# ── Jeux ─────────────────────────────────────────────────
if "game_no" in df.columns and "set_no" in df.columns:
    games_per_match = df.groupby("match_id").apply(
        lambda m: m.groupby(["set_no","game_no"]).ngroups
    )
    total_games = games_per_match.sum()
    print(f"\n🏓 JEUX")
    print(f"   Total jeux    : {total_games}")
    print(f"   Moy/match     : {games_per_match.mean():.1f}")
    print(f"   Min/Max       : {games_per_match.min()} / {games_per_match.max()}")

# ── Points ───────────────────────────────────────────────
pts_per_match = df.groupby("match_id").size()
print(f"\n⚡ POINTS PAR MATCH")
print(f"   Médiane       : {pts_per_match.median():.0f}")
print(f"   Min / Max     : {pts_per_match.min()} / {pts_per_match.max()}")
print(f"   Écart-type    : {pts_per_match.std():.0f}")

# ── Colonnes disponibles ─────────────────────────────────
print(f"\n📋 COLONNES ({len(df.columns)} au total)")
print(f"   {list(df.columns)}")

# ── Valeurs manquantes ───────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    print(f"\n⚠️  VALEURS MANQUANTES")
    for col, n in missing.items():
        print(f"   {col:30s} : {n} ({100*n/len(df):.1f}%)")
else:
    print(f"\n✅ Aucune valeur manquante")

# ── Équilibre des classes ────────────────────────────────
if "Y" in df.columns:
    vc = df["Y"].value_counts(normalize=True)
    print(f"\n⚖️  ÉQUILIBRE DES CLASSES (Y)")
    for k,v in vc.items():
        print(f"   {k} : {v:.1%}")

print("\n" + "=" * 55)