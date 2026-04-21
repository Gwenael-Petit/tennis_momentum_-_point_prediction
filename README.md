# 🎾 Tennis Momentum Prediction — Guide d'utilisation

## Structure du projet

```
tennis_momentum/
├── niveau1_lstm.py            # Baseline LSTM (next-point classification)
├── niveau2_seq2seq.py         # Seq2Seq (séquence de points futurs)
├── niveau3_seq2seq_momentum.py # Seq2Seq + tête momentum
└── niveau4_hydranet.py        # HydraNet complet (état de l'art)
```

---

## Progression recommandée

| Niveau | Modèle | Input | Output | Nouveauté |
|--------|--------|-------|--------|-----------|
| 1 | LSTM | séquence passée | 1 point | baseline |
| 2 | Seq2Seq | séquence passée | séquence points | décodeur auto-régressif |
| 3 | Seq2Seq + Momentum | séq. passée + momentum | points + momentum | variable latente |
| 4 | HydraNet | idem | idem | encoder multi-scale + attention |

---

## Installation

```bash
pip install torch pandas numpy scikit-learn
```

---

## Utilisation

```bash
# Niveau 1
python niveau1_lstm.py usopen.csv

# Niveau 2
python niveau2_seq2seq.py usopen.csv

# Niveau 3
python niveau3_seq2seq_momentum.py usopen.csv

# Niveau 4 (HydraNet)
python niveau4_hydranet.py usopen.csv
```

---

## Définition du Momentum

Le momentum est calculé comme une moyenne pondérée exponentiellement des résultats récents :

```
momentum_t = Σ decay^k * result_{t-k}   pour k = 0..W-1
```

- `result = +1` si joueur 1 gagne le point, `-1` sinon
- `decay = 0.85` (les points récents comptent plus)
- `W = 8` (fenêtre de calcul)
- `momentum ∈ [-1, 1]`

**Interprétation :**
- `momentum > 0` → J1 en momentum positif
- `momentum < 0` → J2 en momentum positif
- `|momentum| → 1` → momentum fort

---

## Architecture HydraNet (Niveau 4)

```
Input (features + momentum)
        │
        ▼
┌──────────────────────────────┐
│  HydraEncoder (multi-scale)  │
│  ├─ ConvBlock  (court terme) │
│  ├─ LSTM       (moyen terme) │
│  └─ Attention  (long terme)  │
└──────────────┬───────────────┘
               │ context vector
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌──────────────┐
│ Point Head │   │ Momentum Head│
│ (classif.) │   │ (régression) │
└────────────┘   └──────────────┘
```

---

## Paramètres clés à ajuster

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `SEQ_LEN` | 10 | Points passés en entrée |
| `PRED_LEN` | 5 | Points futurs à prédire |
| `MOM_WINDOW` | 8 | Fenêtre de calcul du momentum |
| `MOM_DECAY` | 0.85 | Décroissance exponentielle |
| `LAMBDA_MOM` | 0.5 | Poids de la loss momentum |
| `D_MODEL` | 128 | Dimension interne HydraNet |
| `N_HEADS` | 4 | Têtes d'attention |

---

## Pistes d'amélioration pour le mémoire

1. **Momentum comme variable latente apprise** — plutôt que calculée à la main,
   laisser le modèle apprendre sa propre représentation du momentum via un VAE.

2. **Segmentation des échanges** — intégrer la durée des points ou les statistiques
   de l'échange (nb de frappes, distance courrue).

3. **Cross-match transfer** — pré-entraîner sur plusieurs tournois, fine-tuner sur
   un joueur spécifique.

4. **Évaluation du momentum prédictif** — mesurer si le momentum prédit à t
   est un meilleur prédicteur du résultat futur que les features brutes.
