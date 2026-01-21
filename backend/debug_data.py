import torch
import numpy as np
from pathlib import Path

# Charger les données d'entraînement
BASE_DIR = Path(__file__).resolve().parent.parent
training_path = BASE_DIR / "training.pt"

X, y = torch.load(training_path, weights_only=False)

print("=== FORMAT DES DONNÉES D'ENTRAÎNEMENT ===")
print(f"Shape X: {X.shape}")
print(f"Type X: {X.dtype}")
print(f"Min X: {X.min()}")
print(f"Max X: {X.max()}")
print(f"Exemple valeurs X[0][14] (ligne du milieu): {X[0][14][:10]}...")
print()

# Après normalisation (comme dans le notebook)
X_norm = (X / 255) * 2 - 1
print("=== APRÈS NORMALISATION ===")
print(f"Min X_norm: {X_norm.min()}")
print(f"Max X_norm: {X_norm.max()}")
print(f"Exemple valeurs X_norm[0][14]: {X_norm[0][14][:10]}...")
print()

# Vérifier si le fond est noir (0) ou blanc (255)
print("=== ANALYSE DU FOND ET DU TRAIT ===")
# Compter les pixels noirs vs blancs dans l'image brute
img = X[0].numpy()
print(f"Image 0 - Label: {y[0]}")
print(f"Pixels == 0 (noir): {(img == 0).sum()} / {img.size}")
print(f"Pixels > 0 (trait): {(img > 0).sum()} / {img.size}")
print(f"Valeur moyenne pixels non-nuls: {img[img > 0].mean():.1f}")
