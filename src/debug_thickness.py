import torch
import numpy as np
from pathlib import Path

# Charger les données d'entraînement
BASE_DIR = Path(__file__).resolve().parent.parent
training_path = BASE_DIR / "training.pt"

X, y = torch.load(training_path, weights_only=False)

print("=== ÉPAISSEUR DU TRAIT MNIST ===")
print("(Nombre de pixels non-noirs par image)\n")

# Analyser plusieurs images
stroke_pixels = []
for i in range(100):
    img = X[i].numpy()
    non_black = (img > 0).sum()
    stroke_pixels.append(non_black)

stroke_pixels = np.array(stroke_pixels)
print(f"Min pixels du trait: {stroke_pixels.min()}")
print(f"Max pixels du trait: {stroke_pixels.max()}")
print(f"Moyenne pixels du trait: {stroke_pixels.mean():.1f}")
print(f"Écart-type: {stroke_pixels.std():.1f}")

print("\n=== RATIO TRAIT / IMAGE ===")
total_pixels = 28 * 28  # 784
print(f"Ratio moyen: {stroke_pixels.mean() / total_pixels * 100:.1f}%")
print(f"Ratio min: {stroke_pixels.min() / total_pixels * 100:.1f}%")
print(f"Ratio max: {stroke_pixels.max() / total_pixels * 100:.1f}%")

print("\n=== POUR TON CANVAS 280x280 ===")
print("Ton canvas fait 280x280, soit 10x plus grand que 28x28")
print("Après redimensionnement BOX (moyenne), un pixel MNIST = 10x10 pixels sur ton canvas")
print(f"Épaisseur recommandée sur ton canvas: ~{int(np.sqrt(stroke_pixels.mean() / total_pixels) * 280)}px à {int(np.sqrt(stroke_pixels.max() / total_pixels) * 280)}px")

# Vérifier l'épaisseur typique en analysant les lignes
print("\n=== ÉPAISSEUR TYPIQUE DU TRAIT (en pixels) ===")
for i in range(5):
    img = X[i].numpy()
    # Trouver la ligne avec le plus de pixels allumés
    row_sums = (img > 0).sum(axis=1)
    max_row_width = row_sums.max()
    col_sums = (img > 0).sum(axis=0)
    max_col_width = col_sums.max()
    print(f"Image {i} (label={y[i]}): largeur max horizontale={max_row_width}, verticale={max_col_width}")
