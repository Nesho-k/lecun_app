import torch
import numpy as np
from pathlib import Path

# Charger les données d'entraînement
BASE_DIR = Path(__file__).resolve().parent.parent
training_path = BASE_DIR / "training.pt"

X, y = torch.load(training_path, weights_only=False)

print("=== DISTRIBUTION DES PIXELS MNIST (avant normalisation) ===")
# Prendre quelques images
for i in range(5):
    img = X[i].numpy().flatten()
    print(f"\nImage {i} (label={y[i]}):")
    print(f"  Pixels = 0: {(img == 0).sum()}")
    print(f"  Pixels 1-50: {((img > 0) & (img <= 50)).sum()}")
    print(f"  Pixels 51-150: {((img > 50) & (img <= 150)).sum()}")
    print(f"  Pixels 151-254: {((img > 150) & (img < 255)).sum()}")
    print(f"  Pixels = 255: {(img == 255).sum()}")

print("\n=== MNIST N'EST PAS BINAIRE ===")
print("Les images MNIST ont des niveaux de gris intermédiaires (anti-aliasing)")
print("La binarisation n'est donc peut-être pas la bonne approche.")
