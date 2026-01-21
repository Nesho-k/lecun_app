from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

import uvicorn

from src.model import RN

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "saved_models" / "weights.pt"

# Chargement du modèle
model = RN()
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Poids du modèle non trouvés: {WEIGHTS_PATH}")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
model.eval()

# Application FastAPI
app = FastAPI(title="Reconnaissance de chiffres - LeCun CNN")

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Prétraitement de l'image:
    - Conversion en niveaux de gris
    - Redimensionnement en 28x28
    - Normalisation entre -1 et 1
    - Ajout des dimensions batch et canal
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Convertir en niveaux de gris
    image = image.convert("L")

    # Redimensionner en 28x28 avec BOX (moyenne des pixels) - meilleur pour réduction
    image = image.resize((28, 28), Image.Resampling.BOX)

    # Convertir en tensor
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.reshape(28, 28)

    # Normaliser entre -1 et 1 (comme MNIST)
    tensor = (tensor / 255.0) * 2 - 1

    # Ajouter dimensions batch et canal: (1, 1, 28, 28)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """
    Endpoint de prédiction.
    Reçoit une image PNG et retourne la prédiction + scores (logits).
    """
    # Vérifier le type de fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)

        # DEBUG: Analyser l'image reçue
        t = tensor[0, 0]  # Retirer dimensions batch et canal
        print("=== DEBUG IMAGE REÇUE ===")
        print(f"Min: {t.min():.2f}, Max: {t.max():.2f}")
        print(f"Pixels == -1 (noir): {(t == -1).sum().item()} / {t.numel()}")
        print(f"Pixels > -1 (trait): {(t > -1).sum().item()} / {t.numel()}")
        if (t > -1).sum() > 0:
            print(f"Valeur moyenne pixels du trait (normalisé): {t[t > -1].mean():.2f}")

        # Prédiction
        with torch.no_grad():
            logits = model(tensor)

        scores = logits[0].tolist()
        prediction = int(torch.argmax(logits, dim=1).item())

        return {"prediction": prediction, "scores": scores}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Endpoint de santé pour vérifier que l'API fonctionne."""
    return {"status": "ok"}


#if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000)
