# Reconnaissance de chiffres manuscrits – CNN end-to-end (PyTorch, FastAPI)

Projet de Deep Learning appliqué à la vision par ordinateur : développement d’un modèle de reconnaissance de chiffres manuscrits basé sur un CNN, exposé via une API et utilisable en temps réel via une interface web interactive.

🔗 **Démo live** : https://lecun-app-front.vercel.app/



## Contexte

Ce projet s’inspire des travaux fondateurs de Yann LeCun (1989) sur la reconnaissance de chiffres manuscrits, adaptés dans un environnement moderne (PyTorch, API, frontend interactif).

Objectif : construire un système complet de Machine Learning, depuis l’entraînement du modèle jusqu’à son utilisation par un utilisateur final.



## Objectifs

- Développer un modèle de classification d’images performant (CNN)  
- Construire un pipeline ML complet (data → modèle → API → interface)  
- Permettre une prédiction en temps réel  
- Mettre en évidence l’impact du preprocessing  



## Compétences démontrées

| Compétence | Ce qui est fait |
|----------|----------------|
| Deep Learning | CNN (Conv2D, pooling, dense), entraînement PyTorch |
| Data preprocessing | Normalisation, encodage, adaptation images utilisateur |
| Backend | API REST avec FastAPI |
| Frontend | Interface React avec Canvas |
| Computer Vision | Traitement d’image (resize, grayscale) |



## Dataset

- Source : MNIST  
- Volume : 70 000 images (60K train / 10K test)  
- Format : 28x28 pixels (grayscale)  
- Classes : 0 à 9  



## Modèle CNN

Architecture :
Input (28x28)
→ Conv + Tanh
→ AvgPool
→ Conv + Tanh
→ AvgPool
→ Flatten
→ Dense (10)




## Entraînement

- Optimiseur : SGD + momentum  
- Learning rate : 0.01  
- Batch size : 5  
- Epochs : 20  
- Loss : MSE  

### Résultats

- Accuracy : ~98–99%  
- Loss : 0.03 → 0.01  



## API (FastAPI)

### Endpoint

`POST /predict-file`

### Pipeline

1. Upload image  
2. Prétraitement :
   - grayscale  
   - resize (280 → 28)  
   - normalisation [-1,1]  
3. Prédiction  
4. Retour résultat  



## Interface Web

- Canvas interactif (dessin utilisateur)  
- Ajustement de l’épaisseur du trait  
- Prédiction en temps réel  



## Problèmes rencontrés

**Différence MNIST vs dessin utilisateur**  
→ Ajustement du preprocessing  

**Resize dégradant**  
→ Utilisation de l’algorithme BOX  

**Trait trop fin**  
→ Calibration 20–30 px  

**Normalisation critique**  
→ `(pixel / 255) * 2 - 1`  



## Stack technique

- PyTorch  
- FastAPI  
- React  
- Tailwind  
- Pillow  



## Architecture
Frontend (React)
↓
API FastAPI
↓
Prétraitement
↓
Modèle CNN
↓
Prédiction




## Installation

### Backend

```bash
git clone https://github.com/Nesho-k/lecun_app.git
cd project
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```



## Résultat

Application complète permettant :
- de dessiner un chiffre
- d’obtenir une prédiction instantanée
- de comprendre un pipeline ML end-to-end


## Auteur

**Nesho Kanthakumar**
Étudiant en Data Science 
[GitHub](https://github.com/Nesho-k) · [LinkedIn](https://www.linkedin.com/in/nesho-kanthakumar-6354512a6/)
