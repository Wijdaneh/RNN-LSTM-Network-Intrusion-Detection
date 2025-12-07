# 🧠 Détection d'Anomalies Séquentielles – RNN / LSTM / GRU  
Analyse de données en temps réel & interface Streamlit

<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/lstm/lstm.png" width="350"/>
</p>

---

## 🔖 Badges

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-RNN%2FLSTM%2FGRU-red"/>
  <img src="https://img.shields.io/badge/Status-Active-success"/>
</p>

---

## 📌 Description du Projet

Ce projet implémente un système intelligent basé sur **RNN / LSTM / GRU** pour :

- analyser des données réseau ou séquentielles,  
- détecter des comportements anormaux,  
- classer ou prédire des événements dans un flux,  
- offrir une visualisation avancée via **Streamlit**.

Il simule un scénario "temps réel" en traitant des traces PCAP ou des séries temporaires chargées par l’utilisateur.

---

## 🎯 Objectifs

- 🏗 **Construire un pipeline complet** : prétraitement → entraînement → inférence  
- 📡 **Analyser un flux réseau PCAP comme un trafic en temps réel**  
- 🔍 **Détection d’anomalies et classification séquentielle**  
- 🎨 **Proposer une interface visuelle professionnelle** avec Streamlit  
- 📊 **Offrir des graphiques avancés** pour comprendre les modèles  
- 🧠 **Comprendre l’impact du choix du modèle (RNN/LSTM/GRU)**  

---

## 🛣 Roadmap

| Tâche | Statut |
|------|--------|
| Prétraitement du dataset | ✔️ Terminé |
| Entraînement LSTM | ✔️ Terminé |
| Interface Streamlit (upload PCAP) | ✔️ Terminé |
| Visualisations avancées ML | ✔️ Terminé |
| Support du mode "pseudo temps réel" | ✔️ Terminé |
| Ajout du modèle GRU | ✔️ Terminé |
| Ajout de l’analyse statistique des paquets | 🔄 En progrès |
| Intégration d’un tableau de bord interactif avancé | 🔜 À venir |
| Mode capture live depuis l’interface | 🔜 À venir |

---

## 📂 Structure du Projet

```
RNN_LSTM_Projet/
│
├── data/
│   ├── attack/
│   └── labeled/
│
├── models/
│   ├── lstm_model.h5
│   ├── rnn_model.h5
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── visualize.py
│   └── predict.py
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Technologies Utilisées

| Technologie | Rôle |
|------------|------|
| **TensorFlow / Keras** | Modèles RNN, LSTM, GRU |
| **Streamlit** | Interface web |
| **Scikit-learn** | Normalisation, métriques |
| **Matplotlib, Seaborn, Plotly** | Visualisation |
| **Scapy** | Lecture & parsing PCAP |
| **Pandas / NumPy** | Prétraitement des données |

---

## 🔧 Installation & Exécution

### 1️⃣ Cloner le projet  
```bash
git clone https://github.com/<Ton-GitHub>/RNN_LSTM_Projet.git
cd RNN_LSTM_Projet
```

### 2️⃣ Créer un environnement virtuel  
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Installer les dépendances  
```bash
pip install -r requirements.txt
```

### 4️⃣ Lancer Streamlit  
```bash
cd streamlit_app
streamlit run app.py
```

---

## 📊 Visualisations Incluses

- Courbes **Loss / Accuracy**
- Matrice de confusion
- Précision, Recall, F1
- ROC / AUC
- Heatmaps des séquences
- Graphiques dynamiques Streamlit

---

## 🧪 Fonctionnalités de l’App

✔ Upload d’un fichier PCAP (jusqu’à >200 MB)  
✔ Extraction automatique des caractéristiques  
✔ Analyse séquentielle via LSTM / GRU  
✔ Visualisation temps réel simulé  
✔ Affichage des prédictions modèle  
✔ Tableau de bord dynamique

---

## 👤 Auteur

**Wijdane Hachani**  
Étudiante en ingénierie informatique – Cybersécurité & IA  
Développement Machine Learning, Deep Learning & Streamlit

---

## 📜 Licence  
**MIT License**

