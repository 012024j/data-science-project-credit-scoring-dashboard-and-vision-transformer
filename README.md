# Projet Data Science : Dashboard de Credit Scoring & Veille Technique

## 📝 Description du Projet

Ce projet a pour objectif de développer un **dashboard interactif** pour aider les chargés de relation client à visualiser et interpréter les décisions d’octroi de crédit. Il comprend également une **veille technique** portant sur une nouvelle méthode de modélisation de données d’images ou de textes.

---

## 🎯 Objectifs pédagogiques

- Réaliser un tableau de bord à destination d’un public non technique.
- Présenter de manière intelligible les résultats d’un modèle de classification de crédit.
- Implémenter et comparer une nouvelle approche de modélisation à une méthode existante (veille technique).
- Rédiger une note méthodologique claire et accessible sur l’approche testée.
- Déployer une application de visualisation dans le cloud (Render, Streamlit Cloud…).

---

## 📊 Partie 1 : Conception d’un Dashboard de Credit Scoring

### 🧠 Contexte

L’entreprise **Prêt à dépenser** souhaite accroître la **transparence** de ses décisions d’octroi de crédit. Le tableau de bord doit permettre aux chargés de clientèle de :

- visualiser le score de crédit d’un client,
- expliquer la prédiction,
- comparer un profil client avec des groupes similaires.

### ⚙️ Technologies utilisées

- **Python 3.9+**
- **Streamlit** pour l’interface utilisateur
- **Pandas**, **NumPy**, **Matplotlib**, **Plotly** pour la manipulation et la visualisation
- **Scikit-learn** pour la modélisation
- **SHAP** pour l’interprétabilité
- **Render.com** ou **Streamlit Cloud** pour le déploiement

### 📦 Fonctionnalités principales

- Visualisation du score de crédit sous forme de jauge
- Analyse locale et globale des variables explicatives (SHAP)
- Comparaison d’un client à la population globale et à des groupes similaires
- Graphiques mono- et bi-variés interactifs
- Saisie et modification des données pour recalculer la prédiction (optionnel)
- Déploiement en ligne

---

## 🧪 Partie 2 : Veille Technique – Classification d’Images

### 🎯 Objectif

Tester une **architecture récente de Deep Learning** (moins de 5 ans) sur un jeu de données d’images utilisé précédemment. Comparaison avec une approche plus classique de type CNN (VGG16).

### 🔬 Méthodologie

1. **Choix d’un modèle de l’état de l’art** : Vision Transformer (ViT)
2. **Sources bibliographiques** : Arxiv, PapersWithCode, Medium, etc.
3. **Expérimentation** :
   - Prétraitement des images
   - Extraction des features avec CNN vs ViT
   - Évaluation comparative via classification report et matrice de confusion
4. **Rédaction d’une note méthodologique**

### 📈 Résultats

- Le modèle **ViT** surpasse **VGG16** sur toutes les métriques (précision, rappel, f1-score)
- Meilleure robustesse, meilleure capacité à distinguer les classes proches
- Interprétation visuelle plus claire grâce aux cartes d’attention

---

🚀 Déploiement
L’application est déployée sur Render/Streamlit Cloud.

📚 Ressources utilisées
Streamlit Documentation

PapersWithCode - Vision Transformer

SHAP Explainer

Scikit-learn Documentation

🙋‍♂️ Auteurs
Oumou Faye — Data Scientist
Mentor : Medina Hadjem


