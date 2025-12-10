# 🎮 PUBG Win Prediction — Research‑Grade Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/ML-Model-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Live%20App-Online)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Ops-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)
![Scikit‑Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Tools-f7931e?logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)
![GitHub Stars](https://img.shields.io/badge/GitHub-Stars-lightgrey?style=social)
![GitHub Forks](https://img.shields.io/badge/GitHub-Forks-lightgrey?style=social)
![Live App](https://img.shields.io/badge/Streamlit‑App%20Link-blue?logo=streamlit)-brightgreen)

### 🔗 **Live Demo:** [https://pubg-game-win-prediction-app.streamlit.app/](https://pubg-game-win-prediction-app.streamlit.app/)

### 🔧 Technology Stack Logos

<p align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="55" />
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="55" />
  <img src="https://numpy.org/images/logo.svg" width="65" />
  <img src="https://pandas.pydata.org/static/img/pandas_mark.svg" width="60" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="80" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" width="55" />
  <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="120" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/SHAP_logo.png" width="70" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/CatBoostLogo.png" width="80" />
</p>

## 📌 **Overview****

This repository contains a fully developed **ML system for predicting PUBG match outcomes** (win probability / final placement) using engineered gameplay statistics and boosted decision models. The project follows a **research‑style workflow**—from data exploration to feature engineering, modeling, explainability, and deployment.

A complete **interactive Streamlit web app** is included, offering:

* 🔹 **Single‑player prediction interface**
* 🔹 **Batch CSV inference** with downloadable results
* 🔹 **EDA tools** including heatmaps, distributions, and high‑kill analysis
* 🔹 **SHAP‑based explainability**
* 🔹 **Model Card** documenting assumptions & reproducibility

This README is written in a **portfolio‑ready format**, suitable for GitHub and recruiters reviewing end‑to‑end ML engineering skills.

---

## 🧠 Problem Statement

For example, **given mid‑match player statistics such as damage dealt, movement distance, boosts used, and kills achieved**, predicting the likely placement can help analysts understand performance patterns, coaches refine player strategy, and game designers study balance dynamics.

Given PUBG player‑match statistics, predict the **final match outcome** (win probability or placement score). The model leverages engineered behavioral features representing:

* Movement patterns
* Aggression (kills, headshots, damage)
* Resource usage (boosts/heals)
* Efficiency metrics (damage per kill, kill‑without‑movement indicator)

This resembles real‑world esports analytics, with applications in:

* Player performance modeling
* Strategy optimization
* Esports coaching tools
* Player ranking systems

---

## 📂 Project Structure

```
├── streamlit_pubg_app.py        # Full Streamlit Web App  
├── PUBG Game Prediction.ipynb   # Research + training notebook
├── pubg_best_model.pkl          # (user provided) ML model
├── pubg_preprocess.pkl          # Optional preprocessing pipeline
├── requirements.txt             # Dependencies
└── README.md
```
## 🏛️ System Architecture Diagram

```mermaid
graph TD;
    A[User Input via Streamlit UI]
    B[Feature Engineering Layer]
    C[Preprocessing Pipeline]
    D[CatBoost ML Model]
    E[Prediction Output]
    F[SHAP Explainability Module]

    A --> B --> C --> D --> E
    D --> F
---

## 🏗️ Data Pipeline

### 1️⃣ **Data Cleaning**

* Handled missing numeric values
* Removed non-essential text fields for model training
* Processed extreme outliers (kills, distances)

### 2️⃣ **Feature Engineering (Core Innovation)**

Engineered features used across prediction, batch inference, and SHAP:

* `totalDistance = walk + ride + swim`
* `killswithoutMoving = kills>0 & totalDistance==0`
* `headshotRate = headshotKills/kills` (safe division)
* `damage_per_kill = damageDealt/kills`

These features capture meaningful in‑game behavior patterns that directly influence prediction quality.

---

## 🤖 Model Development

This project uses publicly available PUBG match statistics (typically hundreds of thousands of player‑match records) sourced from community datasets, providing a large and diverse foundation for training a robust predictive model.

The final model (stored as `pubg_best_model.pkl`) is typically a **CatBoost / Gradient Boosting Regressor**.

### ✔️ Why Gradient Boosting?

* Handles nonlinear interactions
* Works well with skewed esports data
* Robust to missing values
* Produces interpretable feature importance

### ✔️ Training Notebook Includes:

* Hyperparameter optimization
* Train–validation split analysis
* Error curve diagnostics
* SHAP explainability

---

## 🧪 Evaluation Metrics

Depending on problem framing:

* **Regression:** MAE, RMSE, R²
* **Classification-like (win probability):** Log Loss, AUC

The best model achieved strong generalisation with balanced errors across aggressive and passive playstyles.

---

## 🌐 Deployment — Streamlit App

**End-to-end deployment workflow:**

1. **User Input (Streamlit UI):** Player stats (kills, distances, damage, boosts, heals, headshots) are entered via interactive widgets.
2. **Feature Engineering Layer:** The app computes `totalDistance`, `killswithoutMoving`, `headshotRate`, and `damage_per_kill` on the fly.
3. **Preprocessing Pipeline:** If available, the saved `pubg_preprocess.pkl` transforms the engineered features (scaling/encoding/selection).
4. **Model Inference:** The `pubg_best_model.pkl` CatBoost/GBM model generates a placement score or win probability.
5. **UI Output:** Predictions are surfaced back to the user as metrics, tables, and visual components (metrics card, batch CSV download, SHAP plots).

### **Live App:** [https://pubg-game-win-prediction-app.streamlit.app/](https://pubg-game-win-prediction-app.streamlit.app/)

The UI includes:

### 🟦 Home Dashboard

* Project overview
* Model load status
* Documentation links

### 🟩 Single Prediction

Interactive sliders for:

* kills, damage, distances, boosts, heals, headshots
* Auto‑engineered features applied under the hood

### 🟧 Batch Prediction

* Upload CSV
* Auto feature engineering + preprocessing + inference
* Downloadable results

### 🟨 EDA Module

* Correlation heatmap
* Kills distribution analysis
* High‑kill segmentation

### 🟥 Explainability (SHAP)

* Global feature importance
* SHAP summary plots
* Behavioural feature impact visualisation

### 🟪 Model Card

* Intended use
* Limitations
* Reproducibility steps

---

## 🔬 Explainability — SHAP Results

The most influential **engineered features** in this project are:

* `totalDistance` — captures overall mobility and map presence
* `damage_per_kill` — reflects combat efficiency
* `headshotRate` — approximates mechanical skill and accuracy
* `killswithoutMoving` — flags suspicious or risky behaviour patterns

SHAP analysis shows how these features shape the model’s decisions:

* **Higher totalDistance** generally pushes predictions towards **better placements**, as players who move more tend to survive longer.
* **Higher damage_per_kill** increases win probability, indicating efficient damage conversion into kills.
* **Higher headshotRate** is associated with stronger performance, aligning with the idea of skilled aimers doing better overall.
* A **positive killswithoutMoving flag** often pulls predictions downward, signalling unrealistic or low‑quality scenarios.

By combining these engineered features with SHAP explanations, the project not only predicts outcomes but also provides **behaviour‑level insights** into *why* certain players are more likely to win.

---

## 🔁 Reproducibility Instructions

### **Training**

```python
import joblib
joblib.dump(model, 'pubg_best_model.pkl')
joblib.dump(preprocess, 'pubg_preprocess.pkl')
```

### **Running the App Locally**

```bash
pip install -r requirements.txt
streamlit run streamlit_pubg_app.py
```

### **Model Inputs** (for inference)

Numeric-only features:

```
kills, damageDealt, walkDistance, rideDistance, swimDistance,
boosts, heals, headshotKills
```

Engineered automatically:

```
totalDistance, killswithoutMoving, headshotRate, damage_per_kill
```

---

## 🧾 Requirements

Example `requirements.txt`:

```
streamlit
pandas
numpy
joblib
matplotlib
seaborn
shap
catboost
scikit-learn
```

---

## ⚠️ Limitations

* Not intended for ranking real esports players
* Model trained on public data — gameplay dynamics may differ
* MatchType not currently used as a categorical input
* Extreme outliers may reduce prediction stability

---

## 📜 License

This project is open-source for educational and portfolio demonstration purposes.

---

## ⭐ Acknowledgements

* PUBG public dataset community
* Streamlit for rapid prototyping
* SHAP library authors
* CatBoost research team

---

## 🙌 Want to Improve This Project?

You can contribute by:

* Adding matchType embeddings
* Improving handling of team-based matches
* Building a leaderboard visualization
* Enhancing the real-time inference engine

---

### 🎯 **This README is portfolio-ready and demonstrates skills in:**

* Data engineering
* Feature design
* Applied machine learning
* Explainability (SHAP)
* Model deployment
* Software engineering
* Technical documentation

