from __future__ import annotations

import os
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ---------------------------------------------------------------------
# Streamlit layout config
# ---------------------------------------------------------------------
st.set_page_config(page_title="PUBG Win Prediction", layout="wide")

MODEL_PATH = "pubg_best_model.pkl"
PREPROCESS_PATH = "pubg_preprocess.pkl"

# ---------------------------------------------------------------------
# Load model & preprocessor
# ---------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load model from {path}: {e}")
        return None


@st.cache_resource(ttl=3600)
def load_preprocessor(path: str = PREPROCESS_PATH):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load preprocessor from {path}: {e}")
        return None


model = load_model()
preprocess = load_preprocessor()

# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------
def _get_model_num_features(model) -> Optional[int]:
    """
    Try to infer how many features the trained model expects.
    Works for CatBoost; falls back to sklearn-style if possible.
    """
    # 1) CatBoost internal object
    try:
        obj = getattr(model, "_object", None)
        if obj is not None and hasattr(obj, "_get_num_feature"):
            return obj._get_num_feature()
    except Exception:
        pass

    # 2) If feature names are stored (sklearn-like)
    try:
        names = getattr(model, "feature_names_", None)
        if names is not None and len(names) > 0:
            return len(names)
    except Exception:
        pass

    # 3) If feature_names_in_ exists (sklearn)
    try:
        names_in = getattr(model, "feature_names_in_", None)
        if names_in is not None and len(names_in) > 0:
            return len(names_in)
    except Exception:
        pass

    return None


def _prepare_for_model(model, X) -> np.ndarray:
    """
    Convert X (DataFrame or array) to a NumPy array whose number of columns
    matches what the model expects. If we have too few columns, pad with zeros;
    if too many, truncate extra columns.
    This avoids CatBoost errors like:
        'Feature k is present in model but not in pool'
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    expected = _get_model_num_features(model)
    if expected is None or X_arr.ndim != 2:
        # We don't know, just return as-is
        return X_arr

    current = X_arr.shape[1]

    if current == expected:
        return X_arr

    if current < expected:
        # pad with zeros on the right
        pad_width = expected - current
        pad = np.zeros((X_arr.shape[0], pad_width), dtype=X_arr.dtype)
        X_arr = np.hstack([X_arr, pad])
    elif current > expected:
        # truncate extra columns
        X_arr = X_arr[:, :expected]

    return X_arr

def engineer_features_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized feature engineering that mirrors your notebook logic.
    Adapt this function if your notebook uses more/different features.
    """
    df = df_in.copy()

    # totalDistance = ride + walk + swim
    if {"rideDistance", "walkDistance", "swimDistance"}.issubset(df.columns):
        df["totalDistance"] = (
            df["rideDistance"].fillna(0.0)
            + df["walkDistance"].fillna(0.0)
            + df["swimDistance"].fillna(0.0)
        )
    else:
        df["totalDistance"] = 0.0

    # killswithoutMoving flag: kills > 0 and totalDistance == 0
    if "kills" in df.columns:
        df["killswithoutMoving"] = (
            (df["kills"].fillna(0) > 0) & (df["totalDistance"] == 0)
        ).astype(int)
    else:
        df["killswithoutMoving"] = 0

    # headshotRate = headshotKills / kills (safe divide)
    if {"headshotKills", "kills"}.issubset(df.columns):
        kills_safe = df["kills"].replace(0, np.nan)
        df["headshotRate"] = (df["headshotKills"] / kills_safe).fillna(0.0)
    else:
        df["headshotRate"] = 0.0

    # damage_per_kill
    if {"damageDealt", "kills"}.issubset(df.columns):
        kills_safe = df["kills"].replace(0, np.nan)
        df["damage_per_kill"] = (df["damageDealt"] / kills_safe).fillna(0.0)
    else:
        df["damage_per_kill"] = 0.0

    return df


def engineer_features_single(data: dict) -> pd.DataFrame:
    """Convenience wrapper for single-row input."""
    df = pd.DataFrame([data])
    return engineer_features_df(df)


# ---------------------------------------------------------------------
# Feature alignment helper (for sklearn-style models)
# ---------------------------------------------------------------------
def _align_features_for_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    If the model exposes feature_names_in_ (typical sklearn),
    align DataFrame columns to that list and add any missing ones with 0.

    For CatBoost, feature_names_in_ usually does not exist; in that case
    we simply return df unchanged and rely on the numeric order.
    """
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return df

    out = df.copy()
    for col in feature_names:
        if col not in out.columns:
            out[col] = 0
    return out[list(feature_names)]


# ---------------------------------------------------------------------
# Safe prediction wrapper – key fix for CatBoost
# ---------------------------------------------------------------------
def safe_predict(model, X):
    """
    Robust prediction wrapper:
    - Aligns/pads/truncates features to match model expectations.
    - Always passes NumPy arrays to CatBoost (no column names).
    """
    try:
        # If X is a DataFrame, keep your engineered columns first
        if isinstance(X, pd.DataFrame):
            X_prepared = engineer_features_df(X) if "totalDistance" not in X.columns else X
        else:
            X_prepared = X

        data = _prepare_for_model(model, X_prepared)

        preds = model.predict(data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data)
        except Exception:
            proba = None

    return np.asarray(preds), proba


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.title("🎮 PUBG Win Prediction ")

menu = st.sidebar.selectbox(
    "Choose page",
    ["Home", "Single Prediction", "Batch Prediction", "EDA", "Explainability", "Model Card"],
)

# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
if menu == "Home":
    st.header("Project Overview")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.write(
            """
This app predicts PUBG match outcome (e.g., final placement or win probability)
using a pre-trained machine learning model. It features:
- 🔹 Single-player interactive prediction  
- 🔹 Batch CSV predictions + downloadable results  
- 🔹 EDA tools (correlation heatmaps, distributions, high-kill analysis)  
- 🔹 SHAP-based model explainability  
- 🔹 A simple Model Card with reproducibility notes  

Core engineered features include:
`totalDistance`, `killswithoutMoving`, `headshotRate`, and `damage_per_kill`.
Adapt these to match my training notebook exactly for best results.
            """
        )

    with c2:
        if model is None:
            st.warning(
                "Model NOT found. Place `pubg_best_model.pkl` in this folder and restart the app."
            )
        else:
            st.success("✅ Model loaded successfully.")
        if preprocess is not None:
            st.info("Preprocessing pipeline detected (`pubg_preprocess.pkl`).")


# ---------------------------------------------------------------------
# SINGLE PREDICTION
# ---------------------------------------------------------------------
elif menu == "Single Prediction":
    st.header("Single Player Prediction")

    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            kills = st.number_input("kills", min_value=0, max_value=100, value=0)
            damage = st.number_input(
                "damageDealt", min_value=0.0, max_value=20000.0, value=0.0, step=1.0
            )
            walk = st.number_input(
                "walkDistance", min_value=0.0, max_value=20000.0, value=0.0, step=0.01
            )

        with col2:
            ride = st.number_input(
                "rideDistance", min_value=0.0, max_value=20000.0, value=0.0, step=0.01
            )
            swim = st.number_input(
                "swimDistance", min_value=0.0, max_value=20000.0, value=0.0, step=0.01
            )
            boosts = st.number_input("boosts", min_value=0, max_value=100, value=0)

        with col3:
            heals = st.number_input("heals", min_value=0, max_value=100, value=0)
            headshotKills = st.number_input(
                "headshotKills", min_value=0, max_value=100, value=0
            )
            # NOTE: matchType is used only for UI storytelling, NOT fed to the model
            matchType = st.selectbox("matchType (not used in model)", ["solo", "duo", "squad"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None:
            st.error("No model available. Train and save `pubg_best_model.pkl` first.")
        else:
            # Build raw feature dict (without matchType – we assume numeric-only model)
            raw_data = {
                "kills": kills,
                "damageDealt": damage,
                "walkDistance": walk,
                "rideDistance": ride,
                "swimDistance": swim,
                "boosts": boosts,
                "heals": heals,
                "headshotKills": headshotKills,
            }

            feat_df = engineer_features_single(raw_data)

            # Apply preprocessing if available; otherwise align & predict directly
            if preprocess is not None:
                try:
                    X = preprocess.transform(feat_df)
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    X = feat_df
            else:
                X = feat_df

            preds, proba = safe_predict(model, X)
            pred_val = float(preds[0])

            st.metric("Predicted value", f"{pred_val:.4f}")
            if proba is not None and proba.ndim == 2 and proba.shape[1] > 1:
                prob = float(proba[0, 1])
                st.write(f"Estimated win probability: **{prob:.3f}**")


# ---------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------
elif menu == "Batch Prediction":
    st.header("Batch CSV Predictions")

    uploaded = st.file_uploader(
        "Upload a CSV with PUBG features (kills, damageDealt, distances, boosts, heals, etc.). "
        "Text columns like matchType will be ignored for the model.",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            # 1. Read & show preview
            raw = pd.read_csv(uploaded)
            st.write("Preview of uploaded file:")
            st.dataframe(raw.head())

            if model is None:
                st.error("Model not loaded. Train and save `pubg_best_model.pkl` first.")
            else:
                # 2. Apply SAME feature engineering used during training
                feats_engineered = engineer_features_df(raw)

                # 3. Keep ONLY numeric features for the model
                X_numeric = feats_engineered.select_dtypes(include=[np.number]).copy()

                if X_numeric.shape[1] == 0:
                    st.error("No numeric columns available for prediction.")
                else:
                    # 4. Apply preprocessor if available
                    if preprocess is not None:
                        try:
                            X_for_model = preprocess.transform(X_numeric)
                        except Exception as e:
                            st.warning(
                                f"Preprocessor failed on uploaded data ({e}). "
                                "Falling back to raw numeric features."
                            )
                            X_for_model = X_numeric
                    else:
                        X_for_model = X_numeric

                    # 5. Predict using safe_predict (handles CatBoost feature shape)
                    preds, probas = safe_predict(model, X_for_model)

                    # 6. Build output DataFrame for user
                    out = raw.copy()   # keep original columns, incl. matchType
                    out["prediction"] = preds
                    if (
                        probas is not None
                        and probas.ndim == 2
                        and probas.shape[1] > 1
                    ):
                        out["win_prob"] = probas[:, 1]

                    # 7. Show and allow download
                    st.write("### Results preview")
                    st.dataframe(out.head())

                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions as CSV",
                        csv,
                        file_name="pubg_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Failed to process uploaded CSV: {e}")


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    uploaded = st.file_uploader(
        "Upload a CSV for EDA (typically your training data or a sample).",
        type=["csv"],
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())

        num = df.select_dtypes(include=[np.number])

        if num.shape[1] == 0:
            st.warning("No numeric columns found for EDA.")
        else:
            st.subheader("Correlation Heatmap (numeric only)")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if "kills" in df.columns:
            st.subheader("Kills Distribution (Histogram)")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(df["kills"], bins=30, ax=ax2)
            st.pyplot(fig2)

            st.subheader("High Kills (≥ 15) — Binned Counts")
            high = df[df["kills"] >= 15].copy()
            if high.shape[0] > 0:
                bins = [15, 20, 25, 30, 40, 100]
                labels = ["15–19", "20–24", "25–29", "30–39", "40+"]
                high["kills_bin"] = pd.cut(
                    high["kills"], bins=bins, labels=labels, right=False
                )
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                sns.countplot(x="kills_bin", data=high, ax=ax3)
                st.pyplot(fig3)
            else:
                st.info("No rows with kills ≥ 15 in this dataset.")


# ---------------------------------------------------------------------
# EXPLAINABILITY (SHAP)
# ---------------------------------------------------------------------
elif menu == "Explainability":
    st.header("Model Explainability (SHAP)")

    if model is None:
        st.error("Model not loaded. Place `pubg_best_model.pkl` in this folder.")
    else:
        st.markdown(
            """
Upload a **sample of your PUBG dataset** (a few hundred to a few thousand rows).
The app will compute **SHAP values** to show which features drive the model’s predictions.
            """
        )

        sample_file = st.file_uploader(
            "Upload CSV for SHAP analysis", type=["csv"]
        )

        if sample_file is not None:
            try:
                # 1. Load & preview
                df_shap = pd.read_csv(sample_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df_shap.head())

                # 2. Apply SAME feature engineering as training
                df_engineered = engineer_features_df(df_shap)

                # Numeric features only
                X_raw = df_engineered.select_dtypes(include=[np.number]).copy()
                X_raw = X_raw.dropna()

                if len(X_raw) == 0:
                    st.error("No valid numeric rows after feature engineering.")
                else:
                    # 3. Sample for speed
                    sample_size = min(500, len(X_raw))
                    X_raw_sample = X_raw.sample(sample_size, random_state=42)

                    # 4. Apply preprocessor if available
                    if preprocess is not None:
                        try:
                            X_for_model = preprocess.transform(X_raw_sample)
                        except Exception as e:
                            st.warning(
                                f"Preprocess transform failed for SHAP; "
                                f"using raw numeric features instead. ({e})"
                            )
                            X_for_model = X_raw_sample.to_numpy()
                    else:
                        X_for_model = X_raw_sample.to_numpy()

                    # 5. Force feature count to match CatBoost model
                    X_for_model = _prepare_for_model(model, X_for_model)

                    st.info(
                        f"Using {X_for_model.shape[0]} rows and "
                        f"{X_for_model.shape[1]} features for SHAP analysis."
                    )

                    # 6. Build SHAP explainer & compute values
                    with st.spinner("Computing SHAP values… this may take a moment."):
                        explainer = shap.Explainer(
                            lambda X: model.predict(_prepare_for_model(model, X)),
                            X_for_model,
                        )
                        shap_values = explainer(X_for_model)

                    # 7. Global SHAP feature importance (bar)
                    st.subheader("Global SHAP Feature Importance")

                    fig_bar = plt.figure(figsize=(8, 5))
                    shap.plots.bar(shap_values, max_display=15, show=False)
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                    # 8. SHAP summary / beeswarm plot
                    #    IMPORTANT: use X_for_model here, not X_raw_sample
                    st.subheader("SHAP Summary Plot (feature impact)")

                    fig_sum = plt.figure(figsize=(8, 5))
                    shap.summary_plot(
                        shap_values,
                        X_for_model,          # <--- same data used for shap_values
                        max_display=15,
                        show=False,
                    )
                    st.pyplot(fig_sum)
                    plt.close(fig_sum)

                    st.caption(
                        "Each dot is a player. Color = feature value (blue=low, red=high). "
                        "Position on x-axis = how much that feature pushed the prediction "
                        "higher or lower."
                    )

            except Exception as e:
                st.error(f"Failed to run SHAP analysis: {e}")
        else:
            st.info("Upload a CSV to start SHAP analysis.")

            
# ---------------------------------------------------------------------
# MODEL CARD
# ---------------------------------------------------------------------
elif menu == "Model Card":
    st.header("Model Card & Reproducibility")

    st.markdown(
        """
**Model file:** `pubg_best_model.pkl`  
**Intended use:** Educational / portfolio demo / experimentation.  
**Not intended for:** Real-money esports ranking, commercial decision systems, or high-stakes use.  

**Data:** PUBG match/player statistics (kills, damage, distances, items, etc.).  
The model is typically a gradient-boosting style model (e.g., CatBoost).
        """
    )

    st.subheader("How to Reproduce")
    st.markdown(
        """
1. Open your training notebook (`PUBG Game Prediction.ipynb`).  
2. Train your final model and save it:

   ```python
   import joblib
   joblib.dump(model, "pubg_best_model.pkl")        # required
   joblib.dump(preprocess, "pubg_preprocess.pkl")   # optional, if you have a pipeline
   """
   )


