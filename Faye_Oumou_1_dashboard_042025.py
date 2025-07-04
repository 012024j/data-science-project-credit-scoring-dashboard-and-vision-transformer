import streamlit as st

# Première instruction de streamlit
st.set_page_config(
    page_title="Dashboard Crédit",
    layout="wide",
    page_icon="credit_score_demo.png" 
)

# IMPORTS 
import base64
import io
from PIL import Image
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import scipy.special
from streamlit.runtime.scriptrunner import get_script_run_ctx

# CSS 
st.markdown("""
<style>
div[data-baseweb="select"] {max-width: 180px !important; margin-left: auto !important; margin-right: auto !important;}
.css-1cpxqw2 {max-width: 200px !important; margin-left: auto !important; margin-right: auto !important;}
input, select, textarea {font-size: 16px !important; font-weight: bold !important; text-align: center !important;}
label {display: flex; justify-content: center; font-size: 16px !important; font-weight: 600 !important; margin-bottom: 5px !important;}
div.stButton > button:first-child {background-color: black; color: white; font-weight: bold; width: 100%; height: 45px; border-radius: 10px;}
select {font-weight: bold !important; text-align-last: center !important;}
.pred-result-box {display: flex; flex-direction: column; align-items: center; margin-top: 20px;}
.stTabs [data-baseweb="tab-list"] {justify-content: center !important; width: 100%; gap: 60px;}
.stTabs [data-baseweb="tab"] > div {font-size: 20px !important; font-weight: 700 !important; font-family: 'Roboto', 'Arial', sans-serif !important; color: #333 !important; padding: 10px 0; transition: all 0.3s ease; text-align: center !important; border: 2px solid transparent; border-radius: 10px;}
.stTabs [data-baseweb="tab"][aria-selected="true"] > div {color: red !important; border-bottom: 4px solid red !important;}
.stTabs [data-baseweb="tab"] > div:hover {color: #e60000 !important; text-decoration: underline; border-color: #e60000 !important;}
</style>
""", unsafe_allow_html=True)

# ------ Session State ------
if "show_pred" not in st.session_state:
    st.session_state["show_pred"] = False
if "show_shap" not in st.session_state:
    st.session_state["show_shap"] = False
if "show_shap_global_menu0" not in st.session_state:
    st.session_state["show_shap_global_menu0"] = False

# ------ Data & Image ------
@st.cache_data
def load_data():
    return pd.read_csv("application_test.csv")

# Chargement du modèle avec gestion d'erreur
@st.cache_resource
def load_model():
    try:
        return joblib.load("Best_XGBoost_Business_Model.pkl")
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

data = load_data()
model = load_model()

API_URL = "https://streamlit-fastapi-app.onrender.com"
endpoint = f"{API_URL}/predict"

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

image_path = "credit_score_demo.png"
try:
    image = Image.open(image_path)
    image_b64 = image_to_base64(image)
except FileNotFoundError:
    st.warning("Image de démonstration introuvable.")
    image_b64 = ""

st.markdown(f"""
<div style="display: flex; flex-direction: column; align-items: center; margin-top: -60px;">
    <h2 style="margin-bottom: 10px; font-weight: bold; color: #000; font-size: 35px;">
        Évaluation de la solvabilité du client
    </h2>
    <img src="data:image/png;base64,{image_b64}" width="280" style="border-radius: 10px;"/>
</div>
""", unsafe_allow_html=True)

# ------ Tabs ------
menu = st.tabs([
    "Critères du client et son score de solvabilité", 
    "Comparaison du client avec la population étudiée", 
    "Influence du profil client sur la décision finale de prêt"
])

NOM_FEATURES_CLEAN = {
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Mensualité (€)",
    "CREDIT_TERM": "Durée de remboursement (mois)",
    "EXT_SOURCE_1": "Score de solvabilité externe n°1",
    "EXT_SOURCE_2": "Score de solvabilité externe n°2",
    "EXT_SOURCE_3": "Score de solvabilité externe n°3",
    "DAYS_BIRTH": "Âge (années)",
    "CODE_GENDER_M": "Genre",
    "CNT_CHILDREN": "Nombre d'enfants à charge",
    "DAYS_EMPLOYED": "Années d'ancienneté (emploi)",
    "CREDIT_INCOME_PERCENT": "Ratio crédit / revenu",
    "ANNUITY_INCOME_PERCENT": "Ratio mensualité / revenu",
    "DAYS_EMPLOYED_PERCENT": "Ratio ancienneté / âge",
    "NAME_INCOME_TYPE_Working": "Type de revenu",
    "REGION_RATING_CLIENT_W_CITY": "Note région (avec ville)",
    "REGION_RATING_CLIENT": "Note région",
    "REG_CITY_NOT_WORK_CITY": "Travail dans une autre ville",
    "FLAG_OWN_REALTY": "Propriétaire d'un bien immobilier",
    "OCCUPATION_TYPE_Laborers": "Travailleur de la classe ouvrière"
}
NOM_FEATURES_CLEAN_INV = {v: k for k, v in NOM_FEATURES_CLEAN.items()}

# ------ Onglet 0 ------
with menu[0]:
    st.markdown("<div style='text-align: center; font-size: 35px; font-weight: 600; margin-bottom: 10px;'>Saisir les caractéristiques du client</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Saisie des montants principaux valeur par défault
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Revenu annuel (€)</div>", unsafe_allow_html=True)
        montant_brut = 100000
        montant_formate = f"{montant_brut:,}".replace(",", ".")
        revenu_str = st.text_input("", value=montant_formate, key="income")
        try:
            amt_income = int(revenu_str.replace(".", "").replace(" ", ""))
        except ValueError:
            amt_income = 0
    with col2:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Montant du crédit (€)</div>", unsafe_allow_html=True)
        montant_brut = 5000
        montant_formate = f"{montant_brut:,}".replace(",", ".")
        revenu_str = st.text_input("", value=montant_formate, key="amt_credit")
        try:
            amt_credit = int(revenu_str.replace(".", "").replace(" ", ""))
        except ValueError:
            amt_credit = 0
    with col3:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Mensualité du crédit (€)</div>", unsafe_allow_html=True)
        amt_annuity = st.number_input("", value=100, key="annuity")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Âge (années)</div>", unsafe_allow_html=True)
        age = st.number_input("", value=45, key="age")
    with col5:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Durée de remboursement (mois)</div>", unsafe_allow_html=True)
        credit_term = st.number_input("", value=24, key="term")
    with col6:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Année d'ancienneté (emploi)</div>", unsafe_allow_html=True)
        anciennete = st.number_input("", value=20, key="anciennete")

    col7, col8, col9 = st.columns(3)
    with col7:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Score de solvabilité externe n°1</div>", unsafe_allow_html=True)
        ext1 = st.number_input("", value=0.95, key="ext1")
    with col8:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Score de solvabilité externe n°2</div>", unsafe_allow_html=True)
        ext2 = st.number_input("", value=0.96, key="ext2")
    with col9:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Score de solvabilité externe n°3</div>", unsafe_allow_html=True)
        ext3 = st.number_input("", value=0.97, key="ext3")

    col10, col11, col12 = st.columns(3)
    with col10:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Nombre d'enfants à charge</div>", unsafe_allow_html=True)
        cnt_children = st.number_input("  ", min_value=0, value=0, key="cnt_children")
    with col11:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Note de la région (ville 1: faible, 3: bonne)</div>", unsafe_allow_html=True)
        region_city_rating = st.selectbox("  ", [1, 2, 3], index=2, key="region_city")
    with col12:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Note de la région (générale 1: faible, 3: bonne)</div>", unsafe_allow_html=True)
        region_rating = st.selectbox("  ", [1, 2, 3], index=2, key="region_general")

    col13, col14, col15, col16, col17 = st.columns(5)
    with col13:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Genre</div>", unsafe_allow_html=True)
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
        with subcol2:
            genre = st.radio(" ", ["Homme", "Femme"], index=0, key="genre", label_visibility="collapsed")
        code_gender_m = int(genre == "Homme")
    with col14:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Travaille-t-il dans une autre ville ?</div>", unsafe_allow_html=True)
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
        with subcol2:
            city_diff = int(st.radio(" ", ["Oui", "Non"], index=1, key="travail_ville", label_visibility="collapsed") == "Oui")
    with col15:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Propriétaire d'un bien immobilier ?</div>", unsafe_allow_html=True)
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
        with subcol2:
            own_realty = int(st.radio(" ", ["Oui", "Non"], index=0, key="realty", label_visibility="collapsed") == "Oui")
    with col16:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Travailleur manuel ?</div>", unsafe_allow_html=True)
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
        with subcol2:
            laborer = int(st.radio(" ", ["Oui", "Non"], index=1, key="laborer", label_visibility="collapsed") == "Oui")
    with col17:
        st.markdown("<div style='text-align: center; font-size: 21px;'>Type de revenu</div>", unsafe_allow_html=True)
        subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
        with subcol2:
            income_type = st.radio("Typologie de revenu", ["Travail", "Autre revenu"], index=0, key="income_type", label_visibility="collapsed")
            income_working = int(income_type == "Travail")

    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

    # Calcul des ratios
    days_birth = -age * 365
    days_employed = -anciennete * 365
    credit_income_percent = amt_credit / amt_income if amt_income else 0
    annuity_income_percent = amt_annuity / amt_income if amt_income else 0
    days_employed_percent = abs(days_employed) / abs(days_birth) if days_birth else 0

    col_ratio1, col_ratio2, col_ratio3 = st.columns(3)
    with col_ratio1:
        st.markdown(f"<div style='background:#d0ebff; padding:18px; border-radius:5px; text-align:center;'><b>💳 Crédit / Revenu : {credit_income_percent*100:.2f}%</b></div>", unsafe_allow_html=True)
    with col_ratio2:
        st.markdown(f"<div style='background:#d3f9d8; padding:18px; border-radius:5px; text-align:center;'><b>📅 Mensualité / Revenu : {annuity_income_percent*100:.2f}%</b></div>", unsafe_allow_html=True)
    with col_ratio3:
        st.markdown(f"<div style='background:#fff3bf; padding:18px; border-radius:5px; text-align:center;'><b>⏳ Ancienneté / Âge : {days_employed_percent*100:.2f}%</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    features = {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "CREDIT_INCOME_PERCENT": credit_income_percent,
        "ANNUITY_INCOME_PERCENT": annuity_income_percent,
        "CREDIT_TERM": credit_term,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "DAYS_BIRTH": days_birth,
        "CODE_GENDER_M": code_gender_m,
        "CNT_CHILDREN": cnt_children,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_EMPLOYED_PERCENT": days_employed_percent,
        "NAME_INCOME_TYPE_Working": income_working,
        "REGION_RATING_CLIENT_W_CITY": region_city_rating,
        "REGION_RATING_CLIENT": region_rating,
        "REG_CITY_NOT_WORK_CITY": city_diff,
        "FLAG_OWN_REALTY": own_realty,
        "OCCUPATION_TYPE_Laborers": laborer,
    }
    st.session_state["features"] = features

    # --- Boutons ---
    spacer_left, col1, spacer_middle, col2, spacer_right = st.columns([2, 3, 0.5, 3, 2])
    with col1:
        if st.button("Évaluer la Solvabilité du client", key="btn_predire_centered"):
            st.session_state["show_pred"] = True
    with col2:
        if st.button("🔁 Réinitialiser le formulaire client", key="reset_form"):
            st.session_state.clear()
            st.rerun()

    # --- Résultat prédiction & SHAP global ---
    if st.session_state.get("show_pred"):
        # Mise en place la gestion d'erreur pour la requête API
        try:
            response = requests.post(endpoint, json={"features": features}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                bg_color = "#d3f9d8" if result['classe'] == "accepté" else "#ffe3e3"
                text_color = "#2f9e44" if result['classe'] == "accepté" else "#c92a2a"
                st.markdown(f"<div style='text-align: center; margin: 20px 0;'><span style='background-color: {bg_color}; color: {text_color}; padding: 10px 20px; font-size: 16px; font-weight: bold; border-radius: 8px;'>Décision d'octroi de crédit : {result['classe']}</span></div>", unsafe_allow_html=True)
                
                col_centered = st.columns([1, 2, 1])
                with col_centered[1]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["proba_defaut"] * 100,
                        title={"text": "Probabilité de défaut (%)"},
                        number={"suffix": " %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "crimson"},
                            "steps": [
                                {"range": [0, 30], "color": "#d4f4dd"},
                                {"range": [30, 70], "color": "#fff4ce"},
                                {"range": [70, 100], "color": "#f9dcdc"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": result["proba_defaut"] * 100
                            }
                        }
                    ))
                    fig.update_layout(margin=dict(t=30, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erreur API : {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion API : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

        # Utilisation du modèle chargé
        if model is not None:
            # --- Interprétation SHAP global ---
            st.markdown("""
            <div style='text-align: center; font-size: 16px; margin-top: 10px; margin-bottom: 30px;'>
                 ℹ️ <b>Interprétation du score de solvabilité :</b> Le score de solvabilité est un indicateur numérique. On l'utilise pour évaluer la solvabilité ou le risque d'un client à partir d'informations externes, comme des données de bureau de crédit ou d'autres bases de données financières. Plus le score est élevé (proche de 1), plus la probabilité de remboursement est forte.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <h2 style='text-align: center; margin-top: 30px;'>
            Top 10 des critères qui influencent le plus l'accord de crédit
            </h2>
            <div style='text-align: center; font-size: 25px; color: red; font-style: italic; margin-top: -0.3px; margin-bottom: -0.3px;'>
            Selon l'ensemble des clients étudiés
            </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 3])
                with col2:
                    st.markdown("""
                    <style>
                    details summary {font-size: 28px !important; font-weight: 1000 !important;}
                    </style>
                    """, unsafe_allow_html=True)
                    with st.expander("ℹ️ **Signification des valeurs: critères client ?**", expanded=False):
                        st.markdown("""
- **Genre :** 1 = Homme, 0 = Femme  
- **Travail :** 1 = Oui, 0 = Non  
- **Travailleur manuel :** 1 = Oui, 0 = Non  
- **Autre ville :** 1 = Oui, 0 = Non  
- **Propriétaire :** 1 = Oui, 0 = Non  
- **Type de revenu :** 1 = Travail salarié, 0 = Autre (pension, aide…)  
- **Scores de solvabilité :** proche de 1 = client plus fiable  
- **Ratios (ex. crédit / revenu) :** plus c'est bas, mieux c'est
                        """)

            try:
                sample = data.sample(n=min(1000, len(data)), random_state=42)
                X_sample = sample[features.keys()]
                booster_model = model.named_steps["clf"]
                shap_vals = shap.Explainer(booster_model)(X_sample)
                abs_vals = np.abs(shap_vals.values).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Critère": [NOM_FEATURES_CLEAN.get(k, k) for k in features.keys()],
                    "Importance": abs_vals
                }).sort_values(by="Importance", ascending=False).head(10)
                
                norm = Normalize(vmin=shap_df["Importance"].min(), vmax=shap_df["Importance"].max())
                colors = cm.Blues(norm(shap_df["Importance"]))
                
                fig, ax = plt.subplots(figsize=(11.5, 6))
                bars = ax.barh(shap_df["Critère"], shap_df["Importance"], color=colors, height=0.9)
                ax.set_xlabel("Importance moyenne (population étudiée)")
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
                ax.invert_yaxis()
                for spine in ["top", "right", "bottom", "left"]:
                    ax.spines[spine].set_visible(False)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', ha='left', fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                <div style='text-align: center; font-size: 16px; margin-top: 10px; margin-bottom: 30px;'>
                     ℹ️ <b>Interprétation globale :</b> Ce graphique présente l'impact moyen de chaque variable sur la décision de crédit, en analysant l'ensemble des clients. Les variables les plus influentes sont affichées en haut. Plus la valeur absolue est grande, plus l'effet est déterminant sur la décision.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse SHAP : {e}")

# ------ Onglet 1 : Comparaison ------
with menu[1]:
    # Titre centré
    st.markdown("""
        <div style='text-align: center; font-size: 35px; font-weight: 600; margin-bottom: 0.2px;'>
            Sélectionnez le critère client à comparer avec la population de référence :
        </div>
    """, unsafe_allow_html=True)

    data_display = data.copy()
    NOM_FEATURES_CLEAN_INV = {v: k for k, v in NOM_FEATURES_CLEAN.items()}

    # Hack CSS pour centrer le texte dans le selectbox
    st.markdown("""
        <style>
        div[data-baseweb="select"] {
            max-width: 350px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        div[data-baseweb="select"] > div {
            justify-content: center !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Selectbox centrée avec la légende juste au-dessus
    spacer1, select_col, spacer2 = st.columns([3, 2, 3])
    with select_col:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 4px;">
                <span style='font-size: 16px;'>
                    <b> Détails sur les valeurs des caractéristiques</b>
                    <span title="Genre : 1 = Homme, 0 = Femme&#10;
Travail : 1 = Oui, 0 = Non&#10;
Travailleur manuel : 1 = Oui, 0 = Non&#10;
Autre ville : 1 = Oui, 0 = Non&#10;
Propriétaire : 1 = Oui, 0 = Non&#10;
Type de revenu : 1 = Travail salarié, 0 = Autre (pension, aide…)&#10;
Scores de solvabilité : proche de 1 = client plus fiable&#10;
Ratios (ex. crédit / revenu) : plus c'est bas, mieux c'est" 
                          style="cursor: help; display: inline-block; margin-left: 8px;">
                        ℹ️
                    </span>
                </span>
            </div>
        """, unsafe_allow_html=True)

        var_label = st.selectbox(" ", list(NOM_FEATURES_CLEAN.values()), label_visibility="collapsed", key="select_var")

    var = NOM_FEATURES_CLEAN_INV[var_label]

    # Fonction pour formater les valeurs
    def formatter_valeur(var_label, valeur):
        label = var_label.lower()
        if "ratio" in label or " / " in label:
            return f"{round(valeur * 100, 2)} %"
        elif "âge" in label:
            return f"{int(valeur)} ans"
        elif "ancienneté emploi" in label or "days_employed" in label:
            return f"{abs(valeur) // 365} ans"
        elif "mois" in label or "durée du crédit" in label:
            return f"{int(valeur)} mois"
        elif any(x in label for x in ["travail", "propriétaire", "classe ouvrière", "type de revenu"]):
            return str(int(valeur))
        elif ("revenu" in label and "type" not in label and "ratio" not in label) or "montant" in label or "mensualité" in label:
            return f"{int(valeur):,} €".replace(",", " ")
        elif "score" in label:
            return f"{round(valeur, 2)}"
        elif "note région" in label:
            return f"{int(valeur)}"
        elif "enfants" in label:
            return f"{int(valeur)}"
        else:
            return str(valeur)

    # === Nettoyage et transformation selon la variable sélectionnée par le chargé de relations clients ===
    if "features" in st.session_state:
        features = st.session_state["features"].copy()
        
        if var == "DAYS_BIRTH":
            data_display["DAYS_BIRTH"] = abs(data_display["DAYS_BIRTH"]) // 365
            features["DAYS_BIRTH"] = abs(features["DAYS_BIRTH"]) // 365

        elif var == "DAYS_EMPLOYED":
            data_display = data_display.copy()
            data_display.loc[data_display["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
            data_display["DAYS_EMPLOYED"] = abs(data_display["DAYS_EMPLOYED"]) // 365

            if features["DAYS_EMPLOYED"] == 365243:
                features["DAYS_EMPLOYED"] = np.nan
            else:
                features["DAYS_EMPLOYED"] = abs(features["DAYS_EMPLOYED"]) // 365

        elif var == "DAYS_EMPLOYED_PERCENT":
            data_display = data_display.copy()
            data_display.loc[data_display["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
            data_display["DAYS_EMPLOYED_PERCENT"] = (
                abs(data_display["DAYS_EMPLOYED"]) / abs(data_display["DAYS_BIRTH"])
            )

            if features["DAYS_EMPLOYED"] == 365243:
                features["DAYS_EMPLOYED_PERCENT"] = np.nan
            else:
                features["DAYS_EMPLOYED_PERCENT"] = abs(features["DAYS_EMPLOYED"]) / abs(features["DAYS_BIRTH"])

        # Graphique de comparaison
        if var in data.columns:
            # Nettoyage spécifique pour les variables notées sur 1, 2, 3 (classification des la region et de la ville)
            if var in ["REGION_RATING_CLIENT_W_CITY", "REGION_RATING_CLIENT"]:
                data_display = data_display[data_display[var].isin([1, 2, 3])]
            
            fig = px.box(data_display, y=var, points="all", labels={var: f"<b>{var_label}</b>"})

            valeur_client = features[var]
            texte_client = formatter_valeur(var_label, valeur_client)

            fig.add_scatter(
                x=[0],
                y=[valeur_client],
                mode="markers",
                marker=dict(color="red", size=14),
                name="Client",
                text=[texte_client],
                hoverinfo="text"
            )

            fig.update_layout(
                margin=dict(l=40, r=40, t=30, b=30),
                yaxis=dict(
                    title_font=dict(size=16, family="Arial", color="black"),
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='lightgrey',
                    showline=False
                ),
                xaxis=dict(
                    showgrid=False,
                    showline=False
                ),
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Variable non disponible dans le dataset.")

        # === Graphe croisé ===
        st.markdown("""
            <div style='text-align: center; font-size: 35px; font-weight: 600; margin-bottom: 10px;'>
                Visualisez deux critères du dossier client à comparer :
            </div>
        """, unsafe_allow_html=True)

        # Critère de sélection n°1
        st.markdown("""
                <div style="text-align: center; margin-bottom: 4px;">
                    <span style='font-size: 16px;'>
                        <b> Détails sur les valeurs des caractéristiques</b>
                        <span title="Genre : 1 = Homme, 0 = Femme&#10;
Travail : 1 = Oui, 0 = Non&#10;
Travailleur manuel : 1 = Oui, 0 = Non&#10;
Autre ville : 1 = Oui, 0 = Non&#10;
Propriétaire : 1 = Oui, 0 = Non&#10;
Type de revenu : 1 = Travail salarié, 0 = Autre (pension, aide…)&#10;
Scores de solvabilité : proche de 1 = client plus fiable&#10;
Ratios (ex. crédit / revenu) : plus c'est bas, mieux c'est" 
                              style="cursor: help; display: inline-block; margin-left: 8px;">
                            ℹ️
                        </span>
                    </span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='text-align: center; font-size: 25px; font-weight: 600;'>Critère n°1</div>", unsafe_allow_html=True)
        spacer1, select_col1, spacer2 = st.columns([3, 2, 3])
        with select_col1:
            x_label = st.selectbox(" ", list(NOM_FEATURES_CLEAN_INV.keys()), key="x", label_visibility="collapsed")
        x = NOM_FEATURES_CLEAN_INV[x_label]

        st.markdown("<div style='text-align: center; font-size: 25px; font-weight: 600;'>Critère n°2</div>", unsafe_allow_html=True)
        spacer1, select_col2, spacer2 = st.columns([3, 2, 3])
        with select_col2:
            y_label = st.selectbox("  ", list(NOM_FEATURES_CLEAN_INV.keys()), key="y", label_visibility="collapsed")
        y = NOM_FEATURES_CLEAN_INV[y_label]

        # Prétraitement si nécessaire
        for var_check in set([x, y]):
            if var_check == "DAYS_BIRTH":
                data_display["DAYS_BIRTH"] = abs(data_display["DAYS_BIRTH"]) // 365
                features["DAYS_BIRTH"] = abs(features["DAYS_BIRTH"]) // 365

            elif var_check == "DAYS_EMPLOYED":
                data_display.loc[data_display["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
                data_display["DAYS_EMPLOYED"] = abs(data_display["DAYS_EMPLOYED"]) // 365

                if features["DAYS_EMPLOYED"] == 365243:
                    features["DAYS_EMPLOYED"] = np.nan
                else:
                    features["DAYS_EMPLOYED"] = abs(features["DAYS_EMPLOYED"]) // 365

        # Affichage graphique
        if x in data.columns and y in data.columns:
            # Formattage spécifique pour le graphe si l'une des variables correspond à un ratio
            data_display_graph = data_display.copy()
            features_graph = features.copy()
            
            for ratio_var in [x, y]:
                if ratio_var in ["DAYS_EMPLOYED_PERCENT", "CREDIT_INCOME_PERCENT", "ANNUITY_INCOME_PERCENT"]:
                    data_display_graph[ratio_var] = data_display_graph[ratio_var] * 100
                    features_graph[ratio_var] = features_graph[ratio_var] * 100

            # Création du nuage de points
            fig2 = px.scatter(
                data_display_graph,
                x=x,
                y=y,
                opacity=0.7,
                labels={x: x_label, y: y_label}
            )

            fig2.update_traces(
                marker=dict(
                    size=6,
                    line=dict(width=1, color="white")
                ),
                selector=dict(mode="markers")
            )

            # Formatage conditionnel des axes
            for axis_var, axis_label in [(x, "x"), (y, "y")]:
                if axis_var in ["DAYS_EMPLOYED_PERCENT", "CREDIT_INCOME_PERCENT", "ANNUITY_INCOME_PERCENT"]:
                    if axis_label == "x":
                        fig2.update_xaxes(tickformat=".1f", ticksuffix="%")
                    else:
                        fig2.update_yaxes(tickformat=".1f", ticksuffix="%")

            # Ajout du point client (positionnement sur le graphique)
            fig2.add_scatter(
                x=[features_graph[x]],
                y=[features_graph[y]],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Client"
            )

            fig2.update_layout(
                margin=dict(l=40, r=40, t=10, b=30),
                xaxis=dict(
                    title=dict(text=x_label, font=dict(size=16, color='black', family='Arial', weight='bold')),
                    showgrid=False,
                    showline=False
                ),
                yaxis=dict(
                    title=dict(text=y_label, font=dict(size=16, color='black', family='Arial', weight='bold')),
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='lightgrey',
                    showline=False
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Variables non disponibles dans le dataset.")

        # Comparaison avec clients similaires
        st.markdown("""
            <div style='text-align: center; font-size: 35px; font-weight: 600; margin-bottom: 10px;'>
                Profil du client comparé à des clients similaires
            </div>
        """, unsafe_allow_html=True)

        features_autorisees = {
            "AMT_INCOME_TOTAL": "Revenu (€)",
            "AMT_CREDIT": "Crédit (€)",
            "AMT_ANNUITY": "Mensualité (€)",
            "EXT_SOURCE_1": "Score solvabilité 1",
            "EXT_SOURCE_2": "Score solvabilité 2",
            "EXT_SOURCE_3": "Score solvabilité 3",
            "CREDIT_TERM": "Durée de remboursement (mois)"
        }

        features_autorisees_inv = {v: k for k, v in features_autorisees.items()}

        st.markdown("""
            <div style="text-align: center; margin-bottom: 6px;">
                <span style='font-size: 16px; font-weight: 600;'>
                    Sélectionnez un critère à comparer
                    <span title="ℹ️ Le score de solvabilité est un indicateur numérique. On l'utilise pour évaluer la solvabilité ou le risque d'un client à partir d'informations externes, comme des données de bureau de crédit ou d'autres bases de données financières. Plus le score est élevé (proche de 1), plus la probabilité de remboursement est forte."
                        style="cursor: help; display: inline-block; margin-left: 8px;">
                        ℹ️
                    </span>
                </span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 4, 2])
        with col2:
            selected_feature_label = st.selectbox(
                " ",
                list(features_autorisees.values()),
                index=0,
                label_visibility="collapsed",
                key="comparaison_feature"
            )

        selected_feature = features_autorisees_inv[selected_feature_label]

        # Données similaires
        data_clean = data[list(features.keys())].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean)
        client_vector = scaler.transform([list(features.values())])
        distances = euclidean_distances(client_vector, X_scaled).flatten()
        indices_similaires = distances.argsort()[:10]
        clients_similaires = data_clean.iloc[indices_similaires].copy()

        clients_similaires["Score de similarité avec le client"] = (
            (1 - distances[indices_similaires] / distances.max()) * 100
        ).round(0)

        # Formater les scores solvabilité
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if col in clients_similaires.columns:
                clients_similaires[col] = clients_similaires[col].round(2)

        # Graphique barres
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        clients_similaires_sorted = clients_similaires.sort_values(by=selected_feature)
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=clients_similaires_sorted.index.astype(str),
            y=clients_similaires_sorted[selected_feature],
            text=[
                f"{v:,.2f}" if "solvabilité" in selected_feature_label else
                f"{v:,.0f} €" if "(€" in selected_feature_label else
                f"{v:,.0f}"
                for v in clients_similaires_sorted[selected_feature]
            ],
            texttemplate="<b>%{text}</b>",
            textposition="outside",
            textfont=dict(size=16, color="black"),
            cliponaxis=False,
            name="<b>Clients similaires</b>",
            marker=dict(color=clients_similaires_sorted[selected_feature], colorscale="Blues")
        ))

        # Ajout du client actuel
        fig_bar.add_trace(go.Bar(
            x=["<b>Client actuel</b>"],
            y=[features[selected_feature]],
            text=[
                f"{features[selected_feature]:,.2f}" if "solvabilité" in selected_feature_label else
                f"{features[selected_feature]:,.0f} €" if "(€" in selected_feature_label else
                f"{features[selected_feature]:,.0f}"
            ],
            texttemplate="<b>%{text}</b>",
            textposition="outside",
            textfont=dict(size=16, color="black"),
            name="<b>Client actuel</b>",
            marker_color="crimson"
        ))

        y_max = max(clients_similaires_sorted[selected_feature].max(), features[selected_feature])
        fig_bar.update_layout(
            title=dict(
                text=f"<b>Critère sélectionné : {selected_feature_label}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=25)
            ),
            xaxis_title="<b>Clients</b>",
            yaxis_title=f"<b>{selected_feature_label}</b>",
            yaxis=dict(
                gridcolor="lightgrey",
                gridwidth=0.5,
                tickfont=dict(size=15),
                title_font=dict(size=14, color='black'),
                range=[0, y_max * 1.25],
            ),
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center",
                font=dict(size=16, family="Arial", color="black"),
                title_font=dict(size=16, color="black", family="Arial"),
                itemclick="toggleothers",
                itemdoubleclick="toggle"
            ),
            template="simple_white",
            margin=dict(t=50, b=60, l=30, r=25),
            uniformtext_minsize=10,
            uniformtext_mode='show',
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # Interprétation
        st.markdown("""
            <div style='text-align: center; font-size: 16px; margin-top: 10px; margin-bottom: 30px;'>
                ℹ️ <b>Les principaux critères de comparaisons :</b> Revenu du client / Montant du crédit demandé / Mensualité / Les scores de solvabilité (1,2 et 3).
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; font-size: 15px; margin-top: -10px; margin-bottom: 30px;'>
            <i>Les clients similaires sont sélectionnés automatiquement en fonction des critères numériques principaux (revenu, crédit, mensualité, scores de solvabilité) à l'aide d'une mesure de distance (euclidienne).</i>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Veuillez d'abord saisir les informations du client dans l'onglet 1.")

# ------ Onglet 2 : Influence du profil client sur la décision ------
with menu[2]:
    st.markdown("<div style='text-align: center; font-size: 35px; font-weight: 600; margin-bottom: 10px;'>Influence du profil client sur la décision finale</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='
    text-align: center;
    font-size: 22px;
    background-color: #e6f2ff;
    border-radius: 8px;
    padding: 12px 8px;
    margin-bottom: 12px;
    color: #333;
    font-weight: 500;
'>
    Cette section permet d'analyser l'impact de chaque caractéristique sur la décision de crédit.
</div>
""", unsafe_allow_html=True)

    # Vérifie que le profil client est bien saisie
    if "features" not in st.session_state:
        st.warning("Veuillez d'abord saisir les informations du client dans l'onglet 1.")
    else:
        features = st.session_state["features"]

        # Charger le modèle une seule fois
        @st.cache_resource
        def load_model():
            return joblib.load("Best_XGBoost_Business_Model.pkl")
        model = load_model()
        booster_model = model.named_steps["clf"]
        explainer = shap.Explainer(booster_model)

        X_client = np.array([list(features.values())])
        shap_values = explainer(X_client)

        base_val = shap_values.base_values[0]
        sum_shap = shap_values.values[0].sum()
        logit_final = base_val + sum_shap
        proba_shap = scipy.special.expit(logit_final)

        feature_names_readable = [NOM_FEATURES_CLEAN.get(f, f) for f in features.keys()]
        # La fonction formatter_valeur doit etre définit précédemment
        shap_impact = shap_values.values[0].tolist()

        shap_df = pd.DataFrame({
            "Critère d'influence": feature_names_readable,
            "Valeur": [formatter_valeur(f, features[f]) for f in features.keys()],
            "Facteurs déterminants dans la décision de crédit du client": shap_impact
        }).sort_values(by="Facteurs déterminants dans la décision de crédit du client", key=np.abs, ascending=False).head(10)

        shap_df["Critère d'influence"] = shap_df["Critère d'influence"].apply(lambda x: f"<b>{x}</b>")

        table_html = shap_df.to_html(
            index=False,
            escape=False,
            border=0,
            classes='shap-table',
            justify='center',
            float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else ""
        )

        # Style visuel du tableau explicatif  
        st.markdown("""
            <style>
            .shap-table {
                font-size: 18px;
                font-family: Arial, sans-serif;
                width: 80%;
                border-collapse: collapse;
                margin: 0 auto;
            }
            .shap-table th {
                text-align: center;
                padding: 12px;
                background-color: #000000;
                color: white;
                font-size: 20px;
            }
            .shap-table td {
                text-align: center;
                padding: 10px;
            }
            .shap-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .shap-table tr:hover {
                background-color: #eef;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='text-align: center; font-size: 40px; font-weight: 600; margin-top: -25px; margin-bottom: 0px;'>
                Synthèse des caractéristiques du demandeur
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; margin-bottom: 4px;">
                <span style='font-size: 16px;'>
                    <b> Détails sur les valeurs des caractéristiques</b>
                    <span title="Genre : 1 = Homme, 0 = Femme&#10;
Travail : 1 = Oui, 0 = Non&#10;
Travailleur manuel : 1 = Oui, 0 = Non&#10;
Autre ville : 1 = Oui, 0 = Non&#10;
Propriétaire : 1 = Oui, 0 = Non&#10;
Type de revenu : 1 = Travail salarié, 0 = Autre (pension, aide…)&#10;
Scores de solvabilité : proche de 1 = client plus fiable&#10;
Ratios (ex. crédit / revenu) : plus c’est bas, mieux c’est" 
                          style="cursor: help; display: inline-block; margin-left: 8px;">
                        ℹ️
                    </span>
                </span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='display: flex; justify-content: center;'>" f"{table_html}" "</div>", unsafe_allow_html=True)

        # Option de téléchargement des données pour le chargé de relation client
        csv_data = shap_df.copy()
        csv_data["Critère d'influence"] = csv_data["Critère d'influence"].str.replace("<b>", "").str.replace("</b>", "")
        csv_bytes = csv_data.to_csv(index=False).encode("utf-8")

        col1, col2, col3 = st.columns([2.5, 2.5, 1])
        with col2:
            st.download_button(
                label="📥 Télécharger les données au format CSV",
                data=csv_bytes,
                file_name="explication_shap_locale.csv",
                mime="text/csv",
                key="shap_download",
                help="Cliquez pour enregistrer le tableau",
            )
        # Graphique 
        top_n = 10
        shap_df_clean = shap_df.copy()
        shap_df_clean["Critère d'influence"] = shap_df_clean["Critère d'influence"].str.replace("<b>", "", regex=False).str.replace("</b>", "", regex=False)
        top_shap_df = shap_df_clean.head(top_n).sort_values("Facteurs déterminants dans la décision de crédit du client", ascending=True)

        abs_norm = Normalize(
            vmin=top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs().min(),
            vmax=top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs().max()
        )
        colors = cm.Blues(abs_norm(top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs()))

        st.markdown("<h2 style='text-align: center; color: black;'>Les critères ayant influencé la décision</h2>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(
            top_shap_df["Critère d'influence"],
            top_shap_df["Facteurs déterminants dans la décision de crédit du client"],
            color=colors
        )

        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

        ax.set_xlabel("Effet estimé sur l’acceptation\n(Valeurs négatives : effet défavorable – Valeurs positives : effet favorable)", fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9, pad=8)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}",
                    va='center',
                    ha='left' if width > 0 else 'right',
                    fontsize=8, color='black')

        plt.subplots_adjust(left=0.15)
        st.pyplot(fig)

        # Interprétation visuelle
        st.markdown("""
        <div style='text-align: center; font-size: 22px; margin-top: 25px;'>
            ℹ️<b>Interprétation :</b> Ce graphique montre les éléments qui ont le plus influencé la décision pour ce client. 
            Même si certains indicateurs paraissent bons (comme une bonne solvabilité), d'autres critères peuvent venir contrebalancer cette impression. 
            Les barres vers la gauche indiquent ce qui a réduit les chances d’acceptation, et celles vers la droite ce qui les a renforcées. 
            C’est l’ensemble du profil qui compte dans la décision finale.
        </div>
        """, unsafe_allow_html=True)

