import streamlit as st
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Charger les modèles depuis HuggingFace
path_reg = hf_hub_download(
    repo_id="untiiis/game_sale",
    filename="rf_reg.sav"
)
rf_reg = joblib.load(path_reg)

path_clf = hf_hub_download(
    repo_id="untiiis/game_sale",
    filename="rf_clf.sav"
)
rf_clf = joblib.load(path_clf)

st.title("Prédiction des ventes globales d’un jeu vidéo")

# Récupération des top catégories depuis le pipeline
top_platforms = rf_reg.named_steps["top"].top_platforms_
top_genres = rf_reg.named_steps["top"].top_genres_
top_publishers = rf_reg.named_steps["top"].top_publishers_

# Listes complètes (pour affichage)
all_platforms = [
    'Wii','NES','GB','DS','X360','PS3','PS2','SNES','GBA','PS4','3DS','N64','PS',
    'XB','PC','2600','PSP','XOne','WiiU','GC','GEN','DC','PSV','SAT','SCD','WS',
    'NG','TG16','3DO','GG','PCFX'
]

all_genres = list(top_genres)
all_publishers = list(top_publishers) + ['Others']

st.header("Caractéristiques du jeu")

platform = st.selectbox("Plateforme du jeu :", all_platforms)
genre = st.selectbox("Genre :", all_genres)
publisher = st.selectbox("Éditeur :", all_publishers)

# Remplacement par Others si hors top catégories
if platform not in top_platforms:
    platform = "Others"

if genre not in top_genres:
    genre = "Others"

year = st.slider("Année de sortie", min_value=1980, max_value=2020, step=1)
u_score = st.slider("Note des joueurs (sur 10)", min_value=0, max_value=10)
c_score = st.slider("Note des critiques (sur 100)", min_value=0, max_value=100)

# Conversion interne
u_score_norm = u_score / 10
c_score_norm = c_score / 100

# Récapitualtif
st.subheader("Vos choix")
st.write(f"**Plateforme :** {platform}")
st.write(f"**Genre :** {genre}")
st.write(f"**Éditeur :** {publisher}")
st.write(f"**Année de sortie :** {year}")
st.write(f"**Note des joueurs :** {u_score} / 10")
st.write(f"**Note des critiques :** {c_score} / 100")

# Bouton
if st.button("Analyser le potentiel du jeu"):

    # Construction de l'échantillon
    sample = pd.DataFrame([{
        "Platform": platform,
        "Year_of_Release": year,
        "Genre": genre,
        "Publisher": publisher,
        "User_Score": u_score_norm,
        "Critic_Score": c_score_norm
    }])

    # Prédictions
    pred_class = rf_clf.predict(sample)[0]
    pred_sales = rf_reg.predict(sample)[0]

    # Interprétation des classes
    interpretation = {
        0: "Performance faible (mais rien n’est jamais perdu !)",
        1: "Performance correcte",
        2: "Excellente performance attendue"
    }

    st.subheader("Résultats de la prédiction")
    st.write(f"**Analyse qualitative :** {interpretation[pred_class]}")
    st.write(f"**Estimation des ventes globales :** environ **{pred_sales:.2f} millions** d’exemplaires")
