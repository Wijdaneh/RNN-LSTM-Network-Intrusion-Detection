import streamlit as st
import pandas as pd
import numpy as np
import scapy.all as scapy
from tensorflow.keras.models import load_model
import time
import io

# -------------------------------------------------------------------
# CONFIGURATION DE LA PAGE (doit être la première commande Streamlit)
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Analyse PCAP en Temps Réel - RNN/LSTM/GRU",
    layout="wide",
    page_icon="📡"
)

# -------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# -------------------------------------------------------------------

def extract_features(packet):
    """Extraction simple des features pour démonstration."""
    try:
        src = packet[scapy.IP].src
        dst = packet[scapy.IP].dst
        proto = packet[scapy.IP].proto
        length = len(packet)

        return [hash(src) % 1000, hash(dst) % 1000, proto, length]
    except:
        return None


def load_pcap(file):
    """Charge un fichier PCAP uploadé."""
    data = file.read()
    return scapy.rdpcap(io.BytesIO(data))


def load_prediction_model(model_path="model.h5"):
    """Charge un modèle ML (LSTM/RNN/GRU)."""
    return load_model(model_path)


# -------------------------------------------------------------------
# TITRE ET GUI
# -------------------------------------------------------------------

st.title("📡 Analyse PCAP en Temps Réel")
st.subheader("Analyse des paquets réseau à l’aide de modèles RNN / LSTM / GRU")

uploaded_file = st.file_uploader("Choisir un fichier PCAP", type=["pcap", "pcapng"])

model_choice = st.selectbox(
    "Choisir le modèle de prédiction :",
    ["LSTM", "GRU", "RNN"]
)

start_button = st.button("🚀 Lancer l'analyse")

# -------------------------------------------------------------------
# LOGIQUE PRINCIPALE
# -------------------------------------------------------------------

if uploaded_file and start_button:
    st.info("📂 Chargement du fichier PCAP...")
    packets = load_pcap(uploaded_file)

    st.success(f"Fichier chargé ! Nombre de paquets : {len(packets)}")

    st.info("📦 Chargement du modèle ML...")
    model = load_prediction_model("model.h5")
    st.success("Modèle chargé avec succès !")

    st.subheader("📊 Analyse en temps réel")

    placeholder_table = st.empty()
    placeholder_alert = st.empty()

    results = []

    for i, packet in enumerate(packets):

        features = extract_features(packet)
        if features is None:
            continue

        X = np.array(features).reshape(1, 1, 4)

        prediction = model.predict(X, verbose=0)[0][0]
        label = "🔴 Attaque" if prediction > 0.5 else "🟢 Normal"

        results.append({
            "Packet": i,
            "Prediction": float(prediction),
            "Label": label
        })

        df = pd.DataFrame(results)

        placeholder_table.dataframe(df, height=400)

        if label == "🔴 Attaque":
            placeholder_alert.error(f"🚨 Alerte : Activité suspecte détectée au paquet {i} !")

        time.sleep(0.05)  # simulation temps réel

    st.success("Analyse terminée ✔️")

