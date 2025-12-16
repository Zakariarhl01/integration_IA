import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json 
import os 

# --- CONFIGURATION DES CHEMINS (RELATIFS À LA RACINE DU PROJET) ---
DATA_PATH = "./data/energiTech_par_turbine.csv"
MODEL_PATH = "./models/model_classification.pkl"
METRICS_PATH = "./results/evaluation_metrics.json"

# --- 1. CHARGEMENT DES DONNÉES ET DU MODÈLE ---
try:
    df_data = pd.read_csv(DATA_PATH)
    model_A = joblib.load(MODEL_PATH)
    print("✅ Données et modèle chargés.")
except FileNotFoundError:
    print(f"❌ ERREUR : Assurez-vous que '{DATA_PATH}' et '{MODEL_PATH}' existent.")
    exit()

# --- 2. PRÉPARATION DES DONNÉES DE TEST ---
FEATURES = ['wind_speed', 'vibration_level', 'temperature', 'power_output', 'maintenance_done']
TARGET = 'failure_within_7d' 

if not all(col in df_data.columns for col in FEATURES + [TARGET]):
    print("❌ ERREUR : Colonnes de features ou de cible manquantes dans le fichier CSV.")
    exit()

X_test = df_data[FEATURES]
y_true = df_data[TARGET]

# --- 3. PRÉDICTION ET ÉVALUATION ---
y_pred = model_A.predict(X_test)

# A. Calcul des métriques
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Classe 0 (Pas Panne)', 'Classe 1 (Panne)'], output_dict=True)

# Sauvegarde des métriques pour Streamlit
metrics_data = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(), 
    "classification_report": report
}

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

try:
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("✅ Métriques d'évaluation enregistrées pour l'affichage Streamlit.")
except Exception as e:
    print(f"❌ ERREUR lors de l'enregistrement des métriques : {e}")
