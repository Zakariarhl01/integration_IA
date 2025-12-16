import pandas as pd
import joblib 
import numpy as np 
import os
import json 

# Configure l'affichage de la console à 4 décimales pour la vérification (optionnel, mais propre)
pd.set_option('display.float_format', '{:.4f}'.format) 

# --- CONFIGURATION DES CHEMINS (RELATIFS À LA RACINE DU PROJET - POUR EXECUTION VIA main.py) ---
DATA_PATH = "./data/energiTech_par_turbine.csv"
MODEL_PATH = "./models/model_classification.pkl" 
RESULTS_PATH = "./results/anomalies_non_gerees_final.csv"
DETECTION_STATS_PATH = "./results/detection_stats.json" 

# --- 1. CHARGEMENT DES ACTIFS ---
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Données chargées : energiTech_par_turbine.csv")
except FileNotFoundError:
    print(f"❌ ERREUR FATALE : Le fichier '{DATA_PATH}' est introuvable. Arrêt du script.")
    exit()

try:
    model_A = joblib.load(MODEL_PATH)
    print("✅ Modèle de classification chargé : model_classification.pkl")
except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier '{MODEL_PATH}' est introuvable. Impossible d'ajouter les prédictions.")
    model_A = None


# --- 2. INFÉRENCE DU MODÈLE A ---
if model_A:
    FEATURES = ['wind_speed', 'vibration_level', 'temperature', 'power_output', 'maintenance_done']
    
    if all(f in df.columns for f in FEATURES):
        X_predict = df[FEATURES]
        
        predictions = model_A.predict(X_predict)

        probabilities = model_A.predict_proba(X_predict)[:, 1] 

        df['prediction_panne_7j'] = predictions
        df['proba_panne'] = probabilities
        print("✅ Colonnes de prédiction ajoutées (précision maximale conservée).")
    else:
        print("❌ Certaines colonnes de capteurs requises par le modèle sont manquantes. Prédictions ignorées.")


# --- 3. DÉTECTION D'ANOMALIES (MÉTHODE IQR & ZÉRO) ---
numeric_cols = df.select_dtypes(include='number').columns

cols_to_exclude = [
    'turbine_id', 
    'maintenance_done', 
    'failure_within_7d', 
    'time_to_failure_days', 
    'prediction_panne_7j',
    'proba_panne'
]

numeric_cols_for_anomaly_detection = list(set(numeric_cols) - set(cols_to_exclude))

anomalies = pd.DataFrame()

for col in numeric_cols_for_anomaly_detection:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_lower = df[(df[col] < lower) & (df[col] > 0.0)].copy()
    outliers_upper = df[(df[col] > upper)].copy()
    outliers_zero = df[(df[col] == 0.0)].copy() 
    
    dfs_to_concat = []
    
    if not outliers_lower.empty:
        outliers_lower['anomaly_column'] = col
        outliers_lower['anomaly_type'] = 'lower'
        dfs_to_concat.append(outliers_lower)      
        
    if not outliers_upper.empty:
        outliers_upper['anomaly_column'] = col
        outliers_upper['anomaly_type'] = 'upper'
        dfs_to_concat.append(outliers_upper) 
        
    if not outliers_zero.empty:
        outliers_zero['anomaly_column'] = col
        outliers_zero['anomaly_type'] = 'zero_detection_issue'
        dfs_to_concat.append(outliers_zero)

    if dfs_to_concat:
        anomalies = pd.concat([anomalies] + dfs_to_concat, ignore_index=True)


# --- Sauvegarde des Statistiques de Détection Brute (pour Streamlit) ---
if not anomalies.empty:
    # Calculer les statistiques de l'ensemble 'anomalies'
    detection_stats = anomalies['anomaly_type'].value_counts().to_dict()
    detection_stats['total_anomalies_detectees'] = len(anomalies)

    try:
        os.makedirs(os.path.dirname(DETECTION_STATS_PATH), exist_ok=True)
        with open(DETECTION_STATS_PATH, 'w') as f:
            json.dump(detection_stats, f, indent=4)
        print(f"✅ Statistiques de détection brute enregistrées dans {os.path.basename(DETECTION_STATS_PATH)}.")
    except Exception as e:
        print(f"❌ ERREUR lors de l'enregistrement des statistiques de détection : {e}")
else:
    print("ℹ️ Aucune anomalie de capteur détectée. Statistiques ignorées.")


# --- 4. FILTRAGE ET TRI FINAL ---
anomalies_non_gerees = anomalies[
    (anomalies['maintenance_done'] == 0) 
].copy()

# Trier par risque de panne 
anomalies_non_gerees = anomalies_non_gerees.sort_values(
    by='proba_panne', ascending=False
)

# --- 5. ENREGISTREMENT DU FICHIER FINAL ---
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# L'enregistrement CSV conserve la précision maximale flottante
anomalies_non_gerees.to_csv(RESULTS_PATH, index=False)
print(f"\n✅ Fichier de résultats final généré : {os.path.basename(RESULTS_PATH)}")