import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# L'importation de toutes les métriques est nécessaire pour un rapport complet

print("--- TEST ET ÉVALUATION DU MODÈLE DE CLASSIFICATION (MODÈLE A) ---")

# --- 1. CHARGEMENT DES DONNÉES ET DU MODÈLE ---
try:
    df_data = pd.read_csv("../data/energiTech_par_turbine.csv")
    model_A = joblib.load('../models/model_classification.pkl')
    print("✅ Données et modèle chargés.")
except FileNotFoundError:
    print("❌ ERREUR : Assurez-vous que 'energiTech_par_turbine.csv' et 'model_classification.pkl' existent.")
    exit()

# --- 2. PRÉPARATION DES DONNÉES DE TEST ---
FEATURES = ['wind_speed', 'vibration_level', 'temperature', 'power_output', 'maintenance_done']

# La cible (y_true) est la vérité terrain que le modèle essaie de prédire
TARGET = 'failure_within_7d' 

# Vérification que toutes les colonnes nécessaires existent
if not all(col in df_data.columns for col in FEATURES + [TARGET]):
    print("❌ ERREUR : Colonnes de features ou de cible manquantes dans le fichier CSV.")
    exit()

X_test = df_data[FEATURES]
y_true = df_data[TARGET]

# --- 3. PRÉDICTION ET ÉVALUATION ---

# Obtenir les prédictions du modèle sur les données
y_pred = model_A.predict(X_test)

print("\n--- RÉSULTATS D'ÉVALUATION ---")

# A. Score de précision (Accuracy)
accuracy = accuracy_score(y_true, y_pred)
print(f"📊 Précision globale (Accuracy) : {accuracy:.4f}")

# B. Matrice de Confusion
cm = confusion_matrix(y_true, y_pred)
print("\n📋 Matrice de Confusion :")
print(cm)
print("   (Ligne = Réel, Colonne = Prédit)")
# 

# C. Rapport de Classification (Précision, Rappel, F1-Score)
print("\n📝 Rapport de Classification :")
print(classification_report(y_true, y_pred, target_names=['Classe 0 (Pas Panne)', 'Classe 1 (Panne)']))

print("\n--- FIN DU TEST ---")