import subprocess
import os
import sys

# --- CONFIGURATION ---
PYTHON_EXECUTABLE = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
TRI_SCRIPT = os.path.join(SCRIPT_DIR, "trie_du_csv.py") 
TEST_SCRIPT = os.path.join(SCRIPT_DIR, "test_model.py")
DETECTION_SCRIPT = os.path.join(SCRIPT_DIR, "detection_anomalie.py")
STREAMLIT_SCRIPT = os.path.join(SCRIPT_DIR, "streamlit_app.py")

# Définir le répertoire parent (la racine du projet)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


print("--- DÉMARRAGE DU PIPELINE IA ENERGI-TECH ---")

# 0. NOUVEAU : Exécuter le script de tri et préparation des données
print("\n0. Préparation des données (Tri et Sauvegarde)...")
try:
    subprocess.run(
        [PYTHON_EXECUTABLE, TRI_SCRIPT],
        cwd=ROOT_DIR,
        capture_output=False,
        text=True,
        check=True
    )
    print("✅ Tri et préparation des données TERMINEE.")

except subprocess.CalledProcessError as e:
    print(f"\n❌ ERREUR lors de l'exécution de tri_du_csv.py : {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    print("\n❌ ERREUR : Le script 'tri_du_csv.py' est introuvable. Assurez-vous qu'il est dans le dossier 'scripts/'.")
    sys.exit(1)


# 1. Exécuter le script de test pour vérifier la performance
print("\n1. Évaluation des performances du Modèle A (pour le rapport de conformité)...")
try:
    subprocess.run(
        [PYTHON_EXECUTABLE, "-W", "ignore", TEST_SCRIPT], 
        cwd=ROOT_DIR, 
        capture_output=False, 
        text=True,
        check=True
    )
    print("✅ Évaluation du modèle TERMINEE.")

except subprocess.CalledProcessError as e:
    print(f"\n❌ ERREUR lors de l'exécution de test_model.py : {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    print("\n❌ ERREUR : Le script 'test_model.py' est introuvable.")
    sys.exit(1)


# 2. Exécuter le script de détection d'anomalies (Inférence Batch)
print("\n2. Exécution de l'Inférence Batch et Détection d'Anomalies (Génération du CSV et des Statistiques)...")
try:
    subprocess.run(
        [PYTHON_EXECUTABLE, "-W", "ignore", DETECTION_SCRIPT], 
        cwd=ROOT_DIR, 
        capture_output=False, 
        text=True,
        check=True
    )
    print("✅ Inférence Batch TERMINEE. Fichiers de résultats mis à jour dans le dossier 'results/'.")

except subprocess.CalledProcessError as e:
    print(f"\n❌ ERREUR lors de l'exécution de detection_anomalie.py : {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    print("\n❌ ERREUR : Le script 'detection_anomalie.py' est introuvable.")
    sys.exit(1)


print("\n--- PIPELINE PRÊT ---")
print("3. Lancement de l'interface Streamlit...")
print("    Exécutez cette commande dans votre terminal :")
print(f"   python3 -m streamlit run streamlit_app.py")