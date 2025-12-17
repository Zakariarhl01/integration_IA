🚀 Projet EnergiTech : Pipeline de Maintenance Prédictive IA

Ce projet implémente un pipeline complet pour la maintenance prédictive des éoliennes. Il combine la détection d'anomalies de capteurs (méthode IQR et Zéro Détection) avec un modèle d'apprentissage automatique (RandomForestClassifier, nommé Modèle A) pour identifier les risques de panne imminente. L'objectif final est de fournir un tableau de bord interactif (via Streamlit) pour prioriser les interventions de maintenance.


⚙️ Architecture du Projet
Le pipeline est orchestré par le script main.py  et suit ces étapes :


Préparation des Données (trie_du_csv.py) : Trie les données brutes (energiTech_maintenance_sample.csv ) par turbine et par date. Le fichier trié est sauvegardé dans data/energiTech_par_turbine.csv.


Évaluation du Modèle (test_model.py) : Charge le Modèle A pré-entraîné (model_classification.pkl ), évalue ses performances (Accuracy, Recall, Precision) et sauvegarde les métriques dans results/evaluation_metrics.json.


Inférence et Détection d'Anomalies (detection_anomalie.py) :

Exécute la détection d'anomalies (IQR & Zéro) sur les données.

Exécute le Modèle A pour obtenir la proba_panne (risque à 7 jours).

Filtre les anomalies déjà gérées (maintenance_done=0).

Sauvegarde les alertes finales dans results/anomalies_non_gerees_final.csv et les statistiques brutes dans results/detection_stats.json.


Visualisation (streamlit_app.py) : Lit les fichiers générés par les étapes 2 et 3 pour afficher le tableau de bord.

🛠️ Installation du Projet

1. Cloner le Dépôt

git clone [https://github.com/Zakariarhl01/integration_IA.git]
cd integration_IA

2. Création de l'Environnement Virtuel (Recommandé)

python3 -m venv venv
source venv/bin/activate  # Sous Linux/macOS
# Pour Windows : venv\Scripts\activate

3. Installation des Dépendances

Installez toutes les bibliothèques nécessaires à partir de votre requirements.txt :
pip3 install -r requirements.txt

4. Structure des Fichiers Clés

Assurez-vous que la structure de vos dossiers de données et de modèles est la suivante :

.
├── data/
│   ├── energiTech_maintenance_sample.csv  # Fichier source initial 
│   └── energiTech_par_turbine.csv         # Fichier trié (généré par trie_du_csv.py) 
├── models/
│   └── model_classification.pkl           # Modèle IA (Modèle A - RandomForestClassifier) 
├── results/
│   ├── anomalies_non_gerees_final.csv     # Alertes finales (généré) 
│   ├── detection_stats.json               # Stats brutes de détection (généré) 
│   └── evaluation_metrics.json            # Rapport du Modèle A (généré) 
└── scripts/
    ├── main.py
    ├── trie_du_csv.py
    ├── test_model.py
    ├── detection_anomalie.py
    └── streamlit_app.py
▶️ Exécution du Pipeline
Le pipeline complet est lancé via le script main.py.

1. Lancer le Pipeline (Étapes 0, 1, et 2)
Exécutez cette commande depuis le dossier scripts/ ou en référençant le chemin :

python3 scripts/main.py
Le terminal affichera les messages de succès pour le tri, le test du modèle et la détection d'anomalies.

2. Lancer le Tableau de Bord Streamlit (Étape 3)
Une fois que main.py a terminé, lancez l'interface pour ouvrir le tableau de bord dans votre navigateur:

python3 -m streamlit run scripts/streamlit_app.py
Le tableau de bord affichera :

Le Rapport de Conformité du Modèle A (Précision, Rappel, Matrice de Confusion).

Les Statistiques Brutes des capteurs.

Le TOP 5 des alertes prioritaires (filtrées par maintenance_done=0).