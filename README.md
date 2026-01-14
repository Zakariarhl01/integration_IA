🚀 Projet EnergiTech : Backend d'Inférence IA Sécurisé
Ce projet implémente un pipeline complet de maintenance prédictive pour les éoliennes. Il est conçu comme un service backend d'inférence robuste, respectant les exigences de sécurité et de documentation d'un environnement industriel.

⚙️ Architecture du Système
Le système suit une architecture de type Batch Processing, orchestrée par un point d'entrée unique. Il intègre une détection d'anomalies hybride (Statistiques + Machine Learning).

🔒 Sécurité et Authentification
Conformément au cahier des charges, l'accès au moteur d'inférence est protégé par un Token de sécurité.

Mécanisme : Vérification de la variable d'environnement ENERGI_PRO_API_KEY.

Protection : Sans une clé valide, le script renvoie une erreur 403 Forbidden et interrompt le traitement pour protéger les modèles propriétaires.

📄 Contrat d'Interface (Documentation API)
Le service suit un contrat d'interface strict pour garantir l'interopérabilité :

Entrées (Input)
Fichier CSV structuré contenant : wind_speed, vibration_level, temperature, power_output.

Sorties (Output)
Prédictions : anomalies_non_gerees_final.csv (incluant proba_panne et anomaly_type).

Supervision : detection_stats.json (statistiques en temps réel pour le Dashboard).

▶️ Guide d'Exécution
1. Configuration de la Sécurité
Créez un fichier .env à la racine ou exportez la clé dans votre terminal :

export ENERGI_PRO_API_KEY="ET-PRO-2026-CONFIDENTIAL"
2. Lancement du Pipeline (Backend)
Le pipeline est lancé via le script orchestrateur. Il valide l'authentification, prépare les données, évalue le modèle et génère les inférences.

ENERGI_PRO_API_KEY="ET-PRO-2026-CONFIDENTIAL" python3 scripts/main.py

Sur Windows (PowerShell) :

$env:ENERGI_PRO_API_KEY="ET-PRO-2026-CONFIDENTIAL"; python scripts/main.py

3. Visualisation (Frontend)
Une fois les résultats générés, lancez l'interface de supervision :

streamlit run app/streamlit_app.py

🛠 Structure des Dossiers

.
├── data/           # Données SCADA brutes et préparées
├── models/         # Modèles IA (.pkl) - Accès restreint
├── results/        # Sorties d'inférence et rapports JSON
├── scripts/        # Moteur d'inférence et orchestrateur
├── docs/           # Contrat d'interface détaillé
├── app/            # Dashboard Streamlit
└── .env            # Fichier de configuration des secrets