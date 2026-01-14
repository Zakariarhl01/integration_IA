📄 CONTRAT D'INTERFACE - BACKEND ENERGI-PRO
Ce document définit les spécifications techniques du moteur d'inférence ENERGI-PRO. Il sert de référence pour l'intégration du backend avec le Dashboard ou tout autre système tiers.

1. Description du Service
Le backend est un moteur d'inférence de type Batch CLI. Il traite les logs de maintenance et les données SCADA pour identifier les risques de pannes éoliennes sous 7 jours via une approche hybride (Machine Learning + Seuils Statistiques).

2. Protocole de Sécurité
L'accès au moteur est protégé. Toute exécution nécessite la validation d'un jeton d'authentification.

Méthode : Variable d'environnement.

Clé : ENERGI_PRO_API_KEY

Valeur attendue : ET-PRO-2026-CONFIDENTIAL

Note : En cas d'absence de clé, le service renvoie un code d'erreur 403 Forbidden et interrompt le traitement.

Voici le contenu complet pour ton fichier contrat_interface.md. Ce document est ta pièce maîtresse pour prouver au professeur que ton script est un service professionnel et documenté.

📄 CONTRAT D'INTERFACE - BACKEND ENERGI-PRO
Ce document définit les spécifications techniques du moteur d'inférence ENERGI-PRO. Il sert de référence pour l'intégration du backend avec le Dashboard ou tout autre système tiers.

1. Description du Service
Le backend est un moteur d'inférence de type Batch CLI. Il traite les logs de maintenance et les données SCADA pour identifier les risques de pannes éoliennes sous 7 jours via une approche hybride (Machine Learning + Seuils Statistiques).

2. Protocole de Sécurité
L'accès au moteur est protégé. Toute exécution nécessite la validation d'un jeton d'authentification.

Méthode : Variable d'environnement.

Clé : ENERGI_PRO_API_KEY

Valeur attendue : ET-PRO-2026-CONFIDENTIAL

Note : En cas d'absence de clé, le service renvoie un code d'erreur 403 Forbidden et interrompt le traitement.

3. Spécification des Entrées (Request Contract)
Le fichier source doit être au format CSV et situé dans ./data/. Il doit obligatoirement contenir les colonnes suivantes :

Champ,            Type,    Contrainte,   Description
turbine_id,       Integer, Requis,       Identifiant unique de l'éolienne.
wind_speed,       Float,   > 0,          Vitesse du vent mesurée par l'anémomètre.
vibration_level,  Float,   Non nul,      Niveau de vibration de la nacelle.
temperature,      Float,   Celsius,      Température interne des composants.
power_output,     Float,   kW,           Puissance électrique produite.
maintenance_done, Boolean, 0 ou 1,       Indique si une maintenance a déjà eu lieu.

4. Spécification des Sorties (Response Contract)

4.1. Fichier de Résultats (results/anomalies_non_gerees_final.csv)
Le service génère un fichier enrichi avec les prédictions de l'IA :

Champ,          Type,          Description
proba_panne,    Float [0-1],   Probabilité de défaillance calculée par l'IA.
anomaly_type,   String,        "Libellé de l'anomalie (Upper, Lower, Zero Detection)."
anomaly_column, String,        Capteur ayant déclenché l'alerte statistique.

4.2. Flux de Supervision (results/detection_stats.json)
Pour le monitoring, un fichier JSON résume l'exécution :

{
    "upper": 47,
    "lower": 40,
    "zero_detection_issue": 19,
    "total_anomalies_detectees": 106
}

5. Codes de Retour (Status Codes)
Le script communique son état au système via des codes de sortie standardisés :

Code 0 : SUCCESS - Traitement terminé sans erreur.

Code 1 : GENERAL_ERROR - Erreur de logique ou fichier manquant.

Code 403 : AUTH_FAILED - Clé API invalide ou absente.