import streamlit as st
import pandas as pd
import json 
import os 
import numpy as np

# --- CONFIGURATION (Chemins relatifs au script Streamlit) ---
FINAL_CSV_PATH = "../results/anomalies_non_gerees_final.csv"
METRICS_JSON_PATH = "../results/evaluation_metrics.json" 
DETECTION_STATS_PATH = "../results/detection_stats.json" 

# --- 1. FONCTIONS DE CHARGEMENT DES DONNÉES (AVEC CACHE) ---

@st.cache_data
def load_anomalies_data():
    """
    Charge le fichier CSV des anomalies à traiter.
    Convertit la colonne 'proba_panne' en float pour le tri et le formatage esthétique.
    """
    if not os.path.exists(FINAL_CSV_PATH):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(FINAL_CSV_PATH)
        
        if 'date_measure' in df.columns:
            df['date_measure'] = pd.to_datetime(df['date_measure'])
            
        # Colonne de risque en % (float) pour le tri et le formatage visuel du dataframe
        if 'proba_panne' in df.columns:
            df['Proba_Panne_Pct'] = (df['proba_panne'].astype(float) * 100).round(2)
        
        df = df.rename(columns={
            'turbine_id': 'ID Turbine',
            'date_measure': 'Date Mesure',
            'anomaly_column': 'Capteur Anormal',
            'anomaly_type': 'Type Anomalie',
            'time_to_failure_days': 'Jours Avant Panne (Sim.)',
        })
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier {FINAL_CSV_PATH} : {e}")
        return pd.DataFrame()

@st.cache_data
def load_evaluation_metrics():
    """Charge le fichier JSON des métriques d'évaluation du modèle."""
    if not os.path.exists(METRICS_JSON_PATH):
        return None
    try:
        with open(METRICS_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier {METRICS_JSON_PATH} : {e}")
        return None

@st.cache_data
def load_detection_stats():
    """Charge le fichier JSON des statistiques de détection."""
    if not os.path.exists(DETECTION_STATS_PATH):
        return None
    try:
        with open(DETECTION_STATS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier {DETECTION_STATS_PATH} : {e}")
        return None

# --- 2. CHARGEMENT ET INITIALISATION ---

st.set_page_config(layout="wide", page_title="EnergiTech - Tableau de Bord Maintenance Prédictive")

df_alertes = load_anomalies_data()
metrics = load_evaluation_metrics()
stats = load_detection_stats()

st.title("⚡ EnergiTech - Tableau de Bord Maintenance Prédictive")
st.markdown("Interface d'aide à la décision pour la priorisation des interventions de maintenance.")


# --- A. RAPPORT DE CONFORMITÉ MODÈLE A  ---

if metrics:
    with st.expander("📚 Rapport d'Évaluation Modèle A (Conformité)", expanded=False):
        
        st.markdown("##### 📈 Synthèse des performances")
        report = metrics['classification_report']
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Précision Globale (Accuracy)", f"{metrics['accuracy']:.4f}", help="Pourcentage d'observations correctement classées.")
        
        if 'Classe 1 (Panne)' in report:
            panne_metrics = report['Classe 1 (Panne)']
            
            # Utilisation de couleurs pour souligner le Recall (FN) et la Precision (FP)
            col2.metric("Rappel (Recall) - Panne", f"{panne_metrics['recall']:.2f}", 
                        delta_color="inverse", 
                        delta="🎯 Minimiser les FN")
            
            col3.metric("Précision - Panne", f"{panne_metrics['precision']:.2f}",
                        delta_color="normal", 
                        delta="✅ Minimiser les FP")
            
            col4.metric("F1-Score - Panne", f"{panne_metrics['f1-score']:.2f}", delta="Équilibre des deux")
            
        st.markdown("##### 📋 Matrice de Confusion")
        cm_data = metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm_data,
            index=['Réel: Pas Panne (0)', 'Réel: Panne (1)'],
            columns=['Prédit: Pas Panne (0)', 'Prédit: Panne (1)']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        st.markdown('---')
        tn, fp, fn, tp = cm_data[0][0], cm_data[0][1], cm_data[1][0], cm_data[1][1]
        
        st.markdown(f"**🔴 Faux Négatifs (FN) :** `{fn}` pannes réelles manquées (Risque critique).")
        st.markdown(f"**🟡 Faux Positifs (FP) :** `{fp}` fausses alertes (Coût en inspections inutiles).")
    st.markdown("---") 


# --- B. STATISTIQUES DE DÉTECTION BRUTES (Esthétique : métriques plus visuelles) ---

if stats:
    total_anomalies = stats.get('total_anomalies_detectees', 0)
    filtered_anomalies = len(df_alertes) 
    managed_anomalies = total_anomalies - filtered_anomalies
    total_zero_issues = stats.get('zero_detection_issue', 0)
    
    # Calcul des stats pour le problème de Zéro Détection
    filtered_zero_issues = df_alertes[df_alertes['Type Anomalie'] == 'zero_detection_issue'].shape[0]
    managed_zero_issues = total_zero_issues - filtered_zero_issues
    
    with st.expander("📊 Synthèse des Statistiques de Détection (Brutes vs Gérées)", expanded=False):
        
        st.info(f"Le pipeline a détecté **{total_anomalies}** anomalies au total. Seulement **{filtered_anomalies}** sont affichées ci-dessous car elles n'ont pas encore été suivies d'une maintenance.")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        col_s1.metric(
            label="Anomalies Détectées (Total)", 
            value=total_anomalies,
            help="Total des cas où un capteur a dépassé l'IQR ou a renvoyé zéro."
        )

        col_s2.metric(
            label="Anomalies *Gérées* (Maintenance OK)", 
            value=managed_anomalies,
            delta=f"{(managed_anomalies / total_anomalies * 100):.1f} % traité",
            delta_color="normal",
            help="Anomalies pour lesquelles une maintenance est déjà enregistrée."
        )
        
        col_s3.metric(
            label="Problèmes de 'Zéro Capteur' (Total)", 
            value=total_zero_issues,
            help="Nombre total de valeurs enregistrées à zéro (avant filtrage)."
        )
        
        col_s4.metric(
            label="Zéro Capteur *Actifs*", 
            value=filtered_zero_issues,
            delta_color="inverse",
            delta=f"{filtered_zero_issues} à traiter",
            help="Cas de 'Zéro Capteur' nécessitant une intervention."
        )
    st.markdown("---") 


# --- C. ALERTES ET ACTIONS PRIORITAIRES (Tableau Interactif + Esthétique) ---

if df_alertes.empty:
    st.info("Le tableau de bord est vide. Exécutez le pipeline via main.py pour générer les données.")

else:
    
    st.subheader(f"🚨 Alertes Actives à Prioriser ({len(df_alertes)} cas non gérés)")

    # 1. Widgets interactifs
    col_w1, col_w2, col_w3, col_w4 = st.columns([1, 1, 2, 2])
    
    # Sélecteur de tri
    sort_option = col_w1.radio(
        "Trier le tableau par :",
        ('Risque de Panne', 'Date'),
        index=0, 
        horizontal=True
    )
    
    # Affichage du Risque Maximal (plus impactant en début de section)
    max_risk_value = df_alertes['Proba_Panne_Pct'].max()
    col_w3.metric(label="Risque Maximal Actif", value=f"{max_risk_value} %", delta="Urgence 🔥", delta_color="inverse")
    
    # Champ de recherche
    search_query = col_w4.text_input("🔍 Rechercher (ID Turbine, Capteur, Type...) :", value="")

    # 2. Application du tri
    if sort_option == 'Risque de Panne':
        sort_column = 'Proba_Panne_Pct' 
        ascending = False
    else:
        sort_column = 'Date Mesure'
        ascending = False

    df_display = df_alertes.sort_values(by=sort_column, ascending=ascending)
    
    # 3. Application du filtre de recherche s'il y en a une
    if search_query:
        search_query = search_query.lower()
        df_display = df_display[
            df_display['ID Turbine'].astype(str).str.contains(search_query) |
            df_display['Type Anomalie'].str.lower().str.contains(search_query) |
            df_display['Capteur Anormal'].str.lower().str.contains(search_query)
        ]
        col_w3.info(f"{len(df_display)} résultats trouvés.")


    # 4. Affichage du tableau interactif complet avec style
    
    # Colonnes finales à afficher
    columns_to_show = [
        'ID Turbine', 
        'Date Mesure', 
        'Proba_Panne_Pct', 
        'Capteur Anormal', 
        'Type Anomalie', 
        'Jours Avant Panne (Sim.)', 
        'technician_id'
    ]
    
    # Règle de style : appliquer une couleur dégradée à la colonne de probabilité
    styled_df = df_display[columns_to_show].style.background_gradient(
        cmap='RdYlGn_r', 
        subset=['Proba_Panne_Pct'], 
        vmin=df_alertes['Proba_Panne_Pct'].min(), 
        vmax=df_alertes['Proba_Panne_Pct'].max()
    ).format({
        'Proba_Panne_Pct': "{:.2f} %" 
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#007bff'), ('color', 'white')]} 
    ])
    
    # Affichage du dataframe stylisé
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
    )
    
    st.caption("Le tri par défaut est basé sur le Risque de Panne. Les couleurs indiquent la criticité (Rouge = Urgence, Vert = Moins Critique).")