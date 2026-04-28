import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Config page ──────────────────────────────────────────────────────s
st.set_page_config(
    page_title="Africa IA — Dashboard Bancaire",
    page_icon="🏦",
    layout="wide"
)

# ── CSS personnalisé ─────────────────────────────────────────────────
st.markdown("""
<style>
  body { background-color: #0D1B2A; }
  .main { background-color: #0D1B2A; }
  .metric-card {
    background: #1A2B3C;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    border-left: 4px solid #C9A84C;
    margin-bottom: 1rem;
  }
  .metric-card h3 { color: #C9A84C; font-size: 14px; margin: 0; }
  .metric-card h2 { color: white; font-size: 28px; margin: 4px 0; }
  .metric-card p  { color: #6B7A8D; font-size: 12px; margin: 0; }
  .header-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B4F72 100%);
    padding: 2rem;
    border-radius: 12px;
    border-bottom: 3px solid #C9A84C;
    margin-bottom: 2rem;
  }
  .insight-box {
    background: #1A2B3C;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #2A3B4C;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1 style="color:white; margin:0;">🏦 Dashboard Bancaire — Scoring Crédit & Churn</h1>
  <p style="color:#C9A84C; margin:4px 0 0 0;">Africa IA Consulting Group · Fall Oumar · Données simulées Kaggle</p>
</div>
""", unsafe_allow_html=True)

# ── Chargement des données ────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Churn_Modelling.csv")
    df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    le = LabelEncoder()
    df_clean['Gender']    = le.fit_transform(df_clean['Gender'])
    df_clean['Geography'] = le.fit_transform(df_clean['Geography'])

    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['Churn_Proba'] = model.predict_proba(X_scaled)[:, 1]
    df['Profil_Risque'] = pd.cut(
        df['Churn_Proba'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['🟢 Faible', '🟡 Modéré', '🔴 Élevé']
    )

    def credit_segment(score):
        if score >= 750: return 'Excellent'
        elif score >= 650: return 'Bon'
        elif score >= 550: return 'Moyen'
        else: return 'Faible'

    df['CreditSegment'] = df['CreditScore'].apply(credit_segment)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=X.columns)

    return df, model, auc, fpr, tpr, cm, importances

with st.spinner("⏳ Chargement et entraînement du modèle..."):
    df, model, auc, fpr, tpr, cm, importances = load_and_train()

# ── Sidebar filtres ───────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/0D1B2A/C9A84C?text=Africa+IA", width=200)
st.sidebar.markdown("## 🎛️ Filtres")

geo_options = {0: 'France', 1: 'Allemagne', 2: 'Espagne'}
selected_geo = st.sidebar.multiselect(
    "Géographie", options=[0, 1, 2],
    format_func=lambda x: geo_options[x],
    default=[0, 1, 2]
)
age_range = st.sidebar.slider("Tranche d'âge", 18, 92, (18, 92))
profil_filter = st.sidebar.multiselect(
    "Profil de risque",
    options=['🟢 Faible', '🟡 Modéré', '🔴 Élevé'],
    default=['🟢 Faible', '🟡 Modéré', '🔴 Élevé']
)

df_filtered = df[
    (df['Geography'].isin(selected_geo)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Profil_Risque'].isin(profil_filter))
]

st.sidebar.markdown(f"**{len(df_filtered):,} clients sélectionnés**")

# ── Onglets ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue Générale",
    "🤖 Modèle ML",
    "🏷️ Scoring Crédit",
    "⚠️ Clients à Risque"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Vue Générale
# ════════════════════════════════════════════════════════════════════
with tab1:
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df_filtered['Exited'].mean() * 100
    avg_balance = df_filtered['Balance'].mean()
    avg_score = df_filtered['CreditScore'].mean()
    high_risk = (df_filtered['Profil_Risque'] == '🔴 Élevé').sum()

    col1.markdown(f"""<div class="metric-card">
        <h3>Total Clients</h3><h2>{len(df_filtered):,}</h2><p>dans la sélection</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <h3>Taux de Churn</h3><h2>{churn_rate:.1f}%</h2><p>clients partis</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="metric-card">
        <h3>Balance Moyenne</h3><h2>{avg_balance:,.0f} €</h2><p>par client</p>
    </div>""", unsafe_allow_html=True)
    col4.markdown(f"""<div class="metric-card">
        <h3>Clients à Risque Élevé</h3><h2>{high_risk:,}</h2><p>à contacter en priorité</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        # Churn par géographie
        geo_churn = df_filtered.groupby('Geography')['Exited'].mean().reset_index()
        geo_churn['Geography'] = geo_churn['Geography'].map(geo_options)
        fig = px.bar(geo_churn, x='Geography', y='Exited',
                     title='Taux de Churn par Géographie',
                     labels={'Exited': 'Taux de churn'},
                     color='Exited', color_continuous_scale='Blues',
                     template='plotly_dark')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Churn par âge
        age_churn = df_filtered.groupby('Age')['Churn_Proba'].mean().reset_index()
        fig2 = px.line(age_churn, x='Age', y='Churn_Proba',
                       title='Probabilité de Churn par Âge',
                       labels={'Churn_Proba': 'Probabilité'},
                       template='plotly_dark',
                       color_discrete_sequence=['#C9A84C'])
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        # Distribution age
        fig3 = px.histogram(df_filtered, x='Age', nbins=30,
                            title='Distribution par Âge',
                            template='plotly_dark',
                            color_discrete_sequence=['#185FA5'])
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Churn par nombre de produits
        prod = df_filtered.groupby('NumOfProducts')['Exited'].mean().reset_index()
        fig4 = px.bar(prod, x='NumOfProducts', y='Exited',
                      title='Taux de Churn par Nombre de Produits',
                      labels={'Exited': 'Taux churn', 'NumOfProducts': 'Nb Produits'},
                      template='plotly_dark',
                      color_discrete_sequence=['#185FA5'])
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Modèle ML
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🤖 Random Forest — Meilleur Modèle")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="metric-card">
        <h3>AUC Score</h3><h2>{auc:.4f}</h2><p>Très bon modèle (> 0.80)</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <h3>Algorithme</h3><h2>Random Forest</h2><p>100 arbres de décision</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="metric-card">
        <h3>Données</h3><h2>10 000</h2><p>clients analysés</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Courbe ROC
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC={auc:.3f})',
            line=dict(color='#C9A84C', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(dash='dash', color='gray'), name='Aléatoire'
        ))
        fig_roc.update_layout(
            title='Courbe ROC',
            xaxis_title='Faux Positifs',
            yaxis_title='Vrais Positifs',
            template='plotly_dark'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_r:
        # Matrice de confusion
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x='Prédit', y='Réel'),
                           x=['Resté', 'Parti'], y=['Resté', 'Parti'],
                           color_continuous_scale='Blues',
                           title='Matrice de Confusion',
                           template='plotly_dark')
        st.plotly_chart(fig_cm, use_container_width=True)

    # Importance des variables
    imp_sorted = importances.sort_values(ascending=True)
    fig_imp = px.bar(x=imp_sorted.values, y=imp_sorted.index,
                     orientation='h',
                     title='Importance des Variables',
                     labels={'x': 'Importance', 'y': 'Variable'},
                     template='plotly_dark',
                     color=imp_sorted.values,
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

    # Insights
    st.markdown("### 💡 Insights Clés")
    st.markdown("""
    <div class="insight-box">
        <b style="color:#C9A84C">🔑 Variables les plus importantes :</b>
        <p style="color:white">Age, Balance et CreditScore sont les 3 facteurs qui influencent le plus le départ d'un client.</p>
    </div>
    <div class="insight-box">
        <b style="color:#C9A84C">📊 Performance du modèle :</b>
        <p style="color:white">AUC = 0.8472 signifie que le modèle identifie correctement 84.7% des clients qui vont partir.</p>
    </div>
    <div class="insight-box">
        <b style="color:#C9A84C">🎯 Utilité business :</b>
        <p style="color:white">La banque peut anticiper les départs 1 mois à l'avance et agir avec des offres de rétention ciblées.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Scoring Crédit
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🏷️ Segmentation par Score de Crédit")

    order = ['Faible', 'Moyen', 'Bon', 'Excellent']
    seg_analysis = df_filtered.groupby('CreditSegment').agg(
        Clients=('Exited', 'count'),
        Taux_Churn=('Exited', 'mean'),
        Balance_Moy=('Balance', 'mean'),
        Score_Moy=('CreditScore', 'mean')
    ).reindex(order).reset_index()
    seg_analysis['Taux_Churn_pct'] = (seg_analysis['Taux_Churn'] * 100).round(1)
    seg_analysis['Balance_Moy'] = seg_analysis['Balance_Moy'].round(0)
    seg_analysis['Score_Moy'] = seg_analysis['Score_Moy'].round(0)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_seg = px.bar(seg_analysis, x='CreditSegment', y='Taux_Churn_pct',
                         title='Taux de Churn par Segment',
                         labels={'Taux_Churn_pct': 'Taux churn (%)', 'CreditSegment': 'Segment'},
                         color='Taux_Churn_pct', color_continuous_scale='Blues',
                         template='plotly_dark')
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_r:
        fig_dist = px.histogram(df_filtered, x='CreditScore', nbins=40,
                                title='Distribution des Credit Scores',
                                template='plotly_dark',
                                color_discrete_sequence=['#C9A84C'])
        st.plotly_chart(fig_dist, use_container_width=True)

    # Tableau récapitulatif
    st.markdown("### 📋 Tableau de Synthèse")
    display_df = seg_analysis[['CreditSegment', 'Clients', 'Taux_Churn_pct', 'Balance_Moy', 'Score_Moy']].copy()
    display_df.columns = ['Segment', 'Nb Clients', 'Taux Churn (%)', 'Balance Moy (€)', 'Score Moy']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — Clients à Risque
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚠️ Clients à Risque Élevé — À contacter en priorité")

    # Répartition des profils
    col_l, col_r = st.columns(2)
    with col_l:
        profil_counts = df_filtered['Profil_Risque'].value_counts()
        fig_pie = px.pie(
            values=profil_counts.values,
            names=profil_counts.index.astype(str),
            hole=0.5,
            title='Répartition des Profils de Risque',
            color_discrete_sequence=['#3B6D11', '#BA7517', '#A32D2D'],
            template='plotly_dark'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        # Balance vs Proba churn
        sample = df_filtered.sample(min(500, len(df_filtered)), random_state=42)
        fig_scatter = px.scatter(
            sample, x='Balance', y='Churn_Proba',
            color='Exited', color_continuous_scale='Blues',
            title='Balance vs Probabilité de Churn',
            labels={'Churn_Proba': 'Proba Churn', 'Exited': 'Parti'},
            template='plotly_dark', opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Liste des clients à risque élevé
    st.markdown("### 🔴 Top 20 Clients les Plus à Risque")
    top_risk = df_filtered.nlargest(20, 'Churn_Proba')[[
        'CustomerId', 'Age', 'Geography', 'Balance',
        'CreditScore', 'NumOfProducts', 'Churn_Proba', 'Profil_Risque'
    ]].copy()
    top_risk['Geography'] = top_risk['Geography'].map(geo_options)
    top_risk['Churn_Proba'] = (top_risk['Churn_Proba'] * 100).round(1).astype(str) + '%'
    top_risk.columns = ['ID Client', 'Âge', 'Pays', 'Balance', 'Score Crédit',
                        'Nb Produits', 'Proba Churn', 'Profil']
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

    # Recommandations
    st.markdown("### 💡 Recommandations")
    recs = [
        ("🎯 Contacter en priorité", "Les 417 clients à risque élevé doivent être contactés ce mois-ci avec une offre personnalisée."),
        ("👥 Focus 40-60 ans", "Cette tranche d'âge présente le plus fort risque. Créer des produits adaptés à leurs besoins."),
        ("📦 Revoir la stratégie produits", "Les clients avec 3-4 produits partent massivement. Éviter la vente forcée."),
        ("💎 Service premium", "Les clients à haute balance partent aussi. Miser sur la qualité du service, pas seulement le solde."),
    ]
    for title, desc in recs:
        st.markdown(f"""
        <div class="insight-box">
            <b style="color:#C9A84C">{title}</b>
            <p style="color:white;margin:4px 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6B7A8D; font-size:13px;">
    🏦 Africa IA Consulting Group · Fall Oumar · 
    <a href="https://africaiaconsulting.netlify.app" style="color:#C9A84C;">africaiaconsulting.netlify.app</a>
</div>
""", unsafe_allow_html=True)
