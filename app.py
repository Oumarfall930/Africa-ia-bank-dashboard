import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Africa AI Dashboard",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS ULTRA MODERNE
# ─────────────────────────────────────────────
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0D1B2A, #1B263B);
}

/* HEADER */
.header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 20px;
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.header img {
    width: 90px;
}

.title {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.subtitle {
    color: #C9A84C;
}

/* KPI CARDS */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}

.card h3 {
    color: #C9A84C;
    font-size: 14px;
}

.card h2 {
    color: white;
}

/* SECTION */
.section {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #0D1B2A;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER AVEC LOGO
# ─────────────────────────────────────────────
if os.path.exists("logo.jpeg"):
    logo_path = "logo.jpeg"
else:
    logo_path = ""

st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{open(logo_path, "rb").read().encode("base64").decode() if logo_path else ''}">
    <div>
        <div class="title">Dashboard Bancaire IA</div>
        <div class="subtitle">Africa AI Consulting Group · Fall Oumar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():

    if not os.path.exists("churn.csv"):
        st.error("❌ Fichier churn.csv introuvable")
        st.stop()

    df = pd.read_csv("churn.csv")

    df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    df_clean = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)

    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    df['Churn_Proba'] = model.predict_proba(X_scaled)[:, 1]

    df['Profil_Risque'] = pd.cut(
        df['Churn_Proba'],
        bins=[0,0.3,0.6,1],
        labels=['Faible','Modéré','Élevé']
    )

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    cm = confusion_matrix(y_test, model.predict(X_test))

    return df, auc, fpr, tpr, cm

df, auc, fpr, tpr, cm = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Filtres")

geo = st.sidebar.multiselect(
    "Pays",
    df['Geography'].unique(),
    default=df['Geography'].unique()
)

age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (18, 92))

risk = st.sidebar.multiselect(
    "Risque",
    df['Profil_Risque'].unique(),
    default=df['Profil_Risque'].unique()
)

df_filtered = df[
    (df['Geography'].isin(geo)) &
    (df['Age'].between(age[0], age[1])) &
    (df['Profil_Risque'].isin(risk))
]

if len(df_filtered) == 0:
    st.warning("Aucune donnée")
    st.stop()

# ─────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────
c1,c2,c3 = st.columns(3)

c1.markdown(f"<div class='card'><h3>Clients</h3><h2>{len(df_filtered)}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><h3>Churn</h3><h2>{df_filtered['Exited'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><h3>Balance</h3><h2>{df_filtered['Balance'].mean():.0f}€</h2></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────
col1,col2 = st.columns(2)

with col1:
    fig = px.bar(df_filtered.groupby('Geography')['Exited'].mean().reset_index(),
                 x='Geography', y='Exited', title="Churn par pays")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.histogram(df_filtered, x='Age', title="Distribution âge")
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# ROC
# ─────────────────────────────────────────────
st.subheader("Performance modèle")

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash')))
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TOP RISQUE
# ─────────────────────────────────────────────
st.subheader("Clients à risque")

top = df_filtered.nlargest(20, 'Churn_Proba')
st.dataframe(top)

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
st.download_button(
    "Télécharger CSV",
    top.to_csv(index=False),
    "clients_risque.csv"
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("© Africa AI Consulting Group")
