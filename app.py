import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import base64
import os

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="Africa AI Dashboard", layout="wide")

# ───────────────────────── LOGO ─────────────────────────
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = ""
if os.path.exists("logo.png"):
    logo = get_base64("logo.png")

# ───────────────────────── CSS PREMIUM ─────────────────────────
st.markdown(f"""
<style>

body {{
    background: #071421;
    color: white;
}}

.block-container {{
    padding-top: 1rem;
}}

.header {{
    background: linear-gradient(90deg,#071421,#0D2A4A);
    padding:20px;
    border-radius:15px;
    border:1px solid #C9A84C;
    display:flex;
    align-items:center;
    justify-content:space-between;
}}

.header-left {{
    display:flex;
    align-items:center;
    gap:20px;
}}

.header img {{
    width:80px;
}}

.title {{
    font-size:26px;
    font-weight:bold;
}}

.subtitle {{
    color:#C9A84C;
}}

.card {{
    background:#0D1B2A;
    padding:20px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);
    text-align:center;
}

.card h3 {{
    color:#C9A84C;
    font-size:13px;
}}

.card h2 {{
    font-size:26px;
}}

.section {{
    background:#0D1B2A;
    padding:15px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);
}}

[data-testid="stSidebar"] {{
    background:#071421;
}}

</style>

<div class="header">
    <div class="header-left">
        <img src="data:image/png;base64,{logo}">
        <div>
            <div class="title">DASHBOARD BANCAIRE</div>
            <div class="subtitle">SCORING CRÉDIT & CHURN</div>
        </div>
    </div>
    <div style="text-align:right">
        <div style="color:#C9A84C;">Africa AI Consulting</div>
        <div style="font-size:12px;">Données simulées</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── DATA ─────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("churn.csv")

    df_clean = df.drop(columns=['RowNumber','CustomerId','Surname'])
    df_clean = pd.get_dummies(df_clean, columns=['Geography','Gender'], drop_first=True)

    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    df['Proba'] = model.predict_proba(X_scaled)[:,1]
    df['Risque'] = pd.cut(df['Proba'], bins=[0,0.3,0.6,1], labels=['Faible','Modéré','Élevé'])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    fpr,tpr,_ = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    return df, auc, fpr, tpr

df, auc, fpr, tpr = load()

# ───────────────────────── SIDEBAR ─────────────────────────
st.sidebar.markdown("## FILTRES")

geo = st.sidebar.multiselect("Géographie", df['Geography'].unique(), default=df['Geography'].unique())
age = st.sidebar.slider("Âge", 18, 92, (18,92))
risk = st.sidebar.multiselect("Risque", df['Risque'].unique(), default=df['Risque'].unique())

df_f = df[
    (df['Geography'].isin(geo)) &
    (df['Age'].between(age[0], age[1])) &
    (df['Risque'].isin(risk))
]

# ───────────────────────── KPI ─────────────────────────
c1,c2,c3,c4 = st.columns(4)

c1.markdown(f"<div class='card'><h3>CLIENTS</h3><h2>{len(df_f)}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><h3>CHURN</h3><h2>{df_f['Exited'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><h3>BALANCE</h3><h2>{df_f['Balance'].mean():.0f} €</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><h3>RISQUE ÉLEVÉ</h3><h2>{(df_f['Risque']=='Élevé').sum()}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────── GRAPHIQUES ─────────────────────────
col1,col2 = st.columns(2)

with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    fig = px.bar(df_f.groupby('Geography')['Exited'].mean().reset_index(),
                 x='Geography', y='Exited', title="Churn par pays")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    fig2 = px.histogram(df_f, x='Age', title="Distribution âge")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── ROC ─────────────────────────
st.markdown("<div class='section'>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"AUC={auc:.3f}"))
fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash='dash')))
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── TABLE ─────────────────────────
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Clients à risque")
st.dataframe(df_f.nlargest(20,'Proba'))
st.markdown("</div>", unsafe_allow_html=True)
