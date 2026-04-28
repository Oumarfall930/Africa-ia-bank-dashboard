import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import base64
import os

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="Africa AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# ───────────────────────── LOGO ─────────────────────────
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = ""
if os.path.exists("logo.png"):
    logo = get_base64("logo.png")

logo_tag = f'<img src="data:image/png;base64,{logo}" style="width:90px;height:90px;object-fit:contain;">' if logo else '<div style="width:90px;height:90px;background:linear-gradient(135deg,#1a4a8a,#0D2A4A);border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid #C9A84C;font-size:28px;color:#C9A84C;font-weight:bold;">AI</div>'

# ───────────────────────── CSS ─────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    background-color: #071421 !important;
    color: #E0E6F0 !important;
    font-family: 'Inter', sans-serif !important;
}}

.block-container {{
    padding: 0.5rem 1.5rem 2rem 1.5rem !important;
    max-width: 100% !important;
}}

/* ── HEADER ── */
.top-header {{
    background: linear-gradient(90deg, #071421 0%, #0D2A4A 60%, #071421 100%);
    border: 1px solid #C9A84C;
    border-radius: 14px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}}
.header-brand {{
    display: flex;
    align-items: center;
    gap: 18px;
}}
.brand-name {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: white;
    letter-spacing: 2px;
    line-height: 1.1;
}}
.brand-sub {{
    color: #C9A84C;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
}}
.brand-tags {{
    display: flex;
    gap: 12px;
    margin-top: 6px;
    font-size: 10px;
    color: #7a99bb;
    letter-spacing: 1px;
}}
.divider-v {{
    width: 1px;
    height: 70px;
    background: #C9A84C55;
    margin: 0 24px;
}}
.header-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: white;
    letter-spacing: 2px;
    line-height: 1.0;
}}
.header-subtitle {{
    color: #C9A84C;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 2px;
}}
.header-tagline {{
    color: #7a99bb;
    font-size: 11px;
    margin-top: 4px;
    letter-spacing: 1px;
}}
.header-tagline span {{
    color: #C9A84C;
}}
.header-right {{
    text-align: right;
    font-size: 12px;
    color: #7a99bb;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 10px 16px;
    background: rgba(255,255,255,0.03);
}}
.header-right .date-label {{
    font-size: 10px;
    color: #C9A84C;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.live-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4CAF50;
    border-radius: 50%;
    margin-right: 5px;
}}

/* ── KPI CARDS ── */
.kpi-card {{
    background: #0D1B2A;
    border: 1px solid #1a3a5c;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    min-height: 110px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
}}
.kpi-blue::after {{ background: #2196F3; }}
.kpi-red::after {{ background: #F44336; }}
.kpi-green::after {{ background: #4CAF50; }}
.kpi-gold::after {{ background: #C9A84C; }}
.kpi-icon {{
    font-size: 30px;
    flex-shrink: 0;
    width: 52px;
    height: 52px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.kpi-icon-blue {{ background: rgba(33,150,243,0.15); }}
.kpi-icon-red {{ background: rgba(244,67,54,0.15); }}
.kpi-icon-green {{ background: rgba(76,175,80,0.15); }}
.kpi-icon-gold {{ background: rgba(201,168,76,0.15); }}
.kpi-label {{
    font-size: 11px;
    color: #7a99bb;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}}
.kpi-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: white;
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px;
    color: #4a6a8a;
    margin-top: 2px;
}}

/* ── TABS ── */
.tab-bar {{
    display: flex;
    gap: 8px;
    margin: 18px 0 14px 0;
}}
.tab-item {{
    padding: 10px 22px;
    border-radius: 10px;
    border: 1px solid #1a3a5c;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #7a99bb;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s;
}}
.tab-active {{
    background: linear-gradient(135deg, #1a4a8a, #0D2A4A);
    border-color: #2196F3;
    color: white;
}}

/* ── SECTION ── */
.section {{
    background: #0D1B2A;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
}}
.section-title {{
    font-size: 13px;
    font-weight: 600;
    color: #C9A84C;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: #071421 !important;
    border-right: 1px solid #1a3a5c !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding: 1rem !important;
}}

.sidebar-section {{
    background: #0D1B2A;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}}
.sidebar-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 2px;
    color: white;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.sidebar-label {{
    font-size: 12px;
    color: #7a99bb;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.count-box {{
    background: #0D1B2A;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}}
.count-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #C9A84C;
}}
.count-label {{
    font-size: 12px;
    color: #7a99bb;
}}
.export-btn {{
    background: linear-gradient(135deg, #0D2A4A, #1a3a5c);
    border: 1px solid #2196F3;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    cursor: pointer;
    color: white;
    font-size: 13px;
    font-weight: 600;
}}

/* ── INSIGHT CARDS ── */
.insight-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}}
.insight-icon {{
    font-size: 22px;
    flex-shrink: 0;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.insight-text {{
    font-size: 12px;
    color: #a0b8cc;
    line-height: 1.5;
}}
.insight-text span {{
    color: #C9A84C;
    font-weight: 600;
}}

/* ── Streamlit overrides ── */
.stMultiSelect > div {{
    background: #0D1B2A !important;
    border-color: #1a3a5c !important;
}}
.stSlider .st-bx {{ background: #C9A84C !important; }}
div[data-testid="metric-container"] {{
    background: transparent !important;
}}
</style>

{{% raw %}}
<div class="top-header">
    <div class="header-brand">
        {logo_tag}
        <div>
            <div class="brand-name">AFRICA AI</div>
            <div class="brand-sub">— Consulting Group —</div>
            <div class="brand-tags">
                <span>📊 DATA</span>
                <span>🤖 ARTIFICIAL INTELLIGENCE</span>
                <span>☁️ CLOUD SOLUTIONS</span>
            </div>
        </div>
    </div>
    <div class="divider-v"></div>
    <div>
        <div class="header-title">DASHBOARD BANCAIRE</div>
        <div class="header-subtitle">SCORING CRÉDIT & CHURN</div>
        <div class="header-tagline">TRANSFORMING DATA INTO <span>INTELLIGENT SOLUTIONS</span> FOR AFRICA</div>
    </div>
    <div class="header-right">
        <div class="date-label">📅 Mise à jour</div>
        <div style="font-size:14px;color:white;font-weight:600;margin-top:4px;">23 Mai 2024</div>
        <div style="margin-top:8px;"><span class="live-dot"></span>Données simulées</div>
    </div>
</div>
{{% endraw %}}
""".replace("{% raw %}", "").replace("{% endraw %}", ""), unsafe_allow_html=True)

# ───────────────────────── DATA ─────────────────────────
@st.cache_data
def load():
    if os.path.exists("churn.csv"):
        df = pd.read_csv("churn.csv")
    else:
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'RowNumber': range(1, n+1),
            'CustomerId': np.random.randint(1e7, 9e7, n),
            'Surname': ['Client']*n,
            'CreditScore': np.random.randint(300, 850, n),
            'Geography': np.random.choice(['France','Allemagne','Espagne'], n, p=[0.5,0.25,0.25]),
            'Gender': np.random.choice(['Male','Female'], n),
            'Age': np.random.randint(18, 92, n),
            'Tenure': np.random.randint(0, 10, n),
            'Balance': np.random.uniform(0, 250000, n),
            'NumOfProducts': np.random.choice([1,2,3,4], n, p=[0.5,0.35,0.1,0.05]),
            'HasCrCard': np.random.choice([0,1], n),
            'IsActiveMember': np.random.choice([0,1], n),
            'EstimatedSalary': np.random.uniform(10000, 200000, n),
            'Exited': np.random.choice([0,1], n, p=[0.8,0.2]),
        })

    df_clean = df.drop(columns=['RowNumber','CustomerId','Surname'])
    df_clean = pd.get_dummies(df_clean, columns=['Geography','Gender'], drop_first=True)

    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['Proba'] = model.predict_proba(X_scaled)[:, 1]
    df['Risque'] = pd.cut(df['Proba'], bins=[0, 0.3, 0.6, 1], labels=['Faible','Modéré','Élevé'])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    return df, auc, fpr, tpr, cm, feat_imp

df, auc, fpr, tpr, cm, feat_imp = load()

# ───────────────────────── PLOTLY THEME ─────────────────────────
PLOT_BG = "#0D1B2A"
PAPER_BG = "#0D1B2A"
FONT_COLOR = "#E0E6F0"
GRID_COLOR = "#1a3a5c"
GOLD = "#C9A84C"
BLUE = "#2196F3"

def style_fig(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=FONT_COLOR), x=0),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig

# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ FILTRES</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">🌍 Géographie</div>', unsafe_allow_html=True)
    geo = st.multiselect("", df['Geography'].unique(), default=list(df['Geography'].unique()), label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">👤 Tranche d\'âge</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#4a6a8a;">18 - 92 ans</div>', unsafe_allow_html=True)
    age = st.slider("", 18, 92, (18, 92), label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">⚠️ Profil de risque</div>', unsafe_allow_html=True)
    risk = st.multiselect("", ['Faible','Modéré','Élevé'], default=['Faible','Modéré','Élevé'], label_visibility="collapsed")

    df_f = df[
        (df['Geography'].isin(geo)) &
        (df['Age'].between(age[0], age[1])) &
        (df['Risque'].isin(risk))
    ]

    st.markdown(f"""
    <div class="count-box">
        <div style="font-size:22px;margin-bottom:4px;">👥</div>
        <div class="count-value">{len(df_f):,}</div>
        <div class="count-label">clients sélectionnés</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="export-btn">
        ⬇️ Exporter la sélection<br>
        <span style="font-size:11px;color:#7a99bb;">Télécharger en CSV</span>
    </div>
    """, unsafe_allow_html=True)

    csv = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("", csv, "clients_selection.csv", "text/csv", key="dl", help="Télécharger", use_container_width=True)

# ───────────────────────── KPI ─────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card kpi-blue">
        <div class="kpi-icon kpi-icon-blue">👥</div>
        <div>
            <div class="kpi-label">TOTAL CLIENTS</div>
            <div class="kpi-value">{len(df_f):,}</div>
            <div class="kpi-sub">dans la sélection</div>
        </div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card kpi-red">
        <div class="kpi-icon kpi-icon-red">📉</div>
        <div>
            <div class="kpi-label">TAUX DE CHURN</div>
            <div class="kpi-value">{df_f['Exited'].mean()*100:.1f}%</div>
            <div class="kpi-sub">clients partis</div>
        </div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card kpi-green">
        <div class="kpi-icon kpi-icon-green">💳</div>
        <div>
            <div class="kpi-label">BALANCE MOYENNE</div>
            <div class="kpi-value">{df_f['Balance'].mean():,.0f} €</div>
            <div class="kpi-sub">par client</div>
        </div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card kpi-gold">
        <div class="kpi-icon kpi-icon-gold">⚠️</div>
        <div>
            <div class="kpi-label">CLIENTS À RISQUE ÉLEVÉ</div>
            <div class="kpi-value">{(df_f['Risque']=='Élevé').sum():,}</div>
            <div class="kpi-sub">à contacter en priorité</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ───────────────────────── TABS ─────────────────────────
st.markdown("""
<div class="tab-bar">
    <div class="tab-item tab-active">📊 Vue Générale</div>
    <div class="tab-item">🤖 Modèle ML</div>
    <div class="tab-item">🏷️ Scoring Crédit</div>
    <div class="tab-item">⚠️ Clients à Risque</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── ROW 1: 4 charts ─────────────────────────
g1, g2, g3, g4 = st.columns(4)

with g1:
    churn_geo = df_f.groupby('Geography')['Exited'].mean().reset_index()
    churn_geo['Exited_pct'] = churn_geo['Exited'] * 100
    fig = go.Figure(go.Bar(
        x=churn_geo['Geography'], y=churn_geo['Exited_pct'],
        marker_color=[BLUE]*len(churn_geo),
        text=[f"{v:.1f}%" for v in churn_geo['Exited_pct']],
        textposition='outside', textfont=dict(size=11, color=FONT_COLOR)
    ))
    style_fig(fig, "🌍 Taux de Churn par Géographie")
    fig.update_layout(yaxis_ticksuffix='%', showlegend=False, height=260)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    age_churn = df_f.groupby('Age')['Proba'].mean().reset_index()
    fig2 = go.Figure(go.Scatter(
        x=age_churn['Age'], y=age_churn['Proba']*100,
        fill='tozeroy', line=dict(color=GOLD, width=2),
        fillcolor='rgba(201,168,76,0.15)'
    ))
    style_fig(fig2, "👤 Probabilité de Churn par Âge")
    fig2.update_layout(yaxis_ticksuffix='%', height=260)
    st.plotly_chart(fig2, use_container_width=True)

with g3:
    fig3 = go.Figure(go.Histogram(
        x=df_f['Age'], nbinsx=20,
        marker_color=BLUE, opacity=0.85
    ))
    style_fig(fig3, "📊 Distribution par Âge")
    fig3.update_layout(height=260)
    st.plotly_chart(fig3, use_container_width=True)

with g4:
    prod_churn = df_f.groupby('NumOfProducts')['Exited'].mean().reset_index()
    colors = [BLUE, BLUE, '#9B59B6', '#9B59B6'][:len(prod_churn)]
    fig4 = go.Figure(go.Bar(
        x=prod_churn['NumOfProducts'].astype(str),
        y=prod_churn['Exited']*100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in prod_churn['Exited']],
        textposition='outside', textfont=dict(size=11, color=FONT_COLOR)
    ))
    style_fig(fig4, "📦 Taux de Churn par Nb de Produits")
    fig4.update_layout(
        yaxis_ticksuffix='%', showlegend=False, height=260,
        xaxis_title="Nombre de produits"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ───────────────────────── ROW 2: ROC + CM + Feature Imp + Insights ─────────────────────────
r1, r2, r3, r4 = st.columns([1.2, 1, 1.2, 1])

with r1:
    st.markdown('<div class="section-title">📈 Performance du Modèle</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, name=f"Random Forest (AUC = {auc:.3f})",
        line=dict(color=GOLD, width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], name="Aléatoire",
        line=dict(color="#4a6a8a", dash='dash', width=1)
    ))
    style_fig(fig_roc, "Courbe ROC")
    fig_roc.update_layout(
        height=300,
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        yaxis_tickformat='.0%', xaxis_tickformat='.0%',
        legend=dict(x=0.4, y=0.1, font=dict(size=10))
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with r2:
    st.markdown('<div class="section-title">🔲 Matrice de Confusion</div>', unsafe_allow_html=True)
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=['Resté','Parti'], y=['Resté','Parti'],
        colorscale=[[0,'#0D1B2A'],[1,'#2196F3']],
        showscale=True,
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color='white')
    ))
    style_fig(fig_cm, "Matrice de Confusion")
    fig_cm.update_layout(
        height=300,
        xaxis_title="Prédit",
        yaxis_title="Réel",
        xaxis=dict(side='top')
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with r3:
    st.markdown('<div class="section-title">📊 Importance des Variables</div>', unsafe_allow_html=True)
    top_feat = feat_imp.tail(10)
    fig_fi = go.Figure(go.Bar(
        x=top_feat.values, y=top_feat.index,
        orientation='h',
        marker_color=GOLD, opacity=0.9
    ))
    style_fig(fig_fi, "")
    fig_fi.update_layout(height=300, xaxis_title="Importance")
    st.plotly_chart(fig_fi, use_container_width=True)

with r4:
    st.markdown('<div class="section-title">💡 Insights Clés</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon" style="background:rgba(201,168,76,0.15);">🔑</div>
        <div class="insight-text">
            <span>Age, Balance et CreditScore</span> sont les 3 facteurs qui influencent le plus le départ d'un client.
        </div>
    </div>
    <div class="insight-card">
        <div class="insight-icon" style="background:rgba(76,175,80,0.15);">📈</div>
        <div class="insight-text">
            <span>AUC = {auc:.3f}</span> signifie que le modèle identifie correctement {auc*100:.1f}% des clients qui vont partir.
        </div>
    </div>
    <div class="insight-card">
        <div class="insight-icon" style="background:rgba(33,150,243,0.15);">🎯</div>
        <div class="insight-text">
            La banque peut anticiper les départs <span>1 mois à l'avance</span> et agir avec des offres ciblées.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ───────── CSS ─────────
st.markdown("""
<style>

.header {
    background: linear-gradient(90deg,#071421,#0D2A4A);
    padding:20px;
    border-radius:15px;
    border:1px solid #C9A84C;
    display:flex;
    align-items:center;
    justify-content:space-between;
}

.header-left {
    display:flex;
    align-items:center;
    gap:20px;
}

.header img {
    width:80px;
}

.title {
    font-size:26px;
    font-weight:bold;
}

.subtitle {
    color:#C9A84C;
    font-size:14px;
}

.card {
    background:#0D1B2A;
    padding:20px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);
    text-align:center;
}

.card h3 {
    color:#C9A84C;
    font-size:13px;
}

.card h2 {
    font-size:26px;
}

.section {
    background:#0D1B2A;
    padding:15px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] {
    background:#071421;
}

</style>
""", unsafe_allow_html=True)

# ───────── HEADER ─────────
st.markdown(f"""
<div class="header">
    <div class="header-left">
        <img src="data:image/png;base64,{logo}">
        <div>
            <div class="title">DASHBOARD BANCAIRE</div>
            <div class="subtitle">SCORING CRÉDIT & CHURN</div>
        </div>
    </div>

    <div style="text-align:right; line-height:1.4">
        <div style="color:#C9A84C; font-weight:bold;">
            Africa AI Consulting
        </div>
        <div style="font-size:12px; opacity:0.8;">
            Données simulées IA Banking Dashboard
        </div>
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
