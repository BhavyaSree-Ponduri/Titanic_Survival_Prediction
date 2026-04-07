"""
utils.py - Helper functions for Titanic Survival Prediction System
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64, io, os, smtplib
from email.mime.text import MIMEText

# ─── Color Palettes ────────────────────────────────────────────────────────────
LIGHT = {
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "primary": "#4F46E5",
    "secondary": "#22C55E",
    "accent": "#38BDF8",
    "text": "#1E293B",
    "text2": "#64748B",
    "border": "#E2E8F0",
    "survived": "#22C55E",
    "died": "#EF4444",
    "plot_bg": "rgba(255,255,255,0.0)",
    "paper_bg": "rgba(255,255,255,0.0)",
    "font_color": "#1E293B",
    "grid": "#E2E8F0",
}
DARK = {
    "bg": "#0F172A",
    "card": "#1E293B",
    "primary": "#818CF8",
    "secondary": "#4ADE80",
    "accent": "#38BDF8",
    "text": "#F1F5F9",
    "text2": "#CBD5E1",
    "border": "#334155",
    "survived": "#4ADE80",
    "died": "#F87171",
    "plot_bg": "rgba(15,23,42,0.0)",
    "paper_bg": "rgba(15,23,42,0.0)",
    "font_color": "#F1F5F9",
    "grid": "#334155",
}


def get_theme(dark: bool):
    return DARK if dark else LIGHT


def apply_plotly_theme(fig, dark: bool):
    c = get_theme(dark)
    fig.update_layout(
        plot_bgcolor=c["plot_bg"],
        paper_bgcolor=c["paper_bg"],
        font_color=c["font_color"],
        xaxis=dict(gridcolor=c["grid"], linecolor=c["border"]),
        yaxis=dict(gridcolor=c["grid"], linecolor=c["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─── Charts ────────────────────────────────────────────────────────────────────

def survival_by_gender_chart(df: pd.DataFrame, dark: bool):
    c = get_theme(dark)
    grp = df.groupby(['Sex', 'Survived']).size().reset_index(name='count')
    grp['label'] = grp['Survived'].map({0: 'Died', 1: 'Survived'})
    fig = px.bar(
        grp, x='Sex', y='count', color='label',
        color_discrete_map={'Survived': c['survived'], 'Died': c['died']},
        barmode='group', title='Survival by Gender',
        labels={'count': 'Passengers', 'Sex': 'Gender'}
    )
    return apply_plotly_theme(fig, dark)


def survival_by_class_chart(df: pd.DataFrame, dark: bool):
    c = get_theme(dark)
    grp = df.groupby(['Pclass', 'Survived']).size().reset_index(name='count')
    grp['Class'] = grp['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    grp['Status'] = grp['Survived'].map({0: 'Died', 1: 'Survived'})
    fig = px.pie(
        grp, names='Class', values='count', hole=0.4,
        title='Passengers by Class',
        color_discrete_sequence=[c['primary'], c['accent'], c['secondary']]
    )
    return apply_plotly_theme(fig, dark)


def age_distribution_chart(df: pd.DataFrame, dark: bool):
    c = get_theme(dark)
    df2 = df.copy()
    df2['Age'] = pd.to_numeric(df2['Age'], errors='coerce')
    df2 = df2.dropna(subset=['Age'])
    df2['Status'] = df2['Survived'].map({0: 'Died', 1: 'Survived'})
    fig = px.histogram(
        df2, x='Age', color='Status', nbins=20,
        color_discrete_map={'Survived': c['survived'], 'Died': c['died']},
        opacity=0.8, barmode='overlay', title='Age Distribution by Survival',
    )
    return apply_plotly_theme(fig, dark)


def fare_distribution_chart(df: pd.DataFrame, dark: bool):
    c = get_theme(dark)
    df2 = df.copy()
    df2['Fare'] = pd.to_numeric(df2['Fare'], errors='coerce')
    df2['Status'] = df2['Survived'].map({0: 'Died', 1: 'Survived'})
    fig = px.box(df2, x='Status', y='Fare', color='Status',
                 color_discrete_map={'Survived': c['survived'], 'Died': c['died']},
                 title='Fare Distribution by Survival')
    return apply_plotly_theme(fig, dark)


def survival_gauge(probability: float, dark: bool):
    c = get_theme(dark)
    color = c['survived'] if probability >= 50 else c['died']
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        delta={'reference': 50, 'increasing': {'color': c['survived']}, 'decreasing': {'color': c['died']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': c['text2']},
            'bar': {'color': color},
            'bgcolor': c['card'],
            'bordercolor': c['border'],
            'steps': [
                {'range': [0, 40], 'color': '#FEE2E2'},
                {'range': [40, 60], 'color': '#FEF9C3'},
                {'range': [60, 100], 'color': '#DCFCE7'},
            ],
            'threshold': {'line': {'color': c['text'], 'width': 3}, 'value': 50}
        },
        title={'text': "Survival Probability (%)", 'font': {'color': c['text'], 'size': 16}},
        number={'suffix': '%', 'font': {'color': c['text'], 'size': 28}}
    ))
    fig.update_layout(
        paper_bgcolor=c['paper_bg'],
        font_color=c['font_color'],
        height=280,
        margin=dict(l=30, r=30, t=40, b=10)
    )
    return fig


def feature_importance_chart(importances: dict, dark: bool):
    c = get_theme(dark)
    labels = list(importances.keys())
    values = list(importances.values())
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=c['primary'],
        marker_line_color=c['border'], marker_line_width=1
    ))
    fig.update_layout(title="Feature Importance", xaxis_title="Importance")
    return apply_plotly_theme(fig, dark)


def confusion_matrix_chart(cm, dark: bool):
    c = get_theme(dark)
    z = cm if isinstance(cm, list) else cm.tolist()
    labels = ['Died', 'Survived']
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0, c['card']], [1, c['primary']]],
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 18, "color": c['text']},
        showscale=False
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return apply_plotly_theme(fig, dark)


def model_comparison_chart(results: dict, dark: bool):
    c = get_theme(dark)
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    precs = [results[n]['precision'] for n in names]
    recs = [results[n]['recall'] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=names, y=accs, marker_color=c['primary']))
    fig.add_trace(go.Bar(name='Precision', x=names, y=precs, marker_color=c['accent']))
    fig.add_trace(go.Bar(name='Recall', x=names, y=recs, marker_color=c['secondary']))
    fig.update_layout(barmode='group', title='Model Comparison (%)', yaxis_range=[0, 105])
    return apply_plotly_theme(fig, dark)


# ─── Story Generator ───────────────────────────────────────────────────────────

def generate_story(name: str, sex: str, age: float, pclass: int,
                   fare: float, survived: bool, prob: float) -> str:
    gender_word = "woman" if sex.lower() == "female" else "man"
    age_word = "young" if age < 30 else ("middle-aged" if age < 55 else "elderly")
    class_map = {1: "first-class", 2: "second-class", 3: "third-class"}
    class_word = class_map.get(pclass, "third-class")
    result_word = "survived" if survived else "did not survive"
    evac_priority = "high evacuation priority" if (sex.lower() == "female" or pclass == 1) else "lower evacuation priority"

    story = (
        f"**{name}**, a {age_word} {gender_word} travelling in {class_word}, "
        f"had a **{prob:.1f}%** chance of survival. "
        f"As a {gender_word} in {class_word} with {evac_priority}, "
        f"{'access to lifeboats was more likely' if survived else 'access to lifeboats was limited'}. "
        f"Historical records suggest that {name} **{result_word}** the sinking of the Titanic."
    )
    return story


# ─── CSV Download Helper ────────────────────────────────────────────────────────

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


# ─── Email (stub – fill SMTP creds in production) ─────────────────────────────

def send_email(to_email: str, name: str, survived: bool, prob: float) -> bool:
    """Returns True if sent; False otherwise. Configure SMTP_* env vars."""
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    if not smtp_host or not smtp_user:
        return False
    status = "SURVIVED ✅" if survived else "DID NOT SURVIVE ❌"
    body = (
        f"Hello!\n\nYour Titanic survival prediction for '{name}':\n"
        f"Status: {status}\nProbability: {prob:.1f}%\n\n"
        "Generated by Titanic Survival Prediction System."
    )
    msg = MIMEText(body)
    msg['Subject'] = f"Titanic Prediction: {name} – {status}"
    msg['From'] = smtp_user
    msg['To'] = to_email
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        return True
    except Exception:
        return False


# ─── Chatbot ───────────────────────────────────────────────────────────────────

CHATBOT_KB = {
    "survive": (
        "Key factors: **Gender** (women had higher priority), "
        "**Class** (1st class had easier lifeboat access), and **Age** (children were prioritised). "
        "Fare correlates with class and location aboard the ship."
    ),
    "gender": "Women had a significantly higher survival rate (~74%) compared to men (~19%) due to 'women and children first' protocol.",
    "class": "1st class passengers had ~63% survival vs ~47% for 2nd class and ~24% for 3rd class, largely due to deck location.",
    "age": "Children (age < 10) were often given priority. Very elderly passengers had lower survival rates.",
    "fare": "Higher fares often indicated 1st/2nd class, which correlated with better lifeboat access.",
    "lifeboat": "The Titanic carried only 20 lifeboats – enough for ~1,178 people out of 2,224 aboard.",
    "iceberg": "The Titanic struck an iceberg on April 14, 1912 at 11:40 PM ship's time and sank ~2 hr 40 min later.",
    "model": "We use Random Forest, Decision Tree, and Logistic Regression. The best model is auto-selected based on accuracy.",
}

def chatbot_reply(query: str) -> str:
    q = query.lower()
    for key, response in CHATBOT_KB.items():
        if key in q:
            return response
    return (
        "I can answer questions about survival factors, gender, class, age, fare, "
        "lifeboats, the iceberg, or our ML model. Try asking: *What affects survival?*"
    )
