"""
app.py – Titanic Survival Prediction System (Advanced AI Edition)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64, io, os, time

from model import train_all_models, load_model, predict_single, predict_batch, get_feature_importance, FEATURES
from utils import (
    get_theme, apply_plotly_theme,
    survival_by_gender_chart, survival_by_class_chart,
    age_distribution_chart, fare_distribution_chart,
    survival_gauge, feature_importance_chart, confusion_matrix_chart,
    model_comparison_chart, generate_story, dataframe_to_csv_bytes,
    send_email, chatbot_reply
)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic AI Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "dataset.csv"
IMG_PATH  = os.path.join("assets", "titanic.jpg")

# ─── Session State Init ─────────────────────────────────────────────────────────
if "dark_mode"        not in st.session_state: st.session_state.dark_mode = False
if "history"          not in st.session_state: st.session_state.history = []
if "model_results"    not in st.session_state: st.session_state.model_results = None
if "best_model_name"  not in st.session_state: st.session_state.best_model_name = None
if "chat_history"     not in st.session_state: st.session_state.chat_history = []
if "model_obj"        not in st.session_state: st.session_state.model_obj = None
if "scaler_obj"       not in st.session_state: st.session_state.scaler_obj = None


# ─── Theme CSS ─────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    c = get_theme(dark)
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {c['bg']} !important;
        color: {c['text']} !important;
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {c['card']} !important;
        border-right: 1px solid {c['border']};
    }}
    h1, h2, h3, h4 {{ font-family: 'Playfair Display', serif; color: {c['text']}; }}
    .stButton > button {{
        background: linear-gradient(135deg, {c['primary']}, {c['accent']});
        color: #fff !important;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(79,70,229,0.3);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79,70,229,0.45);
    }}
    .metric-card {{
        background: {c['card']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        transition: transform 0.2s;
    }}
    .metric-card:hover {{ transform: translateY(-3px); }}
    .metric-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {c['primary']};
        font-family: 'Playfair Display', serif;
    }}
    .metric-card .label {{ font-size: 0.82rem; color: {c['text2']}; margin-top: 4px; }}
    .story-box {{
        background: {c['card']};
        border-left: 4px solid {c['accent']};
        border-radius: 10px;
        padding: 18px 22px;
        margin: 14px 0;
        color: {c['text']};
    }}
    .survived-badge {{
        display: inline-block;
        background: {c['secondary']};
        color: #fff;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        animation: pop 0.4s ease;
    }}
    .died-badge {{
        display: inline-block;
        background: #EF4444;
        color: #fff;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        animation: pop 0.4s ease;
    }}
    @keyframes pop {{
        0% {{ transform: scale(0.7); opacity: 0; }}
        80% {{ transform: scale(1.08); }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    .section-header {{
        border-bottom: 2px solid {c['border']};
        padding-bottom: 8px;
        margin: 24px 0 16px;
        color: {c['text']};
    }}
    .chat-bubble-user {{
        background: {c['primary']};
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 6px 0;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
    }}
    .chat-bubble-bot {{
        background: {c['card']};
        border: 1px solid {c['border']};
        color: {c['text']};
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 6px 0;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
    }}
    .timeline-step {{
        display: flex;
        gap: 16px;
        margin: 14px 0;
        align-items: flex-start;
    }}
    .timeline-dot {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        border: 2px solid {c['border']};
    }}
    .timeline-content {{
        background: {c['card']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 12px 16px;
        flex: 1;
        color: {c['text']};
    }}
    .timeline-content strong {{ color: {c['primary']}; }}
    [data-testid="stMetricValue"] {{ color: {c['primary']} !important; }}
    </style>
    """, unsafe_allow_html=True)


# ─── Hero Banner ───────────────────────────────────────────────────────────────
def show_hero():
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style="
            position: relative;
            border-radius: 18px;
            overflow: hidden;
            margin-bottom: 28px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.35);
            height: 340px;
        ">
            <img src="data:image/jpeg;base64,{img_b64}"
                 style="width:100%; height:340px; object-fit:cover; object-position:center 40%; filter:brightness(0.7) contrast(1.1);" />
            <div style="
                position: absolute; inset: 0;
                background: linear-gradient(to bottom, rgba(15,23,42,0.25) 0%, rgba(15,23,42,0.78) 100%);
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                text-align: center;
            ">
                <p style="color:#38BDF8; font-size:0.9rem; letter-spacing:3px; text-transform:uppercase; margin:0 0 8px;">AI · Machine Learning · Data Science</p>
                <h1 style="color:#F1F5F9; font-size:2.8rem; margin:0; font-family:'Playfair Display',serif; text-shadow:0 2px 20px rgba(0,0,0,0.6);">
                    🚢 Titanic Survival Predictor
                </h1>
                <p style="color:#CBD5E1; font-size:1.05rem; margin:10px 0 0; max-width:560px; text-shadow:0 1px 8px rgba(0,0,0,0.5);">
                    Advanced AI predicting who would have survived the unsinkable ship
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0F172A,#1E293B);border-radius:18px;padding:60px 20px;text-align:center;margin-bottom:28px;">
            <h1 style="color:#F1F5F9;font-family:'Playfair Display',serif;font-size:2.8rem;margin:0;">🚢 Titanic Survival Predictor</h1>
            <p style="color:#38BDF8;margin:10px 0 0;">Advanced AI · Machine Learning · Real-Time Analytics</p>
        </div>
        """, unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🚢 Navigation")
        st.markdown("---")
        page = st.radio("", ["🏠 Dashboard", "🔮 Prediction", "📈 Model Insights",
                              "📂 Batch Predict", "🧭 Timeline", "🤖 Chatbot"], label_visibility="collapsed")
        st.markdown("---")
        dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()
        st.markdown("---")
        if st.button("🔄 Train / Retrain Models"):
            with st.spinner("Training all models..."):
                results, best, scaler = train_all_models(DATA_PATH)
                model, sc = load_model()
                st.session_state.model_results   = results
                st.session_state.best_model_name = best
                st.session_state.model_obj        = model
                st.session_state.scaler_obj       = sc
            st.success(f"✅ Best: **{best}**")
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        if st.session_state.history:
            for h in st.session_state.history[-5:][::-1]:
                icon = "✅" if h['survived'] else "❌"
                st.markdown(f"{icon} **{h['name']}** — {h['prob']:.1f}%")
        else:
            st.caption("No predictions yet.")
    return page


# ─── Dashboard Tab ─────────────────────────────────────────────────────────────
def page_dashboard(df, dark):
    show_hero()
    c = get_theme(dark)

    # KPI metrics
    total = len(df)
    survived_n = df['Survived'].sum()
    survival_rate = survived_n / total * 100
    avg_age = pd.to_numeric(df['Age'], errors='coerce').mean()

    cols = st.columns(4)
    kpis = [
        ("🧑‍🤝‍🧑", f"{total}", "Total Passengers"),
        ("✅", f"{int(survived_n)}", "Survived"),
        ("📊", f"{survival_rate:.1f}%", "Survival Rate"),
        ("🎂", f"{avg_age:.1f}", "Avg Age"),
    ]
    for col, (icon, val, label) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.8rem">{icon}</div>
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(survival_by_gender_chart(df, dark), use_container_width=True)
    with r1c2:
        st.plotly_chart(survival_by_class_chart(df, dark), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(age_distribution_chart(df, dark), use_container_width=True)
    with r2c2:
        st.plotly_chart(fare_distribution_chart(df, dark), use_container_width=True)


# ─── Prediction Tab ─────────────────────────────────────────────────────────────
def page_prediction(dark):
    c = get_theme(dark)
    st.markdown("## 🔮 Survival Prediction")

    # ── Pre-filled example passengers ──────────────────────────────
    examples = {
        "Custom Input": None,
        "👩 Rose (1st Class, Female)": dict(name="Rose DeWitt Bukater", age=17.0, sex="Female", pclass=1, fare=85.0, sibsp=0, parch=1),
        "👨 Jack (3rd Class, Male)":   dict(name="Jack Dawson",         age=20.0, sex="Male",   pclass=3, fare=7.25, sibsp=0, parch=0),
        "👶 Child (2nd Class)":         dict(name="Baby Smith",          age=4.0,  sex="Female", pclass=2, fare=30.0, sibsp=0, parch=2),
    }
    preset = st.selectbox("🎭 Load Example Passenger", list(examples.keys()))
    ex = examples[preset]

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name   = st.text_input("👤 Passenger Name", value=ex['name'] if ex else "John Doe")
            age    = st.slider("🎂 Age", 1, 80, int(ex['age']) if ex else 30)
            sex    = st.radio("⚧ Gender", ["Male", "Female"], index=(0 if (ex and ex['sex']=="Male") else 1) if ex else 0, horizontal=True)
        with col2:
            pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3], index=(ex['pclass']-1) if ex else 2)
            fare   = st.number_input("💰 Fare (£)", 0.0, 600.0, float(ex['fare']) if ex else 14.0, step=0.5)
        with col3:
            sibsp  = st.number_input("👫 Siblings/Spouse aboard", 0, 8, int(ex['sibsp']) if ex else 0)
            parch  = st.number_input("👶 Parents/Children aboard", 0, 6, int(ex['parch']) if ex else 0)
            email  = st.text_input("📧 Email (optional)")
        submitted = st.form_submit_button("🚀 Predict Survival")

    if submitted:
        with st.spinner("Analysing passenger profile..."):
            time.sleep(0.4)
            if st.session_state.model_obj is None:
                model, scaler = load_model()
                st.session_state.model_obj   = model
                st.session_state.scaler_obj  = scaler
            survived, prob = predict_single(
                pclass, sex, age, sibsp, parch, fare,
                st.session_state.model_obj, st.session_state.scaler_obj
            )

        # Result display
        badge_class = "survived-badge" if survived else "died-badge"
        badge_text  = "✅ SURVIVED" if survived else "❌ DID NOT SURVIVE"
        st.markdown(f'<div style="text-align:center;margin:20px 0;"><span class="{badge_class}">{badge_text}</span></div>', unsafe_allow_html=True)

        lc, rc = st.columns([1, 1])
        with lc:
            st.plotly_chart(survival_gauge(prob, dark), use_container_width=True)
        with rc:
            st.markdown(f"### 📖 Passenger Story")
            story = generate_story(name, sex, age, pclass, fare, survived, prob)
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

        # SHAP-style feature contribution (manual for hackathon speed)
        st.markdown("### 🔍 Feature Contribution")
        feat_vals  = [pclass, (1 if sex.lower()=="female" else 0), age, sibsp, parch, fare]
        feat_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        # Simple scaled contribution heuristic
        weights = [-0.35, 0.45, -0.05, -0.05, 0.02, 0.12]
        contribs = [w * v for w, v in zip(weights, feat_vals)]
        colors = [get_theme(dark)['secondary'] if v > 0 else '#EF4444' for v in contribs]
        fig = go.Figure(go.Bar(x=contribs, y=feat_names, orientation='h',
                               marker_color=colors))
        fig.update_layout(title="Why this prediction? (Feature Contributions)",
                          xaxis_title="Contribution Score")
        st.plotly_chart(apply_plotly_theme(fig, dark), use_container_width=True)

        # Save history
        st.session_state.history.append({"name": name, "survived": survived, "prob": prob})

        # Email
        if email:
            ok = send_email(email, name, survived, prob)
            if ok:
                st.success(f"📧 Result sent to {email}")
            else:
                st.info("📧 Email sending requires SMTP configuration (SMTP_HOST, SMTP_USER, SMTP_PASS env vars).")


# ─── Model Insights Tab ─────────────────────────────────────────────────────────
def page_insights(dark):
    st.markdown("## 📈 Model Insights & Comparison")

    if st.session_state.model_results is None:
        st.info("👆 Click **Train / Retrain Models** in the sidebar first.")
        return

    results = st.session_state.model_results
    best    = st.session_state.best_model_name

    st.success(f"🏆 Best Model: **{best}** — Accuracy: **{results[best]['accuracy']}%**")
    st.plotly_chart(model_comparison_chart(results, dark), use_container_width=True)

    sel = st.selectbox("View details for:", list(results.keys()))
    r   = results[sel]

    mc1, mc2, mc3 = st.columns(3)
    for col, (k, icon) in zip([mc1,mc2,mc3], [("accuracy","📊"),("precision","🎯"),("recall","📡")]):
        col.markdown(f"""<div class="metric-card"><div style="font-size:1.8rem">{icon}</div>
        <div class="value">{r[k]}%</div><div class="label">{k.capitalize()}</div></div>""",
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        imp = get_feature_importance(r['model'])
        st.plotly_chart(feature_importance_chart(imp, dark), use_container_width=True)
    with ic2:
        st.plotly_chart(confusion_matrix_chart(r['confusion_matrix'], dark), use_container_width=True)


# ─── Batch Prediction Tab ───────────────────────────────────────────────────────
def page_batch(dark):
    st.markdown("## 📂 Batch Prediction")
    st.markdown("Upload a CSV with columns: `Pclass, Sex, Age, SibSp, Parch, Fare` (Name optional).")

    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.dataframe(df_up.head(), use_container_width=True)
        if st.button("⚡ Run Batch Prediction"):
            with st.spinner("Predicting..."):
                if st.session_state.model_obj is None:
                    model, scaler = load_model()
                    st.session_state.model_obj  = model
                    st.session_state.scaler_obj = scaler
                result_df = predict_batch(df_up, st.session_state.model_obj, st.session_state.scaler_obj)
            st.success(f"✅ Predicted {len(result_df)} passengers!")
            st.dataframe(result_df, use_container_width=True)
            csv_bytes = dataframe_to_csv_bytes(result_df)
            st.download_button("📥 Download Predictions CSV", csv_bytes, "titanic_predictions.csv", "text/csv")


# ─── Timeline Tab ───────────────────────────────────────────────────────────────
def page_timeline(dark):
    c = get_theme(dark)
    st.markdown("## 🧭 Titanic Story Timeline")

    events = [
        ("🏗️", "#4F46E5", "Construction Complete", "May 31, 1911",
         "RMS Titanic was launched from the Harland & Wolff shipyard in Belfast, Northern Ireland."),
        ("⚓", "#22C55E", "Maiden Voyage Begins", "April 10, 1912 — Southampton",
         "The Titanic set sail with 2,224 passengers and crew, bound for New York City."),
        ("🌊", "#38BDF8", "Crossing the North Atlantic", "April 11–14, 1912",
         "The ship travelled at near full speed (~22 knots) despite iceberg warnings in the area."),
        ("🧊", "#EF4444", "Iceberg Strike", "April 14, 1912 – 11:40 PM",
         "The Titanic struck an iceberg on its starboard side. Five compartments were breached."),
        ("🆘", "#F59E0B", "SOS & Evacuation", "April 15, 1912 – 12:05 AM",
         "Evacuation began. 'Women and children first' protocol was enforced. Only 20 lifeboats available."),
        ("🌊", "#1E293B", "Titanic Sinks", "April 15, 1912 – 2:20 AM",
         "The Titanic broke apart and sank 3.8 km to the ocean floor. 1,517 people perished."),
        ("🚢", "#22C55E", "RMS Carpathia Rescue", "April 15, 1912 – 4:10 AM",
         "RMS Carpathia arrived and rescued 710 survivors, arriving in New York on April 18."),
    ]

    for icon, color, title, time_str, desc in events:
        st.markdown(f"""
        <div class="timeline-step">
            <div class="timeline-dot" style="background:{color}22;border-color:{color};">
                <span>{icon}</span>
            </div>
            <div class="timeline-content">
                <strong>{title}</strong>
                <div style="font-size:0.78rem;color:{c['accent']};margin:2px 0;">{time_str}</div>
                <div style="font-size:0.9rem;color:{c['text2']};margin-top:4px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Deck visualization (text-based since no ship_layout.png)
    st.markdown("---")
    st.markdown("### 🛳️ Titanic Deck Overview")
    deck_data = {
        "Deck": ["Boat Deck (Top)", "A Deck", "B Deck", "C Deck", "D Deck", "E Deck", "F & G Deck (Bottom)"],
        "Class": ["Officers / Lifeboats", "1st Class", "1st Class", "1st / 2nd Class", "2nd Class", "2nd / 3rd Class", "3rd Class / Crew"],
        "Survival %": ["N/A", "~63%", "~63%", "~55%", "~47%", "~30%", "~24%"],
        "Access to Lifeboats": ["✅ Direct", "✅ Easy", "✅ Easy", "⚠️ Moderate", "⚠️ Moderate", "❌ Difficult", "❌ Very Difficult"],
    }
    st.dataframe(pd.DataFrame(deck_data), use_container_width=True)


# ─── Chatbot Tab ────────────────────────────────────────────────────────────────
def page_chatbot(dark):
    c = get_theme(dark)
    st.markdown("## 🤖 Titanic AI Assistant")
    st.caption("Ask me anything about Titanic survival, the ship, or our ML model!")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", placeholder="What affects survival?")
        send = st.form_submit_button("Send 💬")

    if send and user_input.strip():
        reply = chatbot_reply(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", reply))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div style="text-align:right;"><span class="chat-bubble-user">{msg}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div><span class="chat-bubble-bot">{msg}</span></div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        suggestions = ["What affects survival?", "Why did gender matter?", "Tell me about lifeboats", "How does the ML model work?"]
        st.markdown("**💡 Try asking:**")
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}"):
                    reply = chatbot_reply(s)
                    st.session_state.chat_history.append(("user", s))
                    st.session_state.chat_history.append(("bot", reply))
                    st.rerun()


# ─── Passenger Comparison Tool ──────────────────────────────────────────────────
def comparison_tool(dark):
    st.markdown("---")
    st.markdown("### 👥 Passenger Comparison Tool")
    passengers = [
        {"name": "Rose (1st, F, 17)", "pclass": 1, "sex": "Female", "age": 17, "sibsp": 0, "parch": 1, "fare": 85},
        {"name": "Jack (3rd, M, 20)", "pclass": 3, "sex": "Male",   "age": 20, "sibsp": 0, "parch": 0, "fare": 7.25},
        {"name": "Elder (2nd, M, 65)","pclass": 2, "sex": "Male",   "age": 65, "sibsp": 1, "parch": 0, "fare": 26},
    ]
    if st.session_state.model_obj is None:
        model, scaler = load_model()
        st.session_state.model_obj  = model
        st.session_state.scaler_obj = scaler

    rows = []
    for p in passengers:
        _, prob = predict_single(p['pclass'], p['sex'], p['age'], p['sibsp'], p['parch'], p['fare'],
                                 st.session_state.model_obj, st.session_state.scaler_obj)
        rows.append({"Passenger": p['name'], "Class": p['pclass'], "Gender": p['sex'],
                     "Age": p['age'], "Fare": p['fare'], "Survival %": prob})

    df_cmp = pd.DataFrame(rows)
    c = get_theme(dark)
    fig = go.Figure(go.Bar(
        x=df_cmp['Passenger'], y=df_cmp['Survival %'],
        marker_color=[c['survived'] if v >= 50 else c['died'] for v in df_cmp['Survival %']],
        text=df_cmp['Survival %'].apply(lambda x: f"{x}%"),
        textposition='outside'
    ))
    fig.update_layout(title="Survival % Comparison", yaxis_range=[0, 115])
    st.plotly_chart(apply_plotly_theme(fig, dark), use_container_width=True)
    st.dataframe(df_cmp, use_container_width=True)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    dark = st.session_state.dark_mode
    inject_css(dark)

    # Auto-train on first run
    if st.session_state.model_results is None and os.path.exists(DATA_PATH):
        with st.spinner("🚀 Initialising AI models..."):
            results, best, scaler = train_all_models(DATA_PATH)
            model, sc = load_model()
            st.session_state.model_results   = results
            st.session_state.best_model_name = best
            st.session_state.model_obj        = model
            st.session_state.scaler_obj       = sc

    page = sidebar()

    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

    if   "Dashboard" in page:   page_dashboard(df, dark)
    elif "Prediction" in page:
        page_prediction(dark)
        comparison_tool(dark)
    elif "Insights"  in page:   page_insights(dark)
    elif "Batch"     in page:   page_batch(dark)
    elif "Timeline"  in page:   page_timeline(dark)
    elif "Chatbot"   in page:   page_chatbot(dark)


if __name__ == "__main__":
    main()
