# 🚢 Titanic Survival Prediction System — Advanced AI Edition

> A production-grade, visually stunning ML web application built with Python & Streamlit.

---

## ✨ Features

| Category | Details |
|---|---|
| 🔮 Prediction | Real-time survival prediction with probability gauge |
| 📊 Dashboard | Interactive Plotly charts — survival by gender, class, age, fare |
| 🧠 Models | Logistic Regression, Decision Tree, Random Forest (auto-selects best) |
| 📈 Insights | Feature importance, confusion matrix, accuracy/precision/recall |
| 🔍 Explainability | Feature contribution chart (why this prediction was made) |
| 📂 Batch Predict | Upload CSV, predict all rows, download results |
| 🧭 Timeline | Interactive Titanic story timeline |
| 🤖 Chatbot | Rule-based AI assistant for survival Q&A |
| 👥 Comparison | Side-by-side comparison of example passengers |
| 📖 Story | Human-readable narrative for each prediction |
| 🌙 Dark/Light Mode | Full theme toggle with adaptive charts |
| 💾 History | Session-based prediction history in sidebar |

---

## 🚀 Setup & Running

### 1. Clone / Download
```bash
git clone <repo-url>
cd titanic_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
titanic_project/
├── app.py              # Main Streamlit application
├── model.py            # ML training, inference, persistence
├── utils.py            # Charts, story generator, chatbot, email
├── dataset.csv         # Titanic passenger data (300 rows)
├── requirements.txt
├── README.md
└── assets/
    └── titanic.jpg     # Hero dashboard image
```

---

## ⚙️ Configuration

### Email Feature
Set these environment variables to enable email sending:
```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your@email.com
export SMTP_PASS=your_app_password
```

---

## 📸 Screenshots

| Dashboard | Prediction | Model Insights |
|---|---|---|
| *(Hero image + KPI cards + 4 charts)* | *(Form + Gauge + Story + Feature contributions)* | *(Model comparison + confusion matrix)* |

---

## 🧠 Models Used

- **Logistic Regression** — Scaled features, max_iter=500
- **Decision Tree** — max_depth=5 to prevent overfitting
- **Random Forest** — 100 estimators, robust ensemble

The app auto-selects and saves the best performing model.

---

## 📊 Dataset

300 Titanic passengers with features: `Pclass, Sex, Age, SibSp, Parch, Fare, Survived`

---

## 🎨 UI Theme

- **Light Mode**: Soft indigo + sky blue + pastel whites (#F8FAFC background)
- **Dark Mode**: Deep navy (#0F172A) with lighter indigo accents
- **Fonts**: Playfair Display (headings) + Inter (body)

---

*Built for hackathons, portfolios, and learning. Made with ❤️ and Python.*
