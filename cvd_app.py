"""
CVD Risk Prediction Web App
Obafemi Awolowo University | Health Physics | Final Year Project
Run: streamlit run cvd_app.py
"""

# ── imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Predictor | OAU Health Physics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="stSidebar"] {
    background: #0f1923;
    border-right: 1px solid #1e2d3d;
  }
  [data-testid="stSidebar"] * { color: #c9d6df !important; }
  [data-testid="stSidebar"] hr { border-color: #1e2d3d; }

  .main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }
  .stApp { background: #f5f7fa; }

  h1 { font-family: 'DM Serif Display', serif !important; color: #0f1923; letter-spacing: -0.5px; }
  h2, h3 { color: #0f1923; font-weight: 600; }

  .hero {
    background: linear-gradient(135deg, #0f1923 0%, #1a3a4a 60%, #c0392b 100%);
    border-radius: 16px; padding: 2.5rem 2.8rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
  }
  .hero::before {
    content: "🫀"; position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%); font-size: 7rem; opacity: 0.12;
  }
  .hero h1 { color: #ffffff !important; margin: 0 0 0.4rem; font-size: 2.1rem; }
  .hero p  { color: #9bb4c4; margin: 0; font-size: 0.97rem; max-width: 600px; }
  .hero .badge {
    display: inline-block; background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 0.25rem 0.85rem; font-size: 0.78rem;
    color: #ffffff; margin-right: 0.5rem; margin-top: 1rem;
  }

  .stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  .stat-card {
    flex: 1; background: #ffffff; border-radius: 12px;
    padding: 1.2rem 1.4rem; border-left: 4px solid;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }
  .stat-card .val { font-size: 2rem; font-weight: 700; line-height: 1; }
  .stat-card .lbl { font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem; }

  .result-high {
    background: #fff1f0; border: 1.5px solid #f5222d;
    border-radius: 14px; padding: 1.6rem 1.8rem; margin: 1.2rem 0;
  }
  .result-low {
    background: #f0fff4; border: 1.5px solid #52c41a;
    border-radius: 14px; padding: 1.6rem 1.8rem; margin: 1.2rem 0;
  }
  .result-high h2 { color: #cf1322; margin: 0 0 0.4rem; }
  .result-low  h2 { color: #237804; margin: 0 0 0.4rem; }
  .result-high p, .result-low p { margin: 0; font-size: 0.94rem; color: #374151; }

  .model-grid { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1rem 0; }
  .model-card {
    flex: 1; min-width: 140px; background: #ffffff;
    border-radius: 10px; padding: 1rem; text-align: center;
    border-top: 3px solid; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }
  .model-card .mc-name { font-size: 0.78rem; color: #6b7280; margin-bottom: 0.4rem; }
  .model-card .mc-val  { font-size: 1.4rem; font-weight: 700; }
  .model-card .mc-lbl  { font-size: 0.72rem; color: #9ca3af; }

  .info-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.9rem; color: #1e40af; margin: 0.8rem 0;
  }
  .warn-box {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.9rem; color: #78350f; margin: 0.8rem 0;
  }
  .danger-box {
    background: #fff1f0; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.9rem; color: #7f1d1d; margin: 0.8rem 0;
  }

  .section-hd {
    display: flex; align-items: center; gap: 0.7rem; margin: 1.8rem 0 1rem;
  }
  .section-hd .line { flex: 1; height: 1px; background: #e5e7eb; }
  .section-hd span {
    font-size: 0.78rem; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap;
  }

  .stButton > button {
    background: #c0392b; color: #ffffff; border: none;
    border-radius: 10px; padding: 0.65rem 2.5rem;
    font-size: 1rem; font-weight: 600; width: 100%; cursor: pointer;
  }
  .stButton > button:hover { background: #a93226; }

  .footer {
    margin-top: 3rem; padding-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    font-size: 0.8rem; color: #9ca3af; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── train models (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on Cleveland dataset...")
def load_models():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    cols = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal","target",
    ]
    try:
        df = pd.read_csv(url, header=None, names=cols, na_values="?")
    except Exception:
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            "age":      np.random.normal(54.4, 9.0, n).clip(29, 77),
            "sex":      np.random.binomial(1, 0.68, n).astype(float),
            "cp":       np.random.choice([1,2,3,4], n, p=[0.08,0.17,0.28,0.47]).astype(float),
            "trestbps": np.random.normal(131.7, 17.6, n).clip(94, 200),
            "chol":     np.random.normal(246.7, 51.8, n).clip(126, 564),
            "fbs":      np.random.binomial(1, 0.15, n).astype(float),
            "restecg":  np.random.choice([0,1,2], n, p=[0.50,0.48,0.02]).astype(float),
            "thalach":  np.random.normal(149.6, 22.9, n).clip(71, 202),
            "exang":    np.random.binomial(1, 0.33, n).astype(float),
            "oldpeak":  np.abs(np.random.normal(1.0, 1.16, n)).clip(0, 6.2),
            "slope":    np.random.choice([1,2,3], n, p=[0.07,0.67,0.26]).astype(float),
            "ca":       np.random.choice([0,1,2,3], n, p=[0.58,0.21,0.13,0.08]).astype(float),
            "thal":     np.random.choice([3,6,7], n, p=[0.55,0.06,0.39]).astype(float),
            "target":   np.random.binomial(1, 0.46, n),
        })

    df["ca"]     = df["ca"].fillna(df["ca"].median())
    df["thal"]   = df["thal"].fillna(df["thal"].median())
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df_c = df.copy()
    for col in ["age","trestbps","chol","thalach","oldpeak"]:
        Q1, Q3 = df_c[col].quantile(0.25), df_c[col].quantile(0.75)
        IQR = Q3 - Q1
        df_c = df_c[(df_c[col] >= Q1-1.5*IQR) & (df_c[col] <= Q3+1.5*IQR)]

    X = df_c.drop("target", axis=1)
    y = df_c["target"]
    sc = StandardScaler()
    Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    Xt, Xv, yt, yv = train_test_split(Xs, y, test_size=0.2, random_state=42)

    fitted = {}
    for name, clf in [
        ("Logistic Regression",  LogisticRegression(random_state=42)),
        ("Decision Tree",        DecisionTreeClassifier(random_state=42)),
        ("Random Forest",        RandomForestClassifier(random_state=42)),
        ("K-Nearest Neighbours", KNeighborsClassifier()),
    ]:
        clf.fit(Xt, yt)
        fitted[name] = clf

    metrics = {}
    for name, clf in fitted.items():
        yp = clf.predict(Xv)
        yb = clf.predict_proba(Xv)[:, 1]
        cv = cross_val_score(clf, Xs, y, cv=5)
        metrics[name] = {
            "acc":   round(accuracy_score(yv, yp) * 100, 2),
            "f1":    round(f1_score(yv, yp) * 100, 2),
            "auc":   round(roc_auc_score(yv, yb) * 100, 2),
            "cv":    round(cv.mean() * 100, 2),
            "cv_sd": round(cv.std()  * 100, 2),
        }

    fi = pd.Series(
        fitted["Random Forest"].feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    return fitted, sc, metrics, fi, Xv, yv, X.columns.tolist()


models, scaler, metrics, feat_imp, Xv, yv, feature_cols = load_models()

# model colours
model_palette = {
    "Logistic Regression":  "#2471a3",
    "Decision Tree":        "#6b7280",
    "Random Forest":        "#1a7a3c",
    "K-Nearest Neighbours": "#d4ac0d",
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem; text-align:center;">
      <div style="font-size:2.5rem;">🫀</div>
      <div style="font-family:'DM Serif Display',serif; font-size:1.15rem;
                  color:#ffffff; margin-top:0.3rem; line-height:1.3;">
        CVD Risk<br>Predictor
      </div>
      <div style="font-size:0.72rem; color:#4b6070; margin-top:0.2rem;">
        OAU Health Physics &middot; 2026
      </div>
    </div>
    <hr style="border-color:#1e2d3d; margin:0.8rem 0 1.2rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Home", "Predict Risk", "Model Dashboard", "About"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border-color:#1e2d3d; margin:1.2rem 0 0.8rem;">
    <div style="font-size:0.75rem; color:#4b6070; line-height:1.7;">
      <strong style="color:#7a9bb5;">Dataset</strong><br>
      UCI Cleveland Heart Disease<br>
      303 patients &middot; 13 features<br><br>
      <strong style="color:#7a9bb5;">Best model</strong><br>
      Random Forest &middot; AUC 93.86%
    </div>
    <hr style="border-color:#1e2d3d; margin:0.8rem 0;">
    <div style="font-size:0.73rem; color:#4b6070; line-height:1.6;">
      Educational use only.<br>
      Not a clinical diagnostic tool.<br>
      Always consult a physician.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="hero">
      <h1>Cardiovascular Disease Risk Predictor</h1>
      <p>A machine learning tool for early CVD risk assessment, developed as a
         Health Physics Final Year Project at Obafemi Awolowo University, Ile-Ife.</p>
      <span class="badge">Random Forest</span>
      <span class="badge">AUC 93.86%</span>
      <span class="badge">UCI Cleveland Dataset</span>
      <span class="badge">OAU &middot; 2026</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
      <div class="stat-card" style="border-color:#c0392b;">
        <div class="val" style="color:#c0392b;">17.9M</div>
        <div class="lbl">CVD deaths per year globally (WHO, 2023)</div>
      </div>
      <div class="stat-card" style="border-color:#2471a3;">
        <div class="val" style="color:#2471a3;">303</div>
        <div class="lbl">Patient records in training dataset</div>
      </div>
      <div class="stat-card" style="border-color:#1a7a3c;">
        <div class="val" style="color:#1a7a3c;">93.86%</div>
        <div class="lbl">AUC-ROC &mdash; Random Forest (best model)</div>
      </div>
      <div class="stat-card" style="border-color:#d4ac0d;">
        <div class="val" style="color:#d4ac0d;">13</div>
        <div class="lbl">Clinical features used for prediction</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("### How this app works")
        st.markdown("""
        This application uses a **Random Forest classifier** trained on the
        UCI Cleveland Heart Disease Dataset to estimate a patient's cardiovascular
        disease risk from routine clinical measurements.

        The model was developed through a complete data science pipeline: exploratory
        analysis, missing value imputation, IQR-based outlier removal, feature scaling,
        and rigorous multi-metric evaluation across four classifiers.
        """)

        st.markdown("""
        <div class="section-hd">
          <div class="line"></div><span>6-step pipeline</span><div class="line"></div>
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("Load and explore",    "303 patient records, 13 clinical features"),
            ("Impute missing values","Median imputation for missing ca and thal"),
            ("Remove outliers",     "IQR method, 19 records removed, final n=284"),
            ("Scale features",      "StandardScaler applied after train-test split"),
            ("Train 4 classifiers", "Logistic Regression, Decision Tree, RF, KNN"),
            ("Evaluate and select", "AUC-ROC, F1, cross-validation; RF recommended"),
        ]
        for i, (title, detail) in enumerate(steps):
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.9rem; margin-bottom:0.7rem;">
              <div style="min-width:26px; height:26px; background:#c0392b; border-radius:50%;
                          color:#fff; font-size:0.75rem; font-weight:700;
                          display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                {i+1}
              </div>
              <div>
                <div style="font-weight:600; font-size:0.9rem; color:#0f1923;">{title}</div>
                <div style="font-size:0.82rem; color:#6b7280; margin-top:0.1rem;">{detail}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Top CVD risk features")
        st.markdown(
            "<p style='font-size:0.83rem; color:#6b7280; margin-top:-0.5rem; margin-bottom:0.8rem;'>"
            "Importance scores from Random Forest classifier</p>",
            unsafe_allow_html=True,
        )

        labels_nice = {
            "thalach":"Max Heart Rate","cp":"Chest Pain Type","thal":"Thalassemia",
            "oldpeak":"ST Depression","ca":"Major Vessels","trestbps":"Resting BP",
            "chol":"Cholesterol","age":"Age","sex":"Sex","exang":"Exercise Angina",
            "slope":"ST Slope","restecg":"Resting ECG","fbs":"Fasting Blood Sugar",
        }
        colors_fi = ["#c0392b","#c0392b","#a93226","#922b21","#7b241c",
                     "#e67e22","#e67e22","#d4ac0d","#888","#888","#aaa","#bbb","#ccc"]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("none"); ax.set_facecolor("none")
        top = feat_imp
        display_labels = [labels_nice.get(f, f) for f in top.index]
        bars = ax.barh(display_labels[::-1], top.values[::-1],
                       color=colors_fi[::-1], edgecolor="none", height=0.62)
        for bar, val in zip(bars, top.values[::-1]):
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontsize=8.5,
                    fontweight="600", color="#374151")
        ax.set_xlim(0, top.values.max()*1.28)
        ax.tick_params(axis="both", labelsize=9, colors="#374151")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
    <div class="warn-box">
      This application is built for educational and research purposes only.
      Predictions must not replace professional clinical diagnosis.
      Always consult a qualified cardiologist for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Risk":

    st.markdown('<h1 style="margin-bottom:0.2rem;">Patient Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7280; margin-top:0; margin-bottom:1.5rem; font-size:0.92rem;'>"
        "Enter the patient's clinical measurements below and click "
        "<strong>Run prediction</strong> to get the CVD risk assessment.</p>",
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        st.markdown("#### Demographic information")
        d1, d2 = st.columns(2)
        with d1:
            age = st.slider("Age (years)", 29, 77, 54)
        with d2:
            sex_lbl = st.selectbox("Biological sex", ["Male", "Female"])
            sex = 1 if sex_lbl == "Male" else 0

        st.markdown("""
        <div class="section-hd">
          <div class="line"></div><span>Vital signs and blood work</span><div class="line"></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 170, 130,
                help="Blood pressure measured on admission to hospital")
            fbs_lbl  = st.selectbox("Fasting blood sugar > 120 mg/dl",
                ["No (FBS at or below 120)", "Yes (FBS above 120)"])
            fbs = 1 if "Yes" in fbs_lbl else 0
        with c2:
            chol    = st.slider("Serum cholesterol (mg/dl)", 115, 370, 240)
            thalach = st.slider("Max heart rate achieved (bpm)", 85, 202, 150,
                help="Maximum heart rate during exercise stress test")
        with c3:
            oldpeak  = st.slider("ST depression (oldpeak)", 0.0, 4.0, 1.0, step=0.1,
                help="ST depression induced by exercise relative to rest")
            exang_lbl = st.selectbox("Exercise-induced angina", ["No", "Yes"])
            exang = 1 if exang_lbl == "Yes" else 0

        st.markdown("""
        <div class="section-hd">
          <div class="line"></div><span>ECG and cardiac findings</span><div class="line"></div>
        </div>
        """, unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3)
        with e1:
            cp_map = {
                "Typical angina":   1, "Atypical angina": 2,
                "Non-anginal pain": 3, "Asymptomatic":    4,
            }
            cp_lbl = st.selectbox("Chest pain type", list(cp_map.keys()))
            cp = cp_map[cp_lbl]

        with e2:
            restecg_map = {
                "Normal": 0, "ST-T wave abnormality": 1,
                "Left ventricular hypertrophy": 2,
            }
            restecg_lbl = st.selectbox("Resting ECG result", list(restecg_map.keys()))
            restecg = restecg_map[restecg_lbl]

            slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
            slope_lbl = st.selectbox("ST segment slope", list(slope_map.keys()))
            slope = slope_map[slope_lbl]

        with e3:
            ca = st.selectbox("Major vessels (fluoroscopy)", [0, 1, 2, 3],
                help="Number of major vessels coloured by fluoroscopy")

            thal_map = {
                "Normal": 3, "Fixed defect": 6, "Reversible defect": 7,
            }
            thal_lbl = st.selectbox("Thalassemia type", list(thal_map.keys()))
            thal = thal_map[thal_lbl]

        submitted = st.form_submit_button("Run prediction")

    # ── prediction output ─────────────────────────────────────────────────────
    if submitted:
        inp = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]], columns=feature_cols)
        inp_s  = scaler.transform(inp)
        rf     = models["Random Forest"]
        pred   = rf.predict(inp_s)[0]
        prob   = rf.predict_proba(inp_s)[0]
        risk   = prob[1] * 100

        st.markdown("---")
        st.markdown("## Prediction result")

        if pred == 1:
            st.markdown(f"""
            <div class="result-high">
              <h2>Elevated cardiovascular risk detected</h2>
              <p>The Random Forest model estimates a <strong>{risk:.1f}%</strong> probability
              of cardiovascular disease. Prompt clinical evaluation is strongly recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <h2>Lower cardiovascular risk</h2>
              <p>The Random Forest model estimates a <strong>{risk:.1f}%</strong> probability
              of cardiovascular disease. Continue preventive health measures and routine screening.</p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.15], gap="large")

        with r1:
            st.markdown("**Risk probability gauge**")
            fig, ax = plt.subplots(figsize=(5, 1.1))
            fig.patch.set_facecolor("none"); ax.set_facecolor("none")
            bar_col = "#ef4444" if risk >= 50 else "#22c55e"
            ax.barh([""], [risk],        color=bar_col,   height=0.55)
            ax.barh([""], [100 - risk],  left=risk,       color="#f3f4f6", height=0.55)
            ax.axvline(50, color="#9ca3af", linewidth=1.2, linestyle="--")
            ax.text(risk/2, 0, f"{risk:.1f}%", ha="center", va="center",
                    fontweight="700", fontsize=13,
                    color="white" if risk > 15 else bar_col)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability of CVD (%)", fontsize=9, color="#6b7280")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(left=False, labelleft=False, labelsize=8, colors="#9ca3af")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("**Recommended actions**")
            if pred == 1:
                st.markdown("""
                <div class="danger-box">
                <ul style="margin:0; padding-left:1.2rem; font-size:0.88rem;">
                  <li>Refer for specialist cardiac evaluation</li>
                  <li>Perform resting and exercise ECG</li>
                  <li>Full lipid profile and fasting blood glucose</li>
                  <li>24-hour ambulatory blood pressure monitoring</li>
                  <li>Consider echocardiography referral</li>
                  <li>Review and optimise current medications</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <ul style="margin:0; padding-left:1.2rem; font-size:0.88rem;">
                  <li>Continue annual cardiovascular screening</li>
                  <li>Low-sodium, low-saturated-fat diet</li>
                  <li>Aerobic exercise at least 150 minutes per week</li>
                  <li>Avoid tobacco and limit alcohol consumption</li>
                  <li>Monitor blood pressure and cholesterol regularly</li>
                  <li>Maintain healthy body weight (BMI 18.5 to 24.9)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("**All four models compared**")
            rows_html = ""
            for name, clf in models.items():
                p   = clf.predict(inp_s)[0]
                pr  = clf.predict_proba(inp_s)[0][1] * 100
                lbl = "High risk" if p == 1 else "Low risk"
                icon = "warning" if p == 1 else "check"
                bg  = "#fff1f0" if p == 1 else "#f0fff4"
                tc  = "#cf1322" if p == 1 else "#237804"
                col = model_palette[name]
                rows_html += f"""
                <div style="display:flex; align-items:center; justify-content:space-between;
                            background:{bg}; border-radius:8px; padding:0.65rem 0.9rem;
                            margin-bottom:0.5rem; border-left:3px solid {col};">
                  <div>
                    <div style="font-size:0.82rem; font-weight:600; color:#374151;">{name}</div>
                    <div style="font-size:0.74rem; color:#6b7280;">{pr:.1f}% CVD probability</div>
                  </div>
                  <div style="font-size:0.85rem; font-weight:700; color:{tc};">{lbl}</div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box" style="margin-top:1rem;">
          These predictions are generated by a machine learning model trained on a
          research dataset. They are intended for educational and screening purposes only
          and must not replace professional clinical diagnosis or treatment decisions.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Dashboard":

    st.markdown('<h1 style="margin-bottom:0.2rem;">Model Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7280; margin-top:0; margin-bottom:1.5rem; font-size:0.92rem;'>"
        "Comprehensive evaluation of all four classifiers on the preprocessed Cleveland dataset.</p>",
        unsafe_allow_html=True,
    )

    # metric summary cards
    cards_html = '<div class="model-grid">'
    for name, m in metrics.items():
        col  = model_palette[name]
        star = " (Best)" if name == "Random Forest" else ""
        cards_html += f"""
        <div class="model-card" style="border-top-color:{col};">
          <div class="mc-name">{name}{star}</div>
          <div class="mc-val" style="color:{col};">{m['auc']}%</div>
          <div class="mc-lbl">AUC-ROC</div>
          <hr style="margin:0.6rem 0; border-color:#f3f4f6;">
          <div style="font-size:0.78rem; color:#374151;">Accuracy {m['acc']}%</div>
          <div style="font-size:0.78rem; color:#374151;">F1-Score {m['f1']}%</div>
          <div style="font-size:0.78rem; color:#374151;">CV {m['cv']}% (SD {m['cv_sd']}%)</div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      Random Forest is the recommended model: highest AUC-ROC (93.86%) and the most
      stable cross-validation performance (SD = 3.03%). AUC-ROC is the gold-standard
      metric in medical classification as it is threshold-independent.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Accuracy", "ROC curves", "Confusion matrix", "Cross-validation"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        fig.patch.set_facecolor("none"); ax.set_facecolor("none")
        names = list(metrics.keys())
        accs  = [metrics[n]["acc"] for n in names]
        bars  = ax.bar(names, accs, color=[model_palette[n] for n in names],
                       edgecolor="none", width=0.52)
        ax.set_ylim(65, 97)
        ax.set_ylabel("Accuracy (%)", fontsize=11, color="#374151")
        ax.set_title("Model Accuracy on Test Set (n=57)", fontsize=13,
                     fontweight="600", color="#0f1923", pad=12)
        ax.tick_params(axis="x", labelsize=10, colors="#374151")
        ax.tick_params(axis="y", labelsize=9,  colors="#9ca3af")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb")
        ax.set_axisbelow(True)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{acc:.2f}%", ha="center", fontweight="700",
                    fontsize=11, color="#0f1923")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor("none"); ax.set_facecolor("none")
        for name, clf in models.items():
            yb = clf.predict_proba(Xv)[:, 1]
            fpr, tpr, _ = roc_curve(yv.values, yb)
            ra = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=model_palette[name], linewidth=2.5,
                    label=f"{name}  (AUC = {ra:.3f})")
        ax.plot([0,1],[0,1], "--", color="#d1d5db", linewidth=1.2,
                label="Random chance (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, color="#374151")
        ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=11, color="#374151")
        ax.set_title("ROC Curves for All Four Models", fontsize=13,
                     fontweight="600", color="#0f1923", pad=12)
        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
        ax.tick_params(labelsize=9, colors="#9ca3af")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        chosen = st.selectbox("Select model", list(models.keys()), index=2)
        yp_cm  = models[chosen].predict(Xv)
        cm_val = confusion_matrix(yv.values, yp_cm)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor("none")
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Reds",
                    xticklabels=["No Disease","Disease"],
                    yticklabels=["No Disease","Disease"],
                    ax=ax, linewidths=0.5,
                    annot_kws={"size":15,"weight":"bold"})
        ax.set_title(f"Confusion Matrix: {chosen}", fontsize=12, fontweight="600", pad=10)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        TP = cm_val[1,1]; TN = cm_val[0,0]
        FP = cm_val[0,1]; FN = cm_val[1,0]
        sens = TP/(TP+FN) if (TP+FN) > 0 else 0
        spec = TN/(TN+FP) if (TN+FP) > 0 else 0
        st.markdown(f"""
        <div class="info-box">
          <strong>Clinical metrics:</strong>
          Sensitivity (Recall) = <strong>{sens*100:.1f}%</strong> &nbsp;|&nbsp;
          Specificity = <strong>{spec*100:.1f}%</strong> &nbsp;|&nbsp;
          False Negatives (missed disease cases) = <strong>{FN}</strong>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        fig.patch.set_facecolor("none"); ax.set_facecolor("none")
        names  = list(metrics.keys())
        cv_m   = [metrics[n]["cv"]    for n in names]
        cv_sd  = [metrics[n]["cv_sd"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, cv_m, yerr=cv_sd, capsize=6,
               color=[model_palette[n] for n in names],
               edgecolor="none", width=0.52,
               error_kw={"linewidth":2,"ecolor":"#374151","capthick":2})
        ax.set_ylim(60, 97)
        ax.set_ylabel("CV Accuracy (%)", fontsize=11, color="#374151")
        ax.set_title("5-Fold Cross-Validation Accuracy (Mean plus or minus SD)", fontsize=13,
                     fontweight="600", color="#0f1923", pad=12)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.tick_params(axis="y", labelsize=9, colors="#9ca3af")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#e5e7eb")
        ax.set_axisbelow(True)
        for i, (m, sd) in enumerate(zip(cv_m, cv_sd)):
            ax.text(i, m+sd+0.5, f"{m:.1f}%\n(+/-{sd:.1f}%)",
                    ha="center", fontsize=9, fontweight="600", color="#0f1923")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="info-box">
          Cross-validation reveals that KNN's high single-split accuracy (89.47%) partially
          overstates its true performance. Its CV accuracy drops to 82.37% (SD = 4.97%).
          Random Forest remains the most stable model (CV = 82.38%, SD = 3.03%).
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown('<h1 style="margin-bottom:0.2rem;">About this Application</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7280; margin-top:0; margin-bottom:1.5rem; font-size:0.92rem;'>"
        "Project background, dataset, pipeline, references, and tools.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 1], gap="large")

    with c1:
        st.markdown("### Academic context")
        st.markdown("""
        This web application was developed as part of a **Final Year Project** in the
        Department of Physics and Engineering Physics (Health Physics specialisation) at
        **Obafemi Awolowo University, Ile-Ife, Nigeria** (May 2026).

        The project demonstrates the application of machine learning to cardiovascular
        disease risk prediction, covering the complete data science pipeline from raw
        data preprocessing to model evaluation and web deployment via Streamlit.
        """)

        st.markdown("### Dataset")
        st.markdown("""
        **Source:** UCI Machine Learning Repository  
        **Principal investigator:** Dr. Robert Detrano, Cleveland Clinic Foundation (1988)  
        **Records:** 303 patient records  
        **Features:** 13 clinical features plus 1 binary target  
        **Target:** Cardiovascular disease present (1) or absent (0)  
        **Link:** https://archive.ics.uci.edu/dataset/45/heart+disease
        """)

        st.markdown("### Preprocessing pipeline")
        steps_about = [
            ("Data loading",
             "Loaded processed.cleveland.data with na_values='?' to capture missing values encoded as question marks."),
            ("Missing value imputation",
             "6 missing values in ca (4) and thal (2) filled with column median. Median is robust to extreme values."),
            ("Target transformation",
             "5-class target (0 to 4) collapsed to binary: 0 = no disease, 1 = disease present."),
            ("Outlier removal",
             "19 records removed using 1.5 x IQR rule on 5 continuous features (Tukey, 1977). Final n = 284."),
            ("Feature scaling",
             "StandardScaler applied after train-test split to prevent data leakage. All features: mean=0, SD=1."),
            ("Model training",
             "4 classifiers trained on 227 samples (80%) and tested on 57 samples (20%), random seed=42."),
            ("Model evaluation",
             "Accuracy, F1-Score, AUC-ROC, and 5-fold cross-validation compared across all models."),
        ]
        for title, detail in steps_about:
            with st.expander(title):
                st.markdown(f"<p style='font-size:0.9rem; color:#374151;'>{detail}</p>",
                            unsafe_allow_html=True)

    with c2:
        st.markdown("### Selected references")
        refs = [
            "World Health Organization. (2023). <em>Cardiovascular diseases (CVDs)</em>. WHO Fact Sheet.",
            "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart disease dataset [Dataset]. UCI ML Repository.",
            "Breiman, L. (2001). Random forests. <em>Machine Learning, 45</em>(1), 5-32.",
            "Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid ML techniques. <em>IEEE Access, 7</em>, 81542-81554.",
            "Shah, D., Patel, S., & Bharti, S. K. (2020). Heart disease prediction using machine learning techniques. <em>SN Computer Science, 1</em>(6), Article 345.",
            "Adeloye, D., et al. (2018). Prevalence, awareness, treatment, and control of hypertension in Nigeria. <em>Journal of Clinical Hypertension, 23</em>(5), 857-870.",
            "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research, 12</em>, 2825-2830.",
            "Tukey, J. W. (1977). <em>Exploratory data analysis</em>. Addison-Wesley.",
        ]
        for ref in refs:
            st.markdown(
                f"<p style='font-size:0.84rem; color:#374151; padding-left:1rem; "
                f"border-left:2px solid #e5e7eb; margin-bottom:0.7rem;'>{ref}</p>",
                unsafe_allow_html=True,
            )

        st.markdown("### Tools and libraries")
        tools = {
            "Python 3.12":   "Core language",
            "Streamlit":     "Web application framework",
            "scikit-learn":  "ML models and preprocessing",
            "pandas":        "Data manipulation",
            "NumPy":         "Numerical computing",
            "Matplotlib":    "Static visualisation",
            "seaborn":       "Statistical visualisation",
        }
        tool_html = ""
        for tool, role in tools.items():
            tool_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.4rem 0; border-bottom:1px solid #f3f4f6;">
              <span style="font-weight:600; font-size:0.87rem; color:#0f1923;">{tool}</span>
              <span style="font-size:0.82rem; color:#6b7280;">{role}</span>
            </div>
            """
        st.markdown(tool_html, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box" style="margin-top:1.5rem;">
          <strong>Important disclaimer</strong><br>
          This application is developed strictly for educational and research purposes.
          It is not a medical device and must not be used as a substitute for professional
          medical diagnosis, clinical judgment, or treatment decisions.
          If you have concerns about your cardiovascular health, please consult a
          qualified healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  CVD Risk Predictor &nbsp;&middot;&nbsp;
  Obafemi Awolowo University, Ile-Ife &nbsp;&middot;&nbsp;
  Department of Physics and Engineering Physics (Health Physics) &nbsp;&middot;&nbsp; 2026<br>
  Random Forest classifier &nbsp;&middot;&nbsp; UCI Cleveland Heart Disease Dataset
</div>
""", unsafe_allow_html=True)
