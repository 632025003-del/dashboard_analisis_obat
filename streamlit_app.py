import streamlit as st
import pandas as pd
import plotly.express as px

# CONFIG
st.set_page_config(page_title="MedVision AI", page_icon="💊", layout="wide")

# ====== CUSTOM CSS - FUTURISTIC GLASS STYLE ======
st.markdown("""
<style>

body {
    background: radial-gradient(circle at top, #1a237e, #000000);
}

.header-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.big-title {
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(90deg, #7b2ff7, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    color: #ccddff;
    font-size: 18px;
}

.card {
    padding: 25px;
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.metric-label {
    color: #b7c8ff;
    font-size: 15px;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
    background: linear-gradient(90deg, #00eaff, #db00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.footer {
    color: #9bb0ff;
    margin-top: 30px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="header-box">
    <span class="big-title">MedVision AI – Analisis Obat Generasi Berikutnya</span>
    <p class="sub-title">
    Sistem cerdas dengan tampilan futuristik yang membantu menganalisis data obat,
    efek samping, interaksi, dan pola farmakologi dengan dukungan visual AI modern.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ===== METRIC CARDS =====
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <p class="metric-label">Database Obat</p>
        <p class="metric-value">12.480+</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <p class="metric-label">Interaksi Terdeteksi</p>
        <p class="metric-value">2.300+</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <p class="metric-label">Kategori Terklasifikasi</p>
        <p class="metric-value">67</p>
    </div>
    """, unsafe_allow_html=True)


# ===== UPLOAD DATA =====
st.markdown("## 🧪 Upload Dataset Obat (CSV)")

file = st.file_uploader("Unggah data obat:", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset berhasil dimuat!")
    st.dataframe(df, use_container_width=True)

    st.markdown("## 📊 Visualisasi Interaktif")

    col = st.selectbox("Pilih kolom untuk grafik:", df.columns)

    fig = px.scatter(
        df, x=df.index, y=col,
        title=f"Tren Data: {col}",
        template="plotly_dark",
        color_discrete_sequence=px.colors.cyclical.IceFire
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== FOOTER =====
st.markdown("""
<br><center class="footer">
💊 MedVision AI – Dashboard Obat Futuristik  
<br> Dikembangkan untuk kebutuhan penelitian & analisis farmasi modern
</center>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="AI Analisis Obat",
    layout="wide"
)

st.title("💊 AI Analisis Obat — Dashboard")

st.markdown("""
Dashboard ini memungkinkan Anda melakukan:
- 📤 Upload dataset obat  
- ⚙️ Preprocessing otomatis  
- 🤖 Training model RandomForest  
- 📊 Evaluasi model (akurasi, confusion matrix, report)  
- 🌟 Visualisasi Feature Importance  
- 🌀 PCA 2D untuk analisis obat  
- 🔮 Prediksi obat baru  

---  
""")

# ───────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────
st.sidebar.header("📥 Upload Dataset")
file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

st.sidebar.header("⚙️ Pengaturan Model")
test_size = st.sidebar.slider("Proporsi Test (%)", 10, 50, 20) / 100
n_estimators = st.sidebar.slider("Jumlah Trees RandomForest", 50, 500, 200)

# ───────────────────────────────────────────────
# Load Data
# ───────────────────────────────────────────────
if file is not None:
    df = pd.read_csv(file)
    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.sidebar.header("🎯 Pilih Target")
    target = st.sidebar.selectbox("Kolom target", df.columns)

    fitur = [c for c in df.columns if c != target]

    # ───────────────────────────────────────────────
    # Split Data
    # ───────────────────────────────────────────────
    X = df[fitur]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ───────────────────────────────────────────────
    # Train Model
    # ───────────────────────────────────────────────
    st.header("🤖 Training Model")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Akurasi Model", f"{acc:.3f}")

    # ───────────────────────────────────────────────
    # Classification Report
    # ───────────────────────────────────────────────
    st.subheader("📄 Classification Report")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report)

    # ───────────────────────────────────────────────
    # Confusion Matrix
    # ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("🧮 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    st.pyplot(fig)

    # ───────────────────────────────────────────────
    # Feature Importance
    # ───────────────────────────────────────────────
    st.subheader("🌟 Feature Importance")
    importance = pd.DataFrame({
        "Fitur": fitur,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig2 = px.bar(importance, x="Importance", y="Fitur", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

    # ───────────────────────────────────────────────
    # PCA 2D Visualisasi
    # ───────────────────────────────────────────────
    st.subheader("🌀 PCA Visualisasi Obat")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaler.transform(df[fitur]))

    pca_df = pd.DataFrame({
        "PC1": pca_res[:, 0],
        "PC2": pca_res[:, 1],
        "Label": df[target]
    })

    fig3 = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="Label",
        title="Visualisasi PCA 2D"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ───────────────────────────────────────────────
    # Prediksi Data Baru
    # ───────────────────────────────────────────────
    st.header("🔮 Prediksi Data Baru")

    with st.expander("Input data baru untuk prediksi"):
        input_dict = {}
        for f in fitur:
            input_dict[f] = st.number_input(f, float(df[f].min()), float(df[f].max()))
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            st.success(f"Prediksi kelas obat: **{pred}**")

else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
