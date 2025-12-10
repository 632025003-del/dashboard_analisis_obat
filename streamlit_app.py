import streamlit as st
import pandas as pd
import plotly.express as px

# ======================
#   CONFIG THEME
# ======================
st.set_page_config(
    page_title="MedAI – Analisis Obat Cerdas",
    page_icon="🧪",
    layout="wide"
)

# ======================
#   CUSTOM HEADER STYLE
# ======================
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg, #4CB8C4, #3CD3AD);
    padding: 30px;
    border-radius: 12px;
    color: white;
}
.title {
    font-size: 38px;
    font-weight: 900;
}
.subtitle {
    font-size: 18px;
    opacity: 0.9;
}
.card {
    padding: 20px;
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-title {
    font-size: 16px;
    color: #666;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #2C3A47;
}
</style>
""", unsafe_allow_html=True)

# ======================
#   HEADER SECTION
# ======================
st.markdown("""
<div class="header">
    <div class="title">MedAI – Dashboard Analisis Obat</div>
    <div class="subtitle">
        Sistem analitik cerdas untuk memahami karakteristik obat,
        tren penggunaan, serta efektivitas dan risiko berdasarkan data farmasi.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")  # spacing

# ======================
#   FEATURES SUMMARY
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Total Obat Dianalisis</div>
        <div class="metric-value">8,240</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Kategori Farmakologi</div>
        <div class="metric-value">52</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Interaksi Potensial</div>
        <div class="metric-value">1,420+</div>
    </div>
    """, unsafe_allow_html=True)


# ======================
#   DATA UPLOAD SECTION
# ======================
st.markdown("## 📥 Upload Dataset Obat")

file = st.file_uploader("Upload data obat (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset berhasil diunggah!")
    st.dataframe(df, use_container_width=True)

    # Grafik distribusi kolom
    st.markdown("### 📊 Explorasi Cepat")

    col = st.selectbox("Pilih kolom untuk visualisasi:", df.columns)

    fig = px.histogram(df, x=col, title=f"Distribusi: {col}", opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

# ======================
#   FOOTER
# ======================
st.markdown("""
<hr>
<center>
💊 <i>MedAI – Sistem Analisis Obat Cerdas</i>  
<br> Dibuat untuk mempermudah pemahaman farmakologi & data obat
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
