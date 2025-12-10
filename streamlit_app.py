import streamlit as st

st.set_page_config(page_title="Prediksi Obat", layout="wide")

# =======================
# TAMBAHKAN CSS STYLE
# =======================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #2C3E50;
}
.subtext {
    font-size: 18px;
    color: #555;
}
.card {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
.stat-number {
    font-size: 36px;
    font-weight: 700;
    color: #2C3E50;
}
.stat-label {
    color: #777;
}
.grow-tag {
    background-color: #E8F8F5;
    padding: 5px 10px;
    border-radius: 10px;
    color: #1ABC9C;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =======================
# BAGIAN HEADER
# =======================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="big-title">Selamat Datang di Aplikasi Analisis Obat</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="subtext">
    Aplikasi ini menggunakan <b>Machine Learning</b> untuk menganalisis karakteristik obat,
    efek samping, interaksi obat, serta memberikan prediksi klasifikasi obat otomatis.
    Dengan teknologi AI, analisis obat menjadi lebih cepat dan akurat.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Fitur Utama:**  
    ✅ Analisis Kategori Obat  
    ✅ Visualisasi Data Interaktif  
    ✅ Rekomendasi AI  
    ✅ Prediksi Otomatis  
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=200)

# =======================
# STATISTIK GLOBAL STYLE
# =======================
st.markdown("## 📊 Statistik Obat Global")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Jumlah Obat Teregistrasi</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-number">120.000+</p>', unsafe_allow_html=True)
    st.markdown('<p class="grow-tag">↑ Bertambah tiap tahun</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Kategori Obat</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-number">35 Kategori</p>', unsafe_allow_html=True)
    st.markdown('<p class="grow-tag">↑ 5 kategori baru</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Interaksi Obat</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-number">10.500+</p>', unsafe_allow_html=True)
    st.markdown('<p class="grow-tag">↑ Update reguler</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
