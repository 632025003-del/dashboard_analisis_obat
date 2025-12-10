import streamlit as st
import pandas as pd
import plotly.express as px

# CONFIG
st.set_page_config(page_title="MedVision AI", page_icon="💊", layout="wide")
st.markdown("### 🧪 Koleksi Obat Variatif")
colA, colB, colC, colD = st.columns(4)

with colA:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=90)

with colB:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320748.png", width=90)

with colC:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320716.png", width=90)

with colD:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320758.png", width=90)

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

# ===============================
#    SETTING DASHBOARD
# ===============================
st.set_page_config(page_title="Analisis Jenis Obat", page_icon="💊", layout="wide")

st.title("🔍 Analisis Jenis Obat Berdasarkan Nama Obat")
st.write("Ketik nama obat, dan sistem akan otomatis mengidentifikasi jenis obatnya tanpa perlu upload CSV.")

# ===============================
#   MINI DATABASE OBAT
# ===============================
obat_db = {
    "antibiotik": ["amoxicillin", "ciprofloxacin", "cefixime", "clindamycin", "azithromycin", "doxycycline"],
    "analgesik": ["paracetamol", "ibuprofen", "aspirin", "asam mefenamat", "naproxen"],
    "antihipertensi": ["amlodipine", "captopril", "lisinopril", "losartan", "valsartan"],
    "anti alergi": ["cetirizine", "loratadine", "fexofenadine", "diphenhydramine"],
    "antiseptik": ["betadine", "povidone iodine", "alcohol 70", "chlorhexidine"],
    "obat lambung": ["omeprazole", "lansoprazole", "antacid", "ranitidine"],
    "antidiabetik": ["metformin", "glimepiride", "insulin", "acarbose"]
}

deskripsi = {
    "antibiotik": "Obat untuk mengatasi infeksi bakteri. Tidak efektif untuk virus.",
    "analgesik": "Obat untuk meredakan nyeri dan menurunkan demam.",
    "antihipertensi": "Obat untuk mengontrol dan menurunkan tekanan darah.",
    "anti alergi": "Obat untuk meredakan gejala alergi seperti gatal, ruam, dan bersin.",
    "antiseptik": "Obat untuk membunuh kuman pada permukaan kulit atau luka.",
    "obat lambung": "Obat untuk mengatasi maag, asam lambung, dan GERD.",
    "antidiabetik": "Obat untuk mengatur kadar gula darah."
}

peringatan = {
    "antibiotik": "Harus dihabiskan sesuai resep. Tidak boleh berhenti sebelum waktu.",
    "analgesik": "Gunakan sesuai dosis. Hati-hati untuk penderita gangguan lambung.",
    "antihipertensi": "Harus diminum rutin tiap hari.",
    "anti alergi": "Beberapa jenis menyebabkan kantuk.",
    "antiseptik": "Hanya untuk penggunaan luar. Hindari mata dan mulut.",
    "obat lambung": "Pemakaian jangka panjang harus diawasi dokter.",
    "antidiabetik": "Waspadai risiko hipoglikemia. Ikuti anjuran dokter."
}

# ===============================
#     INPUT NAMA OBAT
# ===============================
nama_obat = st.text_input("Masukkan nama obat:", placeholder="contoh: amoxicillin / paracetamol")

# ===============================
#        HASIL ANALISIS
# ===============================
if st.button("Analisis Obat"):
    if not nama_obat:
        st.warning("Masukkan nama obat terlebih dahulu.")
    else:
        nama = nama_obat.lower().strip()
        hasil = None

        # CARI KATEGORI OBAT
        for kategori, daftar in obat_db.items():
            if nama in daftar:
                hasil = kategori
                break

        # ====================
        #  HASIL DITEMUKAN
        # ====================
        if hasil:
            st.success(f"**{nama_obat.title()} termasuk kategori: {hasil.upper()}**")

            st.markdown("### 📘 Penjelasan")
            st.write(deskripsi[hasil])

            st.markdown("### ⚠️ Peringatan")
            st.write(peringatan[hasil])

            st.markdown("### 💊 Contoh Obat Lain Satu Kategori")
            st.write(", ".join([o.title() for o in obat_db[hasil] if o != nama]))

        # ====================
        #    TIDAK DITEMUKAN
        # ====================
        else:
            st.error(f"Jenis obat untuk **{nama_obat}** tidak ditemukan.")
            st.info("Kamu bisa menambahkan obat baru ke daftar jika perlu.")

# ===============================
#          FOOTER
# ===============================
st.markdown("""
<hr>
<center>
💊 Aplikasi Analisis Obat – Tidak perlu upload file | Cukup ketik nama obat  
<br> Dibuat dengan sistem rule-based sederhana dan cepat
</center>
""", unsafe_allow_html=True)
