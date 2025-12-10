import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
#     CONFIG COLORFUL UI
# =========================
st.set_page_config(page_title="Colorful Drug Analyzer", page_icon="💊", layout="wide")

# ========== CUSTOM CSS COLORFUL ==========
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
}

.header-box {
    background: linear-gradient(90deg, #FF9A9E, #FECFEF, #FFCBA4);
    padding: 30px;
    border-radius: 20px;
    color: #4a4a4a;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.big-title {
    font-size: 40px;
    font-weight: 900;
    color: #3F3F3F;
}

.subtext {
    font-size: 18px;
    color: #444;
}

.card {
    background: linear-gradient(135deg, #FFF6B7 0%, #F6416C 100%);
    padding: 18px;
    color: white;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    text-align: center;
}

.card2 {
    background: linear-gradient(135deg, #A1C4FD 0%, #C2E9FB 100%);
    padding: 18px;
    border-radius: 18px;
    color: #333;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #444;
}
</style>
""", unsafe_allow_html=True)


# ========= HEADER COLORFUL =========
st.markdown("""
<div class="header-box">
    <div class="big-title">🌈 Colorful MedCheck – Analisis Jenis Obat dengan Sekali Tap </div>
    <div class="subtext">
        Ketik nama obat yang ingin ditanyakan atau upload CSV obat hasil analisis akan muncul dengan tampilan cantik & penuh warna 😍💊
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")


# ========= GAMBAR-GAMBAR CUTE =========
colA, colB, colC, colD = st.columns(4)

with colA:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320758.png", width=120)

with colB:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=120)

with colC:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320748.png", width=120)

with colD:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=120)


# ========= DATABASE JENIS OBAT =========
obat_db = {
    "antibiotik": ["amoxicillin", "ciprofloxacin", "cefixime", "doxycycline", "azithromycin"],
    "analgesik": ["paracetamol", "ibuprofen", "aspirin", "naproxen"],
    "antihipertensi": ["amlodipine", "captopril", "lisinopril"],
    "anti alergi": ["cetirizine", "loratadine", "fexofenadine"],
    "antiseptik": ["betadine", "chlorhexidine", "povidone iodine"],
    "obat lambung": ["omeprazole", "lansoprazole", "antacid"],
    "antidiabetik": ["metformin", "glimepiride", "insulin"]
}

deskripsi = {
    "antibiotik": "Digunakan untuk mengatasi infeksi bakteri.",
    "analgesik": "Pereda nyeri dan penurun demam.",
    "antihipertensi": "Menurunkan tekanan darah.",
    "anti alergi": "Meredakan gejala alergi seperti gatal & bersin.",
    "antiseptik": "Membersihkan luka dan membunuh kuman.",
    "obat lambung": "Mengatasi maag dan asam lambung.",
    "antidiabetik": "Mengontrol gula darah penderita diabetes."
}

peringatan = {
    "antibiotik": "Harus dihabiskan sesuai resep.",
    "analgesik": "Hati-hati jika punya sakit maag.",
    "antihipertensi": "Jangan berhenti minum tiba-tiba.",
    "anti alergi": "Beberapa membuat mengantuk.",
    "antiseptik": "Hanya untuk luar tubuh.",
    "obat lambung": "Hindari pemakaian lebih dari 3 bulan.",
    "antidiabetik": "Waspadai risiko hipoglikemia."
}


# ========= INPUT MANUAL OBAT =========
st.markdown("## 💬 Ketik Nama Obat")
nama_obat = st.text_input("Masukkan nama obat:", placeholder="cth: amoxicillin / paracetamol / cetirizine")

if st.button("🎉 Analisis Nama Obat"):
    if not nama_obat:
        st.warning("Masukkan nama obat dulu ya 💖")
    else:
        name = nama_obat.lower().strip()
        kategori_ditemukan = None

        for kategori, daftar in obat_db.items():
            if name in daftar:
                kategori_ditemukan = kategori
                break

        if kategori_ditemukan:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>{nama_obat.title()}</h3>
                    <p><b>Kategori:</b> {kategori_ditemukan.title()}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card2">
                    <p><b>Penjelasan:</b><br>{deskripsi[kategori_ditemukan]}</p>
                    <p><b>Peringatan:</b><br>{peringatan[kategori_ditemukan]}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error(f"Obat **{nama_obat}** tidak ditemukan 🌧")


# ====================================================
#        🌸 FITUR BARU: UPLOAD CSV OBAT (COLORFUL)
# ====================================================

st.markdown("## 📥 Upload CSV Obat (Optional)")

csv = st.file_uploader("Upload file CSV (harus memiliki kolom 'nama_obat'):", type=["csv"])

if csv:
    df = pd.read_csv(csv)
    st.success("CSV berhasil dimuat! Berikut datanya:")
    st.dataframe(df, use_container_width=True)

    if "nama_obat" not in df.columns:
        st.error("CSV harus memiliki kolom **nama_obat** 🌸")
    else:
        st.markdown("### 🎨 Hasil Analisis Berdasarkan CSV")

        df["kategori"] = df["nama_obat"].str.lower().apply(
            lambda x: next((kat for kat, daftar in obat_db.items() if x in daftar), "tidak ditemukan")
        )

        st.dataframe(df, use_container_width=True)

        # VISUALISASI COLORFUL
        fig = px.histogram(
            df,
            x="kategori",
            title="🌈 Distribusi Kategori Obat",
            color="kategori",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)


# ========= FOOTER =========
st.markdown("""
<div class="footer">
    💖 Colorful MedCheck – Dashboard Aesthetic, Ceria, dan Penuh Warna  
    <br> Dibuat khusus untuk kamu ✨
</div>
""", unsafe_allow_html=True)
