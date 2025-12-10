import streamlit as st
import pandas as pd

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
    <div class="big-title">🌈 Colorful MedCheck – Analisis Jenis Obat</div>
    <div class="subtext">
        Ketik nama obat dan biarkan sistem menganalisis jenisnya dengan tampilan cantik & penuh warna 😍💊
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
    "obat lambung": "Hindari pemakaian >3 bulan tanpa dokter.",
    "antidiabetik": "Hati-hati hipoglikemia."
}


# ========= INPUT =========
st.markdown("## 💬 Ketik Nama Obat")
nama_obat = st.text_input("Masukkan nama obat:", placeholder="cth: amoxicillin / paracetamol / cetirizine")


# ========= ANALISIS =========
if st.button("🎉 Analisis Jenis Obat"):
    if not nama_obat:
        st.warning("Masukkan nama obat dulu ya 💖")
    else:
        name = nama_obat.lower().strip()
        kategori_ditemukan = None

        # Cek dalam database
        for kategori, daftar in obat_db.items():
            if name in daftar:
                kategori_ditemukan = kategori
                break

        if kategori_ditemukan:
            # ======== CARD WARNA WARNI =========
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

            # Rekomendasi obat sejenis
            st.markdown("### 🌸 Rekomendasi Obat Satu Kategori")
            rekom = [o.title() for o in obat_db[kategori_ditemukan] if o != name]
            st.write(", ".join(rekom))

        else:
            st.error(f"Obat **{nama_obat}** belum tersedia di database 🌧")
            st.info("Aku bisa tambahkan obat baru kalau kamu mau ✨")


# ========= FOOTER =========
st.markdown("""
<div class="footer">
    💖 Colorful MedCheck – Dashboard Aesthetic & Ceria  
    <br> Dibuat khusus untuk kamu ✨
</div>
""", unsafe_allow_html=True)
