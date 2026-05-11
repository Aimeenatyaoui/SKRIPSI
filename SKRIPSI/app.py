import streamlit as st


st.set_page_config(
    page_title="Deteksi Dini DM Tipe 2 (CBR)",
    page_icon="🩺",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
          .app-title { font-size: 1.55rem; font-weight: 750; margin: 0 0 .15rem 0; }
          .app-subtitle { color: rgba(15, 23, 42, .7); margin: 0 0 1rem 0; }

          .card {
            border: 1px solid rgba(2, 6, 23, .10);
            background: #ffffff;
            border-radius: 14px;
            padding: 1rem 1rem;
          }
          .muted { color: rgba(15, 23, 42, .7); }
          .tiny { font-size: .9rem; }
          .kpi {
            border: 1px solid rgba(2, 6, 23, .10);
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border-radius: 14px;
            padding: .9rem 1rem;
          }
          .kpi .label { font-size: .85rem; color: rgba(15, 23, 42, .70); }
          .kpi .value { font-size: 1.25rem; font-weight: 750; margin-top: .2rem; }
          .divider { height: 1px; background: rgba(2, 6, 23, .08); margin: .8rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.markdown('<div class="app-title">Sistem Deteksi Dini Diabetes Mellitus Tipe 2</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Berbasis <b>Case-Based Reasoning</b> (Retrieve–Reuse–Revise–Retain) untuk mendukung skrining awal di Faskes 1.</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1.35, 1], gap="large")

with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Mulai cepat")
    st.markdown(
        """
        - Buka menu **Deteksi** untuk input data pasien (8 atribut).
        - Sistem akan menampilkan estimasi risiko (hasil CBR/ML).
        - User melakukan **Revise** (konfirmasi klinis).
        - Sistem menyimpan ke basis kasus (**Retain**).
        """.strip()
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Catatan implementasi")
    st.markdown(
        """
        Aplikasi ini disiapkan untuk Streamlit Cloud.
        Saat ini UI sudah lengkap; modul prediksi CBR/ML akan kamu lampirkan dan nanti kita integrasikan.
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown('<div class="label">Peran pengguna</div><div class="value">Petugas Faskes 1</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown(
        '<div class="label">Metode</div><div class="value">CBR + Weighted Euclidean Distance</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown(
        '<div class="label">Alur</div><div class="value">Input → Validasi → Normalisasi → Retrieve → Voting → Revise → Retain</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.info("Gunakan navigasi di sidebar (kiri) untuk masuk ke halaman **Deteksi**.")
