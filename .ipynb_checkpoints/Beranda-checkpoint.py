# app.py
import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.title("PETA KESESUAIAN LAHAN DAN RISIKO TANAM")
st.write("Pilih halaman:")

cols = st.columns(2)
with cols[0]:
    st.page_link("pages/kesesuaian_lahan.py", label="Kesesuaian Lahan", icon="🌱")
with cols[1]:
    st.page_link("pages/risiko.py", label="Analisis Risiko", icon="⚠️")

st.divider()
st.caption("Gunakan tautan di atas atau sidebar Pages untuk navigasi.")
