# ================== IMPORT ==================
import json, io, math
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests
import matplotlib.pyplot as plt

# ================== CONFIG & HEADER ==================
st.set_page_config(page_title="Peta Kesesuaian Lahan", layout="wide")
DEFAULT_VIEW = {"lat": -2.5, "lon": 118.0, "zoom": 4.0}

logo_url = "https://lms.bmkg.go.id/pluginfile.php/1/theme_mb2nl/logo/1725958993/logo%20lms.png"
st.markdown("""
<style>
.block-container{padding-top:3rem;}
h2,h3{margin:0.6rem 0;}
#MainMenu {visibility: visible;}
header {visibility: visible;}
.stDeployButton{display:none;}
hr{margin:0.8rem 0 1.1rem;}
.section-title{font-weight:700;margin:6px 0 8px;}
.small{font:12px system-ui;color:#333}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;">
  <img src="{logo_url}" alt="BMKG" style="max-height:90px; display:block; margin:0 auto;"/>
  <h2>Aktualisasi CPNS Golongan IIIA tahun 2025</h2>
  <h3>Direktorat Layanan Iklim Terapan</h3>
  <h3>"PETA KESESUAIAN LAHAN"</h3>
</div>
<hr/>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
@st.cache_data(show_spinner=False)
def load_from_url(url: str):
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def load_geojson_bytes(b: bytes):
    return json.loads(b.decode("utf-8"))

def compute_center_zoom(df_points: pd.DataFrame):
    lon_min, lon_max = float(df_points["lon"].min()), float(df_points["lon"].max())
    lat_min, lat_max = float(df_points["lat"].min()), float(df_points["lat"].max())

    # padding agar kamera mundur sedikit
    PAD = 1.7            # 1.5–2.0 makin besar = makin jauh
    C   = 8              # 10.3–10.8 makin kecil = makin jauh

    lon_span = (lon_max - lon_min) * PAD
    lat_span = (lat_max - lat_min) * PAD
    span = max(lon_span, lat_span)

    zoom = float(np.clip(C - np.log2(max(span, 1e-6)), 4.8, 9.5))  # clamp nyaman nasional
    center = ((lat_min + lat_max)/2.0, (lon_min + lon_max)/2.0)
    return center, zoom, span


def to_num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan

def legend_html():
    cmap = {"S1":[27,158,119], "S2":[31,119,180], "S3":[214,39,40], "N":[255,255,255]}
    order = ["S1","S2","S3","N"]
    items = []
    for k in order:
        c = ",".join(map(str, cmap[k]))
        items.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:rgb({c});border:1px solid #333;margin:0 6px -2px 10px;"></span>{k}'
        )
    return '<div style="font:12px system-ui;">' + " ".join(items) + "</div>"

def make_deck(df_draw, view_state, kelas_field, title, boundary_layer=None):
    """Scatter titik + (opsional) garis batas GeoJSON hitam di atas titik."""
    data = df_draw.copy()
    if kelas_field not in data.columns:
        data[kelas_field] = "N"

    color_map = {"S1":[27,158,119], "S2":[31,119,180], "S3":[214,39,40], "N":[255,255,255]}
    data["color"] = data[kelas_field].map(lambda k: color_map.get(str(k).upper(), [204,204,204]))

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position="[lon, lat]",
        get_fill_color="color",
        pickable=True,
        radius_min_pixels=st.session_state.get("radius", 2),
        radius_max_pixels=st.session_state.get("radius", 2),
        opacity=st.session_state.get("opacity", 0.8),
        stroked=False,
    )

    layers = [scatter]
    if boundary_layer is not None:
        layers.append(boundary_layer)  # terakhir = di atas titik

    tooltip = {
        "html": (
            f"<b>Model</b>: {title}"
            "<br/><b>Kelas</b>: {" + kelas_field + "}"
            "<br/><b>Provinsi</b>: {prov}"
            "<br/><b>LON</b>: {lon}"
            "<br/><b>LAT</b>: {lat}"
            "<br/><b>CH Tahunan</b>: {CH_THN}"
            "<br/><b>Temperatur (mean)</b>: {T2M_mean}"
            "<br/><b>Kemiringan (%)</b>: {slope_percent}"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)

# ================== SUMBER DATA TITIK ==================
with st.sidebar:
    st.header("Sumber GeoJSON (titik)")
    gj_mode = st.radio("Ambil dari:", ["Upload GeoJSON/JSON","URL GeoJSON"], index=0)

geo = None
if gj_mode == "Upload GeoJSON/JSON":
    up = st.sidebar.file_uploader("Upload .geojson / .json", type=["geojson","json"])
    if up: geo = load_geojson_bytes(up.getvalue())
else:
    url = st.sidebar.text_input("URL GeoJSON titik", placeholder="https://.../data.geojson")
    if url: geo = load_from_url(url)

if not geo:
    st.info("Muat GeoJSON titik dulu (upload/URL)."); st.stop()

assert geo.get("type") == "FeatureCollection", "File harus FeatureCollection."
feats = [f for f in geo.get("features", []) if isinstance(f, dict)]
props_df = pd.DataFrame([f.get("properties", {}) for f in feats if f.get("properties")])
if props_df.empty:
    st.error("GeoJSON tidak memiliki properti."); st.stop()

# ================== RISK MODE (GeoJSON hasil df_bulanan pivot) ==================
import re

# --- ekstrak risk_*_YYYY-MM ke format long untuk plotting ---
def risk_geojson_to_long(geo_fc):
    feats = geo_fc.get("features", [])
    rows = []
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "Point": 
            continue
        lon, lat = g.get("coordinates", [None, None])[:2]
        if lon is None or lat is None:
            continue

        p = f.get("properties") or {}
        # meta wilayah
        prov = p.get("provinsi") or p.get("Provinsi") or p.get("PROVINSI") or p.get("PROV") or p.get("WADMPR")
        kab  = p.get("kabkota") or p.get("KabKota") or p.get("KABKOTA")
        kode = p.get("kode_wilayah") or p.get("KODE_WILAYAH")

        # cari semua kolom risk_*_YYYY-MM
        for k, v in p.items():
            m = re.match(r"^(risk_[a-z_]+)_(\d{4}-\d{2})$", str(k))
            if m:
                base, ym = m.group(1), m.group(2)
                try:
                    val = float(v) if v is not None else np.nan
                except Exception:
                    val = np.nan
                rows.append({
                    "lon": float(lon), "lat": float(lat),
                    "provinsi": prov, "kabkota": kab, "kode_wilayah": kode,
                    "metric": base, "month": ym, "value": val
                })
    return pd.DataFrame(rows)

risk_long = risk_geojson_to_long(geo)
if risk_long.empty:
    st.warning("GeoJSON tidak punya kolom risk_*_YYYY-MM."); st.stop()

# --- sidebar kontrol risiko ---
with st.sidebar:
    st.header("Risiko Iklim (bulan)")
    metric_opts = sorted(risk_long["metric"].unique().tolist())
    sel_metric = st.selectbox("Metric", metric_opts, index=metric_opts.index("risk_intensity_mean") if "risk_intensity_mean" in metric_opts else 0)
    month_opts = sorted(risk_long["month"].unique().tolist())
    sel_month = st.selectbox("Bulan (YYYY-MM)", month_opts, index=0)

# filter provinsi (pakai pilihan PROV_COL jika ada, fallback 'provinsi')
prov_col_risk = PROV_COL if PROV_COL in (risk_long.columns) else "provinsi"
if sel_prov:
    risk_view = risk_long[(risk_long["metric"]==sel_metric)&(risk_long["month"]==sel_month)
                          & (risk_long[prov_col_risk].astype(str).isin(sel_prov))]
else:
    risk_view = risk_long[(risk_long["metric"]==sel_metric)&(risk_long["month"]==sel_month)]

if risk_view.empty:
    st.info("Tidak ada titik untuk kombinasi filter tersebut."); st.stop()

# --- skala warna kontinu (precompute ke kolom 'color') ---
v = risk_view["value"].to_numpy(dtype=float)
v = v[np.isfinite(v)]
vmin, vmax = (float(np.nanmin(v)), float(np.nanmax(v))) if v.size else (0.0, 1.0)
if np.isclose(vmin, vmax):
    vmax = vmin + 1e-6

def lerp(a, b, t): return a + (b - a) * t
# palet: biru → hijau → kuning → merah
PALET = [(33, 102, 172), (67, 162, 202), (123, 204, 196), (255, 255, 191), (253, 174, 97), (215, 25, 28)]
def color_from_val(x):
    if not np.isfinite(x): 
        return [200, 200, 200, int(st.session_state.get("opacity", 0.8)*255)]
    t = (x - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
    # pilih segmen
    nseg = len(PALET) - 1
    idx = min(int(t * nseg), nseg - 1)
    tseg = (t * nseg) - idx
    c0 = np.array(PALET[idx], dtype=float)
    c1 = np.array(PALET[idx+1], dtype=float)
    c  = (1 - tseg) * c0 + tseg * c1
    return [int(c[0]), int(c[1]), int(c[2]), int(st.session_state.get("opacity", 0.8)*255)]

risk_draw = risk_view.copy()
risk_draw["color"] = risk_draw["value"].map(color_from_val)

# --- viewstate auto (pakai titik terfilter risiko) ---
center, zoom, span = compute_center_zoom(risk_draw.rename(columns={"lon":"lon","lat":"lat"}))
if sel_prov and len(sel_prov)==1:
    zoom = float(np.clip(zoom + 0.4, 3, 18))
view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom)

# --- layer batas (pakai yang sudah disiapkan di atas) ---
risk_layer = pdk.Layer(
    "ScatterplotLayer",
    risk_draw,
    get_position="[lon, lat]",
    get_fill_color="color",
    pickable=True,
    radius_min_pixels=st.session_state.get("radius", 2),
    radius_max_pixels=st.session_state.get("radius", 2),
    stroked=False,
)

layers = [risk_layer]
if boundary_layer is not None:
    layers.append(boundary_layer)

# --- tooltip dengan nilai risiko ---
tooltip = {
    "html": (
        f"<b>Metric</b>: {sel_metric}<br/>"
        f"<b>Bulan</b>: {sel_month}<br/>"
        "<b>Provinsi</b>: {provinsi}<br/>"
        "<b>Kab/Kota</b>: {kabkota}<br/>"
        "<b>Nilai</b>: {value}"
    ),
    "style": {"backgroundColor": "white", "color": "black"},
}

st.markdown(f"### Peta Risiko – {sel_metric} ({sel_month})")
deck_risk = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)
st.pydeck_chart(deck_risk, use_container_width=True)

# --- legenda sederhana ---
st.caption(f"Rentang nilai: min={vmin:.3g}  max={vmax:.3g}")

# ================== DETEKSI PROVINSI ==================
CAND = ["provinsi","ADM_PROV","PROVINSI","Provinsi","PROV","WADMPR","NAMPROV","NAMA_PROP","NAMA_PROV"]
PROV_COL = next((c for c in CAND if c in props_df.columns), None)
if not PROV_COL:
    st.error("Kolom provinsi tidak ditemukan. Gunakan salah satu: " + ", ".join(CAND)); st.stop()

# ================== FILTER & KONTROL ==================
with st.sidebar:
    st.header("Filter")
    prov_options = sorted(props_df[PROV_COL].dropna().astype(str).unique().tolist())
    sel_prov = st.multiselect("Pilih PROVINSI", prov_options, default=[])

    st.header("Simbolisasi")
    st.session_state["radius"]  = st.slider("Radius titik (px)", 1, 15, 2)
    st.session_state["opacity"] = st.slider("Opasitas titik", 0.1, 1.0, 0.8, 0.1)

    st.header("Batas provinsi (garis)")
    ov_mode = st.radio("Sumber:", ["Tidak ada","URL GeoJSON","Upload GeoJSON"], index=0)
    line_width = st.slider("Tebal garis", 0.5, 4.0, 1.2, 0.1)
    line_alpha = st.slider("Opacity garis", 0.1, 1.0, 0.7, 0.05)

# === Buat layer batas (HITAM) ===
boundary_layer = None
if ov_mode == "URL GeoJSON":
    ov_url = st.sidebar.text_input("URL GeoJSON batas", "")
    if ov_url:
        boundary_layer = pdk.Layer(
            "GeoJsonLayer", ov_url,
            stroked=True, filled=False, pickable=False,
            get_line_color=[0, 0, 0, int(line_alpha*255)],
            get_line_width=1,
            lineWidthUnits="pixels",
            lineWidthScale=1,
            lineWidthMinPixels=int(math.ceil(line_width))
        )
elif ov_mode == "Upload GeoJSON":
    up_bnd = st.sidebar.file_uploader("Upload GeoJSON batas (≤10MB)",
                                      type=["geojson","json"], key="overlay_gj")
    if up_bnd:
        bnd_geo = json.loads(up_bnd.getvalue().decode("utf-8"))
        boundary_layer = pdk.Layer(
            "GeoJsonLayer", bnd_geo,
            stroked=True, filled=False, pickable=False,
            get_line_color=[0, 0, 0, int(line_alpha*255)],
            get_line_width=1,
            lineWidthUnits="pixels",
            lineWidthScale=1,
            lineWidthMinPixels=int(math.ceil(line_width))
        )

# ================== EXTRACT DATA TITIK ==================
bulan = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
CH_cols  = [f"CH_{b}"  for b in bulan]
T2M_cols = [f"T2M_{b}" for b in bulan]
CH_year_candidates = ["CH_THN_calc","CH_THN","CH_Tahun","CH_Tahunan","CH_Annual"]

def pass_filter(p: dict) -> bool:
    return (not sel_prov) or (str(p.get(PROV_COL)) in sel_prov)

def sum_ch(p):
    ch = np.nansum([to_num(p.get(c)) for c in CH_cols if c in p])
    if np.isnan(ch) or ch == 0:
        for k in CH_year_candidates:
            if k in p and not pd.isna(to_num(p.get(k))):
                return to_num(p.get(k))
        return np.nan
    return ch

rows = []
for f in feats:
    g = f.get("geometry", {}) or {}
    if g.get("type") != "Point": continue
    coords = g.get("coordinates", [None, None])
    if not coords or len(coords) < 2 or None in coords[:2]: continue
    p = f.get("properties", {}) or {}
    if not pass_filter(p): continue

    t2m_vals = [to_num(p.get(c)) for c in T2M_cols if c in p]
    t2m_mean = float(np.nanmean(t2m_vals)) if len(t2m_vals) else (to_num(p.get("T2M_mean")) if "T2M_mean" in p else np.nan)

    row = {
        "lon": float(coords[0]), "lat": float(coords[1]),
        "prov": str(p.get(PROV_COL)),
        "kelas": p.get("kelas"),
        "kelas_crisp": p.get("kelas_crisp"),
        "kelas_tebu": p.get("kelas_tebu"),
        "kelas_tebu_crisp": p.get("kelas_tebu_crisp"),
        "slope_percent": p.get("slope_percent"),
        "CH_THN": None if np.isnan(sum_ch(p)) else float(sum_ch(p)),
        "T2M_mean": None if np.isnan(t2m_mean) else t2m_mean,
    }
    for c in CH_cols:  row[c] = to_num(p.get(c))
    for c in T2M_cols: row[c] = to_num(p.get(c))
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    st.warning("Tidak ada titik setelah filter."); st.stop()

# sampling agar ringan
max_points = st.sidebar.slider("Batas titik dirender", 1000, 200000, min(60000, len(df)), 1000)
df_draw = df.sample(max_points, random_state=42).reset_index(drop=True) if len(df) > max_points else df

# string-kan kolom penting
for c in ["kelas","kelas_crisp","kelas_tebu","kelas_tebu_crisp","prov"]:
    if c in df_draw.columns:
        df_draw[c] = df_draw[c].astype(str)

# ================== VIEWSTATE (untuk peta) ==================
dfv = df_draw[df_draw["prov"].astype(str).isin(sel_prov)] if sel_prov else df_draw
if sel_prov and not dfv.empty:
    center, zoom, span = compute_center_zoom(dfv)
    if len(sel_prov) == 1: zoom += 0.4
    if span < 2: zoom += 0.3
    if span < 1: zoom += 0.2
    SPECIAL_ZOOM = {"Bali": 8.0, "DKI Jakarta": 9.0, "DI Yogyakarta": 8.0,
                    "Kepulauan Riau": 7.5, "Bangka Belitung": 7.5}
    if len(sel_prov) == 1 and sel_prov[0] in SPECIAL_ZOOM: zoom = max(zoom, SPECIAL_ZOOM[sel_prov[0]])
    zoom = float(np.clip(zoom, 3, 18))
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom)
else:
    view_state = pdk.ViewState(latitude=DEFAULT_VIEW["lat"], longitude=DEFAULT_VIEW["lon"], zoom=DEFAULT_VIEW["zoom"])

# ================== 1) PETA SPASIAL: PILIH TANAMAN, FILTER KELAS, PETA FUZZY & CRISP ==================
st.markdown('<div class="section-title">Pilih Tanaman</div>', unsafe_allow_html=True)
tanaman = st.selectbox("", ["Sawit","Tebu"], index=0, label_visibility="collapsed")

FIELD_MAP = {"Sawit": {"fuzzy": "kelas","crisp": "kelas_crisp"},
             "Tebu":  {"fuzzy": "kelas_tebu","crisp": "kelas_tebu_crisp"}}
flds = FIELD_MAP.get(tanaman, {})
fld_fuzzy = flds.get("fuzzy"); fld_crisp = flds.get("crisp")
for col in [fld_fuzzy, fld_crisp]:
    if col and col not in df_draw.columns:
        df_draw[col] = np.nan

st.markdown("**Filter kelas & legenda**")
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,3])
with c1: s1 = st.checkbox("S1", True)
with c2: s2 = st.checkbox("S2", True)
with c3: s3 = st.checkbox("S3", True)
with c4: sn = st.checkbox("N",  True)
with c5: st.markdown(legend_html(), unsafe_allow_html=True)
selected_kelas = [k for k, ok in zip(["S1","S2","S3","N"], [s1,s2,s3,sn]) if ok]
if not selected_kelas:
    st.warning("Tidak ada kelas dipilih."); st.stop()

df_fuzzy = df_draw[df_draw[fld_fuzzy].astype(str).str.upper().isin(selected_kelas)]
df_crisp = df_draw[df_draw[fld_crisp].astype(str).str.upper().isin(selected_kelas)]

st.markdown("**Model Fuzzy**")
deck_fuzzy = make_deck(df_fuzzy, view_state, fld_fuzzy, f"{tanaman} – Fuzzy",
                       boundary_layer=boundary_layer)
st.pydeck_chart(deck_fuzzy, use_container_width=True)

st.markdown("**Model Crisp**")
deck_crisp = make_deck(df_crisp, view_state, fld_crisp, f"{tanaman} – Crisp",
                       boundary_layer=boundary_layer)
st.pydeck_chart(deck_crisp, use_container_width=True)

# ================== 2) SUMMARY % KELAS (default Indonesia) ==================
area_list = ["(Seluruh Indonesia)"] + prov_options
default_area = sel_prov[0] if (sel_prov and len(sel_prov)==1 and sel_prov[0] in prov_options) else "(Seluruh Indonesia)"
area_for_summary = st.selectbox("Area summary", area_list, index=area_list.index(default_area))

st.markdown(f"### Ringkasan kelas – {('Indonesia' if area_for_summary == '(Seluruh Indonesia)' else area_for_summary)}")
if area_for_summary == "(Seluruh Indonesia)":
    df_sum = df.copy()
else:
    df_sum = df[df["prov"] == area_for_summary]

def pct_table(dfx, field):
    if dfx.empty: return pd.DataFrame(columns=["Kelas","Jumlah","Persen"])
    total = len(dfx)
    vc = dfx[field].astype(str).value_counts(dropna=False)
    tbl = vc.rename_axis("Kelas").reset_index(name="Jumlah")
    tbl["Persen"] = (tbl["Jumlah"] / total * 100).round(2)
    order = ["S1","S2","S3","N"]
    tbl["Kelas"] = pd.Categorical(tbl["Kelas"], order)
    return tbl.sort_values("Kelas").reset_index(drop=True)

c1, c2 = st.columns(2)
with c1:
    st.caption(f"{tanaman} – Fuzzy ({fld_fuzzy})")
    st.dataframe(pct_table(df_sum, fld_fuzzy), use_container_width=True, height=180)
with c2:
    st.caption(f"{tanaman} – Crisp ({fld_crisp})")
    st.dataframe(pct_table(df_sum, fld_crisp), use_container_width=True, height=180)

# ================== 3) TEMPORAL CH & T2M (berdampingan) ==================
st.divider()
st.subheader("Temporal (rata-rata area, klimatologi 30 tahun)")

# sinkron area temporal dengan area summary
prov_focus = area_for_summary
if prov_focus == "(Seluruh Indonesia)":
    df_area = df.copy(); area_label = "Indonesia"
else:
    df_area = df[df["prov"] == prov_focus].copy(); area_label = prov_focus

ch_avail  = [c for c in CH_cols  if c in df_area.columns]
t2m_avail = [c for c in T2M_cols if c in df_area.columns]

colA, colB = st.columns(2)

if ch_avail:
    ch_mean_area = df_area[ch_avail].mean(axis=0, skipna=True)
    x_labels = [b for b in bulan if f"CH_{b}" in ch_mean_area.index]
    y = [float(ch_mean_area[f"CH_{b}"]) for b in x_labels]

    with colA:
        fig, ax = plt.subplots(figsize=(7, 3))
        x = np.arange(len(x_labels))
        ax.bar(x, y, width=0.7)
        ax.set_xticks(x, labels=x_labels)
        ax.set_title(f"CH Bulanan – {area_label}")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("mm")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
else:
    with colA: st.info("Kolom CH_Jan..CH_Des tidak ditemukan.")

if t2m_avail:
    t2m_mean_area = df_area[t2m_avail].mean(axis=0, skipna=True)
    x = [b for b in bulan if f"T2M_{b}" in t2m_mean_area.index]
    y = [float(t2m_mean_area[f"T2M_{b}"]) for b in x]
    with colB:
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(x, y, marker="o", color="tab:red")
        ax.set_title(f"Temperatur Bulanan – {area_label}")
        ax.set_xlabel("Bulan"); ax.set_ylabel("°C"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
else:
    with colB: st.info("Kolom T2M_Jan..T2M_Des tidak ditemukan.")

# ================== 4) UNDUH (opsional) ==================
filtered_fc = {
    "type": "FeatureCollection",
    "features": [f for f in geo.get("features", [])
                 if f.get("geometry", {}).get("type") == "Point"
                 and (not sel_prov or str(f.get("properties", {}).get(PROV_COL)) in sel_prov)]
}
st.download_button(
    "Download GeoJSON (terfilter)",
    data=json.dumps(filtered_fc, ensure_ascii=False).encode("utf-8"),
    file_name="data_filtered.geojson",
    mime="application/geo+json",
)
