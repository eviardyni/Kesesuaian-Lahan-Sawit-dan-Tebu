# app.py — Peta Risiko Iklim Bulanan (GeoJSON titik)
import json, math, re
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# ================== CONFIG & HEADER ==================
st.set_page_config(page_title="Peta Risiko Tanam Kelapa Sawit", layout="wide")
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
  <h3>"PETA RISIKO IKLIM UNTUK SEKTOR PERKEBUNAN"</h3>
</div>
<hr/>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> dict:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def load_geojson_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8"))

def to_num(x):
    try: return float(x)
    except Exception: return np.nan

def compute_center_zoom(df_points: pd.DataFrame):
    lon_min, lon_max = float(df_points["lon"].min()), float(df_points["lon"].max())
    lat_min, lat_max = float(df_points["lat"].min()), float(df_points["lat"].max())
    PAD = 1.7; C = 8.5
    lon_span = (lon_max - lon_min) * PAD
    lat_span = (lat_max - lat_min) * PAD
    span = max(lon_span, lat_span, 1e-6)
    zoom = float(np.clip(C - np.log2(span), 4, 12))
    center = ((lat_min + lat_max)/2.0, (lon_min + lon_max)/2.0)
    return center, zoom, span

@st.cache_data(show_spinner=False)
def risk_geojson_to_long(geo_fc: dict, prov_key_guess: str|None=None) -> pd.DataFrame:
    feats = geo_fc.get("features", [])
    rows = []
    prov_candidates = ["provinsi","ADM_PROV","PROVINSI","Provinsi","PROV","WADMPR","NAMPROV","NAMA_PROP","NAMA_PROV"]
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "Point": continue
        coords = g.get("coordinates") or []
        if len(coords) < 2: continue
        lon, lat = coords[:2]
        if lon is None or lat is None: continue
        p = f.get("properties") or {}

        prov_col = prov_key_guess or next((c for c in prov_candidates if c in p), None)
        prov = p.get(prov_col) if prov_col else None
        kab  = p.get("kabkota") or p.get("KabKota") or p.get("KABKOTA")
        kode = p.get("kode_wilayah") or p.get("KODE_WILAYAH")

        for k, v in p.items():
            m = re.match(r"^(risk_[a-z_]+)_(\d{4}-\d{2})$", str(k))
            if not m: 
                continue
            metric, month = m.group(1), m.group(2)
            rows.append({
                "lon": float(lon), "lat": float(lat),
                "prov": None if prov is None else str(prov),
                "kabkota": None if kab  is None else str(kab),
                "kode_wilayah": None if kode is None else str(kode),
                "metric": metric, "month": month,
                "value": to_num(v),
            })
    return pd.DataFrame(rows)

def build_boundary_layer(source_geojson_or_url, line_width_px: float, alpha: float):
    rgba = [0,0,0, int(alpha*255)]
    return pdk.Layer(
        "GeoJsonLayer",
        source_geojson_or_url,
        stroked=True, filled=False, pickable=False,
        get_line_color=rgba, get_line_width=1,
        lineWidthUnits="pixels", lineWidthScale=1,
        lineWidthMinPixels=int(math.ceil(line_width_px))
    )

def make_color_mapper(vmin, vmax, opacity):
    # Hijau (baik) → Kuning → Merah (buruk)
    PALET = [(26,150,65), (166,217,106), (255,255,191), (253,174,97), (215,25,28)]
    nseg = len(PALET) - 1
    a = int(opacity * 255)
    def to_color(x):
        if not np.isfinite(x): return [200,200,200,a]
        t = (x - vmin) / max(vmax - vmin, 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        i = min(int(t * nseg), nseg - 1)
        ts = (t * nseg) - i
        c0, c1 = np.array(PALET[i]), np.array(PALET[i+1])
        c = (1 - ts)*c0 + ts*c1
        return [int(c[0]), int(c[1]), int(c[2]), a]
    return to_color

def render_continuous_legend(vmin, vmax, metric_label="nilai"):
    return f"""
    <div style="margin:8px 0 2px; font:12px system-ui;">
      Skala warna ({metric_label}): lebih hijau = lebih layak tanam, lebih merah = berisiko
    </div>
    <div style="
      height:14px; width:100%; max-width:520px; border:1px solid #999; border-radius:3px;
      background: linear-gradient(to right,
        rgb(26,150,65), rgb(166,217,106), rgb(255,255,191), rgb(253,174,97), rgb(215,25,28)
      );
      ">
    </div>
    <div style="display:flex; justify-content:space-between; width:100%; max-width:520px; font:12px system-ui; color:#333;">
      <span>{vmin:.4g}</span>
      <span>{vmax:.4g}</span>
    </div>
    """

def p95(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 95)) if x.size else np.nan

@st.cache_data(show_spinner=False)
def load_geojson_path(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================== SIDEBAR: INPUT ==================
with st.sidebar:
    st.header("Sumber GeoJSON (titik risiko)")
    src_mode = st.radio("Ambil dari:", ["Upload","URL"], index=0)
    geo = None
    if src_mode == "Upload":
        up = st.file_uploader("Upload .geojson / .json", type=["geojson","json"])
        if up: geo = load_geojson_bytes(up.getvalue())
    else:
        url = st.text_input("URL GeoJSON", "")
        if url: geo = load_from_url(url)

if not geo:
    st.info("Muat GeoJSON dulu (upload/URL).")
    st.stop()

if geo.get("type") != "FeatureCollection":
    st.error("File harus GeoJSON FeatureCollection.")
    st.stop()

# ================== PARSE ==================
props_df = pd.DataFrame([f.get("properties", {}) for f in geo.get("features", []) if isinstance(f, dict)])
prov_col = next((c for c in ["provinsi","ADM_PROV","PROVINSI","Provinsi","PROV","WADMPR","NAMPROV","NAMA_PROP","NAMA_PROV"]
                 if c in props_df.columns), None)
risk_long = risk_geojson_to_long(geo, prov_col)

if risk_long.empty:
    st.error("Tidak ditemukan kolom risk_*_YYYY-MM pada properti.")
    st.stop()

# ================== SIDEBAR: FILTER & STYLE ==================
# ================== SIDEBAR: FILTER & STYLE ==================
with st.sidebar:
    st.header("Filter")
    prov_opts = sorted(risk_long["prov"].dropna().astype(str).unique().tolist())
    sel_prov  = st.multiselect("Pilih provinsi", prov_opts, default=[])

    metric_opts = sorted(risk_long["metric"].unique().tolist())
    sel_metric  = st.selectbox("Metric", metric_opts,
                               index=metric_opts.index("risk_intensity_mean") if "risk_intensity_mean" in metric_opts else 0)
    month_opts  = sorted(risk_long["month"].unique().tolist())
    sel_month   = st.selectbox("Bulan (YYYY-MM)", month_opts, index=len(month_opts)-1)

    st.header("Simbolisasi")
    radius_px = st.slider("Radius titik (px)", 1, 15, 3)
    opacity   = st.slider("Opasitas titik", 0.1, 1.0, 0.8, 0.05)

    st.header("Overlay batas")
    ov_mode = st.radio("Sumber batas:", ["Tidak ada","Path lokal","URL","Upload"], index=1)
    line_w  = st.slider("Tebal garis", 0.5, 4.0, 1.2, 0.1)
    line_a  = st.slider("Opacity garis", 0.1, 1.0, 0.7, 0.05)
    boundary_layer = None

    if ov_mode == "Path lokal":
        # default path kamu — boleh diedit di UI
        default_path = r"C:\Users\Evi Ardiyani\Documents\BMKG\CPNS 2025\Kesesuaian Tanaman\Fix Data\38 Provinsi Indonesia - Provinsi.json"
        local_path = st.text_input("Path GeoJSON batas (lokal)", value=default_path)
        if local_path:
            try:
                bnd_geo = load_geojson_path(local_path)
                boundary_layer = build_boundary_layer(bnd_geo, line_w, line_a)
            except Exception as e:
                st.warning(f"Gagal baca GeoJSON batas: {e}")

    elif ov_mode == "URL":
        ov_url = st.text_input("URL GeoJSON batas", "")
        if ov_url:
            boundary_layer = build_boundary_layer(ov_url, line_w, line_a)

    elif ov_mode == "Upload":
        bup = st.file_uploader("Upload GeoJSON batas", type=["geojson","json"], key="bnd_gj")
        if bup:
            try:
                bnd_geo = json.loads(bup.getvalue().decode("utf-8"))
                boundary_layer = build_boundary_layer(bnd_geo, line_w, line_a)
            except Exception as e:
                st.warning(f"Gagal baca upload GeoJSON batas: {e}")

# ================== APPLY FILTER ==================
q = (risk_long["metric"].eq(sel_metric) & risk_long["month"].eq(sel_month))
if sel_prov:
    q &= risk_long["prov"].astype(str).isin(sel_prov)
view = risk_long.loc[q].copy()
if view.empty:
    st.warning("Tidak ada titik untuk kombinasi filter.")
    st.stop()

# sampling opsional agar ringan
with st.sidebar:
    max_points = st.slider("Batas titik", 1000, 200000, min(80000, len(view)), 1000)
if len(view) > max_points:
    view = view.sample(max_points, random_state=42).reset_index(drop=True)
    
# ================== KATEGORI BERDASAR NILAI ==================
# Ambang bisa dijadikan kontrol sidebar jika perlu
TH_LOW, TH_HIGH = 0.33, 0.66  # 0..1 skala risiko

def kategori_from_value(x: float) -> str:
    if not np.isfinite(x): 
        return "Tidak ada data"
    if x < TH_LOW:
        return "Layak Tanam"
    if x < TH_HIGH:
        return "Cukup Berisiko"
    return "Risiko Tinggi"

view["kategori"] = view["value"].map(kategori_from_value)

# ================== COLOR SCALE ==================
vals = view["value"].to_numpy(float)
vals = vals[np.isfinite(vals)]
vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if vals.size else (0.0, 1.0)
if np.isclose(vmin, vmax): vmax = vmin + 1e-6
to_color = make_color_mapper(vmin, vmax, opacity)
view["color"] = view["value"].map(to_color)

# ================== VIEWSTATE ==================
center, zoom, span = compute_center_zoom(view.rename(columns={"lon":"lon","lat":"lat"}))
if sel_prov and len(sel_prov) == 1:
    zoom = float(np.clip(zoom + 0.4, 3, 18))
view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom)

# ================== MAP ==================
risk_layer = pdk.Layer(
    "ScatterplotLayer",
    view,
    get_position="[lon, lat]",
    get_fill_color="color",
    pickable=True,
    radius_min_pixels=radius_px,
    radius_max_pixels=radius_px,
    stroked=False,
)
layers = [risk_layer] + ([boundary_layer] if boundary_layer is not None else [])

tooltip = {
    "html": (
        "<div>"
        "<div style='font-weight:700; font-size:14px;'>{kategori}</div>"
        f"<b>Metric</b>: {sel_metric}<br/>"
        f"<b>Bulan</b>: {sel_month}<br/>"
        "<b>Provinsi</b>: {prov}<br/>"
        "<b>Kab/Kota</b>: {kabkota}<br/>"
        "<b>Nilai</b>: {value}"
        "</div>"
    ),
    "style": {"backgroundColor": "white", "color": "black"},
}


st.subheader("Peta Risiko")
deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)

# Legend warna hijau→merah di bawah peta
st.markdown(render_continuous_legend(vmin, vmax, metric_label=sel_metric), unsafe_allow_html=True)
st.caption(f"Titik: {len(view):,} | Provinsi: {'; '.join(sel_prov) if sel_prov else 'Seluruh Indonesia'}")

# ================== RINGKASAN (Top 10) ==================
st.divider()

# Kumpulkan semua metric untuk bulan terpilih (+ filter provinsi) agar tabel lengkap
month_q = risk_long["month"].eq(sel_month)
if sel_prov:
    month_q &= risk_long["prov"].astype(str).isin(sel_prov)
month_df = risk_long.loc[month_q].copy()

# rata-rata risk_intensity_mean + jumlah titik
df_int = (month_df[month_df["metric"]=="risk_intensity_mean"]
          .groupby(["prov","kabkota"], as_index=False)
          .agg(n_points=("lon","count"), mean_metric=("value","mean")))

# total risk_days_expected (jika ada)
df_days = (month_df[month_df["metric"]=="risk_days_expected"]
           .groupby(["prov","kabkota"], as_index=False)
           .agg(tot_risk_days=("value","sum")))

# p95 risk_peak (jika ada)
df_peak = (month_df[month_df["metric"]=="risk_peak"]
           .groupby(["prov","kabkota"], as_index=False)
           .agg(peak_p95=("value", p95)))

# gabung
agg = (df_int.merge(df_days, on=["prov","kabkota"], how="left")
             .merge(df_peak, on=["prov","kabkota"], how="left"))

for c in ["mean_metric", "tot_risk_days", "peak_p95"]:
    if c in agg.columns:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")

top_high = agg.sort_values("mean_metric", ascending=False).head(10).reset_index(drop=True)
top_low  = agg.sort_values("mean_metric", ascending=True).head(10).reset_index(drop=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**== 10 Kab/Kota Risiko Tertinggi ({sel_month}) ==**")
    st.dataframe(top_high, use_container_width=True, height=420)
with c2:
    st.markdown(f"**== 10 Kab/Kota Risiko Terendah/Rekomendasi Tanam ({sel_month}) ==**")
    st.dataframe(top_low, use_container_width=True, height=420)

# ================== UNDUH TERFILTER ==================
st.divider()
st.markdown("**Unduh GeoJSON (terfilter)**")

def build_filtered_fc(orig_geo: dict, subset_df: pd.DataFrame) -> dict:
    key = set(zip(np.round(subset_df["lon"], 6), np.round(subset_df["lat"], 6)))
    feats_out = []
    for f in orig_geo.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") != "Point": continue
        coords = g.get("coordinates") or []
        if len(coords) < 2: continue
        k = (round(float(coords[0]), 6), round(float(coords[1]), 6))
        if k in key:
            feats_out.append(f)
    return {"type":"FeatureCollection", "features":feats_out}

filtered_fc = build_filtered_fc(geo, view)
st.download_button(
    "Download GeoJSON",
    data=json.dumps(filtered_fc, ensure_ascii=False).encode("utf-8"),
    file_name=f"risk_filtered_{sel_metric}_{sel_month}.geojson",
    mime="application/geo+json",
)
