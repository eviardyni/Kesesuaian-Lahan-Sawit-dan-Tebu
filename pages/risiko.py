# 2_Risiko.py — Peta Risiko + Konsesi (Upload-only) + Tooltip Gabungan
import json, math, re, hashlib, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# =============== CONFIG ===============
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

# =============== HELPERS I/O ===============
@st.cache_data(show_spinner=False)
def load_geojson_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8"))

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> dict:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

def to_num(x):
    try: return float(x)
    except Exception: return np.nan

# =============== HELPERS DATA ===============
@st.cache_data(show_spinner=False)
def risk_geojson_to_long(geo_fc: dict, prov_key_guess: str|None=None) -> pd.DataFrame:
    feats = geo_fc.get("features", [])
    rows = []
    prov_candidates = ["provinsi","ADM_PROV","PROVINSI","Provinsi","PROV","WADMPR","NAMPROV","NAMA_PROP","NAMA_PROV"]
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
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

def make_color_mapper(vmin, vmax, opacity):
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
    <div style="height:14px; width:100%; max-width:520px; border:1px solid #999; border-radius:3px;
      background: linear-gradient(to right,
        rgb(26,150,65), rgb(166,217,106), rgb(255,255,191), rgb(253,174,97), rgb(215,25,28));">
    </div>
    <div style="display:flex; justify-content:space-between; width:100%; max-width:520px; font:12px system-ui; color:#333;">
      <span>{vmin:.4g}</span><span>{vmax:.4g}</span>
    </div>
    """

def p95(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 95)) if x.size else np.nan

def build_boundary_layer(source_geojson_or_url, line_width_px: float, alpha: float):
    rgba = [0,0,0, int(alpha*255)]
    return pdk.Layer("GeoJsonLayer", source_geojson_or_url,
        stroked=True, filled=False, pickable=False,
        get_line_color=rgba, get_line_width=1,
        lineWidthUnits="pixels", lineWidthScale=1,
        lineWidthMinPixels=int(math.ceil(line_width_px))
    )

@st.cache_data(show_spinner=False)
def decorate_concession_features(fc: dict, color_key: str, fill_opacity: float) -> dict:
    """Tambahkan __fill__ & __line__ (RGBA) untuk styling pydeck."""
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        return fc
    feats = []
    for f in fc.get("features", []):
        if not isinstance(f, dict): continue
        props = dict(f.get("properties") or {})
        keyval = str(props.get(color_key, "N/A")) if color_key != "uniform" else "ALL"
        h = hashlib.md5(keyval.encode("utf-8")).hexdigest()
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        r = int(120 + 0.5*(r % 136)); g = int(120 + 0.5*(g % 136)); b = int(120 + 0.5*(b % 136))
        a = int(np.clip(fill_opacity, 0, 1) * 255)
        props["__fill__"] = [r, g, b, a]
        props["__line__"] = [max(0, r-40), max(0, g-40), max(0, b-40), 255]
        # rapikan numeric
        if "area_ha" in props:
            try: props["area_ha"] = float(props["area_ha"])
            except Exception: pass
        feats.append({"type": "Feature", "properties": props, "geometry": f.get("geometry")})
    return {"type": "FeatureCollection", "features": feats}

# ---------- Point in Polygon (tanpa shapely), dengan indeks grid 1° ----------
def _ring_contains_point(ring, x, y):
    # ray casting (lon=x, lat=y); ring: list[[x,y],...]
    inside = False
    n = len(ring)
    if n < 3: return False
    for i in range(n):
        x1,y1 = ring[i][0], ring[i][1]
        x2,y2 = ring[(i+1) % n][0], ring[(i+1) % n][1]
        # cek jika y di antara y1..y2 dan sisi melintas ke kanan titik
        cond = ((y1 > y) != (y2 > y)) and (x < (x2-x1) * (y - y1) / (y2 - y1 + 1e-30) + x1)
        if cond: inside = not inside
    return inside

def _polygon_contains_point(poly_coords, x, y):
    # poly_coords: [ring_exterior, hole1, hole2, ...]
    if not poly_coords: return False
    if not _ring_contains_point(poly_coords[0], x, y):
        return False
    # jika ada hole dan titik jatuh di hole → dianggap di luar
    for hole in poly_coords[1:]:
        if _ring_contains_point(hole, x, y):
            return False
    return True

def _shoelace_area(coords):
    # luas poligon (derajat^2; hanya untuk ranking), absolut
    a = 0.0
    n = len(coords)
    for i in range(n):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1) % n]
        a += x1*y2 - x2*y1
    return abs(a) * 0.5

@st.cache_data(show_spinner=False)
def build_polygon_index(ks_geo: dict, tile_size_deg: float = 1.0):
    """Parse GeoJSON Polygon/MultiPolygon → list poly + grid index sederhana."""
    polys = []  # [{rings: [...], props: {...}, bbox:(minx,miny,maxx,maxy), area_rank:float}, ...]
    feats = ks_geo.get("features", [])
    for f in feats:
        g = f.get("geometry") or {}
        t = g.get("type")
        if t not in ("Polygon","MultiPolygon"): 
            continue
        props = dict(f.get("properties") or {})
        if t == "Polygon":
            geoms = [g.get("coordinates") or []]
        else:
            geoms = g.get("coordinates") or []

        for rings in geoms:
            # pastikan ring ada
            if not rings or not rings[0]: 
                continue
            # bbox
            xs = list(itertools.chain.from_iterable([[p[0] for p in ring] for ring in rings]))
            ys = list(itertools.chain.from_iterable([[p[1] for p in ring] for ring in rings]))
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            # area untuk ranking (pakai exterior saja)
            area_rank = _shoelace_area(rings[0])
            polys.append({"rings": rings, "props": props, "bbox": (minx,miny,maxx,maxy), "area_rank": area_rank})

    # grid index
    grid = {}  # (ix,iy) -> list idx polys
    def _key(x, y):
        return (int(math.floor(x / tile_size_deg)),
                int(math.floor(y / tile_size_deg)))
    for i, p in enumerate(polys):
        minx,miny,maxx,maxy = p["bbox"]
        ix0, ix1 = _key(minx,0)[0], _key(maxx,0)[0]
        iy0, iy1 = _key(0,miny)[1], _key(0,maxy)[1]
        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                grid.setdefault((ix,iy), []).append(i)
    return polys, grid, tile_size_deg

def point_lookup_concession(polys, grid, tile_size_deg, x, y):
    """Cari poligon yang memuat titik (x=lon,y=lat). Pilih dengan area_rank terbesar."""
    ix = int(math.floor(x / tile_size_deg))
    iy = int(math.floor(y / tile_size_deg))
    cand = []
    # tile tetangga (3x3) agar aman di tepi
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            cand.extend(grid.get((ix+dx, iy+dy), []))
    if not cand: 
        return None
    best = None
    best_area = -1.0
    for idx in cand:
        p = polys[idx]
        minx,miny,maxx,maxy = p["bbox"]
        if not (minx <= x <= maxx and miny <= y <= maxy):
            continue
        if _polygon_contains_point(p["rings"], x, y):
            if p["area_rank"] > best_area:
                best = p
                best_area = p["area_rank"]
    return best

# =============== SIDEBAR: TITIK RISIKO (UPLOAD) ===============
with st.sidebar:
    st.header("Data Titik Risiko Tanam (Upload)")
    up_pts = st.file_uploader("Upload GeoJSON titik (.geojson/.json)", type=["geojson","json"], key="risk_pts")
    geo = load_geojson_bytes(up_pts.getvalue()) if up_pts else None

if not geo:
    st.info("Unggah GeoJSON titik risiko dahulu.")
    st.stop()

if geo.get("type") != "FeatureCollection":
    st.error("File titik harus GeoJSON FeatureCollection.")
    st.stop()

# =============== PARSE TITIK ===============
props_df = pd.DataFrame([f.get("properties", {}) for f in geo.get("features", []) if isinstance(f, dict)])
prov_col = next((c for c in ["provinsi","ADM_PROV","PROVINSI","Provinsi","PROV","WADMPR","NAMPROV","NAMA_PROP","NAMA_PROV"]
                 if c in props_df.columns), None)
risk_long = risk_geojson_to_long(geo, prov_col)

if risk_long.empty:
    st.error("Tidak ditemukan kolom risk_*_YYYY-MM pada properti titik.")
    st.stop()

# =============== SIDEBAR: FILTER & STYLE ===============
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

# =============== SIDEBAR: OVERLAY BATAS (opsional) ===============
with st.sidebar:
    st.header("Overlay batas admin (opsional)")
    ov_mode = st.radio("Sumber batas:", ["Tidak ada","Upload","URL"], index=0)
    line_w  = st.slider("Tebal garis", 0.5, 4.0, 1.2, 0.1)
    line_a  = st.slider("Opacity garis", 0.1, 1.0, 0.7, 0.05)
    boundary_layer = None

    try:
        if ov_mode == "Upload":
            bup = st.file_uploader("Upload GeoJSON batas", type=["geojson","json"], key="bnd")
            if bup:
                bnd_geo = load_geojson_bytes(bup.getvalue())
                boundary_layer = build_boundary_layer(bnd_geo, line_w, line_a)
        elif ov_mode == "URL":
            ov_url = st.text_input("URL GeoJSON batas", "")
            if ov_url:
                boundary_layer = build_boundary_layer(ov_url, line_w, line_a)
    except Exception as e:
        st.warning(f"Gagal baca GeoJSON batas: {e}")

# =============== SIDEBAR: KONSESI (UPLOAD) ===============
with st.sidebar:
    st.header("Konsesi Sawit (poligon) — Upload")
    ks_up = st.file_uploader("Upload GeoJSON konsesi (.json/.geojson)", type=["geojson","json"], key="ks")
    ks_fill_op = st.slider("Opasitas isi konsesi", 0.05, 0.95, 0.35, 0.05)
    ks_line = st.slider("Tebal garis konsesi", 0.3, 4.0, 1.0, 0.1)
    ks_color_by = st.selectbox("Warna berdasarkan", ["group_comp","company","type","country","uniform"], index=0)
    ks_geo = None
    ks_layer = None
    polys = grid = None
    tile_size = 1.0
    try:
        if ks_up:
            ks_geo = load_geojson_bytes(ks_up.getvalue())
            # styling layer poligon (tidak pickable agar tooltip selalu dari titik)
            ks_geo_styled = decorate_concession_features(ks_geo, ks_color_by, ks_fill_op)
            ks_layer = pdk.Layer(
                "GeoJsonLayer",
                ks_geo_styled,
                stroked=True, filled=True, pickable=False,
                get_fill_color="properties.__fill__",
                get_line_color="properties.__line__",
                get_line_width=1,
                lineWidthUnits="pixels", lineWidthScale=1,
                lineWidthMinPixels=int(math.ceil(ks_line)),
            )
            polys, grid, tile_size = build_polygon_index(ks_geo)  # untuk join
    except Exception as e:
        st.warning(f"Gagal baca/siapkan GeoJSON konsesi: {e}")

# =============== FILTER TITIK ===============
q = (risk_long["metric"].eq(sel_metric) & risk_long["month"].eq(sel_month))
if sel_prov:
    q &= risk_long["prov"].astype(str).isin(sel_prov)
view = risk_long.loc[q].copy()
if view.empty:
    st.warning("Tidak ada titik untuk kombinasi filter.")
    st.stop()

with st.sidebar:
    max_points = st.slider("Batas titik (sampling)", 1000, 200000, min(80000, len(view)), 1000)
if len(view) > max_points:
    view = view.sample(max_points, random_state=42).reset_index(drop=True)

# =============== KATEGORI & WARNA TITIK ===============
TH_LOW, TH_HIGH = 0.33, 0.66
def kategori_from_value(x: float) -> str:
    if not np.isfinite(x): return "Tidak ada data"
    if x < TH_LOW: return "Layak Tanam"
    if x < TH_HIGH: return "Cukup Berisiko"
    return "Risiko Tinggi"

view["kategori"] = view["value"].map(kategori_from_value)

vals = view["value"].to_numpy(float)
vals = vals[np.isfinite(vals)]
vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if vals.size else (0.0, 1.0)
if np.isclose(vmin, vmax): vmax = vmin + 1e-6
to_color = make_color_mapper(vmin, vmax, opacity)
view["color"] = view["value"].map(to_color)

# =============== JOIN TITIK → POLIGON (opsional, jika konsesi diupload) ===============
def _fmt_area(v):
    if v is None or not np.isfinite(v): return "-"
    try:
        return f"{float(v):,.0f}".replace(",", ".")
    except Exception:
        return str(v)

if ks_geo and polys is not None:
    # siapkan kolom default "-"
    view["company"] = "-"
    view["group_comp"] = "-"
    view["area_ha"] = np.nan

    # loop ringan dengan indeks grid
    for i in range(len(view)):
        x = float(view.at[i, "lon"]); y = float(view.at[i, "lat"])
        hit = point_lookup_concession(polys, grid, tile_size, x, y)
        if hit is not None:
            props = hit["props"]
            comp = props.get("company") or props.get("name") or "-"
            grp  = props.get("group_comp") or "-"
            area = props.get("area_ha")
            if area is None:
                # fallback: estimasi kasar dari exterior (derajat^2 → tidak akurat area-nyata),
                # lebih baik pakai kolom area_ha asli jika tersedia.
                area = np.nan
            view.at[i, "company"] = str(comp)
            view.at[i, "group_comp"] = str(grp)
            view.at[i, "area_ha"] = to_num(area)

else:
    # tanpa konsesi, tetap sediakan kolom supaya tooltip konsisten
    view["company"] = "-"
    view["group_comp"] = "-"
    view["area_ha"] = np.nan

# format tampilan area jadi string
view["area_ha_fmt"] = view["area_ha"].map(_fmt_area)

# =============== VIEWSTATE ===============
center, zoom, span = compute_center_zoom(view)
if sel_prov and len(sel_prov) == 1:
    zoom = float(np.clip(zoom + 0.4, 3, 18))
view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom)

# =============== LAYERS ===============
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

layers = [risk_layer]
if ks_layer is not None:
    layers.append(ks_layer)
if boundary_layer is not None:
    layers.append(boundary_layer)

# =============== MAP (Tooltip gabungan dari layer titik) ===============
tooltip = {
    "html": (
        "<div style='font-size:12px'>"
        "<div style='font-weight:700; font-size:14px;'>{kategori}</div>"
        f"<b>Bulan</b>: {sel_month}<br/>"
        "<b>Provinsi</b>: {prov}<br/>"
        "<b>Kab Kota</b>: {kabkota}<br/>"
        "<hr style='margin:6px 0;'/>"
        "<b>Perusahaan Sawit</b>: {company}<br/>"
        "<b>Luas (Ha)</b>: {area_ha_fmt}"
        "</div>"
    ),
    "style": {"backgroundColor": "white", "color": "black"},
}

st.subheader("Peta Risiko")
deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)

# Legend
st.markdown(render_continuous_legend(vmin, vmax, metric_label=sel_metric), unsafe_allow_html=True)
st.caption(f"Titik: {len(view):,} | Provinsi: {'; '.join(sel_prov) if sel_prov else 'Seluruh Indonesia'}")

# =============== RINGKASAN (Top 10) ===============
st.divider()
month_q = risk_long["month"].eq(sel_month)
if sel_prov:
    month_q &= risk_long["prov"].astype(str).isin(sel_prov)
month_df = risk_long.loc[month_q].copy()

df_int = (month_df[month_df["metric"]=="risk_intensity_mean"]
          .groupby(["prov","kabkota"], as_index=False)
          .agg(n_points=("lon","count"), mean_metric=("value","mean")))

df_days = (month_df[month_df["metric"]=="risk_days_expected"]
           .groupby(["prov","kabkota"], as_index=False)
           .agg(tot_risk_days=("value","sum")))

df_peak = (month_df[month_df["metric"]=="risk_peak"]
           .groupby(["prov","kabkota"], as_index=False)
           .agg(peak_p95=("value", p95)))

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

# =============== UNDUH GEOJSON TERFILTER (TITIK) ===============
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
    return {"type": "FeatureCollection", "features": feats_out}

filtered_fc = build_filtered_fc(geo, view)
st.download_button(
    "Download GeoJSON",
    data=json.dumps(filtered_fc, ensure_ascii=False).encode("utf-8"),
    file_name=f"risk_filtered_{sel_metric}_{sel_month}.geojson",
    mime="application/geo+json",
)
