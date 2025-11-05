# ================== IMPORT ==================
import os, io, zipfile, tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

import streamlit as st
import pydeck as pdk
try:
    import geopandas as gpd
except Exception:
    gpd = None  # aplikasi tetap bisa jalan tanpa geopandas

# Color mapping for classifications with more vibrant colors
COLOR_MAPPING = {
    'S1': [46/255, 204/255, 113/255],    # Emerald green
    'S2': [52/255, 152/255, 219/255],    # Bright blue
    'S3': [231/255, 76/255, 60/255],     # Bright red
    'N':  [189/255, 195/255, 199/255]    # Light grey
}

# ================== CONFIG (WAJIB PALING ATAS) ==================
st.set_page_config(page_title="Visualisasi Iklim & Kesesuaian", layout="wide")

# ================== HEADER (logo tengah, tak kepotong) ==================
logo_url = "https://lms.bmkg.go.id/pluginfile.php/1/theme_mb2nl/logo/1725958993/logo%20lms.png"
st.markdown("""
<style>
.block-container{padding-top:3rem;}
h2,h3{margin:0.6rem 0;}
#MainMenu {visibility: visible;}
.stDeployButton {display:none;}
header {visibility: visible;}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;">
  <img src="{logo_url}" alt="BMKG" style="max-height:90px; display:block; margin:0 auto;"/>
  <h2>Aktualisasi CPNS Golongan IIIA tahun 2025</h2>
  <h3>Direktorat Layanan Iklim Terapan</h3>
  <h3>"PETA KESESUAIAN LAHAN DAN RISIKO IKLIM UNTUK SEKTOR PERKEBUNAN"</h3>
</div>
<hr/>
""", unsafe_allow_html=True)

# ================== Upload data (GeoJSON / Geo files) ==================
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload GeoJSON / Geo file", type=["geojson","json","zip"])
if up is None:
    st.info("Upload file GeoJSON terlebih dulu."); st.stop()

if gpd is None:
    st.error("GeoPandas diperlukan untuk membaca GeoJSON/SHAPE. Pasang paket 'geopandas' terlebih dahulu."); st.stop()

try:
    # Handle ZIP containing shapefile or geojson
    name = (getattr(up, "name", "") or "").lower()
    if name.endswith('.zip'):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(up, 'r') as zf:
            zf.extractall(tmpdir)
        # Prefer .shp if present, else .geojson/.json
        shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith('.shp')]
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
        else:
            geo_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(('.geojson','.json'))]
            if not geo_files:
                raise RuntimeError('Tidak ada file .shp atau .geojson di dalam zip')
            gdf = gpd.read_file(geo_files[0])
    else:
        # Directly read GeoJSON/JSON
        gdf = gpd.read_file(up)

    # Ensure geographic CRS (lat/lon)
    if getattr(gdf, 'crs', None) is not None:
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            # If conversion fails, continue but warn
            st.warning('Gagal konversi CRS ke EPSG:4326; asumsikan data sudah dalam lat/lon')

    # Make a copy and ensure LON/LAT columns exist (extract from geometry if missing)
    df_tmp = gdf.copy()
    if 'LON' not in df_tmp.columns or 'LAT' not in df_tmp.columns:
        if hasattr(df_tmp, 'geometry'):
            try:
                df_tmp['LON'] = df_tmp.geometry.x
                df_tmp['LAT'] = df_tmp.geometry.y
            except Exception as e:
                st.error(f"Gagal mengekstrak koordinat dari geometry: {e}"); st.stop()
        else:
            st.error('Kolom LON dan LAT wajib ada, dan geometry tidak tersedia untuk diekstrak.'); st.stop()

    # Drop geometry column to produce a plain DataFrame for downstream code
    if 'geometry' in df_tmp.columns:
        df_tmp = pd.DataFrame(df_tmp.drop(columns='geometry'))
    df = df_tmp
except Exception as e:
    st.error(f"Gagal membaca GeoJSON/SHAPE: {e}"); st.stop()

# ================== Persiapan kolom ==================
bulan = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
CH_cols  = [f"CH_{b}"  for b in bulan]
T2M_cols = [f"T2M_{b}" for b in bulan]

if "LON" not in df.columns or "LAT" not in df.columns:
    st.error("Kolom LON dan LAT wajib ada."); st.stop()

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["LON","LAT"])

if not set(CH_cols).issubset(df.columns):
    st.error("Kolom CH_Jan..CH_Des tidak lengkap."); st.stop()

df["CH_THN_calc"] = df[CH_cols].sum(axis=1, skipna=True)
ch_sum_id = df[CH_cols].sum(axis=0, skipna=True); ch_sum_id.index = bulan

has_t2m = set(T2M_cols).issubset(df.columns)
if has_t2m:
    df["T2M_mean"] = df[T2M_cols].mean(axis=1, skipna=True)
    t2m_mean_id = df[T2M_cols].mean(axis=0, skipna=True); t2m_mean_id.index = bulan

# Kategori kesesuaian
CAT_ORDER = ["S1","S2","S3","N"]
CAT_CMAP  = ListedColormap(["#1b9e77", "#1f77b4", "#d62728", "#ffffff"])  # hijau, biru, merah, putih

def _prep_cat(dfin, col):
    d = dfin[["LON","LAT",col]].dropna().copy()
    lab2id = {k:i for i,k in enumerate(CAT_ORDER)}
    d["_id"] = d[col].map(lab2id)
    return d[~d["_id"].isna()]

# ================== Sidebar controls ==================
st.sidebar.header("Pengaturan")
ambang   = st.sidebar.number_input("Ambang bulan kering (mm)", 0.0, 500.0, 60.0, 5.0)
size_pt  = st.sidebar.slider("Ukuran titik", 1, 20, 2)
max_slope= st.sidebar.slider("Batas maksimum slope (%)", 50, 300, 125, 5)

# Batas wilayah (opsional)
st.sidebar.subheader("Batas Wilayah (opsional)")
up_batas = st.sidebar.file_uploader("Upload SHP (.zip) atau GeoJSON", type=["zip","geojson","json"])
gdf_batas = None
if up_batas is not None:
    if gpd is None:
        st.warning("GeoPandas belum terpasang. Lewati batas wilayah atau pasang paketnya terlebih dulu.")
    else:
        try:
            if up_batas.name.lower().endswith(".zip"):
                tmpdir = tempfile.mkdtemp()
                with zipfile.ZipFile(up_batas, "r") as zf:
                    zf.extractall(tmpdir)
                shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
                if not shp_files:
                    raise RuntimeError("File .shp tidak ditemukan dalam .zip")
                gdf_batas = gpd.read_file(shp_files[0])
            else:
                gdf_batas = gpd.read_file(up_batas)

            if gdf_batas is not None and gdf_batas.crs is not None:
                gdf_batas = gdf_batas.to_crs(epsg=4326)
        except Exception as e:
            st.warning(f"Gagal membaca batas wilayah: {e}")
            gdf_batas = None

# Zoom terpusat
lon_min, lon_max = float(df["LON"].min()), float(df["LON"].max())
lat_min, lat_max = float(df["LAT"].min()), float(df["LAT"].max())
st.sidebar.subheader("Zoom")
zoom_pct = st.sidebar.slider("Zoom (%)", 50, 200, 100, 5)  # 100 = penuh
center_lon = (lon_min + lon_max) / 2
center_lat = (lat_min + lat_max) / 2
zf = 100.0 / zoom_pct
half_lon = (lon_max - lon_min) * 0.5 * zf
half_lat = (lat_max - lat_min) * 0.5 * zf
xlim = (center_lon - half_lon, center_lon + half_lon)
ylim = (center_lat - half_lat, center_lat + half_lat)

# ================== Helpers ==================
def create_pydeck_layer(data, column_name, tooltip=None):
    """Create a pydeck scatter plot layer with colors based on classification"""
    data = data.copy()
    
    # Convert classification colors to RGB arrays
    def get_color(row):
        cls = row[column_name]
        if cls in COLOR_MAPPING:
            return [int(c * 255) for c in COLOR_MAPPING[cls]]
        return [128, 128, 128]  # Default gray for unknown values
    
    # Apply color mapping to each row
    data['fill_color'] = data.apply(get_color, axis=1)
    
    return pdk.Layer(
        'ScatterplotLayer',
        data,
        get_position=['LON', 'LAT'],
        get_fill_color='fill_color',
        get_radius=size_pt * 50,
        pickable=True,
        opacity=0.8,
        filled=True,
    )

def plot_temporal_data(df_prov, title):
    """Plot temporal data for temperature and rainfall"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot rainfall
    ax1.plot(bulan, [df_prov[f'CH_{b}'].mean() for b in bulan], marker='o', color='blue')
    ax1.set_title(f'Curah Hujan Bulanan - {title}')
    ax1.set_ylabel('mm')
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature if available
    if has_t2m:
        ax2.plot(bulan, [df_prov[f'T2M_{b}'].mean() for b in bulan], marker='o', color='red')
        ax2.set_title(f'Temperatur Bulanan - {title}')
        ax2.set_ylabel('°C')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_spatial_numeric(dfnum, col, title, size_pt=6, cbar_label=None,
                         vmin=None, vmax=None, xlim=None, ylim=None, cmap=None,
                         gdf_outline=None, render=True):
    fig, ax = plt.subplots(figsize=(5.2, 2.4), dpi=140)
    sc = ax.scatter(dfnum["LON"], dfnum["LAT"], c=dfnum[col], s=size_pt,
                    vmin=vmin, vmax=vmax, cmap=cmap)
    if gdf_outline is not None and hasattr(gdf_outline, "empty") and not gdf_outline.empty:
        gdf_outline.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
    cb = fig.colorbar(sc); cb.set_label(cbar_label or col, fontsize=8)
    ax.set_xlabel("LON", fontsize=8); ax.set_ylabel("LAT", fontsize=8)
    ax.set_title(title, fontsize=10, pad=2)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box"); ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout(pad=0.4)
    if render: st.pyplot(fig, use_container_width=False)
    return fig

def plot_kesesuaian_two_figs(dfin, size_pt=2, figsize=(4.3, 1.9), dpi=140,
                             xlim=None, ylim=None, gdf_outline=None, render=True):
    d1 = _prep_cat(dfin, "kelas"); d2 = _prep_cat(dfin, "kelas_crisp")
    if d1.empty or d2.empty:
        st.error("Nilai pada kolom 'kelas' atau 'kelas_crisp' kosong semua.")
        return None, None
    if xlim is None:
        xlim = (min(d1["LON"].min(), d2["LON"].min()),
                max(d1["LON"].max(), d2["LON"].max()))
    if ylim is None:
        ylim = (min(d1["LAT"].min(), d2["LAT"].min()),
                max(d1["LAT"].max(), d2["LAT"].max()))
    def _one(d, title):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(d["LON"], d["LAT"], c=d["_id"], s=size_pt,
                   cmap=CAT_CMAP, vmin=0, vmax=len(CAT_ORDER)-0.001)
        if gdf_outline is not None and hasattr(gdf_outline, "empty") and not gdf_outline.empty:
            gdf_outline.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("LON", fontsize=7); ax.set_ylabel("LAT", fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis="both", labelsize=7, length=3)
        ax.set_aspect("equal", adjustable="box"); ax.margins(0)
        fig.tight_layout(pad=0.4)
        return fig
    f1 = _one(d1, "Fuzzy"); f2 = _one(d2, "Crisp")
    if render:
        st.pyplot(f1, use_container_width=False)
        st.pyplot(f2, use_container_width=False)
    return f1, f2

def download_fig(fig, filename):
    if fig is None: 
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download PNG", data=buf.getvalue(),
                       file_name=filename, mime="image/png")

# ================== UI ==================
st.title("Visualisasi Kesesuaian Lahan Interaktif")

# Select crop type
crop_type = st.selectbox(
    "Pilih Jenis Tanaman",
    ["Pilih jenis tanaman", "Kelapa Sawit", "Tebu"],
    index=0  # Set default to "Pilih jenis tanaman"
)

if crop_type == "Pilih jenis tanaman":
    st.info("Silakan pilih jenis tanaman terlebih dahulu")
    st.stop()

# Select classification type
class_type = st.radio("Pilih Jenis Klasifikasi", ["Fuzzy", "Crisp"])

# Show loading state
with st.spinner(f'Mempersiapkan visualisasi kesesuaian lahan {crop_type.lower()}...'):
    # ================== Plot Interactive Map ==================
    # Determine which classification column to use
    if crop_type == "Kelapa Sawit":
        col_name = "kelas" if class_type == "Fuzzy" else "kelas_crisp"
        title = f"Kesesuaian Lahan Kelapa Sawit ({class_type})"
    else:
        col_name = "kelas_tebu" if class_type == "Fuzzy" else "kelas_tebu_crisp"
        title = f"Kesesuaian Lahan Tebu ({class_type})"

# Create tooltip
tooltip = {
    "html": "<b>Lokasi:</b> {ID_PROV38}<br/>"
            "<b>Koordinat:</b> ({LON:.2f}, {LAT:.2f})<br/>"
            f"<b>Kelas:</b> {{{col_name}}}<br/>"
            "<b>CH Tahunan:</b> {CH_THN_calc:.0f} mm<br/>"
            "<b>Klik untuk detail temporal</b>",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

# Create scatter plot layer with increased size and opacity
scatter_layer = create_pydeck_layer(df, col_name, tooltip)
scatter_layer.get_radius = size_pt * 100  # Make points bigger
scatter_layer.opacity = 1.0  # Make points fully opaque

view_state = pdk.ViewState(
    longitude=center_lon,
    latitude=center_lat,
    zoom=5,
    pitch=0,
)

deck = pdk.Deck(
    layers=[scatter_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style='road'  # Simple style that doesn't require token
)

# Show the map
st.pydeck_chart(deck)

# Legend with styled box
st.markdown("""
<div style="background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h4 style="text-align:center; margin:0 0 10px 0;">Klasifikasi Kesesuaian</h4>
    <div style="text-align:center;">
        <div style="display:inline-block; margin:0 15px;">
            <span style="display:inline-block;width:15px;height:15px;background:#2ecc71;border-radius:50%;margin-right:6px;"></span>
            <span style="vertical-align:middle;">S1 - Sangat Sesuai</span>
        </div>
        <div style="display:inline-block; margin:0 15px;">
            <span style="display:inline-block;width:15px;height:15px;background:#3498db;border-radius:50%;margin-right:6px;"></span>
            <span style="vertical-align:middle;">S2 - Cukup Sesuai</span>
        </div>
        <div style="display:inline-block; margin:0 15px;">
            <span style="display:inline-block;width:15px;height:15px;background:#e74c3c;border-radius:50%;margin-right:6px;"></span>
            <span style="vertical-align:middle;">S3 - Sesuai Marginal</span>
        </div>
        <div style="display:inline-block; margin:0 15px;">
            <span style="display:inline-block;width:15px;height:15px;background:#bdc3c7;border-radius:50%;margin-right:6px;"></span>
            <span style="vertical-align:middle;">N - Tidak Sesuai</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Handle click events
if st.session_state.get('last_clicked_point'):
    point = st.session_state.last_clicked_point
    prov = point.get('ID_PROV38', 'Unknown')
    
    # Filter data for the selected province
    df_prov = df[df['ID_PROV38'] == prov].copy()
    
    if not df_prov.empty:
        fig = plot_temporal_data(df_prov, prov)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected location")


