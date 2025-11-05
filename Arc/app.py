# ================== IMPORT ==================
import os, io, zipfile, tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

import streamlit as st
try:
    import geopandas as gpd
except Exception:
    gpd = None  # aplikasi tetap bisa jalan tanpa geopandas

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

# ================== Upload data ==================
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload (parquet/csv)", type=["parquet","csv"])
if up is None:
    st.info("Upload file terlebih dulu."); st.stop()

fname = (getattr(up, "name", "") or "").lower()
if fname.endswith(".parquet"):
    try:
        # butuh pyarrow atau fastparquet
        df = pd.read_parquet(up)
    except Exception as e:
        st.error(f"Gagal baca parquet: {e}. Coba upload CSV."); st.stop()
elif fname.endswith(".csv"):
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Gagal baca CSV: {e}"); st.stop()
else:
    st.error("Hanya .parquet atau .csv."); st.stop()

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
st.title("Visualisasi Iklim & Kesesuaian")
opsi = st.selectbox(
    "Pilih visualisasi",
    [
        "1. Spatial CH tahunan",
        "2. Temporal jumlah CH bulanan (Indonesia)",
        "3. Spatial rata-rata temperatur",
        "4. Temporal temperatur bulanan",
        "5. Bulan kering (bar x=bulan)",
        "6. Spatial kemiringan (%)",
        "7. Kesesuaian Lahan: fuzzy vs crisp",
    ],
)

# ================== Plot switcher ==================
if opsi.startswith("1"):
    fig = plot_spatial_numeric(
        df, "CH_THN_calc", "Spatial Curah Hujan Tahunan (mm)",
        size_pt=size_pt, cbar_label="CH_THN_calc (mm)",
        xlim=xlim, ylim=ylim, cmap="RdYlBu", gdf_outline=gdf_batas, render=True
    )
    download_fig(fig, "CH_tahunan.png")

elif opsi.startswith("2"):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(bulan, ch_sum_id.values, marker="o")
    ax.set_title("Jumlah CH Bulanan (total Indonesia)")
    ax.set_xlabel("Bulan"); ax.set_ylabel("mm"); ax.grid(True, alpha=0.3)
    st.pyplot(fig); download_fig(fig, "CH_bulanan_total.png")

elif opsi.startswith("3"):
    if not has_t2m:
        st.error("Kolom T2M_Jan..T2M_Des tidak lengkap.")
    else:
        fig = plot_spatial_numeric(
            df, "T2M_mean", "Spatial Rata-rata Temperatur (°C)",
            size_pt=size_pt, cbar_label="T2M_mean (°C)",
            xlim=xlim, ylim=ylim, cmap="coolwarm", gdf_outline=gdf_batas, render=True
        )
        download_fig(fig, "T2M_spasial.png")

elif opsi.startswith("4"):
    if not has_t2m:
        st.error("Kolom T2M_Jan..T2M_Des tidak lengkap.")
    else:
        fig, ax = plt.subplots(figsize=(9,3))
        ax.plot(bulan, t2m_mean_id.values, marker="o")
        ax.set_title("Rata-rata Temperatur Bulanan (Indonesia)")
        ax.set_xlabel("Bulan"); ax.set_ylabel("°C"); ax.grid(True, alpha=0.3)
        y_min = float(np.nanmin(t2m_mean_id.values))
        y_max = float(np.nanmax(t2m_mean_id.values))
        y0 = np.floor(y_min*10)/10.0
        y1 = np.ceil(y_max*10)/10.0
        ax.set_ylim(y0, y1)
        ax.set_yticks(np.arange(y0, y1 + 0.0001, 0.1))
        st.pyplot(fig); download_fig(fig, "T2M_temporal.png")

elif opsi.startswith("5"):
    kering_counts = [int((df[f"CH_{b}"] < ambang).sum()) for b in bulan]
    fig, ax = plt.subplots(figsize=(9,3))
    ax.bar(bulan, kering_counts)
    ax.set_title(f"Jumlah Grid Bulan Kering (< {ambang} mm)")
    ax.set_xlabel("Bulan"); ax.set_ylabel("Jumlah grid")
    st.pyplot(fig); download_fig(fig, "Bulan_kering.png")

elif opsi.startswith("6"):
    if "slope_percent" not in df.columns:
        st.error("Kolom slope_percent tidak ada.")
    else:
        fig = plot_spatial_numeric(
            df, "slope_percent", "Spatial Kemiringan (%)",
            size_pt=size_pt, cbar_label="slope_percent (%)",
            vmin=0, vmax=max_slope, xlim=xlim, ylim=ylim, gdf_outline=gdf_batas, render=True
        )
        download_fig(fig, "Kemiringan_spasial.png")

elif opsi.startswith("7"):
    if "kelas" not in df.columns:
        st.error("Kolom 'kelas' tidak ada.")
    elif "kelas_crisp" not in df.columns:
        st.error("Kolom 'kelas_crisp' tidak ada.")
    else:
        left, mid, right = st.columns([1, 6, 1])
        with mid:
            st.markdown(
                "<h4 style='text-align:center;'>PETA KESESUAIAN LAHAN TANAMAN KELAPA SAWIT</h4>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div style="text-align:center; margin:-6px 0 6px 0;">
                  <span style="display:inline-block;width:12px;height:12px;background:#1b9e77;border-radius:50%;margin-right:6px;"></span>S1
                  &nbsp;&nbsp;&nbsp;
                  <span style="display:inline-block;width:12px;height:12px;background:#1f77b4;border-radius:50%;margin-right:6px;"></span>S2
                  &nbsp;&nbsp;&nbsp;
                  <span style="display:inline-block;width:12px;height:12px;background:#d62728;border-radius:50%;margin-right:6px;"></span>S3
                  &nbsp;&nbsp;&nbsp;
                  <span style="display:inline-block;width:12px;height:12px;background:#ffffff;border:1px solid #999;border-radius:50%;margin-right:6px;"></span>N
                </div>
                """,
                unsafe_allow_html=True
            )
            f1, f2 = plot_kesesuaian_two_figs(
                df, size_pt=size_pt, xlim=xlim, ylim=ylim, gdf_outline=gdf_batas, render=True
            )
            cdl1, cdl2 = st.columns(2)
            with cdl1: download_fig(f1, "Kesesuaian_Fuzzy.png")
            with cdl2: download_fig(f2, "Kesesuaian_Crisp.png")
