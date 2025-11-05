# ==========================================================
# KELAPA SAWIT & TEBU — KLASIFIKASI (FUZZY vs CRISP) + PETA + EKSPOR
# ==========================================================
# pip install pandas numpy matplotlib openpyxl xlsxwriter
# opsional peta batas: pip install geopandas shapely pyproj

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# KONFIG (SESUAIKAN PATH)
# -------------------------
PATH_GRID_XLS = Path(r"C:\Users\Evi Ardiyani\Documents\BMKG\CPNS 2025\DATA NORMAL DARI PAK ROBI\data ch t2m bk slope.xlsx")
SHEET_NAME    = 0
PATH_SHP_PROV = Path(r"C:\Users\Evi Ardiyani\Documents\BMKG\CPNS 2025\Kesesuaian Tanaman\R Script\DATA\INDONESIA\Indonesia_38_Provinsi.shp")  # boleh None
OUT_DIR       = Path(r"C:\output"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# (opsional) nama kolom provinsi di grid, jika mau ringkas per-prov
COL_PROV_GRID = "ID_PROV38"

# -------------------------
# UTIL DASAR
# -------------------------
def load_grid(path_xls: Path, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(path_xls, sheet_name=sheet_name, engine="openpyxl")
    need = {"LON","LAT"}
    if not need.issubset(df.columns):
        raise ValueError(f"Kolom wajib hilang: {need - set(df.columns)}")
    return df

def fuzzy_band(x, n_lo, s3_lo, s1_lo, s1_hi, s3_hi, n_hi):
    """Membership S1 berbentuk 'punggung' (0→1→0) dengan bahu S3 dan N."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    left = (x > n_lo) & (x < s1_lo); y[left] = (x[left]-n_lo)/(s1_lo-n_lo)
    mid  = (x >= s1_lo) & (x <= s1_hi); y[mid] = 1.0
    right= (x > s1_hi) & (x < n_hi);  y[right]= (n_hi - x[right])/(n_hi - s1_hi)
    return np.clip(y, 0, 1)

# -------------------------
# SAWIT: SCORING & KLASIFIKASI
# -------------------------
def score_temp(t):  # °C (mean)
    return fuzzy_band(t, n_lo=20, s3_lo=22, s1_lo=25, s1_hi=28, s3_hi=32, n_hi=35)

def score_ch(ch):   # mm/th
    return fuzzy_band(ch, n_lo=1250, s3_lo=1450, s1_lo=1700, s1_hi=2500, s3_hi=3500, n_hi=4000)

def score_bk(bk):   # bulan kering/tahun (lebih kecil lebih baik)
    x = np.asarray(bk, dtype=float)
    s = np.zeros_like(x)
    s[x <= 2.0] = 1.0
    m = (x > 2.0) & (x <= 3.0); s[m] = np.interp(x[m], [2.0,3.0], [1.0,0.7])
    m = (x > 3.0) & (x <= 4.0); s[m] = np.interp(x[m], [3.0,4.0], [0.7,0.4])
    m = x > 4.0;              s[m] = np.interp(np.minimum(x[m],5.0), [4.0,5.0], [0.4,0.0])
    return np.clip(s, 0, 1)

def score_slope(s): # persen kemiringan (lebih kecil lebih baik)
    x = np.asarray(s, dtype=float)
    v = np.zeros_like(x)
    v[x <= 8] = 1.0
    m = (x > 8) & (x <= 16); v[m] = np.interp(x[m], [8,16], [1.0,0.7])
    m = (x > 16) & (x <= 30);v[m] = np.interp(x[m], [16,30],[0.7,0.4])
    m = x > 30;              v[m] = np.interp(np.minimum(x[m],45.0), [30,45],[0.4,0.0])
    return np.clip(v, 0, 1)

def classify_fuzzy_crisp(df: pd.DataFrame) -> pd.DataFrame:
    """Sawit: tambah score_* , score_total, kelas (fuzzy), kelas_crisp."""
    t_cols = [c for c in df.columns if c.startswith("T2M_")]
    temp_mean = df[t_cols].mean(axis=1) if t_cols else np.nan

    if "CH_THN" in df.columns:
        ch_year = df["CH_THN"]
    else:
        ch_cols = [c for c in df.columns if c.startswith("CH_") and len(c.split("_"))==2]
        ch_year = df[ch_cols].sum(axis=1) if ch_cols else np.nan

    bk = df["BK_max"] if "BK_max" in df.columns else df.get("BK", np.nan)
    slope = df["slope_percent"] if "slope_percent" in df.columns else df.get("slope", np.nan)

    out = df.copy()
    # fuzzy
    out["score_T"]   = score_temp(temp_mean)
    out["score_CH"]  = score_ch(ch_year)
    out["score_BK"]  = score_bk(bk)
    out["score_SLP"] = score_slope(slope)
    out["score_total"] = out[["score_T","score_CH","score_BK","score_SLP"]].min(axis=1)

    def to_class(mu):
        if mu >= 0.75: return "S1"
        if mu >= 0.50: return "S2"
        if mu >= 0.25: return "S3"
        return "N"
    out["kelas"] = out["score_total"].apply(to_class)

    # crisp per faktor → ambil tingkat terendah
    def cls_temp(t):
        return np.select(
            [(t>=25)&(t<=28), ((t>=22)&(t<25))|((t>28)&(t<=32)), ((t>=20)&(t<22))|((t>32)&(t<=35))],
            ["S1","S2","S3"], default="N")
    def cls_ch(ch):
        return np.select(
            [(ch>=1700)&(ch<=2500), ((ch>=1450)&(ch<1700))|((ch>2500)&(ch<=3500)), ((ch>=1250)&(ch<1450))|((ch>3500)&(ch<=4000))],
            ["S1","S2","S3"], default="N")
    def cls_bk(x):
        return np.select([x<2, (x>=2)&(x<=3), (x>3)&(x<=4)], ["S1","S2","S3"], default="N")
    def cls_slope(s):
        return np.select([s<=8, (s>8)&(s<=16), (s>16)&(s<=30)], ["S1","S2","S3"], default="N")

    out["T_cls"]   = cls_temp(temp_mean)
    out["CH_cls"]  = cls_ch(ch_year)
    out["BK_cls"]  = cls_bk(bk)
    out["SLP_cls"] = cls_slope(slope)

    rank = {"N":0,"S3":1,"S2":2,"S1":3}; inv = {v:k for k,v in rank.items()}
    out["kelas_crisp"] = pd.DataFrame({
        "T": out["T_cls"].map(rank),
        "CH": out["CH_cls"].map(rank),
        "BK": out["BK_cls"].map(rank),
        "SLP": out["SLP_cls"].map(rank),
    }).min(axis=1).map(inv)
    return out

# -------------------------
# TEBU: SCORING & KLASIFIKASI
# -------------------------
def score_temp_tebu(t):
    # S1: 25–32, S2: 23–25 atau 32–34, S3: 20–23 atau 34–36
    return fuzzy_band(t, n_lo=20, s3_lo=23, s1_lo=25, s1_hi=32, s3_hi=34, n_hi=36)

def score_ch_tebu(ch):
    # S1: 1400–2200; S2: 1200–1400 / 2200–2800; S3: 1000–1200 / 2800–3200
    return fuzzy_band(ch, n_lo=1000, s3_lo=1200, s1_lo=1400, s1_hi=2200, s3_hi=2800, n_hi=3200)

def score_bk_tebu(bk):
    # S1 ≤2; S2 ≈3; S3 ≈4; >4 turun ke 0
    x = np.asarray(bk, dtype=float)
    s = np.zeros_like(x)
    s[x <= 2.0] = 1.0
    m = (x > 2.0) & (x <= 3.0); s[m] = np.interp(x[m], [2.0,3.0], [1.0,0.7])
    m = (x > 3.0) & (x <= 4.0); s[m] = np.interp(x[m], [3.0,4.0], [0.7,0.4])
    m = x > 4.0;              s[m] = np.interp(np.minimum(x[m],5.0), [4.0,5.0], [0.4,0.0])
    return np.clip(s, 0, 1)

def score_slope_tebu(s):
    # S1 ≤5%; S2 5–8%; S3 8–12%
    x = np.asarray(s, dtype=float)
    v = np.zeros_like(x)
    v[x <= 5] = 1.0
    m = (x > 5) & (x <= 8);   v[m] = np.interp(x[m], [5,8],   [1.0,0.7])
    m = (x > 8) & (x <= 12);  v[m] = np.interp(x[m], [8,12],  [0.7,0.4])
    m = x > 12;               v[m] = np.interp(np.minimum(x[m],20.0), [12,20],[0.4,0.0])
    return np.clip(v, 0, 1)

def classify_tebu_fuzzy_crisp(df: pd.DataFrame) -> pd.DataFrame:
    """Tebu: tambah score_*_tebu , score_total_tebu, kelas_tebu, kelas_tebu_crisp."""
    out = df.copy()

    t_cols = [c for c in out.columns if c.startswith("T2M_")]
    temp_mean = out[t_cols].mean(axis=1) if t_cols else np.nan

    if "CH_THN" in out.columns:
        ch_year = out["CH_THN"]
    else:
        ch_cols = [c for c in out.columns if c.startswith("CH_") and len(c.split("_"))==2]
        ch_year = out[ch_cols].sum(axis=1) if ch_cols else np.nan

    bk    = out["BK_max"] if "BK_max" in out.columns else out.get("BK", np.nan)
    slope = out["slope_percent"] if "slope_percent" in out.columns else out.get("slope", np.nan)

    # FUZZY
    out["score_T_tebu"]   = score_temp_tebu(temp_mean)
    out["score_CH_tebu"]  = score_ch_tebu(ch_year)
    out["score_BK_tebu"]  = score_bk_tebu(bk)
    out["score_SLP_tebu"] = score_slope_tebu(slope)
    out["score_total_tebu"] = out[["score_T_tebu","score_CH_tebu","score_BK_tebu","score_SLP_tebu"]].min(axis=1)

    def to_class(mu):
        if mu >= 0.75: return "S1"
        if mu >= 0.50: return "S2"
        if mu >= 0.25: return "S3"
        return "N"
    out["kelas_tebu"] = out["score_total_tebu"].apply(to_class)

    # CRISP
    def cls_temp_tebu(t):
        return np.select(
            [(t>=25)&(t<=32), ((t>=23)&(t<25))|((t>32)&(t<=34)), ((t>=20)&(t<23))|((t>34)&(t<=36))],
            ["S1","S2","S3"], default="N")
    def cls_ch_tebu(ch):
        return np.select(
            [(ch>=1400)&(ch<=2200), ((ch>=1200)&(ch<1400))|((ch>2200)&(ch<=2800)),
             ((ch>=1000)&(ch<1200))|((ch>2800)&(ch<=3200))],
            ["S1","S2","S3"], default="N")
    def cls_bk_tebu(x):
        return np.select([x<=2, (x>2)&(x<=3), (x>3)&(x<=4)], ["S1","S2","S3"], default="N")
    def cls_slope_tebu(s):
        return np.select([s<=5, (s>5)&(s<=8), (s>8)&(s<=12)], ["S1","S2","S3"], default="N")

    out["T_tebu_cls"]   = cls_temp_tebu(temp_mean)
    out["CH_tebu_cls"]  = cls_ch_tebu(ch_year)
    out["BK_tebu_cls"]  = cls_bk_tebu(bk)
    out["SLP_tebu_cls"] = cls_slope_tebu(slope)

    rank = {"N":0,"S3":1,"S2":2,"S1":3}; inv = {v:k for k,v in rank.items()}
    out["kelas_tebu_crisp"] = pd.DataFrame({
        "T": out["T_tebu_cls"].map(rank),
        "CH": out["CH_tebu_cls"].map(rank),
        "BK": out["BK_tebu_cls"].map(rank),
        "SLP": out["SLP_tebu_cls"].map(rank),
    }).min(axis=1).map(inv)

    return out

# -------------------------
# EXPORT GEOJSON
# -------------------------
def export_to_geojson(hasil: pd.DataFrame, output_path: Path) -> None:
    """
    Export hasil klasifikasi ke format GeoJSON.
    Args:
        hasil: DataFrame hasil klasifikasi dengan kolom LON, LAT
        output_path: Path file output GeoJSON
    """
    try:
        import geopandas as gpd
        # Buat GeoDataFrame dari hasil klasifikasi
        gdf = gpd.GeoDataFrame(
            hasil,
            geometry=gpd.points_from_xy(hasil.LON, hasil.LAT),
            crs="EPSG:4326"
        )
        # Pilih kolom yang akan disimpan
        cols_to_keep = [
            'LON', 'LAT', 
            'kelas', 'kelas_crisp', 
            'kelas_tebu', 'kelas_tebu_crisp',
            'score_total', 'score_total_tebu'
        ]
        cols_to_keep = [col for col in cols_to_keep if col in gdf.columns]
        gdf = gdf[cols_to_keep + ['geometry']]
        # Export ke GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"GeoJSON tersimpan: {output_path}")
    except Exception as e:
        print(f"Gagal export GeoJSON: {e}")

# -------------------------
# PETA (opsional)
# -------------------------
def plot_map_by_class(hasil: pd.DataFrame, kelas_col: str, title: str, out_png: Path, shp_path: Path|None):
    colors = {"S1":"#1b9e77","S2":"#d95f02","S3":"#7570b3","N":"#9e9e9e"}
    try:
        import geopandas as gpd
        if shp_path and shp_path.exists():
            prov = gpd.read_file(shp_path).to_crs("EPSG:4326")
            gdf  = gpd.GeoDataFrame(hasil, geometry=gpd.points_from_xy(hasil.LON, hasil.LAT), crs="EPSG:4326")
            ax = prov.boundary.plot(color="#444", linewidth=0.5, figsize=(8,10))
            for k, sub in gdf.groupby(kelas_col):
                sub.plot(ax=ax, markersize=3, color=colors.get(k,"#9e9e9e"), alpha=0.7, rasterized=True, label=k)
            ax.set_title(title); ax.set_axis_off(); ax.set_aspect("equal")
            ax.legend(title="Kelas", loc="lower left", frameon=True)
            fig = ax.get_figure()
            fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig); return
    except Exception:
        pass
    # fallback tanpa geopandas
    plt.figure(figsize=(8,10))
    for k, sub in hasil.groupby(kelas_col):
        plt.scatter(sub["LON"], sub["LAT"], s=3, c=colors.get(k,"#9e9e9e"), alpha=0.7, linewidths=0, label=k)
    plt.legend(title="Kelas"); plt.title(title); plt.xlabel("LON"); plt.ylabel("LAT")
    plt.axis("equal"); plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # 1) Load data grid
    df = load_grid(PATH_GRID_XLS, sheet_name=SHEET_NAME)

    # 2) Klasifikasi SAWIT
    hasil = classify_fuzzy_crisp(df)

    # 3) Klasifikasi TEBU (tambahkan kolom tebu)
    hasil = classify_tebu_fuzzy_crisp(hasil)

    # 4) Cek ringkas
    print("Sawit (fuzzy):", hasil["kelas"].value_counts(dropna=False).to_dict())
    print("Sawit (crisp):", hasil["kelas_crisp"].value_counts(dropna=False).to_dict())
    print("Tebu  (fuzzy):", hasil["kelas_tebu"].value_counts(dropna=False).to_dict())
    print("Tebu  (crisp):", hasil["kelas_tebu_crisp"].value_counts(dropna=False).to_dict())

    # 5) Peta (opsional, simpan 4 png)
    plot_map_by_class(hasil, "kelas",              "Peta Kesesuaian Sawit (FUZZY)", OUT_DIR/"peta_sawit_fuzzy.png", PATH_SHP_PROV)
    plot_map_by_class(hasil, "kelas_crisp",        "Peta Kesesuaian Sawit (CRISP)", OUT_DIR/"peta_sawit_crisp.png", PATH_SHP_PROV)
    plot_map_by_class(hasil, "kelas_tebu",         "Peta Kesesuaian Tebu (FUZZY)",  OUT_DIR/"peta_tebu_fuzzy.png",  PATH_SHP_PROV)
    plot_map_by_class(hasil, "kelas_tebu_crisp",   "Peta Kesesuaian Tebu (CRISP)",  OUT_DIR/"peta_tebu_crisp.png",  PATH_SHP_PROV)

    # 6) Ekspor Excel hasil grid lengkap
    out_xlsx = OUT_DIR / "klasifikasi_sawit_tebu_grid.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
        hasil.to_excel(w, sheet_name="Hasil_Grid", index=False)
        # (opsional) rekap per-prov jika kolom provinsi tersedia
        if COL_PROV_GRID in hasil.columns:
            def _rekap(df, col):
                order = ["S1","S2","S3","N"]
                tab = pd.crosstab(df[COL_PROV_GRID], df[col]).reindex(columns=order, fill_value=0)
                tab["TOTAL"] = tab.sum(axis=1)
                tab["S1_pct"] = tab["S1"]/tab["TOTAL"]*100
                tab["S1S2_pct"] = (tab["S1"]+tab["S2"])/tab["TOTAL"]*100
                return tab.reset_index()
            _rekap(hasil, "kelas").to_excel(w, sheet_name="Rekap_Sawit", index=False)
            _rekap(hasil, "kelas_tebu").to_excel(w, sheet_name="Rekap_Tebu", index=False)
    print("Excel tersimpan:", out_xlsx)
    
    # 7) Export GeoJSON untuk web mapping
    out_geojson = OUT_DIR / "klasifikasi_sawit_tebu.geojson"
    export_to_geojson(hasil, out_geojson)
