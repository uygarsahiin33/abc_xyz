# app_abc_xyz_jupyter.py
# -*- coding: utf-8 -*-
# Jupyter: SQL -> DataFrame -> ABC-XYZ sınıflaması (+ z-çarpan) + pie chartlar + Excel çıktı

import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

# =============== SQL CONNECTION (BAŞA EKLENEN KISIM) ===============
from whitelotuts.utils import create_selfservis_connection
engine = create_selfservis_connection()

# Sorguyu burada değiştireceksin
query = "SELECT * FROM selfservis_migros.v_item"
df_sql = pd.read_sql(query, engine)

# =============== Inverse Normal CDF (ppf) ===============
def norm_ppf(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+1))

# =============== Settings ===============
REQUIRED_BASE_COLS = [
    "Depo",
    "Mağaza Adı",
    "Ürün",
    "Son 1 Ay Toplam Satış",
    "Varyans",
    "Tedarikçi Güven Endeksi",
    "Lead Time (1 ila 7 gün randomize)"
]
SUGGESTED_GROUP_COLS = ["Ana Grup", "Dördüncü Kırılım"]
OPTIONAL_RESULT_COLS = ["ABC-XYZ Sonucu", "ABC-XYZ'e göre çarpan"]

DEFAULT_SERVICE_LEVEL_GRID = {
    "AX": 0.99,  "AY": 0.975, "AZ": 0.95,
    "BX": 0.975, "BY": 0.95,  "BZ": 0.90,
    "CX": 0.95,  "CY": 0.90,  "CZ": 0.85,
}
DEFAULT_MULTIPLIER_GRID = {
    "AX": 1.88, "AY": 1.64, "AZ": 1.48,
    "BX": 1.34, "BY": 1.28, "BZ": 1.23,
    "CX": 1.13, "CY": 1.04, "CZ": 0.84,
}
ABCXYZ_ORDER = ["AX","AY","AZ","BX","BY","BZ","CX","CY","CZ"]

# =============== Helpers ===============
def to_numeric_strict(s: pd.Series) -> pd.Series:
    ser = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(ser, errors="coerce").fillna(0)

def _percent_rank(series: pd.Series, ascending: bool) -> pd.Series:
    n = series.shape[0]
    if n <= 1:
        return pd.Series(np.zeros(n), index=series.index, dtype=float)
    r = series.rank(method="average", ascending=ascending)
    return (r - 1) / (n - 1)

def group_percent_rank(df: pd.DataFrame, value_col: str, ascending: bool, group_cols: list[str] | None):
    if not group_cols:
        return _percent_rank(df[value_col], ascending=ascending)
    return df.groupby(group_cols, group_keys=False)[value_col]\
             .apply(lambda s: _percent_rank(s, ascending=ascending))

def classify_from_pr(pr: float, cut1=0.20, cut2=0.60, labels=("A","B","C")) -> str:
    if pr <= cut1:
        return labels[0]
    elif pr <= cut2:
        return labels[1]
    else:
        return labels[2]

def compute_abc_xyz(
    df: pd.DataFrame,
    abc_top_cut: float = 0.20,
    abc_mid_cut: float = 0.60,
    xyz_low_cut: float = 0.20,
    xyz_mid_cut: float = 0.60,
    grid_mode: str = "z",                 # "z" (direct) or "csl"
    grid_z: dict | None = None,           # used when grid_mode="z"
    grid_csl: dict | None = None,         # used when grid_mode="csl"
    scope: str = "global",                # "global" | "ana_grup" | "dorduncu_kirilim"
) -> pd.DataFrame:
    """ABC (sales percentile) + XYZ (variance percentile) + multiplier with group scopes."""
    out = df.copy()
    out["Son 1 Ay Toplam Satış"] = to_numeric_strict(out["Son 1 Ay Toplam Satış"])
    out["Varyans"] = to_numeric_strict(out["Varyans"])

    # Scope -> group columns
    if scope == "global":
        group_cols = None
    elif scope == "ana_grup":
        if "Ana Grup" not in out.columns:
            raise ValueError("Analiz kapsamı 'Ana Grup içinde' seçildi ama 'Ana Grup' kolonu yok.")
        group_cols = ["Ana Grup"]
    elif scope == "dorduncu_kirilim":
        if "Dördüncü Kırılım" not in out.columns:
            raise ValueError("Analiz kapsamı 'Dördüncü Kırılım içinde' seçildi ama 'Dördüncü Kırılım' kolonu yok.")
        group_cols = ["Dördüncü Kırılım"]
    else:
        group_cols = None

    # ABC: satış DESC → percent rank by scope
    pr_sales = group_percent_rank(out, "Son 1 Ay Toplam Satış", ascending=False, group_cols=group_cols)
    ABC = pr_sales.apply(lambda v: classify_from_pr(v, abc_top_cut, abc_mid_cut, ("A","B","C")))

    # XYZ: varyans ASC → percent rank by scope
    pr_var = group_percent_rank(out, "Varyans", ascending=True, group_cols=group_cols)
    XYZ = pr_var.apply(lambda v: classify_from_pr(v, xyz_low_cut, xyz_mid_cut, ("X","Y","Z")))

    out["ABC-XYZ Sonucu"] = (ABC + XYZ).astype(str)

    # Multiplier
    if grid_mode == "z":
        zgrid = grid_z or DEFAULT_MULTIPLIER_GRID
        out["ABC-XYZ'e göre çarpan"] = out["ABC-XYZ Sonucu"].map(zgrid).fillna(1.0)
    else:
        csl = grid_csl or DEFAULT_SERVICE_LEVEL_GRID
        target = out["ABC-XYZ Sonucu"].map(csl).fillna(0.90)
        out["ABC-XYZ'e göre çarpan"] = target.apply(norm_ppf)

    out["ABC-XYZ'e göre çarpan"] = out["ABC-XYZ'e göre çarpan"].astype(float).round(2)
    return out

# =============== ÇALIŞTIRMA BLOĞU (JUPYTER) ===============
# Parametreleri buradan ayarla
scope_key = "global"          # "global" | "ana_grup" | "dorduncu_kirilim"
abc_top   = 0.20              # A üst dilim
abc_mid   = 0.60              # B üst dilim
xyz_low   = 0.20              # X alt dilim
xyz_mid   = 0.60              # Y üst dilim
grid_mode = "z"               # "z" (DEFAULT_MULTIPLIER_GRID) veya "csl" (DEFAULT_SERVICE_LEVEL_GRID)
grid_z    = None              # grid_mode="z" ise özelleştir
grid_csl  = None              # grid_mode="csl" ise özelleştir

# SQL DF'ini kopyala ve zorunlu/opsiyonel kolonu kontrol et
df = df_sql.copy()

missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolon(lar) / Missing columns: {missing}")

# Opsiyonel sonuç kolonlarını ekle (yoksa)
for c in OPTIONAL_RESULT_COLS:
    if c not in df.columns:
        df[c] = ""

print(f"Veri yüklendi: {len(df):,} satır")
display(df.head(10))

# Hesapla
out = compute_abc_xyz(
    df,
    abc_top_cut=abc_top,
    abc_mid_cut=abc_mid,
    xyz_low_cut=xyz_low,
    xyz_mid_cut=xyz_mid,
    grid_mode=grid_mode,
    grid_z=grid_z,
    grid_csl=grid_csl,
    scope=scope_key,
)

# --- PIE CHARTLAR (Genel dağılım + A/B/C içi)
print("\nABC–XYZ dağılım grafikleri oluşturuluyor...")

# Genel dağılım
vc_all = out["ABC-XYZ Sonucu"].astype(str).value_counts()
df_all = pd.DataFrame({"ABCXYZ": ABCXYZ_ORDER})
df_all["count"] = df_all["ABCXYZ"].map(vc_all).fillna(0).astype(int)

fig_all = px.pie(
    df_all, names="ABCXYZ", values="count",
    category_orders={"ABCXYZ": ABCXYZ_ORDER},
    hole=0.35, title="Tüm kayıtlar için ABC-XYZ dağılımı"
)
fig_all.update_traces(textposition="inside", textinfo="percent+label")
fig_all.show()

# A/B/C içi kırılımlar
def sub_pie(letter: str, target_order: list[str], title: str):
    sub = out[out["ABC-XYZ Sonucu"].str.startswith(letter)]
    vc = sub["ABC-XYZ Sonucu"].astype(str).value_counts()
    dfx = pd.DataFrame({"ABCXYZ": target_order})
    dfx["count"] = dfx["ABCXYZ"].map(vc).fillna(0).astype(int)
    fig = px.pie(
        dfx, names="ABCXYZ", values="count",
        category_orders={"ABCXYZ": target_order},
        hole=0.35, title=title
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.show()

sub_pie("A", ["AX","AY","AZ"], "A sınıfı içinde dağılım")
sub_pie("B", ["BX","BY","BZ"], "B sınıfı içinde dağılım")
sub_pie("C", ["CX","CY","CZ"], "C sınıfı içinde dağılım")

# Ön izleme tablo (ilk 50 kayıt, sıralı)
cat_order = pd.api.types.CategoricalDtype(categories=ABCXYZ_ORDER, ordered=True)
preview_df = out.copy()
preview_df["ABC-XYZ Sonucu"] = preview_df["ABC-XYZ Sonucu"].astype(cat_order)
sort_cols = [c for c in ["ABC-XYZ Sonucu","Ana Grup","Dördüncü Kırılım","Ürün"] if c in preview_df.columns]
preview_df = preview_df.sort_values(sort_cols, na_position="last")
display(preview_df.head(50))

# Excel çıktısı
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"ABC_XYZ_Sonuclar_{scope_key}_{ts}.xlsx"
with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
    out.to_excel(writer, index=False)
print(f"\n📥 Excel kaydedildi: {fname}")
