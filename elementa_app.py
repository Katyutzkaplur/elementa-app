""
Elementa — Sistema Analítico Colorimétrico Digital
Derechos reservados (Katyutzka Villarreal, 2026)

Herramienta educativa y analítica para estimación colorimétrica por imágenes RGB.
No sustituye métodos instrumentales certificados ni análisis en laboratorios acreditados.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from io import BytesIO
import base64
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Elementa",
    page_icon="[E]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PALETA ──────────────────────────────────
EMERALD  = "#059669"
BLUE_ACC = "#3B82F6"
AMBER    = "#D97706"
ROSE     = "#E11D48"
BG_CARD  = "#1e293b"
BG_DARK  = "#0f172a"
TEXT_SUB = "#94a3b8"
TEXT_MUT = "#64748b"
BORDER   = "#334155"

PLOTLY_TEMPLATE = "plotly_dark"

TYPE_COLORS = {
    "Blanco":           "#F59E0B",   # amber
    "Estandar":         "#10B981",   # green
    "Muestra":          "#3B82F6",   # blue
    "Control":          "#F97316",   # orange
    "Adicion estandar": "#8B5CF6",   # purple
    "Sin asignar":      "#334155",   # dark slate
}
TYPE_COLORS_BGR = {
    "Blanco":           (184, 157,  61),
    "Estandar":         (17,  185, 129),
    "Muestra":          (59,  130, 246),
    "Control":          (249, 115,  22),
    "Adicion estandar": (139,  92, 246),
    "Sin asignar":      (51,   65,  85),
}

# ─── CSS GLOBAL ──────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  * {{ font-family: 'IBM Plex Sans', sans-serif; }}
  code, pre {{ font-family: 'IBM Plex Mono', monospace; }}

  .stApp {{ background-color: {BG_DARK}; color: #e2e8f0; }}
  [data-testid="stSidebar"] {{ background-color: #0a1220; border-right: 1px solid {BORDER}; }}

  .metric-card {{
      background: {BG_CARD}; border-radius: 6px;
      padding: 14px 18px; margin-bottom: 8px;
      border-left: 3px solid {EMERALD};
  }}
  .metric-card h4 {{
      margin: 0 0 4px 0; color: {TEXT_SUB};
      font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.08em; font-weight: 600;
  }}
  .metric-card p  {{ margin: 0; font-size: 1.35rem; font-weight: 700; color: #f1f5f9; }}
  .metric-card small {{ color: {TEXT_MUT}; font-size: 0.75rem; }}

  .stat-explain {{
      color: {TEXT_SUB}; font-size: 0.78rem;
      margin-top: 4px; line-height: 1.45;
      font-style: italic;
  }}

  .badge-pass  {{ background: #052e16; color: #4ade80; padding: 3px 10px; border-radius: 4px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; border: 1px solid #166534; }}
  .badge-fail  {{ background: #450a0a; color: #f87171; padding: 3px 10px; border-radius: 4px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; border: 1px solid #991b1b; }}
  .badge-none  {{ background: #1e293b; color: {TEXT_MUT}; padding: 3px 10px; border-radius: 4px; font-size: 0.78rem; font-weight: 600; border: 1px solid {BORDER}; }}

  .warn-box {{
      background: #1c0a0a; border-left: 3px solid #dc2626;
      border-radius: 4px; padding: 11px 15px; font-size: 0.83rem;
      color: #fca5a5; margin: 10px 0; line-height: 1.5;
  }}
  .info-box {{
      background: #0c1525; border-left: 3px solid {BLUE_ACC};
      border-radius: 4px; padding: 11px 15px; font-size: 0.83rem;
      color: #93c5fd; margin: 10px 0; line-height: 1.5;
  }}
  .success-box {{
      background: #052e16; border-left: 3px solid {EMERALD};
      border-radius: 4px; padding: 11px 15px; font-size: 0.83rem;
      color: #6ee7b7; margin: 10px 0; line-height: 1.5;
  }}

  h1 {{ color: #f1f5f9; font-weight: 700; letter-spacing: -0.02em; }}
  h2 {{ color: #e2e8f0; font-weight: 600; }}
  h3 {{ color: #cbd5e1; font-weight: 600; }}

  .stButton>button {{
      background-color: {EMERALD}; color: #fff;
      border: none; border-radius: 5px;
      padding: 9px 22px; font-weight: 600;
      letter-spacing: 0.03em; font-size: 0.85rem;
  }}
  .stButton>button:hover {{ background-color: #047857; }}

  .step-header {{
      font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
      text-transform: uppercase; color: {EMERALD};
      margin-bottom: 2px;
  }}

  .plate-legend-item {{
      display: inline-flex; align-items: center; gap: 7px;
      margin-right: 16px; font-size: 0.8rem; color: {TEXT_SUB};
  }}
  .plate-legend-dot {{
      width: 12px; height: 12px; border-radius: 2px;
  }}
  .triplicate-tag {{
      background: #1e3a5f; color: #7dd3fc;
      font-size: 0.7rem; padding: 2px 7px; border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace; font-weight: 600;
  }}
  .footer {{
      text-align: center; color: {TEXT_MUT}; font-size: 0.72rem;
      padding: 24px 0 8px 0; margin-top: 48px;
      border-top: 1px solid {BORDER};
  }}
  .divider {{ border: none; border-top: 1px solid {BORDER}; margin: 24px 0; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  LÍMITES NORMATIVOS
# ════════════════════════════════════════════════════════════════
NORMATIVE_LIMITS = {
    "Pb": {
        "NOM-127-SSA1-2021 (agua potable)":        0.01,
        "NOM-001-SEMARNAT-2021 (descarga A)":       0.2,
        "NOM-001-SEMARNAT-2021 (descarga B)":       1.0,
    },
    "Cd": {
        "NOM-127-SSA1-2021 (agua potable)":         0.003,
        "NOM-001-SEMARNAT-2021 (descarga A)":       0.1,
        "NOM-001-SEMARNAT-2021 (descarga B)":       0.2,
    },
    "Cr total": {
        "NOM-127-SSA1-2021 (agua potable)":         0.05,
        "NOM-001-SEMARNAT-2021 (descarga A)":       0.5,
        "NOM-001-SEMARNAT-2021 (descarga B)":       1.0,
    },
    "Cr(VI)": {
        "NOM-127-SSA1-2021 (agua potable)":         0.05,
    },
}

STAT_EXPLANATIONS = {
    "R2": (
        "Coeficiente de determinación (R²). Indica qué proporción de la varianza de la "
        "señal queda explicada por la concentración. Un R² > 0.999 indica linealidad "
        "excelente para métodos cuantitativos."
    ),
    "slope": (
        "Pendiente de la recta de calibración (m en y = mx + b). Representa la "
        "sensibilidad analítica: cuánto cambia la señal digital por unidad de "
        "concentración. Una pendiente negativa es esperada en ensayos donde el "
        "analito reduce la intensidad de color (p. ej. DPPH)."
    ),
    "intercept": (
        "Ordenada al origen (b). El valor esperado de la señal cuando la "
        "concentración es cero. Idealmente cercano al valor del blanco de reactivos."
    ),
    "se": (
        "Error estándar de la pendiente (Sb). Cuantifica la incertidumbre en la "
        "estimación de la pendiente debida a la dispersión de los puntos de "
        "calibración. Valores menores indican mayor precisión del ajuste."
    ),
    "LOD": (
        "Límite de Detección (LOD = 3.3 × sigma / |m|). Concentración mínima "
        "que puede detectarse estadísticamente con 99% de confianza. Las "
        "concentraciones por debajo de este valor son indistinguibles del ruido."
    ),
    "LOQ": (
        "Límite de Cuantificación (LOQ = 10 × sigma / |m|). Concentración "
        "mínima que puede cuantificarse con precisión y exactitud aceptables "
        "(generalmente CV < 10%). Las muestras entre LOD y LOQ pueden detectarse "
        "pero no cuantificarse de manera confiable."
    ),
    "CV": (
        "Coeficiente de Variación (CV = (SD / media) × 100 %). Medida de "
        "precisión relativa para triplicados. CV < 5% indica precisión muy buena; "
        "5–10% aceptable; > 10% requiere revisar la técnica experimental."
    ),
}


# ════════════════════════════════════════════════════════════════
#  FUNCIONES DE PROCESAMIENTO DE IMAGEN
# ════════════════════════════════════════════════════════════════

def load_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def generate_rois_linear(x0, y0, w, h, n, dx, dy) -> list[dict]:
    rois = []
    for i in range(n):
        rois.append({
            "x": int(x0 + i * dx), "y": int(y0 + i * dy),
            "w": int(w), "h": int(h), "label": f"ROI {i+1}",
        })
    return rois


def generate_rois_microplate(x0, y0, w, h, dx, dy, rows=8, cols=12) -> list[dict]:
    rois = []
    row_labels = "ABCDEFGH"
    for r in range(rows):
        for c in range(cols):
            rois.append({
                "x": int(x0 + c * dx), "y": int(y0 + r * dy),
                "w": int(w), "h": int(h),
                "label": f"{row_labels[r]}{c+1}",
            })
    return rois


def draw_rois(image: np.ndarray, rois: list[dict], assignments: dict | None = None) -> np.ndarray:
    img = image.copy()
    for roi in rois:
        tipo = (assignments or {}).get(roi["label"], "Sin asignar")
        color_rgb = TYPE_COLORS_BGR.get(tipo, (51, 65, 85))
        bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(img, (x, y), (x+w, y+h), bgr, 2)
        cv2.putText(img, roi["label"], (x, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1, cv2.LINE_AA)
    return img


def extract_rgb_stats(image: np.ndarray, rois: list[dict]) -> pd.DataFrame:
    records = []
    h_img, w_img = image.shape[:2]
    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x+w), min(h_img, y+h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            r_m = g_m = b_m = r_s = g_s = b_s = np.nan
        else:
            r_m, g_m, b_m = crop[:,:,0].mean(), crop[:,:,1].mean(), crop[:,:,2].mean()
            r_s, g_s, b_s = crop[:,:,0].std(),  crop[:,:,1].std(),  crop[:,:,2].std()
        records.append({
            "ROI": roi["label"],
            "R_mean": round(r_m, 2), "G_mean": round(g_m, 2), "B_mean": round(b_m, 2),
            "R_std":  round(r_s, 2), "G_std":  round(g_s, 2), "B_std":  round(b_s, 2),
        })
    return pd.DataFrame(records)


def calculate_normalized_intensity(df_rgb: pd.DataFrame) -> pd.DataFrame:
    df = df_rgb.copy()
    eps = 1e-9
    total = df["R_mean"] + df["G_mean"] + df["B_mean"] + eps
    df["Total_RGB"] = df["R_mean"] + df["G_mean"] + df["B_mean"]
    df["R_norm"]    = (df["R_mean"] / total) * 100
    df["G_norm"]    = (df["G_mean"] / total) * 100
    df["B_norm"]    = (df["B_mean"] / total) * 100
    return df


def calculate_digital_absorbance(df: pd.DataFrame, blank_label: str | None,
                                   channels: list[str]) -> pd.DataFrame:
    df = df.copy()
    if blank_label is None or blank_label not in df["ROI"].values:
        for ch in channels:
            df[f"A_dig_{ch}"] = np.nan
        return df
    for ch in channels:
        col = f"{ch}_norm" if "_norm" not in ch else ch
        if col not in df.columns:
            df[f"A_dig_{ch}"] = np.nan
            continue
        blank_val = df.loc[df["ROI"] == blank_label, col].values[0]
        eps = 1e-9
        df[f"A_dig_{ch}"] = df[col].apply(
            lambda v: math.log10((blank_val + eps) / (v + eps)) if pd.notna(v) else np.nan
        )
    return df


def fit_calibration_curve(concs: np.ndarray, signals: np.ndarray):
    if len(concs) < 2:
        return None
    mask = ~(np.isnan(concs) | np.isnan(signals))
    concs, signals = concs[mask], signals[mask]
    if len(concs) < 2:
        return None
    slope, intercept, r, p, se = stats.linregress(concs, signals)
    y_pred = slope * concs + intercept
    residuals = signals - y_pred
    return {
        "slope": slope, "intercept": intercept,
        "r2": r**2, "se": se,
        "residuals": residuals, "n": len(concs),
    }


def calculate_lod_loq(cal_result: dict, blank_signals: np.ndarray | None) -> dict:
    m = abs(cal_result["slope"])
    if m < 1e-12:
        return {"LOD": np.nan, "LOQ": np.nan, "sigma_used": np.nan, "proxy": True}
    if blank_signals is not None and len(blank_signals) >= 2:
        sigma = np.std(blank_signals, ddof=1)
        proxy = False
    else:
        sigma = cal_result["se"] if cal_result["se"] else np.nan
        proxy = True
    lod = 3.3 * sigma / m if not np.isnan(sigma) else np.nan
    loq = 10.0 * sigma / m if not np.isnan(sigma) else np.nan
    return {"LOD": lod, "LOQ": loq, "sigma_used": sigma, "proxy": proxy}


def select_best_channel(df: pd.DataFrame, standard_labels: list[str],
                         conc_col: str = "Concentracion") -> dict:
    channels_to_test = ["R_norm", "G_norm", "B_norm"]
    std_df = df[df["ROI"].isin(standard_labels)].copy()
    if conc_col not in std_df.columns or len(std_df) < 2:
        return {"best_channel": "G_norm", "results": {}}
    results = {}
    for ch in channels_to_test:
        a_col = f"A_dig_{ch}"
        if a_col not in std_df.columns:
            continue
        sub = std_df[[conc_col, a_col]].dropna()
        if len(sub) < 2:
            continue
        cal = fit_calibration_curve(sub[conc_col].values.astype(float),
                                    sub[a_col].values.astype(float))
        if cal:
            results[ch] = cal
    if not results:
        return {"best_channel": "G_norm", "results": {}}
    best = max(results, key=lambda k: results[k]["r2"])
    return {"best_channel": best, "results": results}


def standard_addition_analysis(added_concs: np.ndarray, signals: np.ndarray) -> dict | None:
    cal = fit_calibration_curve(added_concs, signals)
    if cal is None:
        return None
    m, b = cal["slope"], cal["intercept"]
    if abs(m) < 1e-12:
        return None
    x_intercept = -b / m
    c_sample = abs(x_intercept)
    cal["x_intercept"] = x_intercept
    cal["c_sample"]    = c_sample
    return cal


def evaluate_normative_status(analyte: str, conc_mg_L: float) -> list[dict]:
    if analyte not in NORMATIVE_LIMITS:
        return [{"norma": "Sin criterio disponible", "limite": None,
                 "status": "Sin criterio", "badge": "none"}]
    results = []
    for norma, limite in NORMATIVE_LIMITS[analyte].items():
        results.append({
            "norma": norma, "limite": limite,
            "status": "Cumple" if conc_mg_L <= limite else "No cumple",
            "badge":  "pass"  if conc_mg_L <= limite else "fail",
        })
    return results


# ════════════════════════════════════════════════════════════════
#  TRIPLICADOS
# ════════════════════════════════════════════════════════════════

def detect_triplicate_groups(assignment_df: pd.DataFrame) -> dict:
    """
    Detecta grupos de triplicados por columna de placa (A1, B1, C1 -> grupo '1').
    Retorna {col_num: [roi_labels]}.
    """
    groups = {}
    for _, row in assignment_df.iterrows():
        roi = row["ROI"]
        col_num = "".join(filter(str.isdigit, roi))
        tipo = row.get("Tipo", "Sin asignar")
        if col_num and tipo != "Sin asignar":
            groups.setdefault(col_num, []).append(roi)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def calculate_triplicate_stats(df_absorbance: pd.DataFrame,
                                assignment_df: pd.DataFrame,
                                groups: dict,
                                signal_col: str) -> pd.DataFrame:
    """Calcula media, DE y CV% por grupo de triplicado."""
    merged = df_absorbance.merge(assignment_df[["ROI","Tipo","Nombre","Concentracion","Unidad","Analito","Factor_dil"]],
                                  on="ROI", how="left")
    records = []
    for col_num, roi_list in sorted(groups.items(), key=lambda x: int(x[0])):
        sub = merged[merged["ROI"].isin(roi_list)]
        if sub.empty or signal_col not in sub.columns:
            continue
        sigs = sub[signal_col].dropna().values
        if len(sigs) == 0:
            continue
        mean_sig = np.mean(sigs)
        sd_sig   = np.std(sigs, ddof=1) if len(sigs) > 1 else np.nan
        cv_sig   = (sd_sig / abs(mean_sig) * 100) if not np.isnan(sd_sig) and abs(mean_sig) > 1e-9 else np.nan
        tipo_repr = sub["Tipo"].iloc[0]
        conc_repr = sub["Concentracion"].iloc[0] if "Concentracion" in sub.columns else np.nan
        nombre_repr = sub["Nombre"].iloc[0] if "Nombre" in sub.columns else ""
        records.append({
            "Grupo":        f"Col. {col_num}",
            "POcillos":     ", ".join(roi_list),
            "N_replicas":   len(sigs),
            "Tipo":         tipo_repr,
            "Concentracion":round(float(conc_repr), 4) if not pd.isna(conc_repr) else np.nan,
            "Media_señal":  round(mean_sig, 5),
            "DE_señal":     round(sd_sig, 5) if not np.isnan(sd_sig) else np.nan,
            "CV_%":         round(cv_sig, 2) if not np.isnan(cv_sig) else np.nan,
        })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN DE PLACA
# ════════════════════════════════════════════════════════════════

def plot_plate_grid(assignment_df: pd.DataFrame, triplicate_groups: dict) -> go.Figure:
    """
    Genera mapa interactivo de la placa coloreado por tipo de pocillo.
    Marca triplicados con un indicador textual.
    """
    row_labels = list("ABCDEFGH")
    tipo_to_roi = {r["ROI"]: r.get("Tipo", "Sin asignar") for _, r in assignment_df.iterrows()}
    conc_to_roi = {r["ROI"]: r.get("Concentracion", np.nan) for _, r in assignment_df.iterrows()}
    nombre_to_roi = {r["ROI"]: r.get("Nombre", "") for _, r in assignment_df.iterrows()}

    # Build replicate index per ROI
    rep_idx = {}
    for col_num, roi_list in triplicate_groups.items():
        for i, roi in enumerate(roi_list):
            rep_idx[roi] = f"Rep. {i+1}"

    # Detect how many rows and cols are in use
    all_rows = sorted(set(roi[0] for roi in assignment_df["ROI"] if roi[0].isalpha()), key=lambda x: row_labels.index(x) if x in row_labels else 99)
    all_cols = sorted(set(int("".join(filter(str.isdigit, roi))) for roi in assignment_df["ROI"] if any(c.isdigit() for c in roi)))

    if not all_rows or not all_cols:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos de placa disponibles", x=0.5, y=0.5, showarrow=False)
        return fig

    n_rows = len(all_rows)
    n_cols = len(all_cols)

    z_matrix    = []
    text_matrix = []
    hover_matrix = []
    color_map_idx = {
        "Blanco":           0,
        "Estandar":         1,
        "Muestra":          2,
        "Control":          3,
        "Adicion estandar": 4,
        "Sin asignar":      5,
    }
    # Numeric color scale
    for row_lbl in all_rows:
        z_row, txt_row, hov_row = [], [], []
        for col_num in all_cols:
            roi_label = f"{row_lbl}{col_num}"
            tipo = tipo_to_roi.get(roi_label, "Sin asignar")
            conc = conc_to_roi.get(roi_label, np.nan)
            nombre = nombre_to_roi.get(roi_label, "")
            rep   = rep_idx.get(roi_label, "")
            z_row.append(color_map_idx.get(tipo, 5))
            display_tipo = tipo if tipo != "Sin asignar" else "--"
            txt_row.append(roi_label)
            hover_text = (
                f"<b>{roi_label}</b><br>"
                f"Tipo: {display_tipo}<br>"
                f"Conc: {conc:.3f}" if not pd.isna(conc) else f"<b>{roi_label}</b><br>Tipo: {display_tipo}"
            )
            if nombre:
                hover_text += f"<br>Nombre: {nombre}"
            if rep:
                hover_text += f"<br>{rep}"
            hov_row.append(hover_text)
        z_matrix.append(z_row)
        text_matrix.append(txt_row)
        hover_matrix.append(hov_row)

    custom_colorscale = [
        [0/5,  "#78350F"],   # Blanco (amber dark)
        [0.8/5,"#78350F"],
        [1/5,  "#052e16"],   # Estandar (green dark)
        [1.8/5,"#052e16"],
        [2/5,  "#1e3a5f"],   # Muestra (blue dark)
        [2.8/5,"#1e3a5f"],
        [3/5,  "#431407"],   # Control (orange dark)
        [3.8/5,"#431407"],
        [4/5,  "#2e1065"],   # Adicion (purple dark)
        [4.8/5,"#2e1065"],
        [5/5,  "#1e293b"],   # Sin asignar (slate dark)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        text=text_matrix,
        texttemplate="%{text}",
        customdata=hover_matrix,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=custom_colorscale,
        showscale=False,
        xgap=3, ygap=3,
        zmin=0, zmax=5,
    ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_cols)),
        ticktext=[str(c) for c in all_cols],
        side="top",
        showgrid=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=all_rows,
        autorange="reversed",
        showgrid=False,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=20, t=60, b=20),
        height=max(220, 50 * n_rows + 80),
        title=dict(text="Distribución de pocillos en placa", font=dict(size=13)),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  GRÁFICAS PLOTLY
# ════════════════════════════════════════════════════════════════

def plot_calibration(concs, signals, cal_result, channel, analyte, unit, lod, loq) -> go.Figure:
    x_min = min(concs) * 0.9 if min(concs) > 0 else min(concs) - abs(min(concs)) * 0.1
    x_max = max(concs) * 1.1
    x_line = np.linspace(x_min, x_max, 300)
    y_line = cal_result["slope"] * x_line + cal_result["intercept"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=concs, y=signals, mode="markers",
        marker=dict(color="#4ade80", size=10, line=dict(color="#1e293b", width=1.5)),
        name="Estándares"
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#60a5fa", width=2.5), name="Regresión lineal"
    ))
    if not np.isnan(lod):
        fig.add_vline(x=lod, line_dash="dot", line_color="#94a3b8",
                      annotation_text=f"LOD = {lod:.3f}", annotation_font_color="#94a3b8",
                      annotation_font_size=10)
    if not np.isnan(loq):
        fig.add_vline(x=loq, line_dash="dot", line_color="#7dd3fc",
                      annotation_text=f"LOQ = {loq:.3f}", annotation_font_color="#7dd3fc",
                      annotation_font_size=10)

    m, b, r2 = cal_result["slope"], cal_result["intercept"], cal_result["r2"]
    sign = "+" if b >= 0 else "-"
    eq   = f"y = {m:.4f}x {sign} {abs(b):.4f}   |   R² = {r2:.5f}"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Curva de calibración — {analyte} | Canal: {channel}", font=dict(size=13)),
        xaxis_title=f"Concentración ({unit})",
        yaxis_title="Absorbancia digital",
        annotations=[dict(x=0.02, y=0.97, xref="paper", yref="paper",
                          text=eq, showarrow=False,
                          font=dict(color="#4ade80", size=11, family="IBM Plex Mono"),
                          bgcolor="rgba(0,0,0,0.5)",
                          bordercolor="#166534", borderwidth=1, borderpad=5)],
        margin=dict(l=50, r=20, t=55, b=50),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    )
    return fig


def plot_standard_addition(added_concs, signals, cal_result, analyte, unit) -> go.Figure:
    x_int = cal_result["x_intercept"]
    c_sam = cal_result["c_sample"]
    m, b  = cal_result["slope"], cal_result["intercept"]
    x_min = min(x_int * 1.4, min(added_concs) - abs(x_int) * 0.2)
    x_max = max(added_concs) * 1.1
    x_line = np.linspace(x_min, x_max, 300)
    y_line = m * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=added_concs, y=signals, mode="markers",
        marker=dict(color="#4ade80", size=10, line=dict(color="#1e293b", width=1.5)),
        name="Adiciones"
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#60a5fa", width=2.5), name="Proyección"
    ))
    fig.add_trace(go.Scatter(
        x=[x_int], y=[0], mode="markers+text",
        marker=dict(color="#f87171", size=13, symbol="x-thin", line=dict(width=3, color="#f87171")),
        text=[f"  C = {c_sam:.4f} {unit}"], textposition="middle right",
        textfont=dict(color="#f87171", size=11, family="IBM Plex Mono"),
        name=f"C muestra: {c_sam:.4f}"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#334155")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Adición de estándar — {analyte}", font=dict(size=13)),
        xaxis_title=f"Concentración añadida ({unit})",
        yaxis_title="Señal (Absorbancia digital)",
        margin=dict(l=50, r=20, t=55, b=50),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    )
    return fig


def plot_channel_comparison(all_channel_res: dict) -> go.Figure:
    channels = list(all_channel_res.keys())
    r2_vals  = [all_channel_res[c]["r2"]  for c in channels]
    se_vals  = [all_channel_res[c]["se"]  for c in channels]
    colors   = ["#4ade80", "#60a5fa", "#f87171"]  # G, B, R-ish

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=r2_vals,
        marker_color=colors[:len(channels)],
        text=[f"{v:.5f}" for v in r2_vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        name="R²",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Comparativa de R² por canal RGB", font=dict(size=13)),
        yaxis=dict(range=[max(0, min(r2_vals) - 0.05), 1.0], title="R²"),
        xaxis_title="Canal",
        margin=dict(l=40, r=20, t=55, b=40),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    )
    return fig


def plot_residuals(concs, cal_result, channel) -> go.Figure:
    residuals = cal_result["residuals"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=concs, y=residuals, mode="markers+lines",
        marker=dict(color="#f97316", size=8, line=dict(color="#1e293b", width=1)),
        line=dict(color="#334155", width=1, dash="dot"),
        name="Residuos"
    ))
    fig.add_hline(y=0, line_color="#94a3b8", line_dash="solid", line_width=1.5)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Análisis de residuos — Canal {channel}", font=dict(size=13)),
        xaxis_title="Concentración",
        yaxis_title="Residuo (observado – calculado)",
        margin=dict(l=50, r=20, t=55, b=50),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN DE REPORTE PDF — PALETA OSCURA
# ════════════════════════════════════════════════════════════════

def fig_to_png_bytes(fig: go.Figure, width=560, height=300) -> bytes | None:
    """Exporta figura Plotly a PNG. Intenta kaleido primero, luego matplotlib."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1.5)
    except Exception:
        return None


def calibration_fig_to_png_matplotlib(cal_result: dict, concs, signals,
                                       channel: str, analyte: str, unit: str,
                                       lod: float, loq: float) -> bytes | None:
    """
    Genera la curva de calibración como PNG usando matplotlib (sin kaleido).
    Paleta completamente oscura, compatible con el PDF de Elementa.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        BG      = "#0f172a"
        CARD    = "#1e293b"
        GREEN   = "#4ade80"
        BLUE    = "#60a5fa"
        RED     = "#f87171"
        ORANGE  = "#fb923c"
        SUBTEXT = "#94a3b8"
        TEXT    = "#e2e8f0"

        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        # Scatter estándares
        ax.scatter(concs, signals, color=GREEN, s=55, zorder=5,
                   edgecolors="#1e293b", linewidths=1.2, label="Estándares")

        # Línea de regresión
        x_min = max(0, min(concs) * 0.9) if min(concs) > 0 else min(concs) - abs(min(concs)) * 0.1
        x_max = max(concs) * 1.1
        x_line = np.linspace(x_min, x_max, 300)
        y_line = cal_result["slope"] * x_line + cal_result["intercept"]
        ax.plot(x_line, y_line, color=BLUE, linewidth=2.2, label="Regresión lineal")

        # LOD / LOQ
        if not np.isnan(lod):
            ax.axvline(lod, color=RED, linestyle=":", linewidth=1.4)
            ax.text(lod, ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0,
                    f"  LOD={lod:.3f}", color=RED, fontsize=7, va="bottom")
        if not np.isnan(loq):
            ax.axvline(loq, color=ORANGE, linestyle=":", linewidth=1.4)
            ax.text(loq, ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0,
                    f"  LOQ={loq:.3f}", color=ORANGE, fontsize=7, va="bottom")

        # Ecuación y R²
        m, b, r2 = cal_result["slope"], cal_result["intercept"], cal_result["r2"]
        sign = "+" if b >= 0 else "-"
        eq_txt = f"y = {m:.4f}x {sign} {abs(b):.4f}   |   R² = {r2:.5f}"
        ax.text(0.03, 0.97, eq_txt, transform=ax.transAxes,
                fontsize=8.5, color=GREEN, va="top", ha="left",
                bbox=dict(facecolor=CARD, edgecolor="#166534", boxstyle="round,pad=0.35"))

        # Etiquetas y estilos
        ax.set_xlabel(f"Concentración ({unit})", color=SUBTEXT, fontsize=9)
        ax.set_ylabel("Absorbancia digital", color=SUBTEXT, fontsize=9)
        ax.set_title(f"Curva de calibración — {analyte}  |  Canal: {channel}",
                     color=TEXT, fontsize=10, pad=8)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.legend(facecolor=CARD, edgecolor="#334155",
                  labelcolor=TEXT, fontsize=8)
        ax.grid(True, color="#1e293b", linewidth=0.6, linestyle="--")

        plt.tight_layout(pad=0.8)
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        buf.seek(0)
        plt.close(fig)
        return buf.read()
    except Exception:
        return None


def generate_pdf_report(analyte: str, method: str,
                         df_rgb: pd.DataFrame | None,
                         df_results: pd.DataFrame | None,
                         cal_result: dict | None,
                         fig_cal: go.Figure | None,
                         annotated_img: np.ndarray | None,
                         triplicate_stats_df: pd.DataFrame | None = None,
                         cal_concs: np.ndarray | None = None,
                         cal_signals: np.ndarray | None = None,
                         cal_channel: str = "",
                         cal_analyte: str = "",
                         cal_unit: str = "mg/L") -> bytes:
    """Genera reporte PDF con paleta oscura profesional y explicaciones estadísticas."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable, KeepTogether)
    from reportlab.lib.colors import HexColor, white, black

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.65*inch, rightMargin=0.65*inch,
                             topMargin=0.70*inch, bottomMargin=0.70*inch)

    # ── Colores corporativos oscuros ──────────────────────────
    NAVY    = HexColor("#0f172a")
    SLATE   = HexColor("#1e293b")
    SLATE2  = HexColor("#263548")
    BORDER_C= HexColor("#334155")
    GREEN_D = HexColor("#052e16")
    GREEN_L = HexColor("#4ade80")
    BLUE_D  = HexColor("#1e3a5f")
    BLUE_L  = HexColor("#93c5fd")
    TEXT_L  = HexColor("#e2e8f0")
    TEXT_M  = HexColor("#94a3b8")
    RED_D   = HexColor("#450a0a")    # fondo alertas — rojo oscuro
    RED_L   = HexColor("#fca5a5")    # texto alertas — rojo claro legible

    styles = getSampleStyleSheet()
    title_sty = ParagraphStyle("T", parent=styles["Title"],
        textColor=GREEN_L, fontSize=17, spaceAfter=2, leading=20,
        fontName="Helvetica-Bold")
    sub_sty = ParagraphStyle("S", parent=styles["Normal"],
        textColor=TEXT_M, fontSize=9, spaceAfter=6, fontName="Helvetica")
    h2_sty = ParagraphStyle("H2", parent=styles["Heading2"],
        textColor=BLUE_L, fontSize=11, spaceAfter=4, spaceBefore=10,
        fontName="Helvetica-Bold")
    h3_sty = ParagraphStyle("H3", parent=styles["Heading3"],
        textColor=GREEN_L, fontSize=9, spaceAfter=3, spaceBefore=6,
        fontName="Helvetica-Bold")
    body_sty = ParagraphStyle("B", parent=styles["BodyText"],
        textColor=TEXT_L, fontSize=8.5, leading=13, fontName="Helvetica")
    italic_sty = ParagraphStyle("I", parent=styles["BodyText"],
        textColor=TEXT_M, fontSize=7.8, leading=12, fontName="Helvetica-Oblique")
    warn_sty = ParagraphStyle("W", parent=styles["BodyText"],
        textColor=RED_L, fontSize=7.8, leading=12, fontName="Helvetica-Oblique")
    foot_sty = ParagraphStyle("F", parent=styles["BodyText"],
        textColor=TEXT_M, fontSize=7, leading=10, alignment=1, fontName="Helvetica")
    mono_sty = ParagraphStyle("M", parent=styles["BodyText"],
        textColor=GREEN_L, fontSize=9, leading=14, fontName="Courier-Bold")

    now = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    story = []

    # ── Encabezado ─────────────────────────────────────────────
    header_data = [[
        Paragraph("ELEMENTA", ParagraphStyle("H", textColor=GREEN_L, fontSize=22,
                                               fontName="Helvetica-Bold")),
        Paragraph(
            f"<b>Reporte de análisis colorimétrico</b><br/>"
            f"<font size='8'>{now}</font><br/>"
            f"<font size='8'>Analito: {analyte}  |  Método: {method}</font>",
            ParagraphStyle("HR", textColor=TEXT_L, fontSize=9, fontName="Helvetica",
                           alignment=2)),
    ]]
    header_tbl = Table(header_data, colWidths=[3.0*inch, 4.3*inch])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), NAVY),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",    (0,0), (-1,0), 2, GREEN_L),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6))

    # ── Advertencia ────────────────────────────────────────────
    warn_data = [[
        Paragraph(
            "<b>AVISO:</b> Elementa realiza estimaciones colorimétricas digitales. "
            "Los resultados NO sustituyen métodos instrumentales en laboratorios acreditados "
            "ni deben usarse para declaraciones de cumplimiento normativo sin confirmación analítica certificada.",
            warn_sty)
    ]]
    warn_tbl = Table(warn_data, colWidths=[7.3*inch])
    warn_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), RED_D),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
    ]))
    story.append(warn_tbl)
    story.append(Spacer(1, 10))

    # ── Imagen anotada ─────────────────────────────────────────
    if annotated_img is not None:
        story.append(Paragraph("Imagen procesada — ROIs definidas", h2_sty))
        pil_img = Image.fromarray(annotated_img)
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=4.5*inch, height=3.0*inch, kind="proportional")
        story.append(rl_img)
        story.append(Spacer(1, 8))

    # ── Tabla RGB ──────────────────────────────────────────────
    if df_rgb is not None and not df_rgb.empty:
        story.append(Paragraph("Datos colorimétricos RGB por región de interés", h2_sty))
        story.append(Paragraph(
            "R_mean, G_mean, B_mean: intensidad media de cada canal (0–255) en la región. "
            "R_norm, G_norm, B_norm: fracción porcentual de cada canal respecto a la suma total RGB, "
            "lo que normaliza la señal frente a cambios de iluminación global.",
            italic_sty))
        story.append(Spacer(1, 4))

        show_cols = [c for c in ["ROI","R_mean","G_mean","B_mean","R_norm","G_norm","B_norm"]
                     if c in df_rgb.columns]
        df_show = df_rgb[show_cols].round(2)

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        tbl_data = [show_cols] + [[fmt(v) for v in row] for row in df_show.values.tolist()]
        col_w = [1.0*inch] + [0.9*inch] * (len(show_cols)-1)
        tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
        row_bg = [SLATE, SLATE2]
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), BLUE_D),
            ("TEXTCOLOR",     (0,0), (-1,0), BLUE_L),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7.5),
            ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_L),
            ("FONTNAME",      (0,1), (-1,-1), "Courier"),
            ("GRID",          (0,0), (-1,-1), 0.4, BORDER_C),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), row_bg),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    # ── Parámetros de calibración ──────────────────────────────
    if cal_result:
        story.append(Paragraph("Parámetros de calibración", h2_sty))
        story.append(Spacer(1, 3))

        cal_items = [
            ("R² — Coeficiente de determinación",
             f"{cal_result['r2']:.5f}",
             STAT_EXPLANATIONS["R2"]),
            ("m — Pendiente",
             f"{cal_result['slope']:.4f}",
             STAT_EXPLANATIONS["slope"]),
            ("b — Intercepto",
             f"{cal_result['intercept']:.4f}",
             STAT_EXPLANATIONS["intercept"]),
            ("Sb — Error estándar de la pendiente",
             f"{cal_result['se']:.5f}",
             STAT_EXPLANATIONS["se"]),
            ("n — Número de estándares",
             str(cal_result.get("n","N/A")),
             "Número de puntos de calibración incluidos en el ajuste lineal."),
            ("LOD — Límite de detección",
             f"{cal_result.get('LOD', float('nan')):.4f}" if not np.isnan(cal_result.get('LOD', float('nan'))) else "N/D",
             STAT_EXPLANATIONS["LOD"]),
            ("LOQ — Límite de cuantificación",
             f"{cal_result.get('LOQ', float('nan')):.4f}" if not np.isnan(cal_result.get('LOQ', float('nan'))) else "N/D",
             STAT_EXPLANATIONS["LOQ"]),
        ]

        for param, value, explanation in cal_items:
            block = [
                [
                    Paragraph(param, ParagraphStyle("CP", textColor=TEXT_M, fontSize=8,
                                                     fontName="Helvetica-Bold")),
                    Paragraph(value, ParagraphStyle("CV", textColor=GREEN_L, fontSize=10,
                                                     fontName="Courier-Bold")),
                ],
                [
                    Paragraph(explanation, italic_sty),
                    "",
                ],
            ]
            btbl = Table(block, colWidths=[4.2*inch, 3.1*inch])
            btbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), SLATE),
                ("BACKGROUND",    (0,1), (-1,1), NAVY),
                ("GRID",          (0,0), (-1,-1), 0.3, BORDER_C),
                ("SPAN",          (0,1), (-1,1)),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ]))
            story.append(KeepTogether([btbl, Spacer(1, 4)]))

        if cal_result.get("lod_proxy"):
            story.append(Paragraph(
                "Nota: LOD y LOQ calculados usando el error estándar residual del ajuste como proxy de sigma_blanco. "
                "Para mayor rigor, incluya al menos 10 réplicas de blanco y use su desviación estándar.",
                warn_sty))
        story.append(Spacer(1, 8))

        # ── Gráfica de calibración ────────────────────────────
        if fig_cal is not None:
            story.append(Paragraph("Gráfica de calibración", h2_sty))
            png_bytes = None
            # 1. Intentar matplotlib (siempre disponible, sin kaleido)
            if cal_concs is not None and cal_signals is not None and cal_result is not None:
                lod_v = cal_result.get("LOD", float("nan"))
                loq_v = cal_result.get("LOQ", float("nan"))
                png_bytes = calibration_fig_to_png_matplotlib(
                    cal_result, cal_concs, cal_signals,
                    cal_channel, cal_analyte, cal_unit, lod_v, loq_v)
            # 2. Fallback: kaleido si matplotlib falla
            if not png_bytes:
                png_bytes = fig_to_png_bytes(fig_cal, width=560, height=310)

            if png_bytes:
                img_buf2 = BytesIO(png_bytes)
                rl_cal = RLImage(img_buf2, width=5.5*inch, height=3.0*inch, kind="proportional")
                story.append(rl_cal)
                story.append(Paragraph(
                    "La línea azul es la regresión lineal ajustada. Los marcadores verdes son los "
                    "estándares. Las líneas verticales punteadas indican el LOD y el LOQ.",
                    italic_sty))
            else:
                story.append(Paragraph(
                    "(No fue posible generar la imagen de la gráfica en este entorno.)",
                    italic_sty))
            story.append(Spacer(1, 8))

    # ── Triplicados ────────────────────────────────────────────
    if triplicate_stats_df is not None and not triplicate_stats_df.empty:
        story.append(Paragraph("Estadísticas de triplicados", h2_sty))
        story.append(Paragraph(
            STAT_EXPLANATIONS["CV"], italic_sty))
        story.append(Spacer(1, 4))

        tri_cols = [c for c in triplicate_stats_df.columns]
        tri_data = [tri_cols] + [[str(v) for v in row] for row in triplicate_stats_df.values.tolist()]
        col_w_tri = [max(0.8*inch, 7.3*inch / len(tri_cols))] * len(tri_cols)
        tbl_tri = Table(tri_data, colWidths=col_w_tri, repeatRows=1)
        tbl_tri.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), GREEN_D),
            ("TEXTCOLOR",     (0,0), (-1,0), GREEN_L),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7.5),
            ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_L),
            ("FONTNAME",      (0,1), (-1,-1), "Courier"),
            ("GRID",          (0,0), (-1,-1), 0.4, BORDER_C),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [SLATE, SLATE2]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]))
        story.append(tbl_tri)
        story.append(Spacer(1, 10))

    # ── Resultados de muestras ─────────────────────────────────
    if df_results is not None and not df_results.empty:
        story.append(Paragraph("Resultados de muestras — concentraciones calculadas", h2_sty))
        story.append(Paragraph(
            "Conc_calc: concentración calculada directamente de la curva de calibración "
            "(x = (y − b) / m). Conc_corregida: valor multiplicado por el factor de dilución "
            "indicado por el usuario. A_dig: absorbancia digital calculada como "
            "log10(I_blanco / I_muestra) en el canal seleccionado.",
            italic_sty))
        story.append(Spacer(1, 4))

        res_cols = [c for c in df_results.columns]
        res_data = [res_cols] + [[str(v) for v in row] for row in df_results.values.tolist()]
        col_w_res = [max(0.7*inch, 7.3*inch / len(res_cols))] * len(res_cols)
        tbl_res = Table(res_data, colWidths=col_w_res, repeatRows=1)
        tbl_res.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), BLUE_D),
            ("TEXTCOLOR",     (0,0), (-1,0), BLUE_L),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7.5),
            ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_L),
            ("FONTNAME",      (0,1), (-1,-1), "Courier"),
            ("GRID",          (0,0), (-1,-1), 0.4, BORDER_C),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [SLATE, SLATE2]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]))
        story.append(tbl_res)
        story.append(Spacer(1, 10))

    # ── Nota científica y pie ──────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_C))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Nota científica: La precisión de las estimaciones colorimétricas digitales "
        "depende de la uniformidad de la iluminación, las características del sensor de la cámara, "
        "la calidad de los reactivos, la linealidad del método y la reproducibilidad de la preparación "
        "de estándares. Para declaraciones de cumplimiento normativo, los resultados deben confirmarse "
        "mediante métodos oficiales en laboratorios acreditados. Consultar siempre la versión vigente "
        "de las normas aplicables en el Diario Oficial de la Federación.",
        warn_sty))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Derechos reservados (Katyutzka, 2026)  |  Elementa — Sistema Analítico Colorimétrico Digital", foot_sty))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════
#  ESTADO DE SESIÓN
# ════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "image": None, "rois": [], "freeze_rois": False,
        "df_rgb": None, "df_norm": None, "df_absorbance": None,
        "assignment_df": None, "blank_label": None,
        "cal_result": None, "best_channel": "G_norm",
        "all_channel_res": {}, "df_results": None,
        "annotated_img": None, "cal_fig": None, "residual_fig": None,
        "sa_fig": None, "sa_result": None,
        "triplicate_groups": {}, "triplicate_stats_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def render_footer():
    st.markdown(
        '<div class="footer">Derechos reservados (Katyutzka, 2026)'
        ' &nbsp;|&nbsp; Elementa — Sistema Analítico Colorimétrico Digital</div>',
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, explain: str, col, border_color: str = EMERALD):
    col.markdown(
        f'<div class="metric-card" style="border-left-color:{border_color};">'
        f'<h4>{title}</h4><p>{value}</p>'
        f'<div class="stat-explain">{explain}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
#  BARRA LATERAL
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{EMERALD};margin-bottom:2px;font-size:1.5rem;'>"
        f"Elementa</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUT};font-size:0.77rem;margin-top:0;letter-spacing:0.06em;'>"
        f"SISTEMA COLORIMÉTRICO DIGITAL</p>", unsafe_allow_html=True)
    st.divider()
    pagina = st.radio(
        "Navegación",
        ["Analisis", "Fundamentos", "Normativa y Fuentes"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"<p style='color:{TEXT_MUT};font-size:0.72rem;line-height:1.6;'>"
        f"Los resultados son estimaciones colorimétricas digitales. "
        f"No sustituyen métodos certificados ni análisis en laboratorios acreditados.</p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
#  PÁGINA 1: ANÁLISIS
# ════════════════════════════════════════════════════════════════

if pagina == "Analisis":

    st.markdown("<h1>Análisis colorimétrico digital</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Flujo de trabajo: '
        '<b>1. Cargar imagen</b> &rarr; '
        '<b>2. Definir ROIs</b> &rarr; '
        '<b>3. Asignar roles</b> &rarr; '
        '<b>4. Calibrar</b> &rarr; '
        '<b>5. Cuantificar</b> &rarr; '
        '<b>6. Evaluar norma</b> &rarr; '
        '<b>7. Exportar</b>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── PASO 1 ────────────────────────────────────────────────
    st.markdown('<p class="step-header">Paso 1 — Cargar imagen</p>', unsafe_allow_html=True)

    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Subir imagen (JPG/PNG)", type=["jpg","jpeg","png"],
                                          label_visibility="visible")
        if uploaded_file:
            st.session_state["image"] = load_image(uploaded_file)
    with col_cam:
        cam_img = st.camera_input("Capturar con cámara")
        if cam_img:
            st.session_state["image"] = load_image(cam_img)

    if st.session_state["image"] is not None:
        img = st.session_state["image"]
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img, caption="Imagen original", use_container_width=True)
        with col_img2:
            if st.session_state["annotated_img"] is not None:
                st.image(st.session_state["annotated_img"],
                         caption="Imagen con ROIs", use_container_width=True)
            else:
                st.markdown('<div class="info-box">Las regiones de interés (ROIs) se visualizarán aquí una vez definidas.</div>',
                            unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Cargue o capture una imagen para comenzar el análisis.</div>',
                    unsafe_allow_html=True)
        render_footer()
        st.stop()

    img = st.session_state["image"]
    h_img, w_img = img.shape[:2]

    # ── PASO 2 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 2 — Definir regiones de interés (ROIs)</p>', unsafe_allow_html=True)

    with st.expander("Acerca de las ROIs", expanded=False):
        st.markdown(
            "Una región de interés (ROI) es el área rectangular dentro de la imagen "
            "de la cual se extraen los valores promedio R, G y B. La precisión del análisis "
            "depende de que cada ROI esté centrada sobre el pocillo o vial correspondiente, "
            "con un tamaño suficiente para promediar el color interno sin capturar los bordes. "
            "Use el modo de bloqueo para fijar las ROIs una vez que estén bien posicionadas."
        )

    # ── Layout lado a lado: controles | imagen ─────────────────
    roi_ctrl_col, roi_img_col = st.columns([1, 1], gap="large")

    with roi_ctrl_col:
        device_type = st.selectbox(
            "Tipo de dispositivo",
            ["Viales lineales", "Microplaca de 96 pocillos", "Personalizado"],
        )
        freeze = st.toggle("Bloquear ROIs (una vez posicionadas correctamente)",
                           value=st.session_state["freeze_rois"])
        st.session_state["freeze_rois"] = freeze

        if not freeze:
            if device_type == "Viales lineales":
                n_rois = st.number_input("Numero de viales", 2, 24, 6, 1)
                x0     = st.slider("X inicial (px)",   0, w_img-1, int(w_img*0.05))
                y0     = st.slider("Y inicial (px)",   0, h_img-1, int(h_img*0.25))
                roi_w  = st.slider("Ancho ROI (px)",   5, 200, 40)
                roi_h  = st.slider("Alto ROI (px)",    5, 300, 60)
                dx     = st.slider("Espaciado X (px)", 0, 300, int(w_img*0.08))
                dy     = st.slider("Espaciado Y (px)", 0, 200, 0)
                rois   = generate_rois_linear(x0, y0, roi_w, roi_h, int(n_rois), dx, dy)

            elif device_type == "Microplaca de 96 pocillos":
                x0    = st.slider("X inicial (px)",    0, w_img-1, int(w_img*0.05))
                y0    = st.slider("Y inicial (px)",    0, h_img-1, int(h_img*0.05))
                roi_w = st.slider("Ancho ROI (px)",    4, 80,  20)
                roi_h = st.slider("Alto ROI (px)",     4, 80,  20)
                dx    = st.slider("Espaciado X (px)", 10, 200, 50)
                dy    = st.slider("Espaciado Y (px)", 10, 200, 50)
                rows  = st.number_input("Filas",    1, 8,  8, 1)
                cols  = st.number_input("Columnas", 1, 12, 12, 1)
                rois  = generate_rois_microplate(x0, y0, roi_w, roi_h, dx, dy, int(rows), int(cols))
            else:
                n_rois = st.number_input("Numero de ROIs", 2, 50, 6, 1)
                x0     = st.slider("X inicial (px)",   0, w_img-1, int(w_img*0.05))
                y0     = st.slider("Y inicial (px)",   0, h_img-1, int(h_img*0.1))
                roi_w  = st.slider("Ancho ROI (px)",   5, 200, 30)
                roi_h  = st.slider("Alto ROI (px)",    5, 200, 30)
                dx     = st.slider("Espaciado X (px)", 0, 300, int(w_img*0.08))
                dy     = st.slider("Espaciado Y (px)", 0, 300, int(h_img*0.08))
                rois   = generate_rois_linear(x0, y0, roi_w, roi_h, int(n_rois), dx, dy)

            st.session_state["rois"] = rois
        else:
            rois = st.session_state["rois"]
            st.markdown(f'<div class="success-box">ROIs bloqueadas — {len(rois)} regiones definidas.</div>',
                        unsafe_allow_html=True)

        if not rois:
            st.markdown('<div class="warn-box">No hay ROIs definidas. Configure los parámetros de posición y tamaño.</div>',
                        unsafe_allow_html=True)

    # Imagen en tiempo real con ROIs dibujadas (columna derecha)
    with roi_img_col:
        assignments_dict = {}
        if st.session_state["assignment_df"] is not None:
            for _, row in st.session_state["assignment_df"].iterrows():
                assignments_dict[row["ROI"]] = row.get("Tipo", "Sin asignar")

        if rois:
            ann_img_preview = draw_rois(img, rois, assignments_dict)
            st.session_state["annotated_img"] = ann_img_preview
            st.image(ann_img_preview,
                     caption="Vista previa en tiempo real — ajusta los controles para reposicionar las ROIs",
                     use_container_width=True)
        else:
            st.image(img, caption="Imagen original (sin ROIs aún)", use_container_width=True)

    if not rois:
        render_footer()
        st.stop()

    # Sincronizar assignments_dict para uso posterior
    assignments_dict = {}
    if st.session_state["assignment_df"] is not None:
        for _, row in st.session_state["assignment_df"].iterrows():
            assignments_dict[row["ROI"]] = row.get("Tipo", "Sin asignar")

    # ── PASO 3 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 3 — Asignar roles y concentraciones</p>', unsafe_allow_html=True)

    with st.expander("Convenciones de asignacion", expanded=False):
        st.markdown(
            """
**Blanco de reactivos:** pocillo con todos los reactivos excepto el analito. Define la señal de referencia (absorbancia digital = 0).

**Estandar:** solución de concentración conocida para construir la curva de calibración. Se requieren al menos 5 niveles de concentración.

**Muestra:** solución de concentración desconocida que se cuantifica con la curva.

**Control:** solución de concentración conocida diferente a los estándares, usada para verificar la exactitud del método.

**Adicion de estandar:** alícuotas de la muestra a las que se añade una cantidad conocida del analito para el método de adición estándar.

**Triplicados:** en diseño de microplaca, los pocillos de la misma columna (p. ej. A1, B1, C1) se tratan automáticamente como réplicas de la misma condición.
            """
        )

    TIPOS = ["Sin asignar","Blanco","Estandar","Muestra","Control","Adicion estandar"]
    ANALITOS = ["Pb","Cd","Cr total","Cr(VI)","DPPH","ABTS","FRAP","Fenoles totales","Otro"]
    UNIDADES = ["mg/L","ug/L","ppm","uM","mM","%","ug/mL","Otro"]

    if (st.session_state["assignment_df"] is None or
            len(st.session_state["assignment_df"]) != len(rois)):
        init_rows = []
        for roi in rois:
            init_rows.append({
                "ROI": roi["label"], "Tipo": "Sin asignar", "Nombre": "",
                "Concentracion": 0.0, "Unidad": "mg/L",
                "Factor_dil": 1.0, "Analito": "Cr(VI)", "Observaciones": "",
            })
        st.session_state["assignment_df"] = pd.DataFrame(init_rows)

    edited_df = st.data_editor(
        st.session_state["assignment_df"],
        column_config={
            "Tipo":    st.column_config.SelectboxColumn("Tipo",    options=TIPOS,    required=True),
            "Unidad":  st.column_config.SelectboxColumn("Unidad",  options=UNIDADES, required=True),
            "Analito": st.column_config.SelectboxColumn("Analito", options=ANALITOS, required=True),
            "Concentracion": st.column_config.NumberColumn("Conc.", min_value=0.0, step=0.001, format="%.4f"),
            "Factor_dil":    st.column_config.NumberColumn("F. Dil.", min_value=0.01, step=0.1, format="%.2f"),
        },
        num_rows="fixed", use_container_width=True, key="assignment_editor",
    )
    st.session_state["assignment_df"] = edited_df

    for _, row in edited_df.iterrows():
        assignments_dict[row["ROI"]] = row.get("Tipo", "Sin asignar")
    ann_img = draw_rois(img, rois, assignments_dict)
    st.session_state["annotated_img"] = ann_img
    # Mostrar imagen actualizada con colores de tipo bajo la tabla de asignación
    st.image(ann_img,
             caption="Imagen actualizada con tipos de ROI asignados — verde=Estándar, azul=Muestra, amber=Blanco",
             use_container_width=True)

    blank_rows  = edited_df[edited_df["Tipo"] == "Blanco"]
    blank_label = blank_rows["ROI"].iloc[0] if len(blank_rows) > 0 else None
    st.session_state["blank_label"] = blank_label

    if blank_label:
        st.markdown(f'<div class="success-box">Blanco de reactivos asignado: <b>{blank_label}</b></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">No se ha marcado ningun ROI como Blanco. La absorbancia digital requiere un blanco de reactivos.</div>',
                    unsafe_allow_html=True)

    # ── Visualización de placa ────────────────────────────────
    if device_type == "Microplaca de 96 pocillos" and not edited_df.empty:
        st.markdown('<p class="step-header" style="margin-top:16px;">Distribucion de pocillos en placa</p>',
                    unsafe_allow_html=True)

        # Leyenda
        legend_html = ""
        for tipo, color in TYPE_COLORS.items():
            label = tipo if tipo != "Sin asignar" else "Sin asignar / Inactivo"
            legend_html += (
                f'<span class="plate-legend-item">'
                f'<span class="plate-legend-dot" style="background:{color};"></span>'
                f'{label}</span>'
            )
        st.markdown(f'<div style="margin-bottom:8px;">{legend_html}</div>',
                    unsafe_allow_html=True)

        triplicate_groups = detect_triplicate_groups(edited_df)
        st.session_state["triplicate_groups"] = triplicate_groups

        plate_fig = plot_plate_grid(edited_df, triplicate_groups)
        st.plotly_chart(plate_fig, use_container_width=True)

        if triplicate_groups:
            n_groups = len(triplicate_groups)
            total_reps = sum(len(v) for v in triplicate_groups.values())
            st.markdown(
                f'<div class="info-box">'
                f'Se detectaron <b>{n_groups} grupos de triplicados</b> '
                f'({total_reps} pocillos en total). '
                f'Los pocillos grises no tienen tipo asignado y se excluyen del análisis.</div>',
                unsafe_allow_html=True,
            )
            # Tabla de triplicados detectados
            with st.expander("Detalle de grupos de triplicados", expanded=False):
                tri_rows = []
                for col_num, roi_list in sorted(triplicate_groups.items(), key=lambda x: int(x[0])):
                    rep_rois = roi_list
                    tipo_repr = edited_df.loc[edited_df["ROI"].isin(rep_rois), "Tipo"].iloc[0] if not edited_df[edited_df["ROI"].isin(rep_rois)].empty else ""
                    conc_repr = edited_df.loc[edited_df["ROI"].isin(rep_rois), "Concentracion"].iloc[0] if not edited_df[edited_df["ROI"].isin(rep_rois)].empty else np.nan
                    tri_rows.append({
                        "Columna": col_num,
                        "Pocillos": ", ".join(rep_rois),
                        "N repl.":  len(rep_rois),
                        "Tipo":     tipo_repr,
                        "Conc.":    round(float(conc_repr), 4) if not pd.isna(conc_repr) else "—",
                    })
                st.dataframe(pd.DataFrame(tri_rows), use_container_width=True, hide_index=True)

    # ── PASO 4 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 4 — Extraccion de color y calibracion</p>', unsafe_allow_html=True)

    with st.expander("Fundamento del calculo de absorbancia digital", expanded=False):
        st.markdown(
            """
La **absorbancia digital** se define por analogía con la ley de Beer-Lambert:

**A_dig = log10(I_blanco / I_muestra)**

donde I es la intensidad normalizada del canal seleccionado (R, G o B expresado como porcentaje de la suma total RGB).
La normalización porcentual mitiga los efectos de cambios globales de iluminación entre capturas.

El sistema evalúa los tres canales y selecciona automáticamente el de **mayor R²** en la curva de calibración,
ya que distintos analitos y reactivos colorimétricos tienen su máximo de absorbancia en diferentes regiones espectrales.
            """
        )

    if st.button("Extraer RGB y calibrar"):
        with st.spinner("Procesando canales de color y ajuste de calibracion..."):
            df_rgb  = extract_rgb_stats(img, rois)
            df_norm = calculate_normalized_intensity(df_rgb)
            channels_for_abs = ["R_norm", "G_norm", "B_norm"]
            df_abs  = calculate_digital_absorbance(df_norm, blank_label, channels_for_abs)
            df_abs  = df_abs.merge(edited_df, on="ROI", how="left")

            st.session_state["df_rgb"]        = df_rgb
            st.session_state["df_norm"]       = df_norm
            st.session_state["df_absorbance"] = df_abs

            std_df = df_abs[df_abs["Tipo"] == "Estandar"].copy()
            sel    = select_best_channel(df_abs, std_df["ROI"].tolist())
            best_ch = sel["best_channel"]
            all_chs = sel["results"]
            st.session_state["best_channel"]    = best_ch
            st.session_state["all_channel_res"] = all_chs

            if best_ch in all_chs and len(std_df) >= 2:
                cal = all_chs[best_ch]
                blank_df   = df_abs[df_abs["Tipo"] == "Blanco"]
                blank_sigs = blank_df[f"A_dig_{best_ch}"].dropna().values if not blank_df.empty else None
                lod_loq_r  = calculate_lod_loq(cal, blank_sigs)
                cal["LOD"]       = lod_loq_r["LOD"]
                cal["LOQ"]       = lod_loq_r["LOQ"]
                cal["lod_proxy"] = lod_loq_r["proxy"]
                st.session_state["cal_result"] = cal

                concs   = std_df["Concentracion"].values.astype(float)
                signals = std_df[f"A_dig_{best_ch}"].values.astype(float)
                unit    = std_df["Unidad"].iloc[0] if not std_df.empty else "mg/L"
                analyte = std_df["Analito"].iloc[0] if not std_df.empty else "Analito"

                cal_fig   = plot_calibration(concs, signals, cal, best_ch, analyte,
                                              unit, cal["LOD"], cal["LOQ"])
                res_fig   = plot_residuals(concs, cal, best_ch)
                st.session_state["cal_fig"]      = cal_fig
                st.session_state["residual_fig"] = res_fig

            # Calcular stats de triplicados si aplica
            triplicate_groups = st.session_state.get("triplicate_groups", {})
            if triplicate_groups and best_ch:
                sig_col = f"A_dig_{best_ch}"
                tri_stats = calculate_triplicate_stats(df_abs, edited_df, triplicate_groups, sig_col)
                st.session_state["triplicate_stats_df"] = tri_stats

            st.markdown('<div class="success-box">Extraccion y calibracion completadas correctamente.</div>',
                        unsafe_allow_html=True)

    # Mostrar resultados de calibración
    if st.session_state["df_absorbance"] is not None:
        df_abs = st.session_state["df_absorbance"]

        with st.expander("Tabla de absorbancias digitales por ROI", expanded=False):
            show_cols = ["ROI","Tipo","R_mean","G_mean","B_mean",
                         "R_norm","G_norm","B_norm",
                         "A_dig_R_norm","A_dig_G_norm","A_dig_B_norm"]
            disp_cols = [c for c in show_cols if c in df_abs.columns]
            st.dataframe(df_abs[disp_cols].round(4), use_container_width=True)

        if st.session_state["all_channel_res"]:
            with st.expander("Comparativa de canales R, G, B", expanded=True):
                ch_rows = []
                for ch, res in st.session_state["all_channel_res"].items():
                    ch_rows.append({
                        "Canal": ch,
                        "R²":           round(res["r2"], 5),
                        "Pendiente m":  round(res["slope"], 4),
                        "Intercepto b": round(res["intercept"], 4),
                        "Error est. Sb":round(res["se"], 5),
                    })
                df_ch = pd.DataFrame(ch_rows).sort_values("R²", ascending=False)
                st.dataframe(df_ch, use_container_width=True, hide_index=True)
                best = st.session_state["best_channel"]
                st.markdown(f'<div class="success-box">Canal seleccionado automaticamente: <b>{best}</b> (R² máximo)</div>',
                            unsafe_allow_html=True)
                ch_comp_fig = plot_channel_comparison(st.session_state["all_channel_res"])
                st.plotly_chart(ch_comp_fig, use_container_width=True)

        if st.session_state["cal_result"]:
            cal = st.session_state["cal_result"]
            c1, c2, c3, c4 = st.columns(4)
            stat_card("R² — Ajuste lineal", f"{cal['r2']:.5f}",
                      "Proporcion de varianza explicada. > 0.999 = excelente linealidad.", c1, EMERALD)
            stat_card("m — Pendiente", f"{cal['slope']:.4f}",
                      "Sensibilidad analitica del metodo.", c2, BLUE_ACC)
            lod_v = cal.get("LOD", float("nan"))
            loq_v = cal.get("LOQ", float("nan"))
            stat_card("LOD", f"{lod_v:.4f}" if not np.isnan(lod_v) else "N/D",
                      "Concentracion minima detectable (3.3 sigma / |m|).", c3, AMBER)
            stat_card("LOQ", f"{loq_v:.4f}" if not np.isnan(loq_v) else "N/D",
                      "Concentracion minima cuantificable con CV < 10% (10 sigma / |m|).", c4, ROSE)

            if cal.get("lod_proxy"):
                st.markdown(
                    '<div class="warn-box">LOD y LOQ calculados usando el error estandar residual como proxy de sigma_blanco. '
                    'Para mayor rigor, incluya replicas del blanco de reactivos (minimo 10) y use su desviacion estandar.</div>',
                    unsafe_allow_html=True)
            if cal["slope"] < 0:
                st.markdown(
                    '<div class="info-box"><b>Pendiente negativa detectada.</b> Es el comportamiento esperado en ensayos '
                    'donde el analito reduce la intensidad del color (p. ej. DPPH, ABTS). La cuantificacion aplica '
                    'la ecuacion x = (y − b) / m sin ninguna modificacion; el resultado es correcto.</div>',
                    unsafe_allow_html=True)

            tab_cal, tab_res = st.tabs(["Curva de calibracion", "Analisis de residuos"])
            with tab_cal:
                st.plotly_chart(st.session_state["cal_fig"], use_container_width=True)
                st.markdown(
                    "La gráfica muestra los estándares (puntos) y el ajuste lineal (línea). "
                    "Las líneas punteadas verticales marcan el LOD y el LOQ. "
                    "Una buena curva de calibración tiene todos los puntos cercanos a la línea y residuos aleatorios.",
                    unsafe_allow_html=True,
                )
            with tab_res:
                if st.session_state["residual_fig"]:
                    st.plotly_chart(st.session_state["residual_fig"], use_container_width=True)
                    st.markdown(
                        "Los residuos deben distribuirse aleatoriamente alrededor de cero sin tendencias sistemáticas. "
                        "Un patrón curvo o en forma de abanico indica no-linealidad o heterocedasticidad, "
                        "respectivamente, y requiere revisión del rango de calibración.",
                        unsafe_allow_html=True,
                    )

        # Estadísticas de triplicados
        if st.session_state.get("triplicate_stats_df") is not None:
            tri_df = st.session_state["triplicate_stats_df"]
            if not tri_df.empty:
                with st.expander("Estadisticas de triplicados", expanded=True):
                    st.dataframe(tri_df, use_container_width=True, hide_index=True)
                    st.markdown(
                        f'<div class="info-box"><b>CV%</b> — {STAT_EXPLANATIONS["CV"]}</div>',
                        unsafe_allow_html=True,
                    )
                    # Destacar grupos con CV alto
                    high_cv = tri_df[tri_df["CV_%"] > 10] if "CV_%" in tri_df.columns else pd.DataFrame()
                    if not high_cv.empty:
                        grupos = ", ".join(high_cv["Grupo"].tolist())
                        st.markdown(
                            f'<div class="warn-box">Los siguientes grupos presentan CV% > 10%, '
                            f'lo que indica baja reproducibilidad: <b>{grupos}</b>. '
                            f'Revise la tecnica de pipeteo y las condiciones de reaccion.</div>',
                            unsafe_allow_html=True,
                        )

    # ── PASO 5 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 5 — Cuantificacion de muestras</p>', unsafe_allow_html=True)

    method_choice = st.radio(
        "Metodo de cuantificacion",
        ["Calibracion externa", "Adicion de estandar"],
        horizontal=True,
    )

    if method_choice == "Calibracion externa":
        with st.expander("Como funciona la calibracion externa", expanded=False):
            st.markdown(
                "La concentración de la muestra se calcula despejando x de la ecuación de calibración: "
                "**x = (A_dig − b) / m**. Si se aplica un factor de dilución, el resultado se multiplica "
                "por dicho factor para obtener la concentración real en la muestra original."
            )
        if st.button("Calcular concentraciones (calibracion externa)"):
            if st.session_state["cal_result"] is None or st.session_state["df_absorbance"] is None:
                st.error("Realice primero la extraccion y calibracion (Paso 4).")
            else:
                cal     = st.session_state["cal_result"]
                df_a    = st.session_state["df_absorbance"].copy()
                best_ch = st.session_state["best_channel"]
                a_col   = f"A_dig_{best_ch}"
                m, b    = cal["slope"], cal["intercept"]

                sample_df = df_a[df_a["Tipo"] == "Muestra"].copy()
                results   = []
                for _, row in sample_df.iterrows():
                    a_val = row.get(a_col, np.nan)
                    if pd.isna(a_val) or abs(m) < 1e-12:
                        conc_raw = np.nan
                    else:
                        conc_raw = (a_val - b) / m
                    dil      = row.get("Factor_dil", 1.0) or 1.0
                    conc_cor = conc_raw * dil if not np.isnan(conc_raw) else np.nan
                    results.append({
                        "Muestra":        row.get("Nombre") or row["ROI"],
                        "ROI":            row["ROI"],
                        "Canal":          best_ch,
                        "A_dig":          round(a_val, 4)    if not np.isnan(a_val)    else "N/D",
                        "Conc_calc":      round(conc_raw, 4) if not np.isnan(conc_raw) else "N/D",
                        "Factor_dil":     dil,
                        "Conc_corregida": round(conc_cor, 4) if not np.isnan(conc_cor) else "N/D",
                        "Unidad":         row.get("Unidad", "mg/L"),
                        "Analito":        row.get("Analito", ""),
                    })
                df_res = pd.DataFrame(results)
                st.session_state["df_results"] = df_res
                st.dataframe(df_res, use_container_width=True, hide_index=True)

    else:  # Adición de estándar
        with st.expander("Como funciona la adicion de estandar", expanded=False):
            st.markdown(
                "El método de adición estándar corrige interferencias de matriz. "
                "Se añaden cantidades conocidas del analito (C_añadida) a alícuotas de la muestra "
                "y se mide la señal resultante. La regresión Señal = m·C_añadida + b se extrapola "
                "al eje X negativo: **C_muestra = |b / m|**."
            )

        sa_n = st.number_input("Numero de puntos de adicion (sin contar la muestra sin adicion)", 2, 8, 3)
        sa_rows = [{"C_anadida": 0.0, "Senal": 0.0}]
        for _ in range(int(sa_n)):
            sa_rows.append({"C_anadida": 0.0, "Senal": 0.0})

        sa_df_edit = st.data_editor(
            pd.DataFrame(sa_rows),
            column_config={
                "C_anadida": st.column_config.NumberColumn("C anadida (mg/L)", step=0.001, format="%.4f"),
                "Senal":     st.column_config.NumberColumn("Senal (A_dig)",    step=0.0001, format="%.5f"),
            },
            use_container_width=True, num_rows="fixed",
        )

        if st.button("Calcular (adicion de estandar)"):
            added = sa_df_edit["C_anadida"].values.astype(float)
            sigs  = sa_df_edit["Senal"].values.astype(float)
            sa_r  = standard_addition_analysis(added, sigs)
            if sa_r is None:
                st.error("No fue posible ajustar la regresion. Revise que los valores ingresados sean correctos y no colineales.")
            else:
                st.session_state["sa_result"] = sa_r
                asgn = st.session_state["assignment_df"]
                analyte_sel = asgn["Analito"].iloc[0] if asgn is not None else "Analito"
                unit_sel    = asgn["Unidad"].iloc[0]  if asgn is not None else "mg/L"
                sa_fig = plot_standard_addition(added, sigs, sa_r, analyte_sel, unit_sel)
                st.session_state["sa_fig"] = sa_fig
                c_est = sa_r["c_sample"]
                stat_card(
                    "Concentracion estimada (adicion de estandar)",
                    f"{c_est:.4f} {unit_sel}",
                    f"R² del ajuste: {sa_r['r2']:.5f}  |  Pendiente: {sa_r['slope']:.4f}",
                    st,
                )
                st.plotly_chart(sa_fig, use_container_width=True)

    # ── PASO 6 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 6 — Evaluacion normativa</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="warn-box">Verifique siempre los limites permisibles en la version oficial vigente '
        'de la norma aplicable. Los valores mostrados son informativos y pueden haber sido actualizados. '
        'Consultar el Diario Oficial de la Federacion (DOF).</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["df_results"] is not None and not st.session_state["df_results"].empty:
        df_res = st.session_state["df_results"]
        for _, row in df_res.iterrows():
            c_val = row.get("Conc_corregida", "N/D")
            an    = row.get("Analito", "")
            if c_val in ("N/D", "N/A") or pd.isna(c_val):
                continue
            st.markdown(f"**{row['Muestra']}** — {an}: `{c_val} {row['Unidad']}`")
            evals = evaluate_normative_status(an, float(c_val))
            for ev in evals:
                lim_txt   = f"{ev['limite']} mg/L" if ev["limite"] else "—"
                badge_cls = ev["badge"]
                st.markdown(
                    f"&nbsp;&nbsp;<span class='badge-{badge_cls}'>{ev['status']}</span> "
                    f"<span style='color:{TEXT_SUB};font-size:0.83rem;'>"
                    f"{ev['norma']} &nbsp;|&nbsp; Limite: {lim_txt}</span>",
                    unsafe_allow_html=True,
                )
    elif st.session_state["sa_result"] is not None:
        sa_r  = st.session_state["sa_result"]
        asgn  = st.session_state["assignment_df"]
        an    = asgn["Analito"].iloc[0] if asgn is not None else ""
        unit  = asgn["Unidad"].iloc[0]  if asgn is not None else "mg/L"
        c_val = sa_r["c_sample"]
        st.markdown(f"**Adicion de estandar** — {an}: `{c_val:.4f} {unit}`")
        evals = evaluate_normative_status(an, float(c_val))
        for ev in evals:
            lim_txt   = f"{ev['limite']} mg/L" if ev["limite"] else "—"
            badge_cls = ev["badge"]
            st.markdown(
                f"&nbsp;&nbsp;<span class='badge-{badge_cls}'>{ev['status']}</span> "
                f"<span style='color:{TEXT_SUB};font-size:0.83rem;'>"
                f"{ev['norma']} &nbsp;|&nbsp; Limite: {lim_txt}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="info-box">Realice la cuantificacion en el Paso 5 para ver la evaluacion normativa.</div>',
                    unsafe_allow_html=True)

    # ── PASO 7 ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="step-header">Paso 7 — Exportar reporte</p>', unsafe_allow_html=True)

    if st.button("Generar reporte PDF"):
        asgn        = st.session_state["assignment_df"]
        analyte_rep = asgn["Analito"].iloc[0] if asgn is not None and not asgn.empty else "N/D"
        method_rep  = "Adicion de estandar" if st.session_state["sa_result"] else "Calibracion externa"

        # Extraer datos crudos de calibración para el gráfico matplotlib
        cal_concs_pdf   = None
        cal_signals_pdf = None
        cal_channel_pdf = st.session_state.get("best_channel", "")
        cal_analyte_pdf = analyte_rep
        cal_unit_pdf    = asgn["Unidad"].iloc[0] if asgn is not None and not asgn.empty else "mg/L"

        df_abs_tmp = st.session_state.get("df_absorbance")
        if df_abs_tmp is not None and cal_channel_pdf:
            std_tmp = df_abs_tmp[df_abs_tmp["Tipo"] == "Estandar"]
            if not std_tmp.empty and "Concentracion" in std_tmp.columns:
                a_col_tmp = f"A_dig_{cal_channel_pdf}"
                if a_col_tmp in std_tmp.columns:
                    cal_concs_pdf   = std_tmp["Concentracion"].values.astype(float)
                    cal_signals_pdf = std_tmp[a_col_tmp].values.astype(float)

        try:
            pdf_bytes = generate_pdf_report(
                analyte             = analyte_rep,
                method              = method_rep,
                df_rgb              = st.session_state["df_norm"],
                df_results          = st.session_state["df_results"],
                cal_result          = st.session_state["cal_result"],
                fig_cal             = st.session_state["cal_fig"],
                annotated_img       = st.session_state["annotated_img"],
                triplicate_stats_df = st.session_state.get("triplicate_stats_df"),
                cal_concs           = cal_concs_pdf,
                cal_signals         = cal_signals_pdf,
                cal_channel         = cal_channel_pdf,
                cal_analyte         = cal_analyte_pdf,
                cal_unit            = cal_unit_pdf,
            )
            b64      = base64.b64encode(pdf_bytes).decode()
            now_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"Elementa_{analyte_rep}_{now_str}.pdf"
            href = (
                f'<a href="data:application/pdf;base64,{b64}" download="{filename}" '
                f'style="background:{EMERALD};color:white;padding:10px 24px;'
                f'border-radius:5px;text-decoration:none;font-weight:700;'
                f'letter-spacing:0.04em;font-size:0.87rem;">'
                f'Descargar reporte PDF</a>'
            )
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al generar el PDF: {e}")

    st.markdown(
        '<div class="warn-box" style="margin-top:20px;">'
        '<b>Nota cientifica:</b> Elementa realiza estimaciones colorimétricas digitales a partir de '
        'imágenes RGB. La precisión depende de la uniformidad de la iluminación, las características '
        'del sensor de la cámara, la calidad de los reactivos, la linealidad del método y la '
        'reproducibilidad en la preparación de estándares. Para declaraciones de cumplimiento '
        'normativo, los resultados deben confirmarse mediante métodos oficiales en laboratorios acreditados.</div>',
        unsafe_allow_html=True,
    )

    render_footer()


# ════════════════════════════════════════════════════════════════
#  PÁGINA 2: FUNDAMENTOS
# ════════════════════════════════════════════════════════════════

elif pagina == "Fundamentos":

    st.markdown("<h1>Fundamentos del análisis colorimétrico digital</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_SUB};'>Base conceptual de los métodos implementados en Elementa. "
        f"Haga clic en cada sección para expandirla.</p>",
        unsafe_allow_html=True,
    )

    TOPICS = {
        "Colorimetria digital: principios y alcance": """
La **colorimetría digital** es el análisis cuantitativo del color de imágenes capturadas digitalmente para estimar
la concentración de un analito. Un sensor CMOS de smartphone captura la luz reflejada o transmitida por una solución
coloreada, descomponiéndola en tres canales de banda ancha: **Rojo (R, aprox. 580–700 nm), Verde (G, aprox. 500–580 nm)
y Azul (B, aprox. 400–500 nm)**.

A diferencia de un espectrofotómetro UV-Vis con resolución de 0.1–2 nm, el sensor integra bandas amplias.
Por eso Elementa implementa la **absorbancia digital** (log10 de la razón de intensidades normalizadas) como señal
analítica, en lugar de la absorbancia espectrofotométrica clásica.

La colorimetría digital ha demostrado resultados con R² > 0.99 para numerosos sistemas colorimétricos
(Folin-Ciocalteu, DPPH, ditizona, entre otros) en condiciones de iluminación controladas.
    """,
        "Comparativa espectrofotometria UV-Vis vs analisis RGB": """
| Criterio | UV-Vis clásico | Análisis RGB (Elementa) |
|---|---|---|
| Fuente de luz | Tungsteno/deuterio controlado | Luz ambiental o LED |
| Resolución espectral | 0.1–2 nm | ~100 nm por canal |
| Exactitud | Alta (±0.001 Abs) | Moderada (dependiente de condiciones) |
| Rango dinámico | Alto | Limitado por sensor CMOS |
| Costo del instrumento | Alto | Bajo (smartphone) |
| Portabilidad | Limitada | Alta |
| Validación requerida | Sí (ISO/NOM) | Sí (experimental, por usuario y condición) |

El análisis RGB es apropiado para **escrutinio rápido, trabajo de campo y educación analítica**.
No reemplaza la confirmación instrumental para declaraciones normativas.
    """,
        "Ley de Beer-Lambert y absorbancia digital": """
La **ley de Beer-Lambert** establece que la absorbancia A = ε·l·c, donde ε es la absortividad molar, l
es el paso óptico y c la concentración. En consecuencia, A varía linealmente con c.

En colorimetría digital se define por analogía:

**A_dig = log10(I_blanco_norm / I_muestra_norm)**

donde I_norm es la intensidad del canal seleccionado normalizada respecto a la suma total RGB.
La normalización porcentual mitiga los efectos de la iluminación no uniforme. El logaritmo preserva
la linealidad en el rango de trabajo, siempre que la concentración no sature el sensor.

La curva de calibración A_dig = m·C + b se ajusta por regresión lineal mínimos cuadrados.
    """,
        "Historia y aplicacion analitica de la ditizona": """
La **ditizona (difeniltiocarbazona)** es un quelante orgánico con alta selectividad para metales pesados.
Su solución en solventes orgánicos es verde intensa; al complejarse con Pb, Cd, Zn, Hg o Cu forma
compuestos de colores vivos (rojo para Pb, amarillo-anaranjado para Cd).

El método ditizona–Pb fue durante décadas el método de referencia para plomo en agua antes de la
espectrometría de absorción atómica. Continúa siendo relevante en kits portátiles de campo y como
sistema modelo en educación analítica. Su reactividad selectiva con metales la hace adecuada para
su uso con Elementa, siempre que se controlen interferencias por otros metales coextractables.
    """,
        "Toxicidad del Cr(VI) y base normativa": """
El **cromo hexavalente (Cr(VI))** es clasificado como carcinógeno del Grupo 1 (IARC, Monografía Vol. 49).
Sus rutas de exposición y efectos incluyen:

- **Inhalación crónica:** cáncer de pulmón, ulceración del tabique nasal.
- **Ingestión:** daño gastrointestinal, hepático y renal; posible carcinogénesis colorrectal.
- **Contacto dérmico:** úlceras crónicas ("úlceras de cromo"), dermatitis de contacto.

Las fuentes antropogénicas incluyen industria galvánica, curtido de cueros, pigmentos industriales
y preservantes de madera (CCA). La **NOM-127-SSA1-2021** fija un límite de 0.05 mg/L de Cr total
en agua potable para México.
    """,
        "Bioacumulacion y biomagnificacion de metales pesados": """
Los metales pesados (Pb, Cd, Hg, As) son **no biodegradables**: se redistribuyen entre sedimentos,
agua y biota sin mineralizarse. La **bioacumulación** ocurre cuando un organismo absorbe el contaminante
a mayor velocidad de la que lo elimina. La **biomagnificación** amplifica la concentración a lo largo
de la cadena trófica:

alga → invertebrado → pez pequeño → pez grande → mamífero tope → humano

Cada nivel puede multiplicar la concentración entre 2 y 10 veces. Esto explica por qué los límites
normativos en agua se expresan en µg/L (partes por billón), órdenes de magnitud por debajo de las
concentraciones de toxicidad aguda.
    """,
        "Ensayos antioxidantes: DPPH, ABTS, FRAP, Folin-Ciocalteu": """
**DPPH** (2,2-difenil-1-picrilhidrazilo): radical libre de color púrpura intenso (λmax ≈ 515 nm).
Al reducirse por un antioxidante, pierde color. La señal **disminuye** con mayor capacidad antioxidante.
Canal más sensible: habitualmente G o B dependiendo del instrumento y condiciones.

**ABTS** (ácido 2,2'-azino-bis(3-etilbenzotiazolín-6-sulfónico), radical catión): color verde-azulado.
Similar al DPPH en principio. Resultados expresados como equivalentes Trolox (TEAC, µmol TE/g).

**FRAP** (Ferric Reducing Antioxidant Power): reduce Fe³⁺ a Fe²⁺ formando el complejo azul
Fe²⁺–tripiridiltriazina (λmax ≈ 593 nm). La señal **aumenta** con mayor capacidad antioxidante.

**Folin-Ciocalteu (fenoles totales):** el reactivo se reduce a un complejo azul en presencia de fenoles.
Señal creciente con concentración. Resultados en equivalentes de ácido gálico (µg GAE/g o mg GAE/100 g).

**Nota sobre pendiente negativa:** en DPPH y ABTS, si se grafica la absorbancia digital del canal más
sensible frente a la concentración antioxidante, la pendiente es negativa. Elementa calcula correctamente
la concentración aplicando x = (y − b) / m con m < 0.
    """,
        "Limitaciones del smartphone como instrumento analitico": """
1. **Iluminación no controlada:** la intensidad, ángulo y temperatura de color de la luz ambiental varían.
   Se recomienda una caja de luz con LED blanco neutro y difusor, con la cámara en posición fija.

2. **No linealidad del sensor CMOS:** a altas intensidades el sensor satura. Ajustar la exposición
   manualmente a un nivel donde los pocillos más intensos no superen 230/255 en ningún canal.

3. **Compresión JPEG:** puede alterar los valores RGB en hasta ±5 unidades. Usar formato PNG o RAW
   cuando el software lo permita.

4. **Variabilidad entre dispositivos:** distintas cámaras tienen curvas de respuesta espectral
   diferentes. La calibración es **específica para cada dispositivo, configuración y condición experimental**.

5. **Deriva temporal:** la misma cámara puede variar entre sesiones por temperatura, actualizaciones
   de firmware o suciedad del lente. Calibrar en cada sesión analítica.

6. **Rango dinámico limitado:** el rango lineal RGB suele ser más estrecho que el del espectrofotómetro.
   Limitar el rango de concentraciones a aquella región donde R² > 0.99.

**Buenas prácticas recomendadas:** caja de luz, exposición y balance de blancos en manual,
triplados de cada nivel de calibración, al menos 5 niveles de estándar, y registro de los
parámetros de captura (ISO, distancia focal, temperatura de color).
    """,
        "Estadistica analitica: R², LOD, LOQ, CV% — Significado practico": """
**R² (coeficiente de determinación):** proporción de la varianza de la señal explicada por la concentración.
- R² ≥ 0.999: excelente, linealidad confirmada.
- 0.995 ≤ R² < 0.999: aceptable para métodos de campo.
- R² < 0.995: revisar preparación de estándares, condiciones de iluminación o rango de calibración.

**LOD (límite de detección):** concentración mínima que produce una señal estadísticamente distinguible
del ruido (criterio 3σ/m). Las muestras cuya concentración calculada cae por debajo del LOD deben
reportarse como "< LOD", no como cero.

**LOQ (límite de cuantificación):** concentración mínima cuantificable con precisión aceptable (10σ/m,
aproximadamente CV < 10%). Las muestras entre LOD y LOQ pueden detectarse pero no cuantificarse de forma
confiable.

**CV% (coeficiente de variación):** para triplicados, CV < 5% indica precisión muy buena; 5–10% aceptable;
> 10% señala problemas técnicos (pipeteo, homogeneidad de mezcla, tiempo de reacción).

**Error estándar de la pendiente (Sb):** incertidumbre en la estimación de la sensibilidad analítica.
Un Sb/m < 1% indica excelente precisión del ajuste.
    """,
    }

    for title, content in TOPICS.items():
        with st.expander(title, expanded=False):
            st.markdown(content)

    render_footer()


# ════════════════════════════════════════════════════════════════
#  PÁGINA 3: NORMATIVA Y FUENTES
# ════════════════════════════════════════════════════════════════

elif pagina == "Normativa y Fuentes":

    st.markdown("<h1>Normativa y fuentes de referencia</h1>", unsafe_allow_html=True)

    st.subheader("Límites permisibles de referencia")
    st.markdown(
        '<div class="warn-box">Los valores mostrados son informativos. '
        'Verificar siempre los límites permisibles en la versión oficial vigente de la norma aplicable. '
        'Los parámetros normativos pueden actualizarse. '
        'Consultar el Diario Oficial de la Federación (DOF): dof.gob.mx</div>',
        unsafe_allow_html=True,
    )

    norm_rows = []
    for analyte, normas in NORMATIVE_LIMITS.items():
        for norma, limite in normas.items():
            norm_rows.append({"Analito": analyte, "Norma": norma, "Limite (mg/L)": limite})
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)

    st.subheader("Bibliografía y referencias técnicas")
    refs = [
        ("NOM-127-SSA1-2021",
         "Agua para uso y consumo humano. Límites permisibles de calidad del agua. DOF 2021.",
         "https://www.dof.gob.mx"),
        ("NOM-001-SEMARNAT-2021",
         "Límites permisibles de contaminantes en descargas de aguas residuales en cuerpos receptores. DOF 2021.",
         "https://www.dof.gob.mx"),
        ("Wrolstad, R.E. et al. (2005)",
         "Handbook of Food Analytical Chemistry. Wiley-Interscience.", ""),
        ("Cardoso Steele, J.L. et al. (2019)",
         "Digital image colorimetry on smartphone for food analysis: A review. Trends in Analytical Chemistry, 111.", ""),
        ("Brand-Williams, W. et al. (1995)",
         "Use of a free radical method to evaluate antioxidant activity. LWT — Food Science and Technology, 28(1).", ""),
        ("IARC Monographs Vol. 49 (1990)",
         "Chromium, Nickel and Welding. IARC, Lyon. [Clasificación Cr(VI) como Grupo 1].",
         "https://monographs.iarc.who.int"),
        ("Cate, D.M. et al. (2015)",
         "Pushing the limits of lateral flow assays with 2-dimensional paper networks. Analytical Chemistry, 87(1).", ""),
        ("BuGiorgio, M. et al. (2020)",
         "Smartphone-based colorimetric analysis for environmental and food monitoring: review. Sensors, 20(22).", ""),
    ]
    for ref_id, ref_text, url in refs:
        if url:
            st.markdown(f"- **{ref_id}:** {ref_text} &nbsp;[Ver referencia]({url})")
        else:
            st.markdown(f"- **{ref_id}:** {ref_text}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Editar límites normativos (sesión actual)")
    st.markdown(
        '<div class="info-box">Los cambios aplican únicamente durante esta sesión. '
        'Para hacerlos permanentes, edite el diccionario <code>NORMATIVE_LIMITS</code> '
        'directamente en el archivo <code>elementa_app.py</code>.</div>',
        unsafe_allow_html=True,
    )

    ed_rows = []
    for analyte, normas in NORMATIVE_LIMITS.items():
        for norma, limite in normas.items():
            ed_rows.append({"Analito": analyte, "Norma": norma, "Limite_mg_L": limite})
    ed_df = st.data_editor(
        pd.DataFrame(ed_rows),
        column_config={
            "Limite_mg_L": st.column_config.NumberColumn(
                "Límite (mg/L)", min_value=0.0, step=0.001, format="%.5f"),
        },
        num_rows="fixed", use_container_width=True, key="norm_editor",
    )

    if st.button("Aplicar cambios a esta sesion"):
        NORMATIVE_LIMITS.clear()
        for _, row in ed_df.iterrows():
            an, no, lim = row["Analito"], row["Norma"], row["Limite_mg_L"]
            if an not in NORMATIVE_LIMITS:
                NORMATIVE_LIMITS[an] = {}
            NORMATIVE_LIMITS[an][no] = lim
        st.markdown('<div class="success-box">Límites normativos actualizados para esta sesión.</div>',
                    unsafe_allow_html=True)

    render_footer()

