"""
Elementa PWA — Sistema Analítico Colorimétrico Digital
Derechos reservados (Katyutzka, 2026)

Herramienta educativa y analítica para estimación colorimétrica por imágenes RGB.
NO sustituye métodos instrumentales certificados ni análisis en laboratorios acreditados.
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
    page_title="Elementa PWA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PALETA DE COLORES ───────────────────────
EMERALD  = "#059669"
BLUE_ACC = "#3B82F6"
BG_CARD  = "#1e293b"
BG_DARK  = "#0f172a"
TEXT_SUB = "#94a3b8"

PLOTLY_TEMPLATE = "plotly_dark"

# ─── CSS GLOBAL ──────────────────────────────
st.markdown(f"""
<style>
  /* Body dark */
  .stApp {{ background-color: {BG_DARK}; color: #e2e8f0; }}
  /* Sidebar */
  [data-testid="stSidebar"] {{ background-color: #111827; }}
  /* Metric cards */
  .metric-card {{
      background: {BG_CARD}; border-radius: 10px;
      padding: 14px 18px; margin-bottom: 8px;
      border-left: 4px solid {EMERALD};
  }}
  .metric-card h4 {{ margin:0; color:{TEXT_SUB}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em; }}
  .metric-card p  {{ margin:0; font-size:1.4rem; font-weight:700; color:#f1f5f9; }}
  /* Badge */
  .badge-green  {{ background:#065f46; color:#6ee7b7; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
  .badge-red    {{ background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
  .badge-gray   {{ background:#374151; color:#9ca3af; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }}
  /* Footer */
  .footer {{
      position: relative; text-align: center;
      color: {TEXT_SUB}; font-size: 0.75rem;
      padding: 20px 0 8px 0; margin-top: 40px;
      border-top: 1px solid #1e293b;
  }}
  /* Warning box */
  .warning-box {{
      background: #1c1917; border-left: 4px solid #f59e0b;
      border-radius: 6px; padding: 12px 16px; font-size: 0.85rem;
      color: #fde68a; margin: 10px 0;
  }}
  /* Info box */
  .info-box {{
      background: #0c1a2e; border-left: 4px solid {BLUE_ACC};
      border-radius: 6px; padding: 12px 16px; font-size: 0.85rem;
      color: #bfdbfe; margin: 10px 0;
  }}
  h1, h2, h3 {{ color: #f1f5f9; }}
  .stButton>button {{
      background-color: {EMERALD}; color: #fff;
      border: none; border-radius: 8px;
      padding: 8px 20px; font-weight: 600;
  }}
  .stButton>button:hover {{ background-color: #047857; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  LÍMITES NORMATIVOS (editables aquí)
# ════════════════════════════════════════════════════════════════
NORMATIVE_LIMITS = {
    # analito: { norma: limite_mg_L }
    "Pb": {
        "NOM-127-SSA1-2021 (agua potable)": 0.01,
        "NOM-001-SEMARNAT-2021 (descarga A)": 0.2,
        "NOM-001-SEMARNAT-2021 (descarga B)": 1.0,
    },
    "Cd": {
        "NOM-127-SSA1-2021 (agua potable)": 0.003,
        "NOM-001-SEMARNAT-2021 (descarga A)": 0.1,
        "NOM-001-SEMARNAT-2021 (descarga B)": 0.2,
    },
    "Cr total": {
        "NOM-127-SSA1-2021 (agua potable)": 0.05,
        "NOM-001-SEMARNAT-2021 (descarga A)": 0.5,
        "NOM-001-SEMARNAT-2021 (descarga B)": 1.0,
    },
    "Cr(VI)": {
        "NOM-127-SSA1-2021 (agua potable)": 0.05,
    },
}

# ════════════════════════════════════════════════════════════════
#  FUNCIONES DE PROCESAMIENTO DE IMAGEN
# ════════════════════════════════════════════════════════════════

def load_image(uploaded_file) -> np.ndarray | None:
    """Carga imagen desde archivo subido y devuelve array RGB uint8."""
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def generate_rois_linear(x0, y0, w, h, n, dx, dy) -> list[dict]:
    """
    Genera n ROIs en disposición lineal.
    Retorna lista de dicts con {x, y, w, h, label}.
    """
    rois = []
    for i in range(n):
        rois.append({
            "x": int(x0 + i * dx),
            "y": int(y0 + i * dy),
            "w": int(w),
            "h": int(h),
            "label": f"ROI {i+1}",
        })
    return rois


def generate_rois_microplate(x0, y0, w, h, dx, dy,
                              rows=8, cols=12) -> list[dict]:
    """Genera ROIs para microplaca 8×12."""
    rois = []
    row_labels = "ABCDEFGH"
    for r in range(rows):
        for c in range(cols):
            rois.append({
                "x": int(x0 + c * dx),
                "y": int(y0 + r * dy),
                "w": int(w),
                "h": int(h),
                "label": f"{row_labels[r]}{c+1}",
            })
    return rois


def draw_rois(image: np.ndarray, rois: list[dict],
              assignments: dict | None = None) -> np.ndarray:
    """
    Dibuja ROIs sobre imagen.
    assignments: {label: tipo} donde tipo ∈ {Blanco, Estándar, Muestra, Control, Adición estándar}
    """
    img = image.copy()
    # BGR colores por tipo
    color_map = {
        "Blanco":            (255, 255, 100),
        "Estándar":          (100, 220, 100),
        "Muestra":           (100, 160, 255),
        "Control":           (255, 160,  80),
        "Adición estándar":  (200,  80, 255),
        "Sin asignar":       (180, 180, 180),
    }
    for roi in rois:
        tipo = (assignments or {}).get(roi["label"], "Sin asignar")
        color = color_map.get(tipo, (180, 180, 180))
        bgr = (color[2], color[1], color[0])   # RGB→BGR para OpenCV
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(img, (x, y), (x+w, y+h), bgr, 2)
        cv2.putText(img, roi["label"], (x, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    return img


def extract_rgb_stats(image: np.ndarray, rois: list[dict]) -> pd.DataFrame:
    """Extrae R, G, B promedio y desviación estándar para cada ROI."""
    records = []
    h_img, w_img = image.shape[:2]
    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        # Clamp dentro de imagen
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
            "R_mean": round(r_m, 2),
            "G_mean": round(g_m, 2),
            "B_mean": round(b_m, 2),
            "R_std":  round(r_s, 2),
            "G_std":  round(g_s, 2),
            "B_std":  round(b_s, 2),
        })
    return pd.DataFrame(records)


def calculate_normalized_intensity(df_rgb: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega intensidad total y canales normalizados (% de suma RGB).
    I_norm_canal = (canal / (R+G+B)) * 100
    """
    df = df_rgb.copy()
    eps = 1e-9
    total = df["R_mean"] + df["G_mean"] + df["B_mean"] + eps
    df["Total_RGB"]  = df["R_mean"] + df["G_mean"] + df["B_mean"]
    df["R_norm"]     = (df["R_mean"] / total) * 100
    df["G_norm"]     = (df["G_mean"] / total) * 100
    df["B_norm"]     = (df["B_mean"] / total) * 100
    return df


def calculate_digital_absorbance(df: pd.DataFrame,
                                  blank_label: str | None,
                                  channels: list[str]) -> pd.DataFrame:
    """
    Calcula absorbancia digital A = log10(I_norm_blanco / I_norm_muestra)
    para cada canal especificado.
    Si I_muestra == 0 o blanco no definido, retorna NaN con advertencia.
    """
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
    """
    Ajuste lineal y = mx + b. Retorna dict con m, b, r2, se, residuals.
    Admite pendiente negativa.
    """
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
        "residuals": residuals,
        "n": len(concs),
    }


def calculate_lod_loq(cal_result: dict,
                      blank_signals: np.ndarray | None) -> dict:
    """
    LOD = 3.3 * sigma_blank / |m|
    LOQ = 10  * sigma_blank / |m|
    Si no hay blancos, usa error estándar residual como proxy.
    """
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


def select_best_channel(df: pd.DataFrame,
                         standard_labels: list[str],
                         conc_col: str = "Concentracion") -> dict:
    """
    Evalúa canales R, G, B (y normalizados) para estándares,
    retorna el mejor por R² y pendiente ≠ 0.
    """
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


def standard_addition_analysis(added_concs: np.ndarray,
                                signals: np.ndarray) -> dict | None:
    """
    Regresión Señal = m*C_add + b
    C_muestra = |b/m| = extrapolación al eje X negativo
    """
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


def evaluate_normative_status(analyte: str, conc_mg_L: float,
                               norm_limits: dict = NORMATIVE_LIMITS) -> list[dict]:
    """Compara concentración con límites normativos disponibles."""
    if analyte not in norm_limits:
        return [{"norma": "Sin criterio disponible", "limite": None,
                 "status": "Sin criterio", "badge": "gray"}]
    results = []
    for norma, limite in norm_limits[analyte].items():
        if conc_mg_L <= limite:
            results.append({"norma": norma, "limite": limite,
                             "status": "Cumple", "badge": "green"})
        else:
            results.append({"norma": norma, "limite": limite,
                             "status": "No cumple", "badge": "red"})
    return results


# ════════════════════════════════════════════════════════════════
#  GRÁFICAS PLOTLY
# ════════════════════════════════════════════════════════════════

def plot_calibration(concs, signals, cal_result, channel, analyte,
                     unit, lod, loq) -> go.Figure:
    x_min = min(concs) * 0.9
    x_max = max(concs) * 1.1
    x_line = np.linspace(x_min, x_max, 200)
    y_line = cal_result["slope"] * x_line + cal_result["intercept"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=concs, y=signals, mode="markers",
        marker=dict(color=EMERALD, size=9, line=dict(color="white", width=1)),
        name="Estándares"
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color=BLUE_ACC, width=2), name="Regresión"
    ))
    # LOD / LOQ
    if not np.isnan(lod):
        fig.add_vline(x=lod, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"LOD={lod:.3f}", annotation_font_color="#f59e0b")
    if not np.isnan(loq):
        fig.add_vline(x=loq, line_dash="dot", line_color="#fb7185",
                      annotation_text=f"LOQ={loq:.3f}", annotation_font_color="#fb7185")

    m, b, r2 = cal_result["slope"], cal_result["intercept"], cal_result["r2"]
    sign = "+" if b >= 0 else "-"
    eq   = f"y = {m:.4f}x {sign} {abs(b):.4f}   R²={r2:.4f}"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Curva de calibración — {analyte} | Canal: {channel}",
        xaxis_title=f"Concentración ({unit})",
        yaxis_title="Absorbancia digital",
        annotations=[dict(x=0.02, y=0.97, xref="paper", yref="paper",
                          text=eq, showarrow=False, font=dict(color=EMERALD, size=11),
                          bgcolor="rgba(0,0,0,0.4)", bordercolor=EMERALD, borderwidth=1)],
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_standard_addition(added_concs, signals, cal_result,
                            analyte, unit) -> go.Figure:
    x_int = cal_result["x_intercept"]
    c_sam = cal_result["c_sample"]
    m, b  = cal_result["slope"], cal_result["intercept"]

    x_min = min(x_int * 1.3, min(added_concs) - abs(x_int)*0.2)
    x_max = max(added_concs) * 1.1
    x_line = np.linspace(x_min, x_max, 300)
    y_line = m * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=added_concs, y=signals, mode="markers",
        marker=dict(color=EMERALD, size=9), name="Adiciones"
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color=BLUE_ACC, width=2), name="Proyección"
    ))
    # Intercepto
    fig.add_trace(go.Scatter(
        x=[x_int], y=[0], mode="markers+text",
        marker=dict(color="#f43f5e", size=12, symbol="x"),
        text=[f"C = {c_sam:.4f} {unit}"], textposition="top right",
        name=f"C muestra ≈ {c_sam:.4f}"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Adición de estándar — {analyte}",
        xaxis_title=f"Concentración añadida ({unit})",
        yaxis_title="Señal",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  GENERACIÓN DE REPORTE PDF
# ════════════════════════════════════════════════════════════════

def generate_pdf_report(analyte: str, method: str,
                         df_rgb: pd.DataFrame | None,
                         df_results: pd.DataFrame | None,
                         cal_result: dict | None,
                         fig_cal: go.Figure | None,
                         annotated_img: np.ndarray | None) -> bytes:
    """Genera PDF con reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable, PageBreak)
    from reportlab.lib.colors import HexColor

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)

    EMERALD_RL = HexColor("#059669")
    BLUE_RL    = HexColor("#3B82F6")
    DARK_RL    = HexColor("#0f172a")
    SLATE_RL   = HexColor("#1e293b")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
        textColor=EMERALD_RL, fontSize=18, spaceAfter=4)
    h2_style    = ParagraphStyle("H2Style", parent=styles["Heading2"],
        textColor=BLUE_RL, fontSize=12, spaceAfter=3)
    body_style  = ParagraphStyle("BodyStyle", parent=styles["BodyText"],
        textColor=colors.white, fontSize=9, leading=13)
    warn_style  = ParagraphStyle("WarnStyle", parent=styles["BodyText"],
        textColor=HexColor("#fde68a"), fontSize=8, leading=12)
    foot_style  = ParagraphStyle("FootStyle", parent=styles["BodyText"],
        textColor=colors.grey, fontSize=7, leading=10, alignment=1)

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    story = []

    # ── Membrete ─────────────────────────────────────────────
    story.append(Paragraph("🔬 Elementa PWA (2026)", title_style))
    story.append(Paragraph("Sistema Analítico Colorimétrico Digital", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=EMERALD_RL))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Fecha:</b> {now} &nbsp;&nbsp; <b>Analito:</b> {analyte} &nbsp;&nbsp; <b>Método:</b> {method}", body_style))
    story.append(Spacer(1, 10))

    # ── Advertencia ──────────────────────────────────────────
    story.append(Paragraph(
        "⚠️ ADVERTENCIA: Elementa PWA realiza estimaciones colorimétricas digitales. "
        "Los resultados NO sustituyen métodos oficiales en laboratorios acreditados. "
        "No utilizar para cumplimiento normativo sin confirmación analítica certificada.",
        warn_style))
    story.append(Spacer(1, 10))

    # ── Imagen anotada ────────────────────────────────────────
    if annotated_img is not None:
        story.append(Paragraph("Imagen procesada con ROIs", h2_style))
        pil_img = Image.fromarray(annotated_img)
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=4.5*inch, height=3*inch, kind="proportional")
        story.append(rl_img)
        story.append(Spacer(1, 8))

    # ── Tabla RGB ─────────────────────────────────────────────
    if df_rgb is not None and not df_rgb.empty:
        story.append(Paragraph("Datos RGB por ROI", h2_style))
        df_show = df_rgb[["ROI","R_mean","G_mean","B_mean","R_norm","G_norm","B_norm"]].round(2)
        tbl_data = [list(df_show.columns)] + df_show.values.tolist()
        tbl = Table(tbl_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), EMERALD_RL),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTSIZE",    (0,0), (-1,-1), 7),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, HexColor("#f0fdf4")]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    # ── Calibración ───────────────────────────────────────────
    if cal_result:
        story.append(Paragraph("Parámetros de Calibración", h2_style))
        cal_data = [
            ["Parámetro", "Valor"],
            ["Pendiente (m)",      f"{cal_result['slope']:.6f}"],
            ["Intercepto (b)",     f"{cal_result['intercept']:.6f}"],
            ["R²",                 f"{cal_result['r2']:.6f}"],
            ["Error estándar (se)",f"{cal_result['se']:.6f}"],
            ["N estándares",       str(cal_result.get('n','N/A'))],
        ]
        tbl2 = Table(cal_data, colWidths=[2.5*inch, 2*inch])
        tbl2.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), BLUE_RL),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(tbl2)
        story.append(Spacer(1, 10))

    # ── Resultados ────────────────────────────────────────────
    if df_results is not None and not df_results.empty:
        story.append(Paragraph("Resultados de Muestras", h2_style))
        tbl_data2 = [list(df_results.columns)] + df_results.values.tolist()
        tbl3 = Table(tbl_data2, repeatRows=1)
        tbl3.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), EMERALD_RL),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTSIZE",    (0,0), (-1,-1), 7),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, HexColor("#f0fdf4")]),
        ]))
        story.append(tbl3)
        story.append(Spacer(1, 10))

    # ── Nota científica ───────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Nota científica: Elementa PWA realiza estimaciones colorimétricas digitales a partir de imágenes RGB. "
        "La precisión depende de iluminación, cámara, reactivos, linealidad del método, preparación de estándares "
        "y validación experimental. Para fines regulatorios, los resultados deben confirmarse mediante métodos "
        "oficiales en laboratorios acreditados. Verificar siempre los límites permisibles en la versión oficial "
        "vigente de la norma aplicable.",
        warn_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Derechos reservados (Katyutzka, 2026)", foot_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════
#  ESTADO DE SESIÓN — INICIALIZACIÓN
# ════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "image":           None,
        "rois":            [],
        "freeze_rois":     False,
        "df_rgb":          None,
        "df_norm":         None,
        "df_absorbance":   None,
        "assignment_df":   None,
        "blank_label":     None,
        "cal_result":      None,
        "best_channel":    "G_norm",
        "all_channel_res": {},
        "df_results":      None,
        "annotated_img":   None,
        "cal_fig":         None,
        "sa_fig":          None,
        "sa_result":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ════════════════════════════════════════════════════════════════
#  FOOTER HELPER
# ════════════════════════════════════════════════════════════════

def render_footer():
    st.markdown(
        '<div class="footer">Derechos reservados (Katyutzka, 2026) &nbsp;|&nbsp; '
        'Elementa PWA — Sistema Analítico Colorimétrico Digital</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
#  BARRA LATERAL — NAVEGACIÓN
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"<h2 style='color:{EMERALD};margin-bottom:4px;'>🔬 Elementa</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_SUB};font-size:0.78rem;margin-top:0;'>Sistema Colorimétrico Digital</p>", unsafe_allow_html=True)
    st.divider()
    pagina = st.radio(
        "Navegación",
        ["🧪 Análisis", "📖 Para saber más", "📋 Fuentes e Información"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"<p style='color:{TEXT_SUB};font-size:0.72rem;'>⚠️ Los resultados son estimaciones analíticas. No sustituyen métodos certificados.</p>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  PÁGINA 1: ANÁLISIS
# ════════════════════════════════════════════════════════════════

if pagina == "🧪 Análisis":

    st.markdown(f"<h1>🧪 Análisis Colorimétrico</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Sigue el flujo: <b>1. Cargar imagen → 2. Definir ROIs → '
        '3. Asignar roles → 4. Calibrar → 5. Cuantificar → 6. Comparar norma → 7. Exportar</b></div>',
        unsafe_allow_html=True,
    )

    # ── PASO 1: Cargar imagen ─────────────────────────────────
    st.subheader("① Cargar imagen")
    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Subir imagen (JPG/PNG)", type=["jpg","jpeg","png"])
        if uploaded_file:
            st.session_state["image"] = load_image(uploaded_file)
    with col_cam:
        cam_img = st.camera_input("O capturar con cámara")
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
                st.info("Las ROIs se mostrarán aquí una vez definidas.")
    else:
        st.info("Carga o captura una imagen para comenzar.")
        render_footer()
        st.stop()

    img = st.session_state["image"]
    h_img, w_img = img.shape[:2]

    # ── PASO 2: Definir ROIs ──────────────────────────────────
    st.divider()
    st.subheader("② Definir regiones de interés (ROIs)")

    device_type = st.selectbox(
        "Tipo de dispositivo",
        ["Viales lineales", "Microplaca de 96 pozos", "Personalizado (viales 2D)"],
    )

    freeze = st.toggle("🔒 Bloquear ROIs", value=st.session_state["freeze_rois"])
    st.session_state["freeze_rois"] = freeze

    if not freeze:
        if device_type == "Viales lineales":
            c1,c2,c3,c4 = st.columns(4)
            n_rois = c1.number_input("N° de muestras", 2, 24, 6, 1)
            x0 = c2.slider("X inicial", 0, w_img-1, int(w_img*0.05))
            y0 = c3.slider("Y inicial", 0, h_img-1, int(h_img*0.25))
            c5,c6,c7,c8 = st.columns(4)
            roi_w = c5.slider("Ancho ROI", 5, 200, 40)
            roi_h = c6.slider("Alto ROI",  5, 300, 60)
            dx    = c7.slider("Espaciado X", 0, 300, int(w_img*0.08))
            dy    = c8.slider("Espaciado Y", 0, 200, 0)
            rois  = generate_rois_linear(x0, y0, roi_w, roi_h, n_rois, dx, dy)

        elif device_type == "Microplaca de 96 pozos":
            c1,c2,c3,c4 = st.columns(4)
            x0    = c1.slider("X inicial",     0, w_img-1, int(w_img*0.05))
            y0    = c2.slider("Y inicial",     0, h_img-1, int(h_img*0.05))
            roi_w = c3.slider("Ancho ROI",     4, 80,  20)
            roi_h = c4.slider("Alto ROI",      4, 80,  20)
            c5,c6,c7,c8 = st.columns(4)
            dx    = c5.slider("Espaciado X",  10, 200, 50)
            dy    = c6.slider("Espaciado Y",  10, 200, 50)
            rows  = c7.number_input("Filas",  1, 8,  8, 1)
            cols  = c8.number_input("Cols", 1, 12, 12, 1)
            rois  = generate_rois_microplate(x0, y0, roi_w, roi_h, dx, dy,
                                             int(rows), int(cols))
        else:  # Personalizado
            c1,c2,c3,c4 = st.columns(4)
            n_rois = c1.number_input("N° ROIs",  2, 50, 6, 1)
            x0     = c2.slider("X inicial", 0, w_img-1, int(w_img*0.05))
            y0     = c3.slider("Y inicial", 0, h_img-1, int(h_img*0.1))
            c5,c6,c7,c8 = st.columns(4)
            roi_w  = c5.slider("Ancho ROI", 5, 200, 30)
            roi_h  = c6.slider("Alto ROI",  5, 200, 30)
            dx     = c7.slider("Espaciado X", 0, 300, int(w_img*0.08))
            dy     = c8.slider("Espaciado Y", 0, 300, int(h_img*0.08))
            rois   = generate_rois_linear(x0, y0, roi_w, roi_h, int(n_rois), dx, dy)

        st.session_state["rois"] = rois
    else:
        rois = st.session_state["rois"]
        st.success(f"ROIs bloqueadas ({len(rois)} regiones).")

    if not rois:
        st.warning("No hay ROIs definidas.")
        render_footer()
        st.stop()

    # Dibujar ROIs y actualizar imagen anotada
    assignments_dict = {}
    if st.session_state["assignment_df"] is not None:
        for _, row in st.session_state["assignment_df"].iterrows():
            assignments_dict[row["ROI"]] = row.get("Tipo", "Sin asignar")

    ann_img = draw_rois(img, rois, assignments_dict)
    st.session_state["annotated_img"] = ann_img
    # Refresh side image
    with col_img2:
        st.image(ann_img, caption="Imagen con ROIs", use_container_width=True)

    # ── PASO 3: Asignar roles ─────────────────────────────────
    st.divider()
    st.subheader("③ Asignar roles y concentraciones")

    TIPOS = ["Sin asignar","Blanco","Estándar","Muestra","Control","Adición estándar"]
    ANALITOS = ["Pb","Cd","Cr total","Cr(VI)","DPPH","ABTS","FRAP","Fenoles totales","Otro"]
    UNIDADES = ["mg/L","µg/L","ppm","µM","mM","%","µg/mL","Otro"]

    # Construir tabla editable
    if (st.session_state["assignment_df"] is None or
            len(st.session_state["assignment_df"]) != len(rois)):
        init_rows = []
        for roi in rois:
            init_rows.append({
                "ROI":           roi["label"],
                "Tipo":          "Sin asignar",
                "Nombre":        "",
                "Concentracion": 0.0,
                "Unidad":        "mg/L",
                "Factor_dil":    1.0,
                "Analito":       "Cr(VI)",
                "Observaciones": "",
            })
        st.session_state["assignment_df"] = pd.DataFrame(init_rows)

    edited_df = st.data_editor(
        st.session_state["assignment_df"],
        column_config={
            "Tipo":    st.column_config.SelectboxColumn("Tipo",    options=TIPOS,    required=True),
            "Unidad":  st.column_config.SelectboxColumn("Unidad",  options=UNIDADES, required=True),
            "Analito": st.column_config.SelectboxColumn("Analito", options=ANALITOS, required=True),
            "Concentracion": st.column_config.NumberColumn("Conc.", min_value=0.0, step=0.001, format="%.4f"),
            "Factor_dil":    st.column_config.NumberColumn("F.Dilución", min_value=0.01, step=0.1, format="%.2f"),
        },
        num_rows="fixed",
        use_container_width=True,
        key="assignment_editor",
    )
    st.session_state["assignment_df"] = edited_df

    # Actualizar imagen con asignaciones
    for _, row in edited_df.iterrows():
        assignments_dict[row["ROI"]] = row.get("Tipo", "Sin asignar")
    ann_img = draw_rois(img, rois, assignments_dict)
    st.session_state["annotated_img"] = ann_img

    blank_rows = edited_df[edited_df["Tipo"] == "Blanco"]
    blank_label = blank_rows["ROI"].iloc[0] if len(blank_rows) > 0 else None
    st.session_state["blank_label"] = blank_label

    if blank_label:
        st.success(f"Blanco de reactivos: **{blank_label}**")
    else:
        st.warning("No se ha marcado ningún ROI como 'Blanco'. "
                   "La absorbancia digital no podrá calcularse hasta seleccionar un blanco.")

    # ── PASO 4: Extracción y Calibración ─────────────────────
    st.divider()
    st.subheader("④ Extracción de color y calibración")

    if st.button("▶ Extraer RGB y calibrar"):
        with st.spinner("Procesando imagen y canales RGB..."):
            df_rgb  = extract_rgb_stats(img, rois)
            df_norm = calculate_normalized_intensity(df_rgb)

            channels_for_abs = ["R_norm", "G_norm", "B_norm"]
            df_abs  = calculate_digital_absorbance(df_norm, blank_label, channels_for_abs)
            df_abs  = df_abs.merge(edited_df, on="ROI", how="left")

            st.session_state["df_rgb"]        = df_rgb
            st.session_state["df_norm"]       = df_norm
            st.session_state["df_absorbance"] = df_abs

            # Calibración
            std_df = df_abs[df_abs["Tipo"] == "Estándar"].copy()
            sel    = select_best_channel(df_abs, std_df["ROI"].tolist())
            best_ch  = sel["best_channel"]
            all_chs  = sel["results"]

            st.session_state["best_channel"]    = best_ch
            st.session_state["all_channel_res"] = all_chs

            if best_ch in all_chs and len(std_df) >= 2:
                cal = all_chs[best_ch]
                # Blancos para LOD/LOQ
                blank_df    = df_abs[df_abs["Tipo"] == "Blanco"]
                blank_sigs  = blank_df[f"A_dig_{best_ch}"].dropna().values if not blank_df.empty else None
                lod_loq_res = calculate_lod_loq(cal, blank_sigs)
                cal["LOD"]  = lod_loq_res["LOD"]
                cal["LOQ"]  = lod_loq_res["LOQ"]
                cal["lod_proxy"] = lod_loq_res["proxy"]
                st.session_state["cal_result"] = cal

                # Generar figura
                concs   = std_df["Concentracion"].values.astype(float)
                signals = std_df[f"A_dig_{best_ch}"].values.astype(float)
                unit    = std_df["Unidad"].iloc[0] if not std_df.empty else "mg/L"
                analyte = std_df["Analito"].iloc[0] if not std_df.empty else "Analito"
                cal_fig = plot_calibration(concs, signals, cal, best_ch, analyte,
                                           unit, cal["LOD"], cal["LOQ"])
                st.session_state["cal_fig"] = cal_fig

            st.success("Extracción y calibración completadas.")

    # Mostrar resultados de extracción
    if st.session_state["df_absorbance"] is not None:
        df_abs = st.session_state["df_absorbance"]
        with st.expander("📊 Tabla RGB y absorbancias digitales", expanded=False):
            show_cols = ["ROI","Tipo","R_mean","G_mean","B_mean",
                         "R_norm","G_norm","B_norm",
                         "A_dig_R_norm","A_dig_G_norm","A_dig_B_norm"]
            disp_cols = [c for c in show_cols if c in df_abs.columns]
            st.dataframe(df_abs[disp_cols].round(4), use_container_width=True)

        if st.session_state["all_channel_res"]:
            with st.expander("📈 Comparativa de canales RGB", expanded=True):
                ch_rows = []
                for ch, res in st.session_state["all_channel_res"].items():
                    ch_rows.append({
                        "Canal": ch,
                        "R²":    round(res["r2"], 5),
                        "Pendiente m": round(res["slope"], 6),
                        "Intercepto b": round(res["intercept"], 6),
                        "Error est. se": round(res["se"], 6),
                    })
                df_ch = pd.DataFrame(ch_rows).sort_values("R²", ascending=False)
                best = st.session_state["best_channel"]
                st.dataframe(df_ch, use_container_width=True)
                st.success(f"Canal seleccionado automáticamente: **{best}** (mayor R²)")

        if st.session_state["cal_fig"]:
            cal = st.session_state["cal_result"]
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.markdown(f'<div class="metric-card"><h4>R²</h4><p>{cal["r2"]:.5f}</p></div>', unsafe_allow_html=True)
            col_m2.markdown(f'<div class="metric-card"><h4>Pendiente m</h4><p>{cal["slope"]:.5f}</p></div>', unsafe_allow_html=True)
            col_m3.markdown(f'<div class="metric-card"><h4>LOD</h4><p>{cal.get("LOD", float("nan")):.4f}</p></div>', unsafe_allow_html=True)
            col_m4.markdown(f'<div class="metric-card"><h4>LOQ</h4><p>{cal.get("LOQ", float("nan")):.4f}</p></div>', unsafe_allow_html=True)
            if cal.get("lod_proxy"):
                st.markdown('<div class="warning-box">⚠️ LOD/LOQ calculados usando error estándar residual (proxy). '
                            'Para mayor rigor, incluya réplicas de blanco en el análisis.</div>',
                            unsafe_allow_html=True)
            if cal["slope"] < 0:
                st.markdown('<div class="info-box">ℹ️ <b>Pendiente negativa detectada.</b> Esto puede ser esperado en ensayos '
                            'antioxidantes (p.ej. DPPH) donde mayor concentración antioxidante reduce la intensidad del color. '
                            'La cuantificación usa x = (y − b) / m sin modificación.</div>', unsafe_allow_html=True)
            st.plotly_chart(st.session_state["cal_fig"], use_container_width=True)

    # ── PASO 5: Cuantificación ────────────────────────────────
    st.divider()
    st.subheader("⑤ Cuantificación de muestras")

    method_choice = st.radio("Método de cuantificación",
                              ["Calibración externa", "Adición de estándar"],
                              horizontal=True)

    if method_choice == "Calibración externa":
        if st.button("▶ Calcular concentraciones"):
            if (st.session_state["cal_result"] is None or
                    st.session_state["df_absorbance"] is None):
                st.error("Realiza primero la extracción y calibración (Paso 4).")
            else:
                cal  = st.session_state["cal_result"]
                df_a = st.session_state["df_absorbance"].copy()
                best_ch = st.session_state["best_channel"]
                a_col   = f"A_dig_{best_ch}"
                m, b    = cal["slope"], cal["intercept"]

                sample_df = df_a[df_a["Tipo"] == "Muestra"].copy()
                results = []
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
                        "A_dig":          round(a_val, 5) if not np.isnan(a_val) else "N/A",
                        "Conc_calc":      round(conc_raw, 5) if not np.isnan(conc_raw) else "N/A",
                        "Factor_dil":     dil,
                        "Conc_corregida": round(conc_cor, 5) if not np.isnan(conc_cor) else "N/A",
                        "Unidad":         row.get("Unidad", "mg/L"),
                        "Analito":        row.get("Analito", ""),
                    })
                df_res = pd.DataFrame(results)
                st.session_state["df_results"] = df_res
                st.dataframe(df_res, use_container_width=True)

    else:  # Adición de estándar
        st.markdown('<div class="info-box">Ingresa la señal de la muestra sin adición y las adiciones de estándar.</div>',
                    unsafe_allow_html=True)
        best_ch = st.session_state.get("best_channel", "G_norm")
        a_col   = f"A_dig_{best_ch}"

        sa_n = st.number_input("Número de adiciones (sin contar muestra base)", 2, 8, 3)
        sa_rows = []
        sa_rows.append({"C_añadida": 0.0, "Señal": 0.0})
        for i in range(int(sa_n)):
            sa_rows.append({"C_añadida": 0.0, "Señal": 0.0})

        sa_df_edit = st.data_editor(
            pd.DataFrame(sa_rows),
            column_config={
                "C_añadida": st.column_config.NumberColumn("C añadida (mg/L)", step=0.001, format="%.4f"),
                "Señal":     st.column_config.NumberColumn("Señal (A_dig)",     step=0.0001, format="%.5f"),
            },
            use_container_width=True, num_rows="fixed",
        )

        if st.button("▶ Análisis de adición de estándar"):
            added = sa_df_edit["C_añadida"].values.astype(float)
            sigs  = sa_df_edit["Señal"].values.astype(float)
            sa_r  = standard_addition_analysis(added, sigs)
            if sa_r is None:
                st.error("No fue posible ajustar la regresión. Revisa los datos ingresados.")
            else:
                st.session_state["sa_result"] = sa_r
                analyte_sel = st.session_state["assignment_df"]["Analito"].iloc[0] if st.session_state["assignment_df"] is not None else "Analito"
                unit_sel    = st.session_state["assignment_df"]["Unidad"].iloc[0]   if st.session_state["assignment_df"] is not None else "mg/L"
                sa_fig = plot_standard_addition(added, sigs, sa_r, analyte_sel, unit_sel)
                st.session_state["sa_fig"] = sa_fig
                c_est = sa_r["c_sample"]
                st.markdown(f'<div class="metric-card"><h4>Concentración estimada (adición estándar)</h4>'
                            f'<p>{c_est:.5f} {unit_sel}</p></div>', unsafe_allow_html=True)
                st.plotly_chart(sa_fig, use_container_width=True)

    # ── PASO 6: Semáforo normativo ────────────────────────────
    st.divider()
    st.subheader("⑥ Semáforo normativo")

    if st.session_state["df_results"] is not None and not st.session_state["df_results"].empty:
        df_res = st.session_state["df_results"]
        st.markdown('<div class="warning-box">⚠️ Verificar siempre los límites permisibles en la versión oficial vigente de la norma aplicable. '
                    'Los valores mostrados son informativos y pueden haber cambiado.</div>', unsafe_allow_html=True)

        for _, row in df_res.iterrows():
            c_val = row.get("Conc_corregida", "N/A")
            an    = row.get("Analito", "")
            if c_val == "N/A" or pd.isna(c_val) or str(c_val) == "N/A":
                continue
            st.markdown(f"**{row['Muestra']}** — {an}: `{c_val} {row['Unidad']}`")
            evals = evaluate_normative_status(an, float(c_val))
            for ev in evals:
                badge_class = f"badge-{ev['badge']}"
                lim_txt = f"{ev['limite']} mg/L" if ev["limite"] else "—"
                st.markdown(
                    f"&nbsp;&nbsp;<span class='{badge_class}'>{ev['status']}</span> "
                    f"<span style='color:{TEXT_SUB};font-size:0.85rem;'>"
                    f"{ev['norma']} | Límite: {lim_txt}</span>",
                    unsafe_allow_html=True,
                )
    elif st.session_state["sa_result"] is not None:
        sa_r  = st.session_state["sa_result"]
        asgn  = st.session_state["assignment_df"]
        an    = asgn["Analito"].iloc[0] if asgn is not None else ""
        unit  = asgn["Unidad"].iloc[0]  if asgn is not None else "mg/L"
        c_val = sa_r["c_sample"]
        st.markdown(f"**Adición estándar** — {an}: `{c_val:.5f} {unit}`")
        evals = evaluate_normative_status(an, float(c_val))
        st.markdown('<div class="warning-box">⚠️ Verificar siempre los límites permisibles en la versión oficial vigente de la norma aplicable.</div>', unsafe_allow_html=True)
        for ev in evals:
            badge_class = f"badge-{ev['badge']}"
            lim_txt = f"{ev['limite']} mg/L" if ev["limite"] else "—"
            st.markdown(
                f"&nbsp;&nbsp;<span class='{badge_class}'>{ev['status']}</span> "
                f"<span style='color:{TEXT_SUB};font-size:0.85rem;'>"
                f"{ev['norma']} | Límite: {lim_txt}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Realiza la cuantificación (Paso 5) para ver la evaluación normativa.")

    # ── PASO 7: Exportar PDF ──────────────────────────────────
    st.divider()
    st.subheader("⑦ Exportar reporte PDF")

    if st.button("📄 Generar reporte PDF"):
        asgn = st.session_state["assignment_df"]
        analyte_rep = asgn["Analito"].iloc[0] if asgn is not None and not asgn.empty else "N/A"
        method_rep  = "Adición de estándar" if st.session_state["sa_result"] else "Calibración externa"
        try:
            pdf_bytes = generate_pdf_report(
                analyte   = analyte_rep,
                method    = method_rep,
                df_rgb    = st.session_state["df_norm"],
                df_results= st.session_state["df_results"],
                cal_result= st.session_state["cal_result"],
                fig_cal   = st.session_state["cal_fig"],
                annotated_img = st.session_state["annotated_img"],
            )
            b64 = base64.b64encode(pdf_bytes).decode()
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"Elementa_reporte_{analyte_rep}_{now_str}.pdf"
            href = (f'<a href="data:application/pdf;base64,{b64}" '
                    f'download="{filename}" '
                    f'style="background:{EMERALD};color:white;padding:10px 22px;'
                    f'border-radius:8px;text-decoration:none;font-weight:600;">'
                    f'⬇ Descargar PDF</a>')
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generando PDF: {e}")

    st.markdown(
        '<div class="warning-box" style="margin-top:20px;">🔬 <b>Nota científica:</b> '
        'Elementa PWA realiza estimaciones colorimétricas digitales a partir de imágenes RGB. '
        'La precisión depende de iluminación, cámara, reactivos, linealidad del método, '
        'preparación de estándares y validación experimental. Para fines regulatorios, '
        'los resultados deben confirmarse mediante métodos oficiales en laboratorios acreditados.</div>',
        unsafe_allow_html=True,
    )

    render_footer()


# ════════════════════════════════════════════════════════════════
#  PÁGINA 2: PARA SABER MÁS
# ════════════════════════════════════════════════════════════════

elif pagina == "📖 Para saber más":

    st.markdown("<h1>📖 Para saber más</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{TEXT_SUB};'>Sección educativa sobre los principios analíticos detrás de Elementa PWA.</p>",
                unsafe_allow_html=True)

    TOPICS = {
        "🎨 ¿Qué es la colorimetría digital?": """
La **colorimetría digital** es el análisis cuantitativo del color de imágenes capturadas digitalmente para estimar
la concentración de un analito. Un smartphone captura la luz reflejada o transmitida por una solución coloreada,
descomponiéndola en tres canales: **Rojo (R), Verde (G) y Azul (B)**. La intensidad de cada canal es proporcional
a la cantidad de color en ese rango espectral.

A diferencia de un espectrofotómetro UV-Vis que barre longitudes de onda individuales con precisión nanométrica,
la cámara integra bandas amplias (≈400–500 nm para B, ≈500–580 nm para G, ≈580–700 nm para R). Por eso se habla
de un **sistema colorimétrico digital calibrado**, no de un espectrofotómetro.
        """,
        "🔬 Diferencia entre espectrofotometría UV-Vis y análisis RGB": """
| Criterio | UV-Vis clásico | Análisis RGB (smartphone) |
|---|---|---|
| Fuente de luz | Tungsteno/deuterio controlado | Luz ambiental o flash LED |
| Detector | Fotodiodo o arreglo CCD | Sensor CMOS (cámara) |
| Resolución espectral | 0.1–2 nm | ~100 nm por canal (banda ancha) |
| Exactitud | Alta (±0.001 Abs) | Moderada (depende de condiciones) |
| Costo | Alto | Bajo (smartphone) |
| Portabilidad | Limitada | Alta |
| Validación requerida | Sí (protocolos ISO/NOM) | Sí (experimental por usuario) |

El análisis RGB es valioso para **escrutinio rápido** y trabajo de campo, pero requiere calibración por lote y
no reemplaza la confirmación instrumental para fines normativos.
        """,
        "🧪 Historia y uso de la ditizona": """
La **ditizona (difeniltiocarbazona)** es un quelante orgánico altamente específico, descubierto a finales del
siglo XIX. Su solución en disolventes orgánicos es verde intensa; al reaccionar con metales pesados como
**Pb, Cd, Zn, Hg, Cu** forma complejos de colores vivos (rojo para Pb, amarillo-anaranjado para Cd, etc.).

Fue ampliamente usada en métodos espectrofotométricos para trazas de metales antes de la difusión de la
espectrometría de absorción atómica. Hoy sigue siendo relevante en análisis de campo y educación analítica.
El método ditizona-Pb es la base de muchos kits portátiles de detección de plomo.
        """,
        "☠️ Toxicidad y efectos del Cr(VI)": """
El **cromo hexavalente (Cr(VI))** es un carcinógeno bien establecido (Grupo 1 IARC). Sus efectos incluyen:

- **Inhalación**: cáncer de pulmón, perforación del tabique nasal.
- **Ingestión**: úlceras gastrointestinales, daño hepático y renal, posible cáncer colorrectal.
- **Piel**: dermatitis, úlceras crónicas (úlceras de cromo).

Fuentes de exposición: industria galvánica, curtido de pieles, pigmentos, preservantes de madera CCA.
La NOM-127-SSA1-2021 establece un límite de **0.05 mg/L de Cr total** en agua potable.
        """,
        "🐟 Bioacumulación de metales pesados": """
Los metales pesados (Pb, Cd, Hg, As) son **no biodegradables**: una vez en el ambiente, se redistribuyen
entre sedimentos, agua y organismos. La **bioacumulación** ocurre cuando un organismo absorbe el metal
más rápido de lo que lo elimina. La **biomagnificación** amplifica la concentración a lo largo de la cadena
trófica: alga → invertebrado → pez pequeño → pez grande → humano.

Esto explica por qué los límites normativos para estos metales en agua son del orden de µg/L (partes por billón),
aunque sus efectos tóxicos agudos ocurran a concentraciones mucho mayores.
        """,
        "🍇 Ensayos antioxidantes: DPPH, ABTS, FRAP, Fenoles totales": """
**DPPH** (2,2-difenil-1-picrilhidrazilo): radical libre de color púrpura intenso. Al reaccionar con un
antioxidante, se reduce y pierde color (decoloración). *La señal disminuye con mayor capacidad antioxidante.*
Canal de mayor sensibilidad: suele ser el canal **R** o **G** según el instrumento.

**ABTS** (ácido 2,2'-azino-bis(3-etilbenzotiazolín-6-sulfónico)): radical catión verde-azulado. Similar al
DPPH, mide la capacidad de decoloración. Los resultados se expresan como equivalentes Trolox (TEAC).

**FRAP** (Ferric Reducing Antioxidant Power): mide la reducción de Fe³⁺ a Fe²⁺ con formación del complejo
azul ferroso-tripiridiltriazina. *La señal aumenta con mayor capacidad antioxidante.*

**Fenoles totales (Folin-Ciocalteu)**: reactivo que se reduce a un complejo azul en presencia de fenoles.
Señal creciente con concentración. Resultados en equivalentes de ácido gálico (GAE).
        """,
        "📉 ¿Por qué algunas curvas antioxidantes tienen pendiente negativa?": """
Depende del canal óptico y del ensayo:

- En **DPPH** el radical tiene máximo de absorbancia ~515 nm (rango verde). Al disminuir la concentración
  del radical, la señal en el canal G disminuye → curva descendente si se grafica señal vs concentración
  antioxidante.
- Si se grafica **absorbancia digital del canal G** vs concentración antioxidante, la pendiente puede ser
  **negativa** porque más antioxidante = menos absorbancia.
- Elementa PWA mantiene la ecuación y = mx + b con m que puede ser negativa, y calcula correctamente
  la concentración con x = (y − b) / m.

La interpretación es clave: una absorbancia digital negativa respecto al blanco significa que la muestra
es *más transparente* que el blanco en ese canal, lo cual es físicamente posible y analíticamente válido.
        """,
        "📱 Limitaciones del smartphone como instrumento analítico": """
1. **Iluminación no controlada**: la luz ambiental varía en intensidad, ángulo y temperatura de color.
   Se recomienda usar caja de luz o condiciones reproducibles.
2. **Sensor CMOS**: respuesta no lineal a altas intensidades (saturación). Usar exposición manual si es posible.
3. **Compresión JPEG**: puede alterar valores RGB. Preferir formato RAW o PNG.
4. **Variabilidad entre dispositivos**: diferentes cámaras tienen diferentes curvas de respuesta espectral.
   La calibración es **específica para cada dispositivo y condición experimental**.
5. **Drift temporal**: la misma cámara puede variar entre sesiones por temperatura, actualización de firmware
   o suciedad del lente.
6. **Rango dinámico limitado**: concentraciones muy altas o muy bajas pueden caer fuera del rango lineal RGB.

**Recomendación**: validar siempre con estándares certificados, mantener condiciones experimentales constantes
y reportar los parámetros de captura (ISO, balance de blancos, distancia focal).
        """,
    }

    for title, content in TOPICS.items():
        with st.expander(title):
            st.markdown(content)

    render_footer()


# ════════════════════════════════════════════════════════════════
#  PÁGINA 3: FUENTES E INFORMACIÓN
# ════════════════════════════════════════════════════════════════

elif pagina == "📋 Fuentes e Información":

    st.markdown("<h1>📋 Fuentes e Información</h1>", unsafe_allow_html=True)

    # ── Tabla de límites normativos ───────────────────────────
    st.subheader("Límites permisibles de referencia")
    st.markdown(
        '<div class="warning-box">⚠️ Los valores mostrados son informativos. '
        'Verificar siempre los límites permisibles en la versión oficial vigente de la norma aplicable. '
        'Los parámetros normativos pueden actualizarse. Consultar DOF (Diario Oficial de la Federación).</div>',
        unsafe_allow_html=True,
    )

    norm_rows = []
    for analyte, normas in NORMATIVE_LIMITS.items():
        for norma, limite in normas.items():
            norm_rows.append({
                "Analito": analyte,
                "Norma":   norma,
                "Límite (mg/L)": limite,
            })
    df_norm_tbl = pd.DataFrame(norm_rows)
    st.dataframe(df_norm_tbl, use_container_width=True)

    # ── Referencias normativas ────────────────────────────────
    st.subheader("Referencias normativas y técnicas")
    refs = [
        ("NOM-127-SSA1-2021",
         "Agua para uso y consumo humano. Límites permisibles de calidad del agua. DOF 2021.",
         "https://www.dof.gob.mx"),
        ("NOM-001-SEMARNAT-2021",
         "Establece los límites permisibles de contaminantes en las descargas de aguas residuales en cuerpos receptores. DOF 2021.",
         "https://www.dof.gob.mx"),
        ("Wrolstad, R.E. et al. (2005)",
         "Handbook of Food Analytical Chemistry. Wiley.",
         ""),
        ("Cardoso Steele, J.L. (2019)",
         "Digital image colorimetry on smartphone for food analysis. Trends in Analytical Chemistry.",
         ""),
        ("Brand-Williams, W. et al. (1995)",
         "Use of a free radical method to evaluate antioxidant activity. LWT Food Science and Technology.",
         ""),
        ("IARC Monographs Vol. 49 (1990)",
         "Chromium, Nickel and Welding. IARC, Lyon. [Cr(VI) Grupo 1].",
         "https://monographs.iarc.who.int"),
        ("PhotoMetrix App",
         "Aplicación de referencia para análisis colorimétrico con smartphone. Múltiples publicaciones académicas.",
         ""),
    ]
    for ref_id, ref_text, url in refs:
        if url:
            st.markdown(f"- **{ref_id}**: {ref_text} [🔗 Ver]({url})")
        else:
            st.markdown(f"- **{ref_id}**: {ref_text}")

    # ── Editar límites normativos ─────────────────────────────
    st.divider()
    st.subheader("Editar límites normativos (sesión actual)")
    st.markdown(
        '<div class="info-box">ℹ️ Los cambios aplican solo durante esta sesión. '
        'Para hacer cambios permanentes, edita el diccionario <code>NORMATIVE_LIMITS</code> '
        'directamente en <code>elementa_app.py</code>.</div>',
        unsafe_allow_html=True,
    )
    ed_rows = []
    for analyte, normas in NORMATIVE_LIMITS.items():
        for norma, limite in normas.items():
            ed_rows.append({"Analito": analyte, "Norma": norma, "Limite_mg_L": limite})
    ed_df = st.data_editor(
        pd.DataFrame(ed_rows),
        column_config={
            "Limite_mg_L": st.column_config.NumberColumn("Límite (mg/L)", min_value=0.0, step=0.001, format="%.5f"),
        },
        num_rows="fixed", use_container_width=True, key="norm_editor",
    )
    if st.button("Aplicar límites editados (sesión)"):
        NORMATIVE_LIMITS.clear()
        for _, row in ed_df.iterrows():
            an, no, lim = row["Analito"], row["Norma"], row["Limite_mg_L"]
            if an not in NORMATIVE_LIMITS:
                NORMATIVE_LIMITS[an] = {}
            NORMATIVE_LIMITS[an][no] = lim
        st.success("Límites actualizados para esta sesión.")

    render_footer()
