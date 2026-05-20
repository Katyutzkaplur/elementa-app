"""
Elementa — Sistema Analítico Colorimétrico Digital
Derechos reservados (Katyutzka Villarreal, 2026)

Software científico educativo para colorimetría digital.
No sustituye métodos instrumentales certificados.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
from scipy import stats
from io import BytesIO
import base64, datetime, math, hashlib, warnings, tempfile, os
warnings.filterwarnings("ignore")

# ─── Zona horaria: America/Mexico_City (GMT-6 / CDT GMT-5 en verano) ─────────
def _tz_cdmx():
    """Retorna objeto timezone para America/Mexico_City. Compatible Python 3.9+."""
    try:
        from zoneinfo import ZoneInfo          # Python 3.9+ / tzdata
        return ZoneInfo("America/Mexico_City")
    except Exception:
        pass
    try:
        import pytz                            # fallback
        return pytz.timezone("America/Mexico_City")
    except Exception:
        pass
    return datetime.timezone(datetime.timedelta(hours=-6))  # offset fijo

def now_cdmx() -> datetime.datetime:
    """Datetime actual en zona horaria de la Ciudad de México."""
    return datetime.datetime.now(_tz_cdmx())

def fmt_cdmx(dt: datetime.datetime | None = None) -> str:
    """Formatea datetime para reportes. Ej: 2026-05-19  21:43:12  (GMT-0600)"""
    if dt is None:
        dt = now_cdmx()
    return dt.strftime("%Y-%m-%d  %H:%M:%S  (GMT%z)")

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Elementa",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta científica
BG         = "#020617"
PRIMARY    = "#0F172A"
SECONDARY  = "#111827"
CARD       = "#1E293B"
CARD2      = "#263546"
ACCENT     = "#2563EB"
SUCCESS    = "#059669"
DANGER     = "#DC2626"
TEXT       = "#E2E8F0"
MUTED      = "#94A3B8"
BORDER     = "#1E293B"
BORDER2    = "#334155"
PLOT_BG    = "#0B1120"

TIPO_SHORT = {
    "Blanco":           "BL",
    "Estandar":         "STD",
    "Muestra":          "SMP",
    "Control":          "CTRL",
    "Adicion estandar": "ADD",
    "Sin asignar":      "--",
}
TIPO_COLORS = {
    "Blanco":           "#0EA5E9",
    "Estandar":         "#059669",
    "Muestra":          "#2563EB",
    "Control":          "#7C3AED",
    "Adicion estandar": "#DB2777",
    "Sin asignar":      "#1E293B",
}
TIPO_COLORS_BGR = {
    "Blanco":           (8, 165, 233),
    "Estandar":         (5, 150, 105),
    "Muestra":          (37, 99, 235),
    "Control":          (124, 58, 237),
    "Adicion estandar": (219, 39, 119),
    "Sin asignar":      (30, 41, 59),
}
TIPOS    = ["Sin asignar","Blanco","Estandar","Muestra","Control","Adicion estandar"]
ANALITOS = ["Cr(VI)","Pb","Cd","Cr total","DPPH","ABTS","FRAP","Fenoles totales","Otro"]
UNIDADES = ["mg/L","ug/L","ppm","uM","mM","%","ug/mL","Otro"]

# ═══════════════════════════════════════════════════════════════════════
#  CSS  —  Inter, dark mode profesional
# ═══════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ background-color:{BG}; color:{TEXT}; font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{
    background:{PRIMARY}; border-right:1px solid {BORDER2};
}}
[data-testid="stSidebar"] * {{ font-family:'Inter',sans-serif; }}

h1 {{ font-size:1.6rem; font-weight:700; color:{TEXT}; letter-spacing:-0.02em; margin-bottom:.2rem; }}
h2 {{ font-size:1.15rem; font-weight:600; color:{TEXT}; letter-spacing:-0.01em; }}
h3 {{ font-size:.95rem; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em; }}

.metric-card {{
    background:{CARD}; border:1px solid {BORDER2}; border-radius:8px;
    padding:16px 18px; transition:border-color .2s;
}}
.metric-card:hover {{ border-color:{ACCENT}; }}
.metric-card .mc-label {{
    font-size:.68rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.09em; color:{MUTED}; margin:0 0 6px 0;
}}
.metric-card .mc-value {{
    font-size:1.45rem; font-weight:700; color:{TEXT}; margin:0; line-height:1.1;
    font-family:'JetBrains Mono',monospace;
}}
.metric-card .mc-interpret {{
    font-size:.72rem; color:{MUTED}; margin:4px 0 0 0; font-style:italic;
}}
.metric-card .mc-explain {{
    font-size:.71rem; color:{MUTED}; margin:6px 0 0 0; line-height:1.45;
    border-top:1px solid {BORDER2}; padding-top:6px;
}}

.info-box {{
    background:{SECONDARY}; border-left:3px solid {ACCENT}; border-radius:0 6px 6px 0;
    padding:10px 14px; font-size:.82rem; color:#93C5FD; margin:8px 0; line-height:1.5;
}}
.warn-box {{
    background:{SECONDARY}; border-left:3px solid {DANGER}; border-radius:0 6px 6px 0;
    padding:10px 14px; font-size:.82rem; color:#FCA5A5; margin:8px 0; line-height:1.5;
}}
.ok-box {{
    background:{SECONDARY}; border-left:3px solid {SUCCESS}; border-radius:0 6px 6px 0;
    padding:10px 14px; font-size:.82rem; color:#6EE7B7; margin:8px 0; line-height:1.5;
}}
.section-label {{
    font-size:.65rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:{ACCENT}; margin:0 0 4px 0;
}}
.stat-block {{
    background:{CARD}; border:1px solid {BORDER2}; border-radius:8px;
    padding:14px 16px; margin-bottom:6px;
}}
.stat-block .sb-param {{
    font-size:.8rem; font-weight:600; color:{TEXT};
    font-family:'JetBrains Mono',monospace;
}}
.stat-block .sb-value {{
    font-size:1.15rem; font-weight:700; color:{ACCENT};
    font-family:'JetBrains Mono',monospace;
}}
.stat-block .sb-interp {{
    font-size:.72rem; font-weight:600; margin:3px 0 0 0;
}}
.stat-block .sb-explain {{
    font-size:.71rem; color:{MUTED}; margin:6px 0 0 0; line-height:1.45;
}}
.badge-pass {{ background:#052e16; color:#4ADE80; padding:2px 9px; border-radius:3px;
    font-size:.75rem; font-weight:700; letter-spacing:.04em; border:1px solid #166534; }}
.badge-fail {{ background:#450a0a; color:#F87171; padding:2px 9px; border-radius:3px;
    font-size:.75rem; font-weight:700; letter-spacing:.04em; border:1px solid #991b1b; }}
.badge-none {{ background:{CARD}; color:{MUTED}; padding:2px 9px; border-radius:3px;
    font-size:.75rem; font-weight:600; border:1px solid {BORDER2}; }}

.footer {{
    text-align:center; color:{MUTED}; font-size:.7rem;
    padding:28px 0 10px; margin-top:48px; border-top:1px solid {BORDER2};
    font-family:'Inter',sans-serif;
}}
.stButton>button {{
    background:{ACCENT}; color:#fff; border:none; border-radius:6px;
    padding:9px 22px; font-weight:600; font-size:.84rem; letter-spacing:.02em;
    font-family:'Inter',sans-serif; transition:background .2s;
}}
.stButton>button:hover {{ background:#1D4ED8; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {BORDER2}; }}
.stTabs [data-baseweb="tab"] {{
    background:transparent; color:{MUTED}; font-weight:500;
    font-size:.85rem; padding:10px 20px; border-radius:6px 6px 0 0;
}}
.stTabs [aria-selected="true"] {{
    background:{CARD} !important; color:{TEXT} !important;
    border-bottom:2px solid {ACCENT} !important;
}}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  LÍMITES NORMATIVOS
# ═══════════════════════════════════════════════════════════════════════

NORMATIVE_LIMITS = {
    "Pb":       {"NOM-127-SSA1-2021 agua potable":0.01,"NOM-001-SEMARNAT-2021 descarga A":0.2,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cd":       {"NOM-127-SSA1-2021 agua potable":0.003,"NOM-001-SEMARNAT-2021 descarga A":0.1,"NOM-001-SEMARNAT-2021 descarga B":0.2},
    "Cr total": {"NOM-127-SSA1-2021 agua potable":0.05,"NOM-001-SEMARNAT-2021 descarga A":0.5,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cr(VI)":   {"NOM-127-SSA1-2021 agua potable":0.05},
}

STAT_EXPL = {
    "R2":        "Coeficiente de determinacion. Indica que fraccion de la varianza en la senal es explicada por el modelo lineal. Rango 0-1; valores cercanos a 1 indican linealidad solida.",
    "slope":     "Pendiente (m). Sensibilidad analitica del metodo: cuanto cambia la senal por unidad de concentracion. Pendiente negativa es valida en ensayos donde el analito reduce el color.",
    "intercept": "Intercepto (b). Valor teorico de la senal a concentracion cero. Idealmente proximo al valor del blanco de reactivos.",
    "se":        "Error estandar de la pendiente (Sb). Incertidumbre en la estimacion de la sensibilidad analitica. Sb/|m| < 1% indica excelente precision del ajuste.",
    "LOD":       "Limite de Deteccion (3.3·sigma/|m|). Concentracion minima estadisticamente distinguible del ruido. Resultados menores al LOD se reportan como < LOD.",
    "LOQ":       "Limite de Cuantificacion (10·sigma/|m|). Concentracion minima cuantificable con CV < 10%. Resultados entre LOD y LOQ son semicuantitativos.",
    "CV":        "Coeficiente de Variacion (100·SD/media). Medida de precision relativa para replicas. CV < 5% excelente; 5-10% aceptable; > 10% requiere revision tecnica.",
}

def interpret_r2(r2):
    if r2 >= 0.999:  return "Linealidad excelente", SUCCESS
    if r2 >= 0.995:  return "Linealidad muy buena", SUCCESS
    if r2 >= 0.990:  return "Ligera dispersion experimental", "#F59E0B"
    if r2 >= 0.980:  return "Calibracion aceptable", "#F59E0B"
    return "Revisar calibracion", DANGER

def interpret_cv(cv):
    if cv is None or math.isnan(cv): return "--", MUTED
    if cv < 5:   return "Excelente", SUCCESS
    if cv < 10:  return "Aceptable", "#F59E0B"
    return "Revisar", DANGER

# ═══════════════════════════════════════════════════════════════════════
#  FUNCIONES NUCLEARES
# ═══════════════════════════════════════════════════════════════════════

def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None: return None
    return np.array(Image.open(uploaded).convert("RGB"))

def generate_rois_linear(x0,y0,w,h,n,dx,dy):
    return [{"x":int(x0+i*dx),"y":int(y0+i*dy),"w":int(w),"h":int(h),"label":f"ROI {i+1}"} for i in range(n)]

def generate_rois_microplate(x0,y0,w,h,dx,dy,rows=8,cols=12):
    rl="ABCDEFGH"
    return [{"x":int(x0+c*dx),"y":int(y0+r*dy),"w":int(w),"h":int(h),"label":f"{rl[r]}{c+1}"}
            for r in range(rows) for c in range(cols)]

# ═══════════════════════════════════════════════════════════════════════
#  DETECCIÓN AUTOMÁTICA DE ROIs
# ═══════════════════════════════════════════════════════════════════════

def sort_wells_to_grid(circles_arr: np.ndarray) -> list[dict]:
    """
    Ordena círculos detectados en una cuadrícula A1-H12.
    Agrupa por fila usando clustering por coordenada Y.
    """
    if len(circles_arr) == 0:
        return []
    row_labels = list("ABCDEFGH")
    # Ordenar por Y (arriba a abajo)
    sorted_c = sorted(circles_arr.tolist(), key=lambda c: c[1])
    # Radio medio para umbral de separación de filas
    mean_r = float(np.mean([c[2] for c in sorted_c]))
    row_gap = max(mean_r * 1.4, 8.0)
    # Agrupar en filas por brecha de coordenada Y
    rows_grouped: list[list] = [[sorted_c[0]]]
    for c in sorted_c[1:]:
        if c[1] - rows_grouped[-1][-1][1] > row_gap:
            rows_grouped.append([c])
        else:
            rows_grouped[-1].append(c)
    # Construir ROIs ordenadas por fila (A→H) y columna (1→12)
    rois = []
    for r_idx, row in enumerate(rows_grouped):
        if r_idx >= len(row_labels):
            break
        for c_idx, (cx, cy, cr) in enumerate(sorted(row, key=lambda c: c[0])):
            rois.append({
                "x": max(0, int(cx - cr)),
                "y": max(0, int(cy - cr)),
                "w": int(cr * 2), "h": int(cr * 2),
                "label": f"{row_labels[r_idx]}{c_idx+1}",
                "_cx": int(cx), "_cy": int(cy), "_cr": int(cr),
            })
    return rois


# ── Presets de sensibilidad ───────────────────────────────────────────────
# Cada preset define (hough_p2, min_i, max_i, min_sat, min_std)
# hough_p2: umbral acumulador Hough (menor = detecta más círculos)
# min/max_i: rango de intensidad media aceptable
# min_sat:   saturación HSV mínima (discrimina soluciones sin color)
# min_std:   desviación estándar mínima (descarta zonas uniformes/vacías)
DETECTION_PRESETS = {
    "Alta (permisiva)":    dict(hough_p2=18, min_i=12, max_i=250, min_sat=2,  min_std=2),
    "Media (recomendada)": dict(hough_p2=28, min_i=22, max_i=240, min_sat=8,  min_std=5),
    "Baja (estricta)":     dict(hough_p2=42, min_i=35, max_i=220, min_sat=18, min_std=9),
}


def filter_well_by_content(img_rgb: np.ndarray, roi: dict,
                            detect_only_filled: bool = True,
                            min_intensity: float = 22,
                            max_intensity: float = 240,
                            min_saturation: float = 8,
                            min_std: float = 5) -> bool:
    """
    Retorna True si el ROI contiene muestra analítica real.

    Descarta automáticamente
    ------------------------
    - Pocillos vacíos (alta intensidad uniforme, baja saturación)
    - Reflejos especulares (intensidad > max_intensity con muy baja desv. std)
    - Regiones oscuras sin muestra (intensidad < min_intensity)
    - Zonas de fondo uniformes (std < min_std)
    - Soluciones incoloras/agua pura (saturación HSV < min_saturation)

    Análisis multicanal
    -------------------
    1. Intensidad media (RGB) — fuera de rango: vacío o reflejo
    2. Desviación estándar (RGB) — demasiado uniforme: vacío / agua
    3. Saturación HSV media — muy baja: incoloro / aire / fondo blanco
    4. Detección de saturación localizada — al menos 20% de píxeles con
       saturación > 15 (descarta soluciones muy diluidas pero con ruido global)
    """
    H, W = img_rgb.shape[:2]
    x1 = max(0, roi["x"]); y1 = max(0, roi["y"])
    x2 = min(W, x1 + roi["w"]); y2 = min(H, y1 + roi["h"])
    crop = img_rgb[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return False

    mean_i = float(crop.mean())

    # ── 1. Rango de intensidad ────────────────────────────────────────
    if mean_i < min_intensity:
        return False   # Demasiado oscuro: pocillo tapado, sombra, suciedad
    if mean_i > max_intensity:
        return False   # Demasiado brillante: reflejo especular, pocillo vacío

    if not detect_only_filled:
        return True    # Modo permisivo: solo filtros geométricos/intensidad

    # ── 2. Variación interna (std) ────────────────────────────────────
    std_i = float(crop.std())
    if std_i < min_std:
        return False   # Zona demasiado uniforme: pocillo vacío / agua / reflejo plano

    # ── 3. Saturación HSV ─────────────────────────────────────────────
    crop8   = crop.astype(np.uint8)
    hsv     = cv2.cvtColor(cv2.cvtColor(crop8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    sat_ch  = hsv[:, :, 1].astype(float)
    mean_sat= float(sat_ch.mean())
    if mean_sat < min_saturation:
        return False   # Sin color: agua clara, pocillo vacío, fondo blanco

    # ── 4. Fracción de píxeles "coloreados" ───────────────────────────
    colored_frac = float((sat_ch > 15).mean())
    if colored_frac < 0.15:
        return False   # Menos del 15% de píxeles tienen color apreciable

    return True


def detect_microplate_rois(img: np.ndarray,
                            min_r: int = 8, max_r: int = 40,
                            sensitivity: int = 28,
                            min_dist: int = 20,
                            detect_only_filled: bool = True,
                            min_intensity: float = 22,
                            max_intensity: float = 240,
                            min_saturation: float = 8,
                            min_std: float = 5) -> list[dict]:
    """
    Detecta pocillos circulares con filtrado inteligente de contenido.

    Paso 1 — Geometría: HoughCircles con CLAHE para robustar ante
             iluminación de smartphone.
    Paso 2 — Contenido: descarta pocillos vacíos, reflejos y ruido
             usando análisis de intensidad, std y saturación HSV.

    Retorna lista de ROIs con etiquetas A1-H12 para pocillos activos.
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=int(min_dist),
        param1=max(30, int(sensitivity * 1.8)),
        param2=int(sensitivity),
        minRadius=int(min_r), maxRadius=int(max_r),
    )
    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    all_rois = sort_wells_to_grid(circles)

    # ── Filtrado por contenido ────────────────────────────────────────
    filtered = []
    for roi in all_rois:
        if filter_well_by_content(
                img, roi,
                detect_only_filled=detect_only_filled,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
                min_saturation=min_saturation,
                min_std=min_std):
            filtered.append(roi)

    return filtered


def detect_vial_rois(img: np.ndarray,
                     min_area: int = 400,
                     max_area_ratio: float = 0.20,
                     sensitivity: int = 50,
                     detect_only_filled: bool = True,
                     min_intensity: float = 22,
                     max_intensity: float = 240,
                     min_saturation: float = 8,
                     min_std: float = 5) -> list[dict]:
    """
    Detecta viales/tubos con umbral adaptativo y filtrado de contenido.

    Paso 1 — Contornos: umbral adaptativo Gaussiano + morfología.
    Paso 2 — Contenido: descarta viales vacíos y ruido usando los
             mismos criterios que detect_microplate_rois.
    """
    H, W     = img.shape[:2]
    max_area = int(H * W * max_area_ratio)
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    c_val  = max(2, int((100 - sensitivity) / 8))
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, c_val)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect      = w / h if h > 0 else 0
        if not (0.15 < aspect < 6.0):
            continue
        compactness = area / (w * h) if w * h > 0 else 0
        if compactness < 0.25:
            continue
        # ── Circularidad del contorno ─────────────────────────────────
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        candidates.append({
            "x": x, "y": y, "w": w, "h": h, "label": "",
            "_area": area, "_circ": circularity})

    # ── Filtrado por contenido ────────────────────────────────────────
    filtered = []
    for roi in candidates:
        if filter_well_by_content(
                img, roi,
                detect_only_filled=detect_only_filled,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
                min_saturation=min_saturation,
                min_std=min_std):
            filtered.append(roi)

    # Ordenar izquierda→derecha
    filtered.sort(key=lambda r: (r["y"] // max(1, H // 6), r["x"]))
    for i, roi in enumerate(filtered):
        roi["label"] = f"ROI {i + 1}"
    return filtered


def validate_rois(rois: list[dict], img_shape: tuple,
                  min_px: int = 4, max_fraction: float = 0.45) -> list[dict]:
    """
    Filtra ROIs fuera de límites o con tamaño inválido.
    Elimina también duplicados por solapamiento excesivo (IoU > 0.5).
    """
    H, W = img_shape[:2]
    valid = []
    for roi in rois:
        w, h = roi["w"], roi["h"]
        x, y = roi["x"], roi["y"]
        if w < min_px or h < min_px:
            continue
        if w > W * max_fraction or h > H * max_fraction:
            continue
        # Clamp a bordes de imagen
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = min(w, W - x)
        h = min(h, H - y)
        if w < min_px or h < min_px:
            continue
        valid.append({**roi, "x": x, "y": y, "w": w, "h": h})

    # Eliminar solapamiento (NMS simple)
    kept = []
    for roi in valid:
        overlap = False
        for k in kept:
            # Intersección sobre unión simplificada
            ix1 = max(roi["x"], k["x"]); iy1 = max(roi["y"], k["y"])
            ix2 = min(roi["x"]+roi["w"], k["x"]+k["w"])
            iy2 = min(roi["y"]+roi["h"], k["y"]+k["h"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = roi["w"]*roi["h"] + k["w"]*k["h"] - inter
                if union > 0 and inter/union > 0.50:
                    overlap = True; break
        if not overlap:
            kept.append(roi)
    return kept


def draw_detected_preview(img: np.ndarray, rois: list[dict],
                           color_ok=(0, 200, 100), color_label=(220, 220, 220)) -> np.ndarray:
    """
    Preview especial para ROIs auto-detectadas: círculo + etiqueta.
    Para círculos detectados usa el centro y radio almacenados.
    """
    out = img.copy()
    for roi in rois:
        cx = roi.get("_cx", roi["x"] + roi["w"] // 2)
        cy = roi.get("_cy", roi["y"] + roi["h"] // 2)
        cr = roi.get("_cr", min(roi["w"], roi["h"]) // 2)
        bgr = (color_ok[2], color_ok[1], color_ok[0])
        cv2.circle(out, (cx, cy), cr, bgr, 2)
        cv2.circle(out, (cx, cy), 2, bgr, -1)
        cv2.putText(out, roi["label"], (roi["x"], max(roi["y"] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (color_label[2], color_label[1], color_label[0]), 1, cv2.LINE_AA)
    return out

def roi_fingerprint(rois):
    return hashlib.md5("|".join(r["label"] for r in rois).encode()).hexdigest()[:12]

def ensure_assignment_df(rois):
    """Crea o actualiza assignment_df preservando ediciones existentes."""
    fp = roi_fingerprint(rois)
    current_fp = st.session_state.get("_asgn_fp", "")
    if fp == current_fp and st.session_state.get("assignment_df") is not None:
        return  # Sin cambios — mantener datos del usuario
    old_map = {}
    old = st.session_state.get("assignment_df")
    if old is not None and "ROI" in old.columns:
        for _, row in old.iterrows():
            old_map[row["ROI"]] = row.to_dict()
    rows = []
    for roi in rois:
        if roi["label"] in old_map:
            rows.append(old_map[roi["label"]])
        else:
            rows.append({"ROI":roi["label"],"Tipo":"Sin asignar","Nombre":"","Concentracion":0.0,
                         "Unidad":"mg/L","Factor_dil":1.0,"Analito":"Cr(VI)","Observaciones":""})
    st.session_state["assignment_df"] = pd.DataFrame(rows)
    st.session_state["_asgn_fp"] = fp

def draw_rois(img, rois, type_map=None):
    out = img.copy()
    for roi in rois:
        tipo = (type_map or {}).get(roi["label"], "Sin asignar")
        rgb  = TIPO_COLORS_BGR.get(tipo, (30,41,59))
        bgr  = (rgb[2], rgb[1], rgb[0])
        x,y,w,h = roi["x"],roi["y"],roi["w"],roi["h"]
        cv2.rectangle(out,(x,y),(x+w,y+h),bgr,2)
        short = TIPO_SHORT.get(tipo, "?")
        cv2.putText(out,f"{roi['label']}",(x,max(y-3,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,bgr,1,cv2.LINE_AA)
    return out

def extract_rgb(img, rois):
    H,W = img.shape[:2]
    rows = []
    for roi in rois:
        x,y,w,h = roi["x"],roi["y"],roi["w"],roi["h"]
        crop = img[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
        if crop.size == 0:
            rows.append({"ROI":roi["label"],"R":np.nan,"G":np.nan,"B":np.nan,
                         "R_sd":np.nan,"G_sd":np.nan,"B_sd":np.nan})
        else:
            rows.append({"ROI":roi["label"],
                         "R":round(crop[:,:,0].mean(),2),"G":round(crop[:,:,1].mean(),2),"B":round(crop[:,:,2].mean(),2),
                         "R_sd":round(crop[:,:,0].std(),2),"G_sd":round(crop[:,:,1].std(),2),"B_sd":round(crop[:,:,2].std(),2)})
    return pd.DataFrame(rows)

def normalize_rgb(df):
    df = df.copy()
    eps = 1e-9
    tot = df["R"]+df["G"]+df["B"]+eps
    df["R_norm"] = (df["R"]/tot)*100
    df["G_norm"] = (df["G"]/tot)*100
    df["B_norm"] = (df["B"]/tot)*100
    df["Total"]  = df["R"]+df["G"]+df["B"]
    return df

def calc_absorbance(df, blank_roi, channels=("R_norm","G_norm","B_norm")):
    df = df.copy()
    for ch in channels:
        col = f"A_{ch}"
        if ch not in df.columns or blank_roi is None or blank_roi not in df["ROI"].values:
            df[col] = np.nan; continue
        bv = float(df.loc[df["ROI"]==blank_roi, ch].values[0])
        if np.isnan(bv): df[col]=np.nan; continue
        eps = 1e-9
        df[col] = df[ch].apply(lambda v: math.log10((bv+eps)/(v+eps)) if pd.notna(v) else np.nan)
    return df

def fit_line(x, y):
    mask = ~(np.isnan(x)|np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x)<2: return None
    m,b,r,_,se = stats.linregress(x,y)
    return {"m":m,"b":b,"r2":r**2,"se":se,"n":len(x),
            "res":y-(m*x+b),"x_fit":x,"y_fit":y}

def select_channel(df, std_rois):
    std = df[df["ROI"].isin(std_rois)].copy()
    if len(std)<2 or "Concentracion" not in std.columns: return "G_norm",{}
    concs = std["Concentracion"].values.astype(float)
    best_ch, best_r2, results = "G_norm", -1, {}
    for ch in ("R_norm","G_norm","B_norm"):
        ac = f"A_{ch}"
        if ac not in std.columns: continue
        sub = std[["Concentracion",ac]].dropna()
        if len(sub)<2: continue
        cal = fit_line(sub["Concentracion"].values.astype(float), sub[ac].values.astype(float))
        if cal:
            results[ch] = cal
            if cal["r2"] > best_r2: best_r2=cal["r2"]; best_ch=ch
    return best_ch, results

def lod_loq(cal, blank_sigs=None):
    m = abs(cal["m"])
    if m < 1e-12: return np.nan, np.nan, True
    sigma = np.std(blank_sigs,ddof=1) if blank_sigs is not None and len(blank_sigs)>=2 else cal["se"]
    proxy = blank_sigs is None or len(blank_sigs)<2
    return 3.3*sigma/m, 10*sigma/m, proxy

def standard_addition(added, sigs):
    cal = fit_line(np.asarray(added,float), np.asarray(sigs,float))
    if cal is None or abs(cal["m"])<1e-12: return None
    xi = -cal["b"]/cal["m"]
    cal["xi"]=xi; cal["c_sample"]=abs(xi)
    return cal

def normative_eval(analyte, conc):
    if analyte not in NORMATIVE_LIMITS:
        return [{"norma":"Sin criterio","limite":None,"status":"Sin criterio","badge":"none"}]
    return [{"norma":n,"limite":lim,
             "status":"Cumple" if conc<=lim else "No cumple",
             "badge":"pass" if conc<=lim else "fail"}
            for n,lim in NORMATIVE_LIMITS[analyte].items()]

def detect_triplicates(asgn_df):
    """Agrupa ROIs por columna de placa. Retorna {col_str: [roi_labels]}."""
    groups = {}
    for _, row in asgn_df.iterrows():
        roi = row["ROI"]
        col = "".join(c for c in roi if c.isdigit())
        if col and row.get("Tipo","Sin asignar") != "Sin asignar":
            groups.setdefault(col, []).append(roi)
    return {k:v for k,v in groups.items() if len(v)>=2}

def triplate_stats(df_merged, groups, sig_col):
    rows = []
    for col, rlist in sorted(groups.items(), key=lambda x:int(x[0])):
        sub  = df_merged[df_merged["ROI"].isin(rlist)]
        if sub.empty or sig_col not in sub.columns: continue
        sigs = sub[sig_col].dropna().values
        if len(sigs)==0: continue
        mean = float(np.mean(sigs))
        sd   = float(np.std(sigs,ddof=1)) if len(sigs)>1 else float("nan")
        cv   = abs(sd/mean)*100 if not math.isnan(sd) and mean!=0 else float("nan")
        info = df_merged[df_merged["ROI"]==rlist[0]].iloc[0] if len(rlist)>0 else {}
        rows.append({"Columna":col,"Pocillos":", ".join(rlist),
                     "N":len(sigs),"Tipo":info.get("Tipo",""),
                     "Concentracion":round(float(info.get("Concentracion",0)),3),
                     "Media":round(mean,4),"SD":round(sd,4) if not math.isnan(sd) else None,
                     "CV_%":round(cv,2) if not math.isnan(cv) else None})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PLOTLY
# ═══════════════════════════════════════════════════════════════════════

_PLTLAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
    font=dict(family="Inter,sans-serif", color=TEXT, size=11),
    margin=dict(l=52,r=20,t=48,b=48),
)

def plot_microplate(asgn_df, tri_groups):
    rl = list("ABCDEFGH")
    tipo_map  = {r["ROI"]:r.get("Tipo","Sin asignar") for _,r in asgn_df.iterrows()}
    conc_map  = {r["ROI"]:r.get("Concentracion",0) for _,r in asgn_df.iterrows()}
    name_map  = {r["ROI"]:r.get("Nombre","") for _,r in asgn_df.iterrows()}
    rep_map   = {}
    for col, rlist in tri_groups.items():
        for i,roi in enumerate(rlist): rep_map[roi] = f"Rep {i+1}/{len(rlist)}"

    # Construir matriz heatmap
    all_rows = sorted(set(r["ROI"][0] for _,r in asgn_df.iterrows() if r["ROI"][0] in rl),
                      key=lambda x: rl.index(x))
    all_cols = sorted(set(int(r["ROI"][1:]) for _,r in asgn_df.iterrows() if r["ROI"][1:].isdigit()))
    if not all_rows or not all_cols:
        return go.Figure()

    idx_map = {"Blanco":1,"Estandar":2,"Muestra":3,"Control":4,"Adicion estandar":5}
    z,txt,hov = [],[],[]
    for rw in all_rows:
        zr,tr,hr = [],[],[]
        for cl in all_cols:
            roi = f"{rw}{cl}"
            tipo = tipo_map.get(roi,"Sin asignar")
            conc = conc_map.get(roi,0)
            nm   = name_map.get(roi,"")
            rep  = rep_map.get(roi,"")
            short = TIPO_SHORT.get(tipo,"--")
            zr.append(idx_map.get(tipo,0))
            tr.append(short)
            ht = f"<b>{roi}</b><br>Tipo: {tipo}<br>Conc: {conc:.3g}"
            if nm: ht += f"<br>{nm}"
            if rep: ht += f"<br>{rep}"
            hr.append(ht)
        z.append(zr); txt.append(tr); hov.append(hr)

    cs = [
        [0/5, "#0B1120"],[0.19/5,"#0B1120"],
        [1/5, "#0C2540"],[1.19/5,"#0C2540"],
        [2/5, "#052e16"],[2.19/5,"#052e16"],
        [3/5, "#0D2159"],[3.19/5,"#0D2159"],
        [4/5, "#2D0A3E"],[4.19/5,"#2D0A3E"],
        [5/5, "#3B0A1F"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, text=txt, texttemplate="%{text}",
        customdata=hov, hovertemplate="%{customdata}<extra></extra>",
        colorscale=cs, showscale=False,
        zmin=0, zmax=5, xgap=2, ygap=2,
        textfont=dict(family="JetBrains Mono",size=9,color=TEXT),
    ))
    fig.update_xaxes(tickvals=list(range(len(all_cols))),ticktext=[str(c) for c in all_cols],
                     side="top",showgrid=False,tickfont=dict(size=9))
    fig.update_yaxes(tickvals=list(range(len(all_rows))),ticktext=all_rows,
                     autorange="reversed",showgrid=False,tickfont=dict(size=9))
    fig.update_layout(**_PLTLAYOUT, height=max(220,48*len(all_rows)+80),
                      title=dict(text="Distribución de pocillos", font=dict(size=12)))
    return fig

def plot_calibration(concs, sigs, cal, ch, analyte, unit, lod, loq):
    x0 = max(0,float(concs.min())*0.85) if float(concs.min())>0 else 0.0
    x1 = float(concs.max())*1.15
    xl = np.linspace(x0,x1,300)
    yl = cal["m"]*xl+cal["b"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=concs,y=sigs,mode="markers",
        marker=dict(color=ACCENT,size=10,line=dict(color=PLOT_BG,width=1.5)),name="Estandares"))
    fig.add_trace(go.Scatter(x=xl,y=yl,mode="lines",
        line=dict(color=SUCCESS,width=2.2),name="Regresion lineal"))
    if not np.isnan(lod):
        fig.add_vline(x=lod,line_dash="dot",line_color=DANGER,
                      annotation_text=f"LOD={lod:.3f}",annotation_font_color=DANGER,annotation_font_size=9)
    if not np.isnan(loq):
        fig.add_vline(x=loq,line_dash="dot",line_color="#F59E0B",
                      annotation_text=f"LOQ={loq:.3f}",annotation_font_color="#F59E0B",annotation_font_size=9)
    m,b,r2 = cal["m"],cal["b"],cal["r2"]
    sgn = "+" if b>=0 else "-"
    eq  = f"A = {m:.4f}·C {sgn} {abs(b):.4f}   |   R² = {r2:.5f}"
    fig.add_annotation(x=0.03,y=0.97,xref="paper",yref="paper",text=eq,
        showarrow=False,font=dict(color="#4ADE80",size=10,family="JetBrains Mono"),
        bgcolor="rgba(11,17,32,.85)",bordercolor=SUCCESS,borderwidth=1,borderpad=5)
    fig.update_layout(**_PLTLAYOUT,
        title=dict(text=f"Curva de calibracion — {analyte} | Canal {ch}",font=dict(size=12)),
        xaxis_title=f"Concentracion ({unit})",yaxis_title="Absorbancia digital")
    return fig

def plot_residuals(concs, cal, ch):
    yfit = cal["m"]*concs+cal["b"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yfit,y=cal["res"],mode="markers",
        marker=dict(color=ACCENT,size=8,line=dict(color=PLOT_BG,width=1)),name="Residuos"))
    fig.add_hline(y=0,line_dash="dash",line_color=MUTED,line_width=1)
    fig.update_layout(**_PLTLAYOUT,height=260,title=dict(text=f"Residuos — Canal {ch}",font=dict(size=12)),
        xaxis_title="Señal predicha",yaxis_title="Residuo (obs − pred)")
    return fig

def plot_channels(ch_results):
    chs = list(ch_results.keys()); r2s=[ch_results[c]["r2"] for c in chs]
    mx = max(r2s)
    colors = [SUCCESS if v==mx else CARD2 for v in r2s]
    fig = go.Figure(go.Bar(x=chs,y=r2s,marker_color=colors,
        text=[f"{v:.5f}" for v in r2s],textposition="outside",
        textfont=dict(color=TEXT,size=10,family="JetBrains Mono")))
    fig.update_layout(**_PLTLAYOUT,height=260,
        title=dict(text="R² por canal RGB",font=dict(size=12)),
        yaxis=dict(range=[max(0,min(r2s)-.05),1.02],title="R²"),xaxis_title="Canal")
    return fig

def plot_std_addition(added, sigs, sa, analyte, unit):
    xi = sa["xi"]; m,b = sa["m"],sa["b"]
    xmin = min(xi*1.3 if xi<0 else -0.1, min(added)-0.1)
    xmax = max(added)*1.1
    xl = np.linspace(xmin,xmax,300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=added,y=sigs,mode="markers",
        marker=dict(color=ACCENT,size=10,line=dict(color=PLOT_BG,width=1.5)),name="Adiciones"))
    fig.add_trace(go.Scatter(x=xl,y=m*xl+b,mode="lines",
        line=dict(color=SUCCESS,width=2),name="Proyeccion"))
    fig.add_trace(go.Scatter(x=[xi],y=[0],mode="markers+text",
        marker=dict(color=DANGER,size=14,symbol="x-thin",line=dict(color=DANGER,width=3)),
        text=[f" C = {sa['c_sample']:.3f} {unit}"],textposition="middle right",
        textfont=dict(color=DANGER,size=10,family="JetBrains Mono"),name="C muestra"))
    fig.add_hline(y=0,line_dash="dash",line_color=BORDER2)
    fig.update_layout(**_PLTLAYOUT,title=dict(text=f"Adicion de estandar — {analyte}",font=dict(size=12)),
        xaxis_title=f"Concentracion añadida ({unit})",yaxis_title="Señal")
    return fig

# ═══════════════════════════════════════════════════════════════════════
#  GRÁFICA MATPLOTLIB PARA PDF  (sin kaleido, siempre funciona)
# ═══════════════════════════════════════════════════════════════════════

def cal_fig_to_png(cal, concs, sigs, ch, analyte, unit, lod, loq):
    """Genera curva de calibracion como PNG con matplotlib. Robusto en Streamlit."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")

        concs = np.asarray(concs,float); sigs = np.asarray(sigs,float)
        mask = ~(np.isnan(concs)|np.isnan(sigs))
        concs,sigs = concs[mask],sigs[mask]
        if len(concs)<2: return None

        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)

        # Puntos
        ax.scatter(concs,sigs,color="#3B82F6",s=60,zorder=5,edgecolors=PLOT_BG,linewidths=1.2,label="Estandares")
        # Recta
        xmin = float(concs.min())*0.85 if float(concs.min())>0 else 0.0
        xmax = float(concs.max())*1.15
        xl = np.linspace(xmin,xmax,300)
        ax.plot(xl, cal["m"]*xl+cal["b"], color="#10B981", linewidth=2.2, label="Regresion lineal")
        # LOD / LOQ
        y_all = np.concatenate([sigs, cal["m"]*xl+cal["b"]])
        ybot,ytop = float(y_all.min()),float(y_all.max())
        ypad = (ytop-ybot)*0.04 if ytop>ybot else 0.01
        ylbl = ybot+ypad
        lod_ok = lod is not None and not (isinstance(lod,float) and math.isnan(lod))
        loq_ok = loq is not None and not (isinstance(loq,float) and math.isnan(loq))
        if lod_ok:
            ax.axvline(lod,color="#DC2626",linestyle=":",linewidth=1.3)
            ax.text(lod,ylbl,f"  LOD={lod:.3f}",color="#DC2626",fontsize=7,va="bottom")
        if loq_ok:
            ax.axvline(loq,color="#F59E0B",linestyle=":",linewidth=1.3)
            ax.text(loq,ylbl,f"  LOQ={loq:.3f}",color="#F59E0B",fontsize=7,va="bottom")
        # Ecuación
        m,b,r2 = cal["m"],cal["b"],cal["r2"]
        sgn = "+" if b>=0 else "-"
        eq  = f"A = {m:.4f}·C {sgn} {abs(b):.4f}   |   R² = {r2:.5f}"
        ax.text(0.03,0.97,eq,transform=ax.transAxes,fontsize=8.5,color="#4ADE80",va="top",ha="left",
                bbox=dict(facecolor=CARD,edgecolor="#166534",boxstyle="round,pad=0.35"))
        ax.set_xlabel(f"Concentracion ({unit})",color=MUTED,fontsize=9)
        ax.set_ylabel("Absorbancia digital",color=MUTED,fontsize=9)
        ax.set_title(f"Curva de calibracion — {analyte} | Canal: {ch}",color=TEXT,fontsize=10,pad=8)
        ax.tick_params(colors=MUTED,labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER2)
        leg = ax.legend(facecolor=CARD,edgecolor=BORDER2,fontsize=8)
        for t in leg.get_texts(): t.set_color(TEXT)
        ax.grid(True,color=CARD,linewidth=0.5,linestyle="--",zorder=0)
        plt.tight_layout(pad=0.8)
        buf = BytesIO()
        plt.savefig(buf,format="png",dpi=160,bbox_inches="tight",facecolor=PLOT_BG,edgecolor="none")
        buf.seek(0); data=buf.read()
        plt.close(fig)
        return data
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE REPORTE PDF
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf(analyte, method, df_rgb, df_results, cal,
                 annotated_img, tri_df, cal_png_bytes, unit="mg/L"):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.units import inch
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                     Paragraph, Spacer, Table, TableStyle,
                                     Image as RLImage, HRFlowable, KeepTogether)

    C = {
        "bg":     HexColor("#020617"),
        "card":   HexColor("#1E293B"),
        "card2":  HexColor("#263546"),
        "accent": HexColor("#2563EB"),
        "green":  HexColor("#059669"),
        "danger": HexColor("#DC2626"),
        "text":   HexColor("#E2E8F0"),
        "muted":  HexColor("#94A3B8"),
        "border": HexColor("#334155"),
        "note_bg":HexColor("#E8EDF3"),
        "note_tx":HexColor("#0A0A0A"),
    }

    buf = BytesIO()

    def bg_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C["bg"])
        canvas.rect(0,0,letter[0],letter[1],fill=1,stroke=0)
        canvas.restoreState()

    doc = BaseDocTemplate(buf,pagesize=letter,
        leftMargin=0.7*inch,rightMargin=0.7*inch,
        topMargin=0.7*inch,bottomMargin=0.7*inch)
    frame = Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="m")
    doc.addPageTemplates([PageTemplate(id="dark",frames=[frame],onPage=bg_page)])

    S = getSampleStyleSheet()
    def ps(n,**kw): return ParagraphStyle(n,parent=S["BodyText"],**kw)
    ts  = ps("T",textColor=white,        fontSize=22,fontName="Helvetica-Bold",spaceAfter=2)
    ss  = ps("ST",textColor=C["muted"],  fontSize=9, fontName="Helvetica",     spaceAfter=8)
    h2s = ps("H2",textColor=C["accent"], fontSize=11,fontName="Helvetica-Bold",spaceBefore=12,spaceAfter=4)
    bs  = ps("B", textColor=C["text"],   fontSize=8, fontName="Helvetica",     leading=12)
    sbs = ps("SB",textColor=C["muted"],  fontSize=7.5,fontName="Helvetica",    leading=11)
    ws  = ps("W", textColor=HexColor("#FCA5A5"),fontSize=8,fontName="Helvetica-Oblique",leading=12)
    fs  = ps("F", textColor=C["muted"],  fontSize=6.5,fontName="Helvetica",    alignment=1)
    ns  = ps("N", textColor=C["note_tx"],fontSize=8,  fontName="Helvetica-Bold",leading=12.5)

    def note(txt):
        t = Table([[Paragraph(txt,ns)]],colWidths=[7.3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),C["note_bg"]),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        return t

    def dark_tbl(data, col_w, hdr_c=None):
        hdr_c = hdr_c or C["accent"]
        t = Table(data,colWidths=col_w,repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hdr_c),
            ("TEXTCOLOR", (0,0),(-1,0),white),
            ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1),7),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C["card"],C["card2"]]),
            ("TEXTCOLOR", (0,1),(-1,-1),C["text"]),
            ("GRID",      (0,0),(-1,-1),0.35,C["border"]),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),5),
            ("FONTNAME",  (0,1),(-1,-1),"Courier"),
        ]))
        return t

    now   = fmt_cdmx()  # Zona horaria America/Mexico_City (GMT-6)
    story = []

    # ── Portada ──────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("ELEMENTA", ts),
        Paragraph(f"<b>Reporte de analisis colorimetrico</b><br/>"
                  f"<font size='8'>{now}</font><br/>"
                  f"<font size='8'>Analito: {analyte}  |  Metodo: {method}</font>",
                  ps("HR",textColor=C["text"],fontSize=9,fontName="Helvetica",alignment=2)),
    ]],colWidths=[3.0*inch,4.3*inch])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C["bg"]),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",(0,0),(-1,0),2,C["accent"]),
    ]))
    story.append(hdr); story.append(Spacer(1,8))

    # ── Aviso ────────────────────────────────────────────────────────
    av = Table([[Paragraph(
        "<b>AVISO:</b> Estimaciones colorimétricas digitales. No sustituyen metodos "
        "instrumentales certificados ni declaraciones de cumplimiento normativo sin "
        "confirmacion analitica acreditada.", ws)]],colWidths=[7.3*inch])
    av.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),HexColor("#450a0a")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(av); story.append(Spacer(1,12))

    # ── Imagen anotada ───────────────────────────────────────────────
    if annotated_img is not None:
        story.append(Paragraph("Imagen procesada — regiones de interes", h2s))
        pil = Image.fromarray(annotated_img); ib=BytesIO(); pil.save(ib,"PNG"); ib.seek(0)
        story.append(RLImage(ib,width=4.8*inch,height=3.0*inch,kind="proportional"))
        story.append(Spacer(1,8))

    # ── Tabla RGB ────────────────────────────────────────────────────
    if df_rgb is not None and not df_rgb.empty:
        story.append(Paragraph("Datos colorimetricos RGB por region de interes", h2s))
        story.append(note("R, G, B: intensidad media del canal (0-255). "
                          "R_norm, G_norm, B_norm: fraccion porcentual de cada canal respecto "
                          "al total, que normaliza la senal frente a cambios de iluminacion global."))
        story.append(Spacer(1,4))
        cols = [c for c in ["ROI","R","G","B","R_norm","G_norm","B_norm"] if c in df_rgb.columns]
        td   = [cols]+[[f"{v:.2f}" if isinstance(v,float) else str(v) for v in row]
                        for _,row in df_rgb[cols].round(2).iterrows()]
        cw   = [0.9*inch]+[0.85*inch]*(len(cols)-1)
        story.append(dark_tbl(td,cw)); story.append(Spacer(1,10))

    # ── Parámetros de calibración ────────────────────────────────────
    if cal:
        story.append(Paragraph("Parametros de calibracion e interpretacion estadistica", h2s))
        lod_v = cal.get("LOD",float("nan")); loq_v = cal.get("LOQ",float("nan"))
        r2_i, _ = interpret_r2(cal["r2"])

        cal_items = [
            ("R²",  f"{cal['r2']:.5f}",  f"Interpretacion: {r2_i}. {STAT_EXPL['R2']}"),
            ("m (Pendiente)",   f"{cal['m']:.4f}", STAT_EXPL["slope"]),
            ("b (Intercepto)",  f"{cal['b']:.4f}", STAT_EXPL["intercept"]),
            ("Sb (Error est.)", f"{cal['se']:.5f}", STAT_EXPL["se"]),
            ("n Estandares",    str(cal.get("n","N/D")), "Numero de puntos de calibracion en el ajuste."),
            ("LOD",  f"{lod_v:.3f}" if not math.isnan(lod_v) else "N/D", STAT_EXPL["LOD"]),
            ("LOQ",  f"{loq_v:.3f}" if not math.isnan(loq_v) else "N/D", STAT_EXPL["LOQ"]),
        ]
        for param, val, expl in cal_items:
            blk = [
                [Paragraph(param, ps("CP",textColor=C["muted"],fontSize=8,fontName="Helvetica-Bold")),
                 Paragraph(val,   ps("CV",textColor=HexColor("#4ADE80"),fontSize=10,fontName="Courier-Bold"))],
                [Paragraph(expl,  ns), ""],
            ]
            bt = Table(blk,colWidths=[4.0*inch,3.3*inch])
            bt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),C["card"]),
                ("BACKGROUND",(0,1),(-1,1),C["note_bg"]),
                ("TEXTCOLOR", (0,1),(0,1), C["note_tx"]),
                ("GRID",(0,0),(-1,-1),0.3,C["border"]),
                ("SPAN",(0,1),(-1,1)),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
            ]))
            story.append(KeepTogether([bt, Spacer(1,4)]))

        if cal.get("lod_proxy"):
            story.append(note("LOD/LOQ calculados con error estandar residual como proxy de sigma_blanco. "
                              "Para mayor rigor, incluya >=10 replicas del blanco."))
        story.append(Spacer(1,8))

    # ── Gráfica de calibración ───────────────────────────────────────
    if cal_png_bytes:
        story.append(Paragraph("Grafica de calibracion", h2s))
        story.append(RLImage(BytesIO(cal_png_bytes),width=5.8*inch,height=3.2*inch))
        story.append(note("Puntos azules: estandares experimentales. "
                          "Linea verde: regresion lineal. "
                          "Lineas punteadas: LOD (rojo) y LOQ (naranja)."))
        story.append(Spacer(1,10))

    # ── Triplicados ──────────────────────────────────────────────────
    if tri_df is not None and not tri_df.empty:
        story.append(Paragraph("Estadisticas por condicion (triplicados)", h2s))
        story.append(note(STAT_EXPL["CV"]))
        story.append(Spacer(1,4))
        cols = list(tri_df.columns)
        td2  = [cols]+[[str(v) if v is not None else "N/D" for v in row] for _,row in tri_df.iterrows()]
        cw2  = [max(0.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dark_tbl(td2,cw2,C["green"])); story.append(Spacer(1,10))

    # ── Resultados ───────────────────────────────────────────────────
    if df_results is not None and not df_results.empty:
        story.append(Paragraph("Resultados de cuantificacion", h2s))
        story.append(note("Conc_calc: calculada de la curva (x=(A-b)/m). "
                          "Conc_corregida: multiplicada por el factor de dilucion."))
        story.append(Spacer(1,4))
        cols = list(df_results.columns)
        td3  = [cols]+[[f"{v:.3f}" if isinstance(v,float) else str(v) for v in row]
                        for _,row in df_results.iterrows()]
        cw3  = [max(0.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dark_tbl(td3,cw3)); story.append(Spacer(1,10))

    # ── Nota científica + pie ────────────────────────────────────────
    story.append(HRFlowable(width="100%",thickness=0.5,color=C["border"]))
    story.append(Spacer(1,5))
    story.append(note(
        "<b>Nota cientifica:</b> La precision de las estimaciones colorimetricas digitales depende "
        "de la uniformidad de iluminacion, caracteristicas del sensor de imagen, calidad de "
        "reactivos, linealidad del metodo y reproducibilidad en la preparacion de estandares. "
        "Para declaraciones de cumplimiento normativo confirmar mediante metodos acreditados. "
        "Consultar siempre la version vigente de las normas aplicables en el DOF."))
    story.append(Spacer(1,8))
    story.append(Paragraph("Derechos reservados (Katyutzka Villarreal, 2026)  |  Elementa — Sistema Analitico Colorimetrico Digital", fs))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

def init_session():
    defs = dict(
        # Imagen y ROIs
        image=None, rois=[], freeze_rois=False, device_type="Viales lineales",
        roi_mode="Manual",          # "Manual" | "Automatico"
        rois_preview=[],            # ROIs detectadas (pendientes de aceptar)
        roi_detection_ran=False,
        # Asignación
        assignment_df=None, _asgn_fp="",
        blank_label=None,
        # Procesamiento
        df_rgb=None, df_norm=None, df_abs=None, df_merged=None,
        # Calibración
        cal_result=None, best_ch="G_norm", all_ch_res={}, tri_groups={}, tri_df=None,
        df_results=None, annotated_img=None,
        cal_fig=None, res_fig=None, sa_fig=None, sa_result=None,
        # Datos crudos de calibración para PDF
        cal_concs=None, cal_sigs=None, cal_unit="mg/L", cal_analyte="", cal_ch="",
        cal_png=None,
    )
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k]=v

init_session()

# ── Helpers UI ─────────────────────────────────────────────────────────

def mc(label, value, interpret=None, explain=None, col=None):
    interp_html = f'<p class="mc-interpret" style="color:{interpret[1] if interpret else MUTED}">{interpret[0] if interpret else ""}</p>' if interpret else ""
    expl_html   = f'<div class="mc-explain">{explain}</div>' if explain else ""
    html = (f'<div class="metric-card"><p class="mc-label">{label}</p>'
            f'<p class="mc-value">{value}</p>{interp_html}{expl_html}</div>')
    (col or st).markdown(html, unsafe_allow_html=True)

def ibox(t): st.markdown(f'<div class="info-box">{t}</div>',unsafe_allow_html=True)
def wbox(t): st.markdown(f'<div class="warn-box">{t}</div>',unsafe_allow_html=True)
def okbox(t): st.markdown(f'<div class="ok-box">{t}</div>',unsafe_allow_html=True)
def slbl(t): st.markdown(f'<p class="section-label">{t}</p>',unsafe_allow_html=True)

def footer():
    st.markdown(
        f'<div class="footer">Derechos reservados (Katyutzka Villarreal, 2026) &nbsp;|&nbsp; '
        f'Elementa — Sistema Analitico Colorimetrico Digital</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{TEXT};margin:0;font-size:1.35rem;font-weight:700;"
        f"letter-spacing:-.02em;'>Elementa</h2>",unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED};font-size:.65rem;font-weight:600;letter-spacing:.1em;"
        f"text-transform:uppercase;margin:2px 0 16px 0;'>Sistema Colorimetrico Digital</p>",
        unsafe_allow_html=True)
    st.divider()
    pagina = st.radio("Seccion",["Analisis","Fundamentos","Normativa y Fuentes"],
                       label_visibility="collapsed")
    st.divider()
    st.markdown(
        f"<p style='color:{MUTED};font-size:.69rem;line-height:1.6;'>"
        f"Estimaciones colorimetricas digitales. "
        f"No sustituyen metodos certificados ni analisis en laboratorios acreditados.</p>",
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  PÁGINA 1: ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════

if pagina == "Analisis":

    st.markdown(f"<h1>Analisis Colorimetrico Digital</h1>",unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-6px;font-size:.85rem;'>"
        f"Calibracion, cuantificacion y evaluacion normativa por imagenes RGB.</p>",
        unsafe_allow_html=True)

    tab_cap, tab_proc, tab_cal, tab_rep = st.tabs(
        ["Captura", "Procesamiento", "Calibracion", "Reporte"])

    # ────────────────────────────────────────────────────────
    #  TAB 1 — CAPTURA
    # ────────────────────────────────────────────────────────
    with tab_cap:
        slbl("Paso 1 — Cargar imagen")
        c1, c2 = st.columns(2)
        with c1:
            uf = st.file_uploader("Subir imagen (JPG/PNG)", type=["jpg","jpeg","png"],
                                   label_visibility="collapsed")
            if uf:
                loaded = load_image(uf)
                if loaded is not st.session_state.get("image"):
                    st.session_state["image"] = loaded
                    st.session_state["rois"]  = []     # Nueva imagen = limpiar ROIs
                    st.session_state["_asgn_fp"] = ""
        with c2:
            cam = st.camera_input("Capturar con camara", label_visibility="collapsed")
            if cam:
                loaded = load_image(cam)
                st.session_state["image"] = loaded
                st.session_state["rois"]  = []
                st.session_state["_asgn_fp"] = ""

        if st.session_state["image"] is None:
            ibox("Cargue o capture una imagen para comenzar.")
            footer(); st.stop()

        img = st.session_state["image"]
        H, W = img.shape[:2]

        st.markdown("<hr style='border-color:#1E293B;margin:16px 0;'>",
                    unsafe_allow_html=True)
        slbl("Paso 2 — Definir regiones de interes (ROIs)")

        # ── Modo: Manual vs Automático ──────────────────────
        roi_mode = st.radio(
            "Modo de deteccion",
            ["Manual", "Deteccion automatica"],
            horizontal=True,
            key="roi_mode_radio",
            help="Manual: sliders de posicion. Automatico: OpenCV detecta pocillos/viales.")
        st.session_state["roi_mode"] = roi_mode

        ctrl_col, img_col = st.columns([1, 1], gap="large")

        # ══════════════════════════════════════════════════
        #  MODO AUTOMÁTICO
        # ══════════════════════════════════════════════════
        if roi_mode == "Deteccion automatica":
            with ctrl_col:
                dev_auto = st.selectbox(
                    "Tipo de dispositivo",
                    ["Microplaca de 96 pocillos", "Viales lineales"],
                    key="dev_auto_sel")
                st.session_state["device_type"] = dev_auto

                # ── Filtrado inteligente ─────────────────────────
                only_filled = st.toggle(
                    "Detectar unicamente pocillos llenos",
                    value=True, key="only_filled",
                    help="Descarta pocillos vacios, reflejos y ruido usando "
                         "analisis de intensidad, desviacion estandar y saturacion HSV.")

                preset_name = st.radio(
                    "Sensibilidad de deteccion",
                    list(DETECTION_PRESETS.keys()),
                    index=1, horizontal=True, key="det_preset",
                    help="Media: recomendado para la mayoria de casos. "
                         "Alta: captura mas pocillos (puede incluir falsos positivos). "
                         "Baja: solo pocillos con señal clara.")
                preset = DETECTION_PRESETS[preset_name]

                # Valores por defecto (antes del expander)
                min_r=8; max_r=40; min_dist=20
                min_area_auto=400; max_area_pct=20; vial_sens=50

                with st.expander("Parametros geometricos avanzados", expanded=False):
                    if dev_auto == "Microplaca de 96 pocillos":
                        min_r   = st.slider("Radio minimo de pocillo (px)", 3, 60, 8,  key="auto_minr")
                        max_r   = st.slider("Radio maximo de pocillo (px)", 8, 120, 40, key="auto_maxr")
                        min_dist= st.slider("Separacion minima entre centros (px)", 5, 100, 20, key="auto_mdist")
                    else:
                        min_area_auto = st.slider("Area minima de contorno (px²)", 100, 5000, 400, key="auto_minarea")
                        max_area_pct  = st.slider("Area maxima (% imagen)", 1, 40, 20, key="auto_maxarea")
                        vial_sens     = st.slider("Sensibilidad de umbral morfologico", 10, 90, 50, key="auto_vsens")

                # Mostrar criterios activos
                if only_filled:
                    st.markdown(
                        f"<div class='info-box' style='font-size:.75rem;'>"
                        f"Filtros activos — "
                        f"Intensidad: {preset['min_i']}–{preset['max_i']} &nbsp;|&nbsp; "
                        f"Sat. HSV min: {preset['min_sat']} &nbsp;|&nbsp; "
                        f"Std min: {preset['min_std']}"
                        f"</div>", unsafe_allow_html=True)

                if st.button("Detectar ROIs", key="btn_detect"):
                    with st.spinner("Procesando con OpenCV..."):
                        if dev_auto == "Microplaca de 96 pocillos":
                            rois_det = detect_microplate_rois(
                                img, min_r=min_r, max_r=max_r,
                                sensitivity=preset["hough_p2"],
                                min_dist=min_dist,
                                detect_only_filled=only_filled,
                                min_intensity=preset["min_i"],
                                max_intensity=preset["max_i"],
                                min_saturation=preset["min_sat"],
                                min_std=preset["min_std"])
                        else:
                            rois_det = detect_vial_rois(
                                img,
                                min_area=min_area_auto,
                                max_area_ratio=max_area_pct/100,
                                sensitivity=vial_sens,
                                detect_only_filled=only_filled,
                                min_intensity=preset["min_i"],
                                max_intensity=preset["max_i"],
                                min_saturation=preset["min_sat"],
                                min_std=preset["min_std"])

                        rois_det = validate_rois(rois_det, img.shape)
                        st.session_state["rois_preview"] = rois_det
                        st.session_state["roi_detection_ran"] = True

                # Resultado de detección
                rois_prev = st.session_state.get("rois_preview", [])
                if st.session_state.get("roi_detection_ran"):
                    if rois_prev:
                        okbox(f"Se detectaron <b>{len(rois_prev)}</b> ROIs. "
                              f"Revise la imagen y haga clic en <b>Aceptar</b>.")
                        st.dataframe(
                            pd.DataFrame([{"ROI": r["label"],
                                           "X": r["x"], "Y": r["y"],
                                           "W": r["w"], "H": r["h"]}
                                          for r in rois_prev]),
                            use_container_width=True, hide_index=True, height=180)

                        if st.button("Aceptar ROIs detectadas", key="btn_accept"):
                            st.session_state["rois"]            = rois_prev
                            st.session_state["freeze_rois"]     = True
                            st.session_state["_asgn_fp"]        = ""
                            st.session_state["rois_preview"]    = []
                            st.session_state["roi_detection_ran"] = False
                            okbox(f"{len(rois_prev)} ROIs aceptadas y bloqueadas.")
                    else:
                        wbox("No se detectaron ROIs. Ajuste los parametros "
                             "(reduzca la Sensibilidad o cambie el radio) y vuelva a intentar.")

                # Opcion para desbloquear
                if st.session_state.get("freeze_rois") and st.session_state.get("rois"):
                    if st.button("Desbloquear y re-detectar", key="btn_unlock"):
                        st.session_state["freeze_rois"] = False
                        st.session_state["rois_preview"] = []
                        st.session_state["roi_detection_ran"] = False

            with img_col:
                rois_preview = st.session_state.get("rois_preview", [])
                rois_accepted= st.session_state.get("rois", [])
                if rois_preview:
                    prev_img = draw_detected_preview(img, rois_preview)
                    st.image(prev_img,
                             caption=f"Preview: {len(rois_preview)} ROIs detectadas (pendientes de aceptar)",
                             use_container_width=True)
                elif rois_accepted and st.session_state.get("freeze_rois"):
                    adf2 = st.session_state.get("assignment_df")
                    tm2  = dict(zip(adf2["ROI"], adf2["Tipo"])) if adf2 is not None else {}
                    ann2 = draw_rois(img, rois_accepted, tm2)
                    st.session_state["annotated_img"] = ann2
                    st.image(ann2,
                             caption=f"ROIs aceptadas ({len(rois_accepted)} regiones)",
                             use_container_width=True)
                else:
                    st.image(img, caption="Imagen original — ejecute la deteccion",
                             use_container_width=True)

        # ══════════════════════════════════════════════════
        #  MODO MANUAL  (st.form para estabilidad total)
        # ══════════════════════════════════════════════════
        else:
            with ctrl_col:
                freeze = st.toggle(
                    "Bloquear ROIs",
                    value=st.session_state["freeze_rois"],
                    key="freeze_tog")
                st.session_state["freeze_rois"] = freeze

                if freeze:
                    rois = st.session_state.get("rois", [])
                    if rois:
                        okbox(f"ROIs bloqueadas — {len(rois)} regiones. "
                              f"Desactive el bloqueo para ajustar.")
                    else:
                        wbox("No hay ROIs bloqueadas. Desactive el bloqueo y configure.")
                else:
                    # ── st.form: los sliders NO generan rerenders ──
                    # El script solo recalcula ROIs al hacer clic en "Aplicar"
                    ibox("Configure los parametros y haga clic en <b>Aplicar ROIs</b>. "
                         "La imagen se actualizara solo al aplicar.")

                    dev = st.selectbox(
                        "Tipo de dispositivo",
                        ["Viales lineales", "Microplaca de 96 pocillos", "Personalizado"],
                        key="device_sel")
                    st.session_state["device_type"] = dev

                    with st.form("roi_manual_form"):
                        if dev == "Viales lineales":
                            n  = st.number_input("N de viales", 2, 24, 6, 1)
                            x0 = st.slider("X inicial (px)", 0, W-1, int(W*.05))
                            y0 = st.slider("Y inicial (px)", 0, H-1, int(H*.25))
                            rw = st.slider("Ancho ROI (px)", 5, 200, 40)
                            rh = st.slider("Alto ROI (px)", 5, 300, 60)
                            dx = st.slider("Espaciado X (px)", 0, 300, int(W*.08))
                            dy = st.slider("Espaciado Y (px)", 0, 200, 0)
                        elif dev == "Microplaca de 96 pocillos":
                            x0  = st.slider("X inicial (px)", 0, W-1, int(W*.05))
                            y0  = st.slider("Y inicial (px)", 0, H-1, int(H*.05))
                            rw  = st.slider("Ancho ROI (px)", 4, 80, 20)
                            rh  = st.slider("Alto ROI (px)", 4, 80, 20)
                            dx  = st.slider("Espaciado X (px)", 10, 200, 50)
                            dy  = st.slider("Espaciado Y (px)", 10, 200, 50)
                            rws = st.number_input("Filas", 1, 8, 8, 1)
                            cls = st.number_input("Columnas", 1, 12, 12, 1)
                        else:
                            n  = st.number_input("N de ROIs", 2, 50, 6, 1)
                            x0 = st.slider("X inicial (px)", 0, W-1, int(W*.05))
                            y0 = st.slider("Y inicial (px)", 0, H-1, int(H*.1))
                            rw = st.slider("Ancho ROI (px)", 5, 200, 30)
                            rh = st.slider("Alto ROI (px)", 5, 200, 30)
                            dx = st.slider("Espaciado X (px)", 0, 300, int(W*.08))
                            dy = st.slider("Espaciado Y (px)", 0, 300, int(H*.08))

                        submitted = st.form_submit_button(
                            "Aplicar ROIs",
                            use_container_width=True)

                    # Solo recalcula ROIs cuando se envia el formulario
                    if submitted:
                        if dev == "Viales lineales" or dev == "Personalizado":
                            new_rois = generate_rois_linear(x0, y0, rw, rh, int(n), dx, dy)
                        else:
                            new_rois = generate_rois_microplate(
                                x0, y0, rw, rh, dx, dy, int(rws), int(cls))
                        st.session_state["rois"] = new_rois
                        # Resetear fingerprint solo si estructura cambio
                        if roi_fingerprint(new_rois) != st.session_state.get("_asgn_fp",""):
                            st.session_state["_asgn_fp"] = ""

            with img_col:
                rois = st.session_state.get("rois", [])
                if rois:
                    ensure_assignment_df(rois)
                    adf  = st.session_state["assignment_df"]
                    tm   = dict(zip(adf["ROI"], adf["Tipo"])) if adf is not None else {}
                    ann  = draw_rois(img, rois, tm)
                    st.session_state["annotated_img"] = ann
                    cap  = (f"ROIs actuales: {len(rois)} regiones" +
                            (" (bloqueadas)" if freeze else " — haga clic en 'Aplicar ROIs' para actualizar"))
                    st.image(ann, caption=cap, use_container_width=True)
                else:
                    st.image(img,
                             caption="Configure los parametros y haga clic en 'Aplicar ROIs'",
                             use_container_width=True)

        footer()

    # ────────────────────────────────────────────────────────
    #  TAB 2 — PROCESAMIENTO
    # ────────────────────────────────────────────────────────
    with tab_proc:
        rois = st.session_state.get("rois",[])
        img  = st.session_state.get("image")

        if not rois or img is None:
            wbox("Defina las ROIs en la pestana <b>Captura</b> primero.")
            footer(); st.stop()

        ensure_assignment_df(rois)
        slbl("Paso 3 — Asignar tipos y concentraciones")
        ibox("Asigne <b>BL</b>=Blanco, <b>STD</b>=Estandar, <b>SMP</b>=Muestra. "
             "Para triplicados use la misma columna (A1/B1/C1).")

        fp = roi_fingerprint(rois)
        dev = st.session_state.get("device_type", "")
        is_plate = (dev == "Microplaca de 96 pocillos")

        # ── Layout: tabla izquierda | grid/imagen derecha ─────────
        tbl_col, vis_col = st.columns([6, 5] if is_plate else [1, 1], gap="large")

        with tbl_col:
            edited = st.data_editor(
                st.session_state["assignment_df"],
                column_config={
                    "Tipo":    st.column_config.SelectboxColumn("Tipo",   options=TIPOS,   required=True),
                    "Unidad":  st.column_config.SelectboxColumn("Unidad", options=UNIDADES,required=True),
                    "Analito": st.column_config.SelectboxColumn("Analito",options=ANALITOS,required=True),
                    "Concentracion": st.column_config.NumberColumn("Conc.",min_value=0.0,step=0.001,format="%.4f"),
                    "Factor_dil":    st.column_config.NumberColumn("F.Dil",min_value=0.01,step=0.1,format="%.2f"),
                },
                num_rows="fixed", use_container_width=True,
                key=f"asgn_ed_{fp}",
            )
            # Guardar inmediatamente — no recrear si fingerprint no cambió
            st.session_state["assignment_df"] = edited

            blank_r = edited[edited["Tipo"] == "Blanco"]
            blank   = blank_r["ROI"].iloc[0] if not blank_r.empty else None
            st.session_state["blank_label"] = blank
            if blank: okbox(f"Blanco: <b>{blank}</b>")
            else:     wbox("Sin blanco asignado.")

        with vis_col:
            if is_plate:
                # ── Grid reactivo: se recalcula con 'edited' en cada rerun ──
                slbl("Grid de placa — actualización en tiempo real")
                legend = " ".join(
                    f'<span style="background:{c};color:{TEXT};padding:2px 7px;'
                    f'border-radius:3px;font-size:.7rem;font-weight:700;margin-right:3px;">'
                    f'{TIPO_SHORT[t]}</span>'
                    for t, c in TIPO_COLORS.items() if t != "Sin asignar")
                st.markdown(legend, unsafe_allow_html=True)

                # Usar 'edited' directamente — garantiza datos del rerun actual
                tri_groups = detect_triplicates(edited)
                st.session_state["tri_groups"] = tri_groups
                st.plotly_chart(
                    plot_microplate(edited, tri_groups),
                    use_container_width=True,
                    key="plate_grid_live")   # key fija evita parpadeo

                if tri_groups:
                    n_g = len(tri_groups)
                    n_w = sum(len(v) for v in tri_groups.values())
                    ibox(f"<b>{n_g} grupos de triplicados</b> ({n_w} pocillos).")
                    with st.expander("Detalle de triplicados"):
                        tri_rows = []
                        for col_k, rlist in sorted(tri_groups.items(),
                                                   key=lambda x: int(x[0])):
                            sub  = edited[edited["ROI"].isin(rlist)]
                            tipo = sub["Tipo"].iloc[0]   if not sub.empty else ""
                            conc = sub["Concentracion"].iloc[0] if not sub.empty else 0
                            tri_rows.append({
                                "Columna": col_k,
                                "Pocillos": ", ".join(rlist),
                                "N": len(rlist), "Tipo": tipo,
                                "Conc": round(float(conc), 4)})
                        st.dataframe(pd.DataFrame(tri_rows),
                                     use_container_width=True, hide_index=True)
            else:
                # Para viales: imagen anotada reactiva
                type_map = dict(zip(edited["ROI"], edited["Tipo"]))
                ann = draw_rois(img, rois, type_map)
                st.session_state["annotated_img"] = ann
                st.image(ann,
                         caption="Imagen con tipos asignados — se actualiza al editar",
                         use_container_width=True)

        # ── Imagen anotada (solo microplaca, debajo de la tabla) ───
        if is_plate:
            type_map = dict(zip(edited["ROI"], edited["Tipo"]))
            ann = draw_rois(img, rois, type_map)
            st.session_state["annotated_img"] = ann

        footer()

    # ────────────────────────────────────────────────────────
    #  TAB 3 — CALIBRACIÓN
    # ────────────────────────────────────────────────────────
    with tab_cal:
        rois = st.session_state.get("rois",[])
        img  = st.session_state.get("image")
        adf  = st.session_state.get("assignment_df")

        if not rois or img is None or adf is None:
            wbox("Complete las pestanas <b>Captura</b> y <b>Procesamiento</b> primero.")
            footer(); st.stop()

        blank = st.session_state.get("blank_label")

        with st.expander("Fundamento — absorbancia digital", expanded=False):
            st.markdown(
                "**A_dig = log10(I_blanco / I_muestra)**\n\n"
                "donde I es la intensidad normalizada del canal seleccionado "
                "(expresado como % del total R+G+B). La normalizacion reduce el efecto "
                "de variaciones globales de iluminacion entre capturas.\n\n"
                "El sistema evalua los tres canales y selecciona automaticamente el de "
                "mayor R², ya que distintos sistemas cromatogenicos tienen su maxima "
                "variacion espectral en distintas regiones del visible.")

        if st.button("Extraer RGB y calibrar", key="btn_cal"):
            with st.spinner("Procesando..."):
                df_rgb  = extract_rgb(img, rois)
                df_norm = normalize_rgb(df_rgb)
                df_abs  = calc_absorbance(df_norm, blank)
                df_merged = df_abs.merge(
                    adf[["ROI","Tipo","Nombre","Concentracion","Unidad","Analito","Factor_dil"]],
                    on="ROI",how="left")

                std  = df_merged[df_merged["Tipo"]=="Estandar"]
                best_ch, ch_res = select_channel(df_merged, std["ROI"].tolist())

                st.session_state.update(dict(
                    df_rgb=df_rgb, df_norm=df_norm, df_abs=df_abs, df_merged=df_merged,
                    best_ch=best_ch, all_ch_res=ch_res))

                if best_ch in ch_res and len(std)>=2:
                    cal = ch_res[best_ch]
                    bsigs = df_merged.loc[df_merged["Tipo"]=="Blanco", f"A_{best_ch}"].dropna().values
                    ld,lq,proxy = lod_loq(cal, bsigs if len(bsigs)>=2 else None)
                    cal.update({"LOD":ld,"LOQ":lq,"lod_proxy":proxy})

                    concs = std["Concentracion"].values.astype(float)
                    sigs  = std[f"A_{best_ch}"].values.astype(float)
                    unit  = std["Unidad"].iloc[0] if not std.empty else "mg/L"
                    an    = std["Analito"].iloc[0] if not std.empty else "Analito"

                    # Guardar PNG de calibracion al momento del calculo
                    png = cal_fig_to_png(cal,concs,sigs,best_ch,an,unit,ld,lq)
                    cal_f  = plot_calibration(concs,sigs,cal,best_ch,an,unit,ld,lq)
                    res_f  = plot_residuals(concs,cal,best_ch)

                    # Triplicados
                    tg = detect_triplicates(adf)
                    td = triplate_stats(df_merged, tg, f"A_{best_ch}") if tg else None

                    st.session_state.update(dict(
                        cal_result=cal, cal_fig=cal_f, res_fig=res_f,
                        cal_concs=concs, cal_sigs=sigs, cal_unit=unit,
                        cal_analyte=an,  cal_ch=best_ch, cal_png=png,
                        tri_groups=tg, tri_df=td))

                    okbox("Extraccion y calibracion completadas.")
                else:
                    wbox("No fue posible calibrar. Verifique que haya al menos 2 estandares asignados.")

        # ── Resultados de calibración ──────────────────────────
        if st.session_state.get("cal_result"):
            cal     = st.session_state["cal_result"]
            best_ch = st.session_state["best_ch"]
            ch_res  = st.session_state["all_ch_res"]

            st.markdown("<hr style='border-color:#1E293B;margin:16px 0;'>",unsafe_allow_html=True)
            slbl("Metricas de calibracion")

            r2_interp = interpret_r2(cal["r2"])
            ld,lq = cal.get("LOD",float("nan")), cal.get("LOQ",float("nan"))
            unit  = st.session_state.get("cal_unit","mg/L")

            col1,col2,col3,col4 = st.columns(4)
            mc("R²",f"{cal['r2']:.5f}", interpret=r2_interp,
               explain=STAT_EXPL["R2"], col=col1)
            mc("Pendiente m",f"{cal['m']:.4f}",
               explain=STAT_EXPL["slope"], col=col2)
            mc("LOD",f"{ld:.3f}" if not math.isnan(ld) else "N/D",
               explain=STAT_EXPL["LOD"], col=col3)
            mc("LOQ",f"{lq:.3f}" if not math.isnan(lq) else "N/D",
               explain=STAT_EXPL["LOQ"], col=col4)

            if cal.get("lod_proxy"):
                ibox("LOD/LOQ calculados con error residual como proxy. "
                     "Para mayor precision incluya triplicados del blanco.")
            if cal["m"] < 0:
                ibox("<b>Pendiente negativa.</b> Comportamiento esperado en ensayos donde el analito "
                     "reduce la intensidad del color (DPPH, ABTS). La cuantificacion aplica x=(A-b)/m sin modificacion.")

            # Tabs de graficas
            t1,t2,t3 = st.tabs(["Curva de calibracion","Residuos","Comparativa canales"])
            with t1:
                if st.session_state.get("cal_fig"):
                    st.plotly_chart(st.session_state["cal_fig"],use_container_width=True)
            with t2:
                if st.session_state.get("res_fig"):
                    st.plotly_chart(st.session_state["res_fig"],use_container_width=True)
                    ibox("Los residuos deben distribuirse aleatoriamente alrededor de cero. "
                         "Un patron sistematico indica no-linealidad o heterocedasticidad.")
            with t3:
                if ch_res:
                    st.plotly_chart(plot_channels(ch_res),use_container_width=True)
                    ch_tab = pd.DataFrame([
                        {"Canal":c,"R²":round(v["r2"],5),
                         "m":round(v["m"],4),"b":round(v["b"],4),"se":round(v["se"],5)}
                        for c,v in ch_res.items()]).sort_values("R²",ascending=False)
                    st.dataframe(ch_tab,use_container_width=True,hide_index=True)
                    okbox(f"Canal seleccionado: <b>{best_ch}</b> (mayor R²)")

            # Triplicados
            if st.session_state.get("tri_df") is not None:
                tri_df = st.session_state["tri_df"]
                if not tri_df.empty:
                    with st.expander("Estadisticas de triplicados", expanded=True):
                        st.dataframe(tri_df,use_container_width=True,hide_index=True)
                        bad = tri_df[tri_df["CV_%"]>10] if "CV_%" in tri_df.columns else pd.DataFrame()
                        if not bad.empty:
                            wbox(f"CV% > 10% en: {', '.join(bad['Columna'].tolist())}. Revisar tecnica.")

            # Tabla de absorbancias
            with st.expander("Tabla de absorbancias digitales por ROI"):
                dm = st.session_state.get("df_merged")
                if dm is not None:
                    cols_show = ["ROI","Tipo","Concentracion","R_norm","G_norm","B_norm",
                                 f"A_{best_ch}"]
                    dc = [c for c in cols_show if c in dm.columns]
                    st.dataframe(dm[dc].round(4),use_container_width=True)
                    csv = dm[dc].to_csv(index=False).encode()
                    st.download_button("Descargar CSV",csv,"elementa_datos.csv","text/csv")

        # ── Cuantificación ─────────────────────────────────────
        st.markdown("<hr style='border-color:#1E293B;margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Cuantificacion de muestras")

        m_choice = st.radio("Metodo",["Calibracion externa","Adicion de estandar"],horizontal=True)

        if m_choice == "Calibracion externa":
            if st.button("Calcular concentraciones", key="btn_quant"):
                cal     = st.session_state.get("cal_result")
                dm      = st.session_state.get("df_merged")
                best_ch = st.session_state.get("best_ch","G_norm")
                if cal is None or dm is None:
                    st.error("Ejecute primero la calibracion."); st.stop()
                m,b = cal["m"],cal["b"]
                samples = dm[dm["Tipo"]=="Muestra"].copy()
                results = []
                for _,row in samples.iterrows():
                    a   = row.get(f"A_{best_ch}",float("nan"))
                    dil = float(row.get("Factor_dil",1) or 1)
                    c_r = (a-b)/m if not math.isnan(a) and abs(m)>1e-12 else float("nan")
                    c_c = c_r*dil if not math.isnan(c_r) else float("nan")
                    results.append({
                        "Muestra":        str(row.get("Nombre","")) or row["ROI"],
                        "ROI":            row["ROI"],
                        "Canal":          best_ch,
                        "A_digital":      round(a,4) if not math.isnan(a) else None,
                        "Conc_calc":      round(c_r,3) if not math.isnan(c_r) else None,
                        "Factor_dil":     dil,
                        "Conc_corregida": round(c_c,3) if not math.isnan(c_c) else None,
                        "Unidad":         str(row.get("Unidad","mg/L")),
                        "Analito":        str(row.get("Analito","")),
                    })
                df_res = pd.DataFrame(results)
                st.session_state["df_results"] = df_res
                st.dataframe(df_res,use_container_width=True,hide_index=True)
                csv2 = df_res.to_csv(index=False).encode()
                st.download_button("Descargar resultados CSV",csv2,"elementa_resultados.csv","text/csv")

        else:
            ibox("Ingrese la senal de la muestra (C_anadida=0) y las adiciones sucesivas.")
            n_add = st.number_input("Numero de adiciones",2,8,3,key="n_add")
            sa_df = st.data_editor(
                pd.DataFrame([{"C_anadida_mg_L":0.0,"Senal_A_dig":0.0}]+
                             [{"C_anadida_mg_L":0.0,"Senal_A_dig":0.0} for _ in range(int(n_add))]),
                column_config={
                    "C_anadida_mg_L": st.column_config.NumberColumn("C añadida",step=0.001,format="%.4f"),
                    "Senal_A_dig":    st.column_config.NumberColumn("Señal",step=0.0001,format="%.5f"),
                },
                num_rows="fixed",use_container_width=True,key="sa_editor")
            if st.button("Calcular por adicion de estandar",key="btn_sa"):
                sa = standard_addition(sa_df["C_anadida_mg_L"].values, sa_df["Senal_A_dig"].values)
                if sa is None:
                    st.error("No se pudo ajustar la regresion. Revise los datos.")
                else:
                    st.session_state["sa_result"] = sa
                    an   = st.session_state.get("cal_analyte","Analito")
                    unit = st.session_state.get("cal_unit","mg/L")
                    sa_f = plot_std_addition(sa_df["C_anadida_mg_L"].values,
                                              sa_df["Senal_A_dig"].values, sa, an, unit)
                    st.session_state["sa_fig"] = sa_f
                    mc("Concentracion estimada",f"{sa['c_sample']:.3f} {unit}",
                       interpret=(f"R² = {sa['r2']:.4f}",SUCCESS))
                    st.plotly_chart(sa_f,use_container_width=True)

        footer()

    # ────────────────────────────────────────────────────────
    #  TAB 4 — REPORTE
    # ────────────────────────────────────────────────────────
    with tab_rep:
        slbl("Paso 6 — Evaluacion normativa")
        wbox("Verificar siempre los limites permisibles en la version oficial vigente "
             "de la norma aplicable (DOF). Los valores mostrados son referenciales.")

        df_res = st.session_state.get("df_results")
        sa_r   = st.session_state.get("sa_result")

        if df_res is not None and not df_res.empty:
            for _,row in df_res.iterrows():
                try:
                    cv = float(row["Conc_corregida"])
                    an = str(row["Analito"])
                    st.markdown(f"**{row['Muestra']}** — {an}: `{cv:.3f} {row['Unidad']}`")
                    for ev in normative_eval(an,cv):
                        lim_s  = f"{ev['limite']:.3g} mg/L" if ev["limite"] else "—"
                        badge  = ev["badge"]
                        status = ev["status"]
                        norma  = ev["norma"]
                        st.markdown(
                            f"&nbsp;&nbsp;<span class='badge-{badge}'>{status}</span> "
                            f"<span style='color:{MUTED};font-size:.82rem;'>"
                            f"{norma} | Limite: {lim_s}</span>",
                            unsafe_allow_html=True)
                except: pass
        elif sa_r:
            an   = st.session_state.get("cal_analyte","Analito")
            unit = st.session_state.get("cal_unit","mg/L")
            cv   = sa_r["c_sample"]
            st.markdown(f"**Adicion de estandar** — {an}: `{cv:.3f} {unit}`")
            for ev in normative_eval(an,cv):
                lim_s  = f"{ev['limite']:.3g} mg/L" if ev["limite"] else "—"
                badge  = ev["badge"]
                status = ev["status"]
                norma  = ev["norma"]
                st.markdown(
                    f"&nbsp;&nbsp;<span class='badge-{badge}'>{status}</span> "
                    f"<span style='color:{MUTED};font-size:.82rem;'>"
                    f"{norma} | Limite: {lim_s}</span>",
                    unsafe_allow_html=True)
        else:
            ibox("Complete la cuantificacion en la pestana <b>Calibracion</b>.")

        st.markdown("<hr style='border-color:#1E293B;margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Paso 7 — Exportar reporte PDF")

        adf  = st.session_state.get("assignment_df")
        an   = adf["Analito"].iloc[0] if adf is not None and not adf.empty else "N/D"
        unit = adf["Unidad"].iloc[0]  if adf is not None and not adf.empty else "mg/L"
        meth = "Adicion de estandar" if sa_r else "Calibracion externa"

        if st.button("Generar reporte PDF", key="btn_pdf"):
            cal_png = st.session_state.get("cal_png")
            # Si no se genero al calibrar, intentar ahora
            if cal_png is None:
                cal_r = st.session_state.get("cal_result")
                if cal_r:
                    cal_png = cal_fig_to_png(
                        cal_r,
                        st.session_state.get("cal_concs"),
                        st.session_state.get("cal_sigs"),
                        st.session_state.get("cal_ch",""),
                        an, unit,
                        cal_r.get("LOD",float("nan")),
                        cal_r.get("LOQ",float("nan")))
            try:
                pdf_b = generate_pdf(
                    analyte=an, method=meth,
                    df_rgb=st.session_state.get("df_norm"),
                    df_results=df_res,
                    cal=st.session_state.get("cal_result"),
                    annotated_img=st.session_state.get("annotated_img"),
                    tri_df=st.session_state.get("tri_df"),
                    cal_png_bytes=cal_png,
                    unit=unit)
                b64  = base64.b64encode(pdf_b).decode()
                fname= f"Elementa_{an}_{now_cdmx():%Y%m%d_%H%M}.pdf"
                href = (f'<a href="data:application/pdf;base64,{b64}" download="{fname}" '
                        f'style="background:{ACCENT};color:white;padding:10px 24px;'
                        f'border-radius:6px;text-decoration:none;font-weight:700;'
                        f'font-size:.85rem;display:inline-block;margin-top:8px;">'
                        f'Descargar reporte PDF</a>')
                st.markdown(href, unsafe_allow_html=True)
                if cal_png:
                    okbox("Grafica de calibracion incluida en el PDF correctamente.")
                else:
                    wbox("Calibre el metodo antes de exportar para incluir la grafica.")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")

        footer()

# ═══════════════════════════════════════════════════════════════════════
#  PÁGINA 2: FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════════

elif pagina == "Fundamentos":

    st.markdown("<h1>Fundamentos del analisis colorimetrico digital</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};'>Base cientifica y conceptual de los metodos implementados en Elementa.</p>",unsafe_allow_html=True)

    TOPICS = {
        "Colorimetria digital — principios y alcance": """
La **colorimetria digital** cuantifica el color de imagenes capturadas digitalmente para estimar la concentracion de un analito.
Un sensor CMOS descompone la luz en tres bandas espectrales amplias: **Rojo (R ~580-700 nm), Verde (G ~500-580 nm), Azul (B ~400-500 nm)**.

A diferencia de un espectrofotometro UV-Vis que resuelve longitudes de onda individuales (0.1-2 nm), el sensor
integra bandas de ~100 nm. Por ello Elementa utiliza la **absorbancia digital** como señal analitica en lugar
de la absorbancia clasica. La tecnica ha demostrado R² > 0.99 para numerosos sistemas cromatogenicos
(Folin-Ciocalteu, DPPH, ditizona) bajo condiciones de iluminacion controladas.""",

        "Ley de Beer-Lambert y absorbancia digital": """
La ley de Beer-Lambert establece que la absorbancia es proporcional a la concentracion del soluto:

> **A = epsilon · l · c**

La **absorbancia digital** se define por analogia:

> **A_dig = log10(I_blanco / I_muestra)**

donde I es la intensidad normalizada (% del total R+G+B) en el canal seleccionado.
La normalizacion porcentual mitiga los efectos de variaciones globales de iluminacion entre capturas.

Una A_dig negativa indica que la muestra es mas intensa (mas clara) que el blanco, 
comportamiento valido en ensayos antioxidantes donde el analito reduce la señal de color.""",

        "Estadistica analitica: R², LOD, LOQ, CV%": """
**R² (coeficiente de determinacion):**
- R² >= 0.999: linealidad excelente
- 0.995 <= R² < 0.999: muy buena, aceptable para metodos de campo
- R² < 0.995: revisar preparacion de estandares o rango de calibracion

**LOD** (limite de deteccion, 3.3·sigma/|m|):
Concentracion minima distinguible del ruido del blanco con 99% de confianza.
Las muestras por debajo del LOD se reportan como "< LOD", no como cero.

**LOQ** (limite de cuantificacion, 10·sigma/|m|):
Concentracion minima cuantificable con CV < 10%. Resultados entre LOD y LOQ son semicuantitativos.

**CV%** (coeficiente de variacion para triplicados):
CV < 5% excelente | 5-10% aceptable | > 10% revisar tecnica de pipeteo.""",

        "Metales pesados: ditizona y Cr(VI)": """
La **ditizona** forma complejos coloreados con metales de transicion (Pb, Cd, Zn, Hg).
En disolvente organico es verde intensa; al reaccionar con Pb(II) forma un complejo rojo-escarlata (~510 nm).

El **Cr(VI)** es carcinogeno del Grupo 1 (IARC). El metodo colorimetrico clasico usa 1,5-difenilcarbazida (DPC)
que forma un complejo rojo-violeta con Cr(VI) reducido, con maximo a ~540 nm.

La **NOM-127-SSA1-2021** establece un limite de 0.05 mg/L de Cr total en agua para uso y consumo humano.
El Cr(VI) es altamente toxico por ingestion (daño hepatico, renal, potencial cancerigeno) e inhalacion (cancer pulmonar).""",

        "Ensayos antioxidantes: DPPH, ABTS, FRAP": """
**DPPH** (2,2-difenil-1-picrilhidrazilo): radical purpura (~515 nm). Al reducirse por un antioxidante pierde color.
La señal **decrece** con mayor capacidad antioxidante. Resultado: IC50 o % inhibicion.

**ABTS**: radical cation verde-azulado (~734 nm). Similar al DPPH, admite antioxidantes polares e hidrofobos.
Resultado: equivalentes Trolox (TEAC, uM TE/g).

**FRAP**: reduce Fe(III) a Fe(II) formando el complejo azul Fe(II)-TPTZ (~595 nm).
La señal **aumenta** con mayor capacidad antioxidante. Resultado: mmol Fe(II) equivalente/g.

**Pendiente negativa en DPPH**: al graficar absorbancia digital del canal sensible vs concentracion
antioxidante, la pendiente es negativa porque mayor concentracion = menor color purpura.
Elementa maneja correctamente esta situacion sin modificar la ecuacion.""",

        "Limitaciones del smartphone como instrumento analitico": """
1. **Iluminacion no controlada**: fuente principal de error. Usar caja de luz con LED blanco neutro y difusor.
2. **Saturacion del sensor**: a intensidades altas (>230/255) la respuesta no es lineal. Ajustar exposicion manualmente.
3. **Compresion JPEG**: puede alterar valores RGB en ±5 unidades. Preferir PNG o RAW.
4. **Variabilidad entre dispositivos**: la calibracion es especifica para cada camara, configuracion y condicion.
5. **Drift temporal**: calibrar en cada sesion analitica.
6. **Rango dinamico**: el rango lineal RGB es mas estrecho que el del espectrofotometro.

**Buenas practicas**: caja de luz, exposicion en manual, balance de blancos fijo, triplicados de cada nivel,
minimo 5 niveles de calibracion, registro de parametros de captura (ISO, distancia, temperatura de color).""",
    }

    for title, content in TOPICS.items():
        with st.expander(title, expanded=False):
            st.markdown(content)

    footer()

# ═══════════════════════════════════════════════════════════════════════
#  PÁGINA 3: NORMATIVA Y FUENTES
# ═══════════════════════════════════════════════════════════════════════

elif pagina == "Normativa y Fuentes":

    st.markdown("<h1>Normativa y fuentes de referencia</h1>",unsafe_allow_html=True)

    slbl("Limites permisibles de referencia")
    wbox("Los valores mostrados son informativos. Verificar siempre en la version oficial vigente "
         "del Diario Oficial de la Federacion (dof.gob.mx). Los limites normativos pueden actualizarse.")

    norm_rows = [{"Analito":a,"Norma":n,"Limite (mg/L)":l}
                  for a,ns in NORMATIVE_LIMITS.items() for n,l in ns.items()]
    st.dataframe(pd.DataFrame(norm_rows),use_container_width=True,hide_index=True)

    st.markdown("<hr style='border-color:#1E293B;margin:20px 0;'>",unsafe_allow_html=True)
    slbl("Referencias normativas y bibliograficas")
    refs = [
        ("NOM-127-SSA1-2021","Agua para uso y consumo humano. Limites permisibles. DOF 2021.","https://www.dof.gob.mx"),
        ("NOM-001-SEMARNAT-2021","Limites en descargas de aguas residuales en cuerpos receptores. DOF 2021.","https://www.dof.gob.mx"),
        ("Cardoso Steele et al. (2019)","Digital image colorimetry on smartphone for food analysis. Trends Anal. Chem. 111.",""),
        ("Brand-Williams et al. (1995)","Free radical method to evaluate antioxidant activity. LWT 28(1).",""),
        ("IARC Monographs Vol. 49 (1990)","Chromium, Nickel and Welding. [Cr(VI) Grupo 1]. IARC, Lyon.","https://monographs.iarc.who.int"),
        ("Miller & Miller (2010)","Statistics and Chemometrics for Analytical Chemistry. 6a ed. Pearson.",""),
    ]
    for rid,rtxt,url in refs:
        if url: st.markdown(f"- **{rid}**: {rtxt} — [Ver]({url})")
        else:   st.markdown(f"- **{rid}**: {rtxt}")

    st.markdown("<hr style='border-color:#1E293B;margin:20px 0;'>",unsafe_allow_html=True)
    slbl("Actualizar limites normativos (sesion actual)")
    ibox("Los cambios aplican solo en esta sesion. Para hacerlos permanentes edite "
         "<code>NORMATIVE_LIMITS</code> en el archivo fuente.")

    ed_rows = [{"Analito":a,"Norma":n,"Limite_mg_L":l}
               for a,ns in NORMATIVE_LIMITS.items() for n,l in ns.items()]
    ed_df = st.data_editor(pd.DataFrame(ed_rows),
        column_config={"Limite_mg_L":st.column_config.NumberColumn("Limite (mg/L)",min_value=0.0,step=0.001,format="%.4f")},
        num_rows="fixed",use_container_width=True,key="norm_editor")

    if st.button("Aplicar cambios a esta sesion"):
        NORMATIVE_LIMITS.clear()
        for _,row in ed_df.iterrows():
            a,n,l = row["Analito"],row["Norma"],row["Limite_mg_L"]
            NORMATIVE_LIMITS.setdefault(a,{})[n]=l
        okbox("Limites normativos actualizados para esta sesion.")

    footer()
