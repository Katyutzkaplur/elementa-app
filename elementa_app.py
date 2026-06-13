"""
Elementa — Sistema Analítico Colorimétrico Digital
Derechos reservados (Katyutzka Villarreal, 2026)

Software científico para colorimetría digital basada en imágenes RGB.
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
import base64, datetime, math, hashlib, warnings, re
warnings.filterwarnings("ignore")

# ─── Zona horaria ─────────────────────────────────────────────────────────────
def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Mexico_City")
    except Exception:
        return datetime.timezone(datetime.timedelta(hours=-6))

def now_mx():
    return datetime.datetime.now(_tz())

def fmt_mx(dt=None):
    return (dt or now_mx()).strftime("%Y-%m-%d  %H:%M:%S  (GMT%z)")

# ══════════════════════════════════════════════════════════════════════════════
#  PALETA Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

BG       = "#020617"
PRIMARY  = "#0F172A"
CARD     = "#1E293B"
CARD2    = "#263546"
ACCENT   = "#2563EB"
SUCCESS  = "#059669"
DANGER   = "#DC2626"
TEXT     = "#E2E8F0"
MUTED    = "#94A3B8"
BORDER   = "#334155"
PLOT_BG  = "#0B1120"

TIPO_COLORS = {
    "Blanco":           "#0EA5E9",
    "Estándar":         "#059669",
    "Muestra":          "#2563EB",
    "Control":          "#7C3AED",
    "Adición estándar": "#DB2777",
    "Sin asignar":      "#1E293B",
}
TIPO_BGR = {
    "Blanco":           (8,  165, 233),
    "Estándar":         (5,  150, 105),
    "Muestra":          (37,  99, 235),
    "Control":          (124, 58, 237),
    "Adición estándar": (219, 39, 119),
    "Sin asignar":      (30,  41,  59),
}
TIPO_SHORT = {
    "Blanco":"BLANK","Estándar":"STD","Muestra":"SMP",
    "Control":"CTRL","Adición estándar":"ADD","Sin asignar":"--",
}
TIPOS    = ["Sin asignar","Blanco","Estándar","Muestra","Control","Adición estándar"]
ANALITOS = ["Cr(VI)","Pb","Cd","Cr total","DPPH","ABTS","FRAP","Nitrógeno amoniacal","Fenoles totales","Otro"]
UNIDADES = ["mg/L","µg/L","ppm","µM","mM","%","µg/mL","Otro"]

NORMATIVE_LIMITS = {
    "Pb":       {"NOM-127-SSA1-2021 agua potable":0.01,"NOM-001-SEMARNAT-2021 descarga A":0.2,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cd":       {"NOM-127-SSA1-2021 agua potable":0.003,"NOM-001-SEMARNAT-2021 descarga A":0.1,"NOM-001-SEMARNAT-2021 descarga B":0.2},
    "Cr total": {"NOM-127-SSA1-2021 agua potable":0.05,"NOM-001-SEMARNAT-2021 descarga A":0.5,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cr(VI)":   {"NOM-127-SSA1-2021 agua potable":0.05},
}

STAT_EXPL = {
    "R2":       "Coeficiente de determinación. Proporción de la varianza de la señal explicada por la concentración. R² >= 0.999 = linealidad excelente para cuantificación analítica.",
    "slope":    "Pendiente (m). Sensibilidad analítica del método: cambio de señal por unidad de concentración. La pendiente negativa es válida en ensayos donde el analito reduce el color (DPPH, ABTS).",
    "intercept": "Intercepto (b). Señal teórica a concentración cero. Idealmente cercano al valor del blanco de reactivos.",
    "se":       "Error estándar de la pendiente (Sb). Incertidumbre en la estimación de la sensibilidad. Sb/|m| menor a 1% indica excelente precisión.",
    "LOD":      "Límite de detección (3.3 sigma/|m|). Concentración mínima distinguible del ruido con 99% de confianza. Resultados < LOD se reportan como no detectados.",
    "LOQ":      "Límite de cuantificación (10 sigma/|m|). Concentración mínima cuantificable con CV < 10%. Resultados entre LOD y LOQ son semicuantitativos.",
    "CV":       "Coeficiente de Variación (SD/media × 100%). Medida de precisión relativa. CV < 5% = excelente; 5-10% = aceptable; > 10% = revisar técnica.",
}

def interpret_r2(r2):
    if r2 >= 0.999: return "Linealidad excelente", SUCCESS
    if r2 >= 0.995: return "Muy buena", SUCCESS
    if r2 >= 0.990: return "Ligera dispersión experimental", "#F59E0B"
    return "Revisar calibración", DANGER

def interpret_slope(m):
    if m > 0:
        return "Relación directa: la señal aumenta con la concentración.", ACCENT
    else:
        return ("Relación inversa: la señal disminuye con la concentración. "
                "Esto puede ser completamente normal dependiendo del canal y "
                "la transformación colorimétrica aplicada (ej. DPPH, nitratos).", "#F59E0B")

# ─── Biblioteca de Protocolos Analíticos ─────────────────────────────────────
PROTOCOL_LIBRARY = {
    "Metales pesados": {
        "Cr(VI) — Difenilcarbazida": dict(
            analito="Cr(VI)", unidad="mg/L",
            principio="El Cr(VI) reacciona con 1,5-difenilcarbazida en medio ácido (pH 1.5-2.0) formando un complejo rojo-violeta intenso.",
            lambda_ref=540, color="Rojo-violeta", canal="G_norm",
            obs="Interferencias: Fe(III), Mo, V, Cu. Preparar patrón en agua ultrapura.",
            ref="NMX-AA-044-SCFI-2014 | APHA Method 3500-Cr B"),
        "Pb — Ditizona": dict(
            analito="Pb", unidad="µg/L",
            principio="El Pb(II) forma un complejo rojo-escarlata con ditizona en solvente orgánico a pH controlado.",
            lambda_ref=510, color="Rojo-escarlata", canal="G_norm",
            obs="Controlar pH con acetato de amonio. Interferencias: Sn, Bi, Tl.",
            ref="NOM-117-SSA1-1994 | APHA Method 3500-Pb"),
        "Cd — Ditizona": dict(
            analito="Cd", unidad="µg/L",
            principio="El Cd(II) forma un complejo amarillo-anaranjado con ditizona a pH 8-9.",
            lambda_ref=518, color="Amarillo-anaranjado", canal="B_norm",
            obs="Enmascarar Cu y Zn con KCN. pH con buffer de tartrato.",
            ref="APHA Method 3500-Cd"),
        "Cu — Neocuproína": dict(
            analito="Cu", unidad="mg/L",
            principio="El Cu(I) forma un complejo naranja-amarillo con neocuproína (2,9-dimetil-1,10-fenantrolina).",
            lambda_ref=457, color="Naranja-amarillo", canal="B_norm",
            obs="Reducir Cu(II) a Cu(I) con hidroxilamina. pH 3.5-4.5.",
            ref="APHA Method 3500-Cu"),
        "Fe — Fenantrolina": dict(
            analito="Fe", unidad="mg/L",
            principio="El Fe(II) forma un complejo naranja-rojo con 1,10-fenantrolina. El Fe(III) se reduce con hidroxilamina.",
            lambda_ref=510, color="Naranja-rojo", canal="G_norm",
            obs="pH 3-9. Interferencias: Cu, Co, Ni a altas concentraciones.",
            ref="APHA Method 3500-Fe B"),
    },
    "Parámetros ambientales": {
        "Nitrógeno amoniacal — Nessler": dict(
            analito="N-NH3", unidad="mg/L",
            principio="El reactivo de Nessler (K2HgI4) reacciona con NH3 formando un complejo amarillo-pardo.",
            lambda_ref=420, color="Amarillo-pardo", canal="B_norm",
            obs="Pendiente puede ser negativa en canal B. pH 6-7. Interferencias: Ca, Mg, Fe.",
            ref="NMX-AA-026-SCFI-2001 | APHA Method 4500-NH3 B"),
        "Nitritos — Griess": dict(
            analito="NO2-N", unidad="mg/L",
            principio="Los nitritos reaccionan con sulfanilamida y NED formando un azo-colorante rosado.",
            lambda_ref=543, color="Rosa-rojo", canal="G_norm",
            obs="pH 1.5-2.5. Sin interferencias significativas a concentraciones habituales.",
            ref="NMX-AA-079-SCFI-2001 | APHA Method 4500-NO2 B"),
        "Nitratos — Reducción con Zn": dict(
            analito="NO3-N", unidad="mg/L",
            principio="Los nitratos se reducen a nitritos con Zn en polvo, luego se detectan por el método de Griess.",
            lambda_ref=543, color="Rosa-rojo", canal="G_norm",
            obs="Usar agua libre de nitratos. Temperatura controlada.",
            ref="APHA Method 4500-NO3 B"),
        "Fosfatos — Ácido ascórbico": dict(
            analito="PO4-P", unidad="mg/L",
            principio="Los fosfatos forman un complejo azul de fosfomolibdeno reducido con ácido ascórbico.",
            lambda_ref=880, color="Azul", canal="R_norm",
            obs="pH 3.8. Interferencias: sulfuros, silicatos a altas concentraciones.",
            ref="NMX-AA-029-SCFI-2001 | APHA Method 4500-P E"),
        "Sulfatos — Turbidimetría": dict(
            analito="SO4", unidad="mg/L",
            principio="Los sulfatos precipitan con BaCl2 formando BaSO4 (turbidez). Medición a 420 nm.",
            lambda_ref=420, color="Turbio/Blanco", canal="B_norm",
            obs="Concentración de SO4 entre 1-40 mg/L. Homogeneizar bien.",
            ref="APHA Method 4500-SO4 E"),
    },
    "Antioxidantes y bioactivos": {
        "Fenoles totales — Folin-Ciocalteu": dict(
            analito="Fenoles totales", unidad="mg GAE/L",
            principio="Los grupos fenólicos reducen el reactivo de Folin-Ciocalteu formando un complejo azul intenso.",
            lambda_ref=765, color="Azul", canal="R_norm",
            obs="pH alcalino con Na2CO3. Incubar 2 h a temperatura ambiente.",
            ref="Singleton & Rossi, 1965 | Folin & Ciocalteu, 1927"),
        "DPPH — Actividad antioxidante": dict(
            analito="DPPH IC50", unidad="%",
            principio="El radical DPPH (púrpura) se reduce por antioxidantes perdiendo color. Señal decreciente.",
            lambda_ref=515, color="Púrpura → decolorado", canal="G_norm",
            obs="Pendiente NEGATIVA esperada. Expresar como IC50 o % inhibición.",
            ref="Brand-Williams et al. 1995"),
        "ABTS — Actividad antioxidante": dict(
            analito="ABTS TEAC", unidad="mM TE/L",
            principio="El radical ABTS•+ (verde-azulado) se reduce por antioxidantes. Señal decreciente.",
            lambda_ref=734, color="Verde-azulado → claro", canal="R_norm",
            obs="Pendiente NEGATIVA esperada. Resultados en equivalentes Trolox (TEAC).",
            ref="Re et al. 1999"),
        "FRAP — Poder reductor férrico": dict(
            analito="FRAP", unidad="mM Fe2+/L",
            principio="El Fe3+-TPTZ se reduce a Fe2+-TPTZ (azul intenso) por antioxidantes. Señal creciente.",
            lambda_ref=593, color="Azul", canal="R_norm",
            obs="Pendiente POSITIVA. pH 3.6 con buffer acetato. 37°C.",
            ref="Benzie & Strain, 1996"),
        "Flavonoides totales — AlCl3": dict(
            analito="Flavonoides", unidad="mg QE/L",
            principio="Los flavonoides forman complejos amarillos con AlCl3 y NaNO2 en medio básico.",
            lambda_ref=510, color="Amarillo-naranja", canal="B_norm",
            obs="Expresar en equivalentes de quercetina (QE). pH alcalino con NaOH.",
            ref="Zhishen et al. 1999"),
        "MDA — TBARS": dict(
            analito="MDA", unidad="nmol/mL",
            principio="El malondialdehído (MDA) reacciona con ácido tiobarbitúrico (TBA) formando un complejo rosa-rojo.",
            lambda_ref=532, color="Rosa-rojo", canal="G_norm",
            obs="Temperatura 95°C durante 60 min. Interferencias: azúcares, otros aldehídos.",
            ref="Ohkawa et al. 1979"),
        "Clorofilas — Arnon": dict(
            analito="Clorofila total", unidad="mg/L",
            principio="Extracción con acetona 80% y medición espectrofotométrica multi-longitud de onda.",
            lambda_ref=663, color="Verde", canal="R_norm",
            obs="Usar canal R para clorofila a, canal G para clorofila b. Proteger de luz.",
            ref="Arnon, 1949 | Lichtenthaler, 1987"),
    },
}

# ─── Calidad de imagen ────────────────────────────────────────────────────────
def check_image_quality(img: np.ndarray) -> dict:
    """Evalúa enfoque, iluminación, saturación y sobreexposición."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness= float(gray.mean())
    bgr       = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv       = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean  = float(hsv[:,:,1].mean())
    overexp   = float((gray > 245).mean()) * 100
    underexp  = float((gray < 10 ).mean()) * 100

    def grade(val, ok_range, warn_range):
        lo_ok,hi_ok     = ok_range
        lo_w, hi_w      = warn_range
        if lo_ok <= val <= hi_ok: return "ok"
        if lo_w  <= val <= hi_w:  return "warn"
        return "fail"

    return {
        "focus":       {"val": round(lap_var,1),  "label": f"Enfoque (varianza Laplaciano): {lap_var:.0f}",
                        "grade": grade(lap_var, (80,1e9), (40,1e9))},
        "brightness":  {"val": round(brightness,1),"label": f"Brillo medio: {brightness:.0f}/255",
                        "grade": grade(brightness, (60,200), (30,230))},
        "saturation":  {"val": round(sat_mean,1),  "label": f"Saturación media (HSV): {sat_mean:.0f}/255",
                        "grade": grade(sat_mean, (20,1e9), (10,1e9))},
        "overexposure":{"val": round(overexp,1),   "label": f"Píxeles sobreexpuestos: {overexp:.1f}%",
                        "grade": grade(overexp, (0,5), (0,15))},
        "underexposure":{"val": round(underexp,1), "label": f"Píxeles subexpuestos: {underexp:.1f}%",
                        "grade": grade(underexp, (0,5), (0,15))},
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Elementa", page_icon=None, layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box;}}
html,body,.stApp{{background:{BG};color:{TEXT};font-family:'Inter',sans-serif;}}
[data-testid="stSidebar"]{{background:{PRIMARY};border-right:1px solid {BORDER};}}
h1{{font-size:1.55rem;font-weight:700;color:{TEXT};letter-spacing:-.02em;margin-bottom:.2rem;}}
h2{{font-size:1.1rem;font-weight:600;color:{TEXT};}}
h3{{font-size:.9rem;font-weight:600;color:{MUTED};text-transform:uppercase;letter-spacing:.07em;}}
.mc{{background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px 18px;transition:border-color .2s;}}
.mc:hover{{border-color:{ACCENT};}}
.mc .lbl{{font-size:.67rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:{MUTED};margin:0 0 5px 0;}}
.mc .val{{font-size:1.4rem;font-weight:700;color:{TEXT};margin:0;font-family:'JetBrains Mono',monospace;}}
.mc .itp{{font-size:.72rem;margin:3px 0 0 0;font-style:italic;}}
.mc .exp{{font-size:.7rem;color:{MUTED};margin:6px 0 0 0;line-height:1.45;border-top:1px solid {BORDER};padding-top:5px;}}
.info-box{{background:{PRIMARY};border-left:3px solid {ACCENT};border-radius:0 6px 6px 0;padding:10px 14px;font-size:.82rem;color:#93C5FD;margin:8px 0;line-height:1.5;}}
.warn-box{{background:{PRIMARY};border-left:3px solid {DANGER};border-radius:0 6px 6px 0;padding:10px 14px;font-size:.82rem;color:#FCA5A5;margin:8px 0;line-height:1.5;}}
.ok-box{{background:{PRIMARY};border-left:3px solid {SUCCESS};border-radius:0 6px 6px 0;padding:10px 14px;font-size:.82rem;color:#6EE7B7;margin:8px 0;line-height:1.5;}}
.slbl{{font-size:.64rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:{ACCENT};margin:0 0 4px 0;}}
.badge-pass{{background:#052e16;color:#4ADE80;padding:3px 10px;border-radius:3px;font-size:.75rem;font-weight:700;border:1px solid #166534;}}
.badge-fail{{background:#450a0a;color:#F87171;padding:3px 10px;border-radius:3px;font-size:.75rem;font-weight:700;border:1px solid #991b1b;}}
.badge-none{{background:{CARD};color:{MUTED};padding:3px 10px;border-radius:3px;font-size:.75rem;font-weight:600;border:1px solid {BORDER};}}
.footer{{text-align:center;color:{MUTED};font-size:.7rem;padding:28px 0 10px;margin-top:48px;border-top:1px solid {BORDER};}}
.stButton>button{{background:{ACCENT};color:#fff;border:none;border-radius:6px;padding:9px 22px;font-weight:600;font-size:.84rem;letter-spacing:.02em;font-family:'Inter',sans-serif;transition:background .2s;}}
.stButton>button:hover{{background:#1D4ED8;}}
.stTabs [data-baseweb="tab-list"]{{gap:2px;border-bottom:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{MUTED};font-weight:500;font-size:.85rem;padding:10px 20px;border-radius:6px 6px 0 0;}}
.stTabs [aria-selected="true"]{{background:{CARD} !important;color:{TEXT} !important;border-bottom:2px solid {ACCENT} !important;}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES NUCLEARES DE IMAGEN
# ══════════════════════════════════════════════════════════════════════════════

def load_image(up):
    if up is None: return None
    return np.array(Image.open(up).convert("RGB"))

def gen_rois_linear(x0,y0,w,h,n,dx,dy):
    return [{"x":int(x0+i*dx),"y":int(y0+i*dy),"w":int(w),"h":int(h),"label":f"ROI {i+1}"} for i in range(n)]

def gen_rois_plate(x0,y0,w,h,dx,dy,rows=8,cols=12):
    rl="ABCDEFGH"
    return [{"x":int(x0+c*dx),"y":int(y0+r*dy),"w":int(w),"h":int(h),"label":f"{rl[r]}{c+1}"}
            for r in range(rows) for c in range(cols)]

def roi_fp(rois):
    return hashlib.md5("|".join(r["label"] for r in rois).encode()).hexdigest()[:12]

def ensure_asgn(rois):
    """
    Preserva los datos de asignación si la geometría de ROIs no cambió.
    Si cambió, migra los datos antiguos por etiqueta de ROI.
    """
    fp = roi_fp(rois)
    old_df = st.session_state.get("assignment_df")
    old_fp = st.session_state.get("_asgn_fp")

    # Si ya tenemos datos y el fingerprint es el mismo, no hacer nada
    if old_df is not None and old_fp == fp:
        return

    # Respaldo de seguridad
    st.session_state["assignment_df_backup"] = old_df.copy() if old_df is not None else None
    st.session_state["roi_config_backup"] = {"rois":rois, "fp":fp}

    # Construir nuevo DataFrame manteniendo datos previos por etiqueta
    old_data = {}
    if old_df is not None:
        for _, row in old_df.iterrows():
            old_data[row["ROI"]] = row.to_dict()

    rows = []
    for roi in rois:
        if roi["label"] in old_data:
            rows.append(old_data[roi["label"]])
        else:
            rows.append({"ROI":roi["label"],"Tipo":"Sin asignar","Nombre":"",
                         "Concentracion":0.0,"Unidad":"mg/L",
                         "Factor_dil":1.0,"Analito":"Cr(VI)","Observaciones":""})
    st.session_state["assignment_df"] = pd.DataFrame(rows)
    st.session_state["_asgn_fp"] = fp

def draw_rois(img, rois, type_map=None, circular=False, diam_map=None):
    """
    Dibuja ROIs sobre la imagen.
    circular=True → contorno circular con máscara interna semitransparente.
    """
    out = img.copy()
    for roi in rois:
        tipo = (type_map or {}).get(roi["label"], "Sin asignar")
        rgb  = TIPO_BGR.get(tipo, (30, 41, 59))
        bgr  = (rgb[2], rgb[1], rgb[0])
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if circular:
            diam = (diam_map or {}).get(roi["label"], min(w, h))
            cx, cy, r = x + w//2, y + h//2, diam//2
            cv2.circle(out, (cx, cy), r, bgr, 2)
            cv2.circle(out, (cx, cy), 2, bgr, -1)
        else:
            cv2.rectangle(out, (x, y), (x+w, y+h), bgr, 2)
        short = TIPO_SHORT.get(tipo, "")
        cv2.putText(out, roi["label"], (x, max(y-3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1, cv2.LINE_AA)
        if short and short != "--" and w >= 25 and h >= 16:
            cx2, cy2 = x + w//2, y + h//2
            cv2.putText(out, short, (cx2-12, cy2+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, bgr, 1, cv2.LINE_AA)
    return out

def extract_rgb(img, rois, circular=False, diam_map=None):
    """Extrae estadísticas RGB. Si circular=True usa máscara circular."""
    H, W = img.shape[:2]; rows = []
    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if circular:
            diam = (diam_map or {}).get(roi["label"], min(w, h))
            r    = diam // 2
            cx, cy = x + w//2, y + h//2
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(W, cx+r), min(H, cy+r)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                rows.append({"ROI":roi["label"],"R":np.nan,"G":np.nan,"B":np.nan,
                             "R_sd":np.nan,"G_sd":np.nan,"B_sd":np.nan}); continue
            Yg, Xg = np.ogrid[:crop.shape[0], :crop.shape[1]]
            mask = ((Yg-(cy-y1))**2 + (Xg-(cx-x1))**2) <= r**2
            if mask.sum() == 0:
                rows.append({"ROI":roi["label"],"R":np.nan,"G":np.nan,"B":np.nan,
                             "R_sd":np.nan,"G_sd":np.nan,"B_sd":np.nan}); continue
            rv = crop[:,:,0][mask]; gv = crop[:,:,1][mask]; bv = crop[:,:,2][mask]
        else:
            crop = img[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
            if crop.size == 0:
                rows.append({"ROI":roi["label"],"R":np.nan,"G":np.nan,"B":np.nan,
                             "R_sd":np.nan,"G_sd":np.nan,"B_sd":np.nan}); continue
            rv = crop[:,:,0].ravel(); gv = crop[:,:,1].ravel(); bv = crop[:,:,2].ravel()
        rows.append({"ROI":roi["label"],
                     "R":round(rv.mean(),2),"G":round(gv.mean(),2),"B":round(bv.mean(),2),
                     "R_sd":round(rv.std(),2),"G_sd":round(gv.std(),2),"B_sd":round(bv.std(),2)})
    return pd.DataFrame(rows)

def extract_extended_channels(img, rois, circular=False, diam_map=None):
    """
    Extrae canales RGB + HSV + LAB para cada ROI.
    Permite barrido completo de canales para selección óptima.
    """
    H, W = img.shape[:2]; rows = []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)

    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if circular:
            diam = (diam_map or {}).get(roi["label"], min(w, h))
            r    = diam // 2
            cx, cy = x + w//2, y + h//2
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(W, cx+r), min(H, cy+r)
            Yg,Xg  = np.ogrid[:y2-y1, :x2-x1]
            mask   = ((Yg-(cy-y1))**2+(Xg-(cx-x1))**2) <= r**2
            sl = (slice(y1,y2), slice(x1,x2))
        else:
            sl   = (slice(max(0,y),min(H,y+h)), slice(max(0,x),min(W,x+w)))
            mask = np.ones((sl[0].stop-sl[0].start, sl[1].stop-sl[1].start), bool)

        def ch_stats(arr2d):
            v = arr2d[sl][mask].ravel() if arr2d[sl].shape[:2] == mask.shape else arr2d[sl].ravel()
            return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)

        rm,rs = ch_stats(img[:,:,0]); gm,gs = ch_stats(img[:,:,1]); bm,bs = ch_stats(img[:,:,2])
        hm,hs = ch_stats(img_hsv[:,:,0]); sm2,ss2 = ch_stats(img_hsv[:,:,1]); vm,vs = ch_stats(img_hsv[:,:,2])
        lm,ls2= ch_stats(img_lab[:,:,0]); am,as2 = ch_stats(img_lab[:,:,1]); blab,bls = ch_stats(img_lab[:,:,2])

        eps=1e-9; tot=rm+gm+bm+eps
        rows.append({"ROI":roi["label"],
            "R":round(rm,2),"G":round(gm,2),"B":round(bm,2),
            "R_sd":round(rs,2),"G_sd":round(gs,2),"B_sd":round(bs,2),
            "R_norm":round(rm/tot*100,3),"G_norm":round(gm/tot*100,3),"B_norm":round(bm/tot*100,3),
            "H":round(hm,2),"S":round(sm2,2),"V":round(vm,2),
            "L":round(lm,2),"a":round(am,2),"b_lab":round(blab,2),
        })
    return pd.DataFrame(rows)

def normalize_rgb(df):
    df=df.copy(); eps=1e-9
    if "R" in df.columns:
        tot = df["R"]+df["G"]+df["B"]+eps
        df["R_norm"]=(df["R"]/tot)*100; df["G_norm"]=(df["G"]/tot)*100; df["B_norm"]=(df["B"]/tot)*100
        df["Total"]=df["R"]+df["G"]+df["B"]
    return df

def calc_absorbance(df, blank_roi, channels=None):
    """Calcula absorbancia digital para todos los canales disponibles."""
    if channels is None:
        channels = [c for c in ALL_CHANNELS if c in df.columns]
    df=df.copy()
    for ch in channels:
        col=f"A_{ch}"
        if ch not in df.columns or blank_roi is None or blank_roi not in df["ROI"].values:
            df[col]=np.nan; continue
        bv = float(df.loc[df["ROI"]==blank_roi,ch].values[0])
        if np.isnan(bv): df[col]=np.nan; continue
        eps=1e-9
        df[col]=df[ch].apply(lambda v: math.log10((bv+eps)/(v+eps)) if pd.notna(v) else np.nan)
    return df

def fit_line(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return None
    m, b, r, _, se = stats.linregress(x, y)
    return {"m": m, "b": b, "r2": r**2, "se": se, "n": len(x),
            "res": y - (m*x + b), "x_fit": x, "y_fit": y}

ALL_CHANNELS = ["R_norm","G_norm","B_norm","H","S","V","L","a","b_lab"]
CHANNEL_LABELS = {
    "R_norm":"R normalizado","G_norm":"G normalizado","B_norm":"B normalizado",
    "H":"Hue (HSV)","S":"Saturación (HSV)","V":"Valor (HSV)",
    "L":"L* (CIELAB)","a":"a* (CIELAB)","b_lab":"b* (CIELAB)",
}

def best_channel(df_merged, channels=None):
    """Selecciona el canal con mayor R² entre todos los canales disponibles."""
    if channels is None:
        channels = [c for c in ALL_CHANNELS if f"A_{c}" in df_merged.columns]
    std = df_merged[df_merged["Tipo"]=="Estándar"].copy()
    if len(std)<2 or "Concentracion" not in std.columns:
        return "G_norm",{}
    concs=std["Concentracion"].values.astype(float)
    res={}
    for ch in channels:
        ac=f"A_{ch}"
        if ac not in std.columns: continue
        sub=std[["Concentracion",ac]].dropna()
        if len(sub)<2 or len(np.unique(sub["Concentracion"].values))<2: continue
        cal=fit_line(sub["Concentracion"].values.astype(float),sub[ac].values.astype(float))
        if cal: res[ch]=cal
    if not res: return "G_norm",{}
    bst=max(res,key=lambda k:res[k]["r2"])
    return bst,res

def calc_lod_loq(cal, blank_sigs=None):
    """Calcula LOD/LOQ usando Sy/x y abs(m)."""
    m=abs(cal["m"])
    if m<1e-12: return np.nan,np.nan,True
    # Si hay réplicas de blanco, usar su desviación estándar, si no usar error residual
    if blank_sigs is not None and len(blank_sigs)>=2:
        sigma = np.std(blank_sigs, ddof=1)
    else:
        # Usar Sy/x (error estándar residual)
        if len(cal["res"]) > 2:
            sigma = np.sqrt(np.sum(cal["res"]**2) / (len(cal["res"]) - 2))
        else:
            sigma = cal["se"]
    proxy = (blank_sigs is None or len(blank_sigs)<2)
    return 3.3*sigma/m, 10*sigma/m, proxy

def std_addition(added,sigs):
    cal=fit_line(np.asarray(added,float),np.asarray(sigs,float))
    if cal is None or abs(cal["m"])<1e-12: return None
    xi=-cal["b"]/cal["m"]
    cal["xi"]=xi; cal["c_sample"]=abs(xi)
    return cal

def norm_eval(analyte,conc):
    if analyte not in NORMATIVE_LIMITS:
        return [{"norma":"Sin criterio","limite":None,"status":"Sin criterio","badge":"none"}]
    return [{"norma":n,"limite":lim,"status":"Cumple" if conc<=lim else "No cumple",
             "badge":"pass" if conc<=lim else "fail"}
            for n,lim in NORMATIVE_LIMITS[analyte].items()]

def detect_triplicates(asgn_df):
    """Detecta grupos de triplicados por columna de placa."""
    groups={}
    for _,row in asgn_df.iterrows():
        roi=row["ROI"]; col="".join(c for c in roi if c.isdigit())
        tipo=row.get("Tipo","Sin asignar")
        if col and tipo!="Sin asignar":
            groups.setdefault(col,[]).append(roi)
    return {k:v for k,v in groups.items() if len(v)>=2}

def triplate_stats(df_merged, groups, sig_col):
    """Calcula media, SD y CV% por grupo de triplicado."""
    rows=[]
    for col_k, rlist in sorted(groups.items(), key=lambda x: int(x[0])):
        sub=df_merged[df_merged["ROI"].isin(rlist)]
        if sub.empty or sig_col not in sub.columns: continue
        sigs=sub[sig_col].dropna().values
        if len(sigs)==0: continue
        mean=float(np.mean(sigs))
        sd  =float(np.std(sigs,ddof=1)) if len(sigs)>1 else float("nan")
        cv  =abs(sd/mean)*100 if not math.isnan(sd) and mean!=0 else float("nan")
        tipo= sub["Tipo"].iloc[0]         if "Tipo"          in sub.columns else ""
        conc= sub["Concentracion"].iloc[0] if "Concentracion" in sub.columns else float("nan")
        nombre=sub["Nombre"].iloc[0]       if "Nombre"        in sub.columns else ""
        rows.append({
            "Grupo":    f"Col. {col_k}",
            "Pocillos": ", ".join(rlist),
            "N":        len(sigs),
            "Tipo":     tipo,
            "Conc":     round(float(conc),3) if not (conc is None or (isinstance(conc,float) and math.isnan(conc))) else None,
            "Media":    round(mean,4),
            "SD":       round(sd,4)  if not math.isnan(sd) else None,
            "CV_%":     round(cv,2)  if not math.isnan(cv) else None,
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

_PLT = dict(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(family="Inter,sans-serif",color=TEXT,size=11),
            margin=dict(l=52,r=20,t=48,b=48))

def plot_plate(asgn_df, tri_groups):
    rl=list("ABCDEFGH")
    tipo_map={r["ROI"]:r.get("Tipo","Sin asignar") for _,r in asgn_df.iterrows()}
    conc_map={r["ROI"]:r.get("Concentracion",np.nan) for _,r in asgn_df.iterrows()}
    nombre_map={r["ROI"]:r.get("Nombre","") for _,r in asgn_df.iterrows()}
    rep_map={}
    for col,rlist in tri_groups.items():
        for i,roi in enumerate(rlist): rep_map[roi]=f"Rep {i+1}/{len(rlist)}"

    all_rows=sorted(set(roi[0] for roi in asgn_df["ROI"] if roi[0] in rl), key=lambda x:rl.index(x))
    all_cols=sorted(set(int("".join(c for c in roi if c.isdigit())) for roi in asgn_df["ROI"] if any(c.isdigit() for c in roi)))
    if not all_rows or not all_cols:
        return go.Figure()

    idx_map={"Blanco":1,"Estándar":2,"Muestra":3,"Control":4,"Adición estándar":5}
    cs=[[0/5,"#0B1120"],[0.18/5,"#0B1120"],
        [1/5,"#0C2540"],[1.18/5,"#0C2540"],
        [2/5,"#052e16"],[2.18/5,"#052e16"],
        [3/5,"#0D2159"],[3.18/5,"#0D2159"],
        [4/5,"#2D0A3E"],[4.18/5,"#2D0A3E"],
        [5/5,"#1e293b"]]

    z,txt,hov=[],[],[]
    for rw in all_rows:
        zr,tr,hr=[],[],[]
        for cl in all_cols:
            roi=f"{rw}{cl}"; tipo=tipo_map.get(roi,"Sin asignar")
            conc=conc_map.get(roi,np.nan); nm=nombre_map.get(roi,""); rep=rep_map.get(roi,"")
            short=TIPO_SHORT.get(tipo,"--")
            zr.append(idx_map.get(tipo,0)); tr.append(f"{short}\n{rw}{cl}")
            ht=f"<b>{roi}</b><br>{tipo}"
            if not (isinstance(conc,float) and math.isnan(conc)): ht+=f"<br>Conc: {conc:.3g}"
            if nm: ht+=f"<br>{nm}"
            if rep: ht+=f"<br>{rep}"
            hr.append(ht)
        z.append(zr); txt.append(tr); hov.append(hr)

    fig=go.Figure(go.Heatmap(z=z,text=txt,texttemplate="%{text}",
        customdata=hov,hovertemplate="%{customdata}<extra></extra>",
        colorscale=cs,showscale=False,xgap=2,ygap=2,zmin=0,zmax=5,
        textfont=dict(family="JetBrains Mono",size=8,color=TEXT)))
    fig.update_xaxes(tickvals=list(range(len(all_cols))),ticktext=[str(c) for c in all_cols],
                     side="top",showgrid=False,tickfont=dict(size=9))
    fig.update_yaxes(tickvals=list(range(len(all_rows))),ticktext=all_rows,
                     autorange="reversed",showgrid=False,tickfont=dict(size=9))
    fig.update_layout(**_PLT,height=max(220,50*len(all_rows)+80),
                      title=dict(text="Mapa de placa",font=dict(size=12)))
    return fig

def plot_cal(concs,sigs,cal,ch,analyte,unit,lod,loq):
    x0=max(0,float(concs.min())*0.85) if float(concs.min())>0 else 0.0
    x1=float(concs.max())*1.15; xl=np.linspace(x0,x1,300)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=concs,y=sigs,mode="markers",
        marker=dict(color=ACCENT,size=10,line=dict(color=PLOT_BG,width=1.5)),name="Estándares"))
    fig.add_trace(go.Scatter(x=xl,y=cal["m"]*xl+cal["b"],mode="lines",
        line=dict(color=SUCCESS,width=2.2),name="Regresión lineal"))
    if not np.isnan(lod): fig.add_vline(x=lod,line_dash="dot",line_color=DANGER,
        annotation_text=f"LOD={lod:.3f}",annotation_font_color=DANGER,annotation_font_size=9)
    if not np.isnan(loq): fig.add_vline(x=loq,line_dash="dot",line_color="#F59E0B",
        annotation_text=f"LOQ={loq:.3f}",annotation_font_color="#F59E0B",annotation_font_size=9)
    m,b,r2=cal["m"],cal["b"],cal["r2"]; sgn="+" if b>=0 else "-"
    eq=f"A = {m:.4f}·C {sgn} {abs(b):.4f}   |   R² = {r2:.5f}"
    fig.add_annotation(x=0.03,y=0.97,xref="paper",yref="paper",text=eq,showarrow=False,
        font=dict(color="#4ADE80",size=10,family="JetBrains Mono"),
        bgcolor="rgba(11,17,32,.85)",bordercolor=SUCCESS,borderwidth=1,borderpad=5)
    fig.update_layout(**_PLT,title=f"Curva de calibración — {analyte} | Canal {ch}",
        xaxis_title=f"Concentración ({unit})",yaxis_title="Absorbancia digital")
    return fig

def plot_residuals(concs,cal,ch):
    yfit=cal["m"]*concs+cal["b"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=yfit,y=cal["res"],mode="markers",
        marker=dict(color=ACCENT,size=8,line=dict(color=PLOT_BG,width=1)),name="Residuos"))
    fig.add_hline(y=0,line_dash="dash",line_color=MUTED,line_width=1)
    fig.update_layout(**_PLT,height=260,title=f"Residuos — Canal {ch}",
        xaxis_title="Señal predicha",yaxis_title="Residuo (obs - pred)")
    return fig

def plot_channels(ch_res):
    """Panel de barrido de canales con R² comparativo."""
    if not ch_res: return go.Figure()
    chs=[c for c in ALL_CHANNELS if c in ch_res]
    if not chs: chs=list(ch_res.keys())
    r2s=[ch_res[c]["r2"]  for c in chs]
    mx =max(r2s)
    clrs=[SUCCESS if v==mx else CARD2 for v in r2s]
    labels=[CHANNEL_LABELS.get(c,c) for c in chs]

    fig=go.Figure()
    fig.add_trace(go.Bar(x=labels,y=r2s,marker_color=clrs,
        text=[f"{v:.5f}" for v in r2s],textposition="outside",
        textfont=dict(color=TEXT,size=9,family="JetBrains Mono"),
        name="R²",hovertemplate="<b>%{x}</b><br>R²=%{y:.5f}<extra></extra>"))
    fig.update_layout(**_PLT,height=300,
        title="Panel de barrido de canales — R² (mayor = mejor canal)",
        yaxis=dict(range=[max(0,min(r2s)-.05),1.02],title="R²"),
        xaxis_title="Canal",xaxis_tickangle=-30)
    return fig

def plot_channel_table(ch_res) -> pd.DataFrame:
    """Tabla comparativa detallada de todos los canales evaluados."""
    rows=[]
    for ch,cal in sorted(ch_res.items(),key=lambda x:-x[1]["r2"]):
        r2i,_=interpret_r2(cal["r2"])
        si,_=interpret_slope(cal["m"])
        rows.append({
            "Canal":       CHANNEL_LABELS.get(ch,ch),
            "R²":          round(cal["r2"],5),
            "Interpretación R²": r2i,
            "Pendiente m": round(cal["m"],4),
            "Tipo pendiente": "Directa" if cal["m"]>0 else "Inversa",
            "Intercepto b":round(cal["b"],4),
            "Error est.":  round(cal["se"],5),
            "N":           cal.get("n",""),
        })
    return pd.DataFrame(rows)

def plot_sa(added,sigs,sa,analyte,unit):
    xi=sa["xi"]; m,b=sa["m"],sa["b"]
    xmin=min(xi*1.3 if xi<0 else -0.1,min(added)-0.1); xmax=max(added)*1.1
    xl=np.linspace(xmin,xmax,300)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=added,y=sigs,mode="markers",
        marker=dict(color=ACCENT,size=10,line=dict(color=PLOT_BG,width=1.5)),name="Adiciones"))
    fig.add_trace(go.Scatter(x=xl,y=m*xl+b,mode="lines",
        line=dict(color=SUCCESS,width=2),name="Proyección"))
    fig.add_trace(go.Scatter(x=[xi],y=[0],mode="markers+text",
        marker=dict(color=DANGER,size=14,symbol="x-thin",line=dict(color=DANGER,width=3)),
        text=[f" C = {sa['c_sample']:.3f} {unit}"],textposition="middle right",
        textfont=dict(color=DANGER,size=10,family="JetBrains Mono"),name="C muestra"))
    fig.add_hline(y=0,line_dash="dash",line_color=BORDER)
    fig.update_layout(**_PLT,title=f"Adición de estándar — {analyte}",
        xaxis_title=f"Concentración añadida ({unit})",yaxis_title="Señal")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICA MATPLOTLIB PARA PDF (sin kaleido)
# ══════════════════════════════════════════════════════════════════════════════

def cal_to_png(cal,concs,sigs,ch,analyte,unit,lod,loq):
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        concs=np.asarray(concs,float); sigs=np.asarray(sigs,float)
        mask=~(np.isnan(concs)|np.isnan(sigs)); concs,sigs=concs[mask],sigs[mask]
        if len(concs)<2: return None
        BG2="#0f172a"; CARD2C="#1e293b"; GRN="#4ade80"; BLU="#60a5fa"
        RED2="#f87171"; ORG="#fb923c"; SUB="#94a3b8"; TXT2="#e2e8f0"
        fig,ax=plt.subplots(figsize=(7.8,3.8))
        fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2)
        ax.scatter(concs,sigs,color=GRN,s=60,zorder=5,edgecolors=BG2,linewidths=1.2,label="Estándares")
        xmn=float(concs.min())*0.85 if float(concs.min())>0 else 0.0
        xmx=float(concs.max())*1.15
        if xmn==xmx: xmn-=0.1; xmx+=0.1
        xl=np.linspace(xmn,xmx,300)
        ax.plot(xl,cal["m"]*xl+cal["b"],color=BLU,linewidth=2.2,label="Regresión lineal")
        y_all=np.concatenate([sigs,cal["m"]*xl+cal["b"]])
        yb,yt=float(y_all.min()),float(y_all.max()); yp=(yt-yb)*0.04 if yt>yb else 0.01
        lod_ok=lod is not None and not (isinstance(lod,float) and math.isnan(lod))
        loq_ok=loq is not None and not (isinstance(loq,float) and math.isnan(loq))
        if lod_ok: ax.axvline(lod,color=RED2,linestyle=":",linewidth=1.3); ax.text(lod,yb+yp,f"  LOD={lod:.3f}",color=RED2,fontsize=7,va="bottom")
        if loq_ok: ax.axvline(loq,color=ORG,linestyle=":",linewidth=1.3); ax.text(loq,yb+yp,f"  LOQ={loq:.3f}",color=ORG,fontsize=7,va="bottom")
        m,b,r2=cal["m"],cal["b"],cal["r2"]; sgn="+" if b>=0 else "-"
        ax.text(0.03,0.97,f"y={m:.4f}x {sgn} {abs(b):.4f}  |  R²={r2:.5f}",
                transform=ax.transAxes,fontsize=8.5,color=GRN,va="top",ha="left",
                bbox=dict(facecolor=CARD2C,edgecolor="#166534",boxstyle="round,pad=0.35"))
        ax.set_xlabel(f"Concentración ({unit})",color=SUB,fontsize=9)
        ax.set_ylabel("Absorbancia digital",color=SUB,fontsize=9)
        ax.set_title(f"Curva de calibración — {analyte} | Canal: {ch}",color=TXT2,fontsize=10,pad=8)
        ax.tick_params(colors=SUB,labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
        leg=ax.legend(facecolor=CARD2C,edgecolor="#334155",fontsize=8)
        for t in leg.get_texts(): t.set_color(TXT2)
        ax.grid(True,color=CARD2C,linewidth=0.5,linestyle="--",zorder=0)
        plt.tight_layout(pad=0.8)
        buf=BytesIO()
        plt.savefig(buf,format="png",dpi=160,bbox_inches="tight",facecolor=BG2,edgecolor="none")
        buf.seek(0); data=buf.read(); plt.close(fig)
        return data
    except Exception:
        return None

def channel_sweep_to_png(channel, cal, concs, sigs, unit):
    """Genera una gráfica de barrido individual para un canal (matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        BG2="#0f172a"; GRN="#4ade80"; BLU="#60a5fa"; SUB="#94a3b8"; TXT2="#e2e8f0"
        fig,ax=plt.subplots(figsize=(4,3))
        fig.patch.set_facecolor(BG2); ax.set_facecolor(BG2)
        ax.scatter(concs,sigs,color=GRN,s=40,zorder=5,edgecolors=BG2,linewidths=1)
        xmin=float(concs.min())*0.85 if float(concs.min())>0 else 0.0
        xmax=float(concs.max())*1.15
        if xmin==xmax: xmin-=0.1; xmax+=0.1
        xl=np.linspace(xmin,xmax,100)
        ax.plot(xl,cal["m"]*xl+cal["b"],color=BLU,linewidth=1.5)
        ax.text(0.05,0.95,f"R²={cal['r2']:.4f}  m={cal['m']:.3f}",transform=ax.transAxes,
                fontsize=7,color=TXT2,va="top",bbox=dict(facecolor="#1e293b",edgecolor="#334155",boxstyle="round"))
        ax.set_title(f"{CHANNEL_LABELS.get(channel,channel)}",color=TXT2,fontsize=8)
        ax.set_xlabel(f"Concentración ({unit})",color=SUB,fontsize=7)
        ax.set_ylabel("Abs. digital",color=SUB,fontsize=7)
        ax.tick_params(colors=SUB,labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
        ax.grid(True,color="#1e293b",linewidth=0.4)
        plt.tight_layout(pad=0.5)
        buf=BytesIO()
        plt.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=BG2,edgecolor="none")
        buf.seek(0); data=buf.read(); plt.close(fig)
        return data
    except:
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  REPORTE PDF (mejorado)
# ══════════════════════════════════════════════════════════════════════════════

def gen_pdf(analyte,method,df_rgb,df_results,cal,annotated_img,tri_df,
            cal_png_bytes,channel_sweep_pngs,selected_channel,unit="mg/L"):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.units import inch
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                     Paragraph, Spacer, Table, TableStyle,
                                     Image as RLImage, HRFlowable, KeepTogether)
    C={"bg":HexColor("#020617"),"card":HexColor("#1E293B"),"card2":HexColor("#263546"),
       "acc":HexColor("#2563EB"),"grn":HexColor("#059669"),"red":HexColor("#DC2626"),
       "txt":HexColor("#E2E8F0"),"mut":HexColor("#94A3B8"),"brd":HexColor("#334155"),
       "nb":HexColor("#E8EDF3"),"nt":HexColor("#0A0A0A")}
    buf=BytesIO()
    def bg(canvas,doc):
        canvas.saveState(); canvas.setFillColor(C["bg"])
        canvas.rect(0,0,letter[0],letter[1],fill=1,stroke=0); canvas.restoreState()
    doc=BaseDocTemplate(buf,pagesize=letter,leftMargin=.7*inch,rightMargin=.7*inch,
                        topMargin=.7*inch,bottomMargin=.7*inch)
    fr=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="m")
    doc.addPageTemplates([PageTemplate(id="dark",frames=[fr],onPage=bg)])
    S=getSampleStyleSheet()
    def ps(n,**kw): return ParagraphStyle(n,parent=S["BodyText"],**kw)
    ts  =ps("T",textColor=white,       fontSize=22,fontName="Helvetica-Bold",spaceAfter=2)
    ss  =ps("ST",textColor=C["mut"],   fontSize=9,fontName="Helvetica",spaceAfter=8)
    h2s =ps("H2",textColor=C["acc"],   fontSize=11,fontName="Helvetica-Bold",spaceBefore=12,spaceAfter=4)
    bs  =ps("B", textColor=C["txt"],   fontSize=8,fontName="Helvetica",leading=12)
    fs  =ps("F", textColor=C["mut"],   fontSize=6.5,fontName="Helvetica",alignment=1)
    ws  =ps("W", textColor=HexColor("#FCA5A5"),fontSize=8,fontName="Helvetica-Oblique",leading=12)
    ni  =ps("NI",textColor=C["nt"],    fontSize=8,fontName="Helvetica-Bold",leading=12.5)
    def note(txt):
        t=Table([[Paragraph(txt,ni)]],colWidths=[7.3*inch])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C["nb"]),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        return t
    def dtbl(data,cw,hc=None):
        t=Table(data,colWidths=cw,repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hc or C["acc"]),
            ("TEXTCOLOR",(0,0),(-1,0),white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),7),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C["card"],C["card2"]]),
            ("TEXTCOLOR",(0,1),(-1,-1),C["txt"]),("FONTNAME",(0,1),(-1,-1),"Courier"),
            ("GRID",(0,0),(-1,-1),.35,C["brd"]),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),5)]))
        return t
    now=fmt_mx()
    story=[]
    # Encabezado
    hdr=Table([[Paragraph("ELEMENTA",ts),
                Paragraph(f"<b>Reporte de análisis colorimétrico</b><br/>"
                          f"<font size='8'>{now}</font><br/>"
                          f"<font size='8'>Analito: {analyte} | Método: {method}</font>",
                          ps("HR",textColor=C["txt"],fontSize=9,fontName="Helvetica",alignment=2))]],
              colWidths=[3.0*inch,4.3*inch])
    hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C["bg"]),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,0),(-1,0),2,C["grn"])]))
    story.append(hdr); story.append(Spacer(1,8))
    # Aviso
    av=Table([[Paragraph("<b>AVISO:</b> Estimaciones colorimétricas digitales. No sustituyen métodos "
                         "instrumentales certificados ni declaraciones de cumplimiento normativo.",ws)]],
             colWidths=[7.3*inch])
    av.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HexColor("#450a0a")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(av); story.append(Spacer(1,12))

    # A) Imagen con ROIs
    if annotated_img is not None:
        story.append(Paragraph("A) Imagen procesada — regiones de interés", h2s))
        pil=Image.fromarray(annotated_img); ib=BytesIO(); pil.save(ib,"PNG"); ib.seek(0)
        story.append(RLImage(ib,width=4.8*inch,height=3.0*inch,kind="proportional"))
        story.append(Spacer(1,8))

    # B) Barrido de canales
    if channel_sweep_pngs and len(channel_sweep_pngs)>0:
        story.append(Paragraph("B) Barrido de canales — selección de longitud de onda digital", h2s))
        # Mostrar hasta 3 canales principales (R,G,B) + otros
        for ch_png in channel_sweep_pngs[:3]:  # limitar a 3 para no alargar demasiado
            story.append(RLImage(BytesIO(ch_png),width=5.5*inch,height=3.0*inch))
            story.append(Spacer(1,4))
        story.append(note("Gráficas de absorbancia digital vs concentración para los canales R_norm, G_norm, B_norm."))
        story.append(Spacer(1,10))

    # Tabla RGB
    if df_rgb is not None and not df_rgb.empty:
        story.append(Paragraph("Datos colorimétricos RGB por región de interés", h2s))
        story.append(note("R, G, B: intensidad media del canal (0–255). "
                          "R_norm, G_norm, B_norm: fracción porcentual de cada canal respecto al total RGB."))
        story.append(Spacer(1,4))
        cols=[c for c in ["ROI","R","G","B","R_norm","G_norm","B_norm"] if c in df_rgb.columns]
        td=[cols]+[[f"{v:.2f}" if isinstance(v,float) else str(v) for v in row]
                   for _,row in df_rgb[cols].round(2).iterrows()]
        cw=[.9*inch]+[.85*inch]*(len(cols)-1)
        story.append(dtbl(td,cw)); story.append(Spacer(1,10))

    # C) Curva de calibración final
    if cal and cal_png_bytes:
        story.append(Paragraph("C) Curva de calibración final optimizada", h2s))
        story.append(RLImage(BytesIO(cal_png_bytes),width=5.8*inch,height=3.2*inch))
        story.append(note(f"Canal seleccionado: {selected_channel}. "
                          "Puntos: estándares. Línea: regresión. Líneas punteadas: LOD/LOQ."))
        story.append(Spacer(1,8))

    # D) Resumen analítico
    if cal:
        story.append(Paragraph("D) Resumen analítico", h2s))
        lod=cal.get("LOD",float("nan")); loq=cal.get("LOQ",float("nan"))
        summary_data=[
            ["Parámetro","Valor"],
            ["Analito",analyte],
            ["Método",method],
            ["Canal usado",CHANNEL_LABELS.get(selected_channel,selected_channel)],
            ["Pendiente (m)",f"{cal['m']:.4f}"],
            ["Intercepto (b)",f"{cal['b']:.4f}"],
            ["R²",f"{cal['r2']:.5f}"],
            ["LOD",f"{lod:.3f}" if not math.isnan(lod) else "N/D"],
            ["LOQ",f"{loq:.3f}" if not math.isnan(loq) else "N/D"],
            ["Nº estándares",str(cal.get("n",""))],
            ["Blanco usado",st.session_state.get("blank_label","No especificado")],
            ["Fecha/hora",now],
        ]
        story.append(dtbl(summary_data,[2.5*inch,4.8*inch],C["grn"]))
        story.append(Spacer(1,10))

    # Triplicados
    if tri_df is not None and not tri_df.empty:
        story.append(Paragraph("Estadísticas de triplicados", h2s))
        story.append(note(STAT_EXPL["CV"])); story.append(Spacer(1,4))
        cols=list(tri_df.columns)
        td2=[cols]+[[str(v) if v is not None else "N/D" for v in row] for _,row in tri_df.iterrows()]
        cw2=[max(.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dtbl(td2,cw2,C["grn"])); story.append(Spacer(1,10))

    # E) Resultados de muestras
    if df_results is not None and not df_results.empty:
        story.append(Paragraph("E) Resultados de cuantificación", h2s))
        story.append(note("Conc_calc: calculada de la curva (x=(A-b)/m). "
                          "Conc_corregida: multiplicada por el factor de dilución."))
        story.append(Spacer(1,4))
        cols=list(df_results.columns)
        td3=[cols]+[[f"{v:.3f}" if isinstance(v,float) else str(v) for v in row]
                    for _,row in df_results.iterrows()]
        cw3=[max(.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dtbl(td3,cw3)); story.append(Spacer(1,10))

    # Pie
    story.append(HRFlowable(width="100%",thickness=0.5,color=C["brd"]))
    story.append(Spacer(1,5))
    story.append(note("<b>Nota científica:</b> La precisión depende de la iluminación, el sensor, "
                      "los reactivos y la preparación de estándares. Para cumplimiento normativo "
                      "confirmar mediante métodos acreditados. Consultar siempre la versión vigente del DOF."))
    story.append(Spacer(1,8))
    story.append(Paragraph("Derechos reservados (Katyutzka Villarreal, 2026)  |  Elementa",fs))
    doc.build(story); buf.seek(0)
    return buf.read()

def sanitize_filename(name):
    """Convierte nombre de analito a formato seguro para archivo."""
    name = name.replace("(", "").replace(")", "").replace(" ", "_")
    name = re.sub(r'[^\w\-_\.]', '', name)
    return name

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init():
    defs=dict(image=None,rois=[],freeze_rois=False,device_type="Viales lineales",
              use_circular=False,global_diam=18,
              assignment_df=None,_asgn_fp="",blank_label=None,
              df_rgb=None,df_norm=None,df_abs=None,df_merged=None,
              cal_result=None,best_ch="G_norm",all_ch={},tri_groups={},tri_df=None,
              df_results=None,annotated_img=None,
              cal_fig=None,res_fig=None,sa_fig=None,sa_result=None,
              cal_concs=None,cal_sigs=None,cal_unit="mg/L",cal_analyte="",cal_ch="",
              cal_png=None,selected_channel="G_norm",
              assignment_df_backup=None,roi_config_backup=None)
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v

init()

# ── Helpers UI ─────────────────────────────────────────────────────────────────

def mc(label,value,interpret=None,explain=None,col=None):
    itp=""
    if interpret: itp=f'<p class="itp" style="color:{interpret[1]}">{interpret[0]}</p>'
    exp=""
    if explain: exp=f'<div class="exp">{explain}</div>'
    html=(f'<div class="mc"><p class="lbl">{label}</p>'
          f'<p class="val">{value}</p>{itp}{exp}</div>')
    (col or st).markdown(html,unsafe_allow_html=True)

def ibox(t): st.markdown(f'<div class="info-box">{t}</div>',unsafe_allow_html=True)
def wbox(t): st.markdown(f'<div class="warn-box">{t}</div>',unsafe_allow_html=True)
def okbox(t): st.markdown(f'<div class="ok-box">{t}</div>',unsafe_allow_html=True)
def slbl(t): st.markdown(f'<p class="slbl">{t}</p>',unsafe_allow_html=True)
def footer():
    st.markdown('<div class="footer">Derechos reservados (Katyutzka Villarreal, 2026) | Elementa — Sistema Analítico Colorimétrico Digital</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT};margin:0;font-size:1.5rem;font-weight:700;letter-spacing:-.02em;'>Elementa</h2>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin:2px 0 16px 0;'>Sistema Colorimétrico Digital</p>",unsafe_allow_html=True)
    st.divider()
    pagina=st.radio("Sección",
        ["Tutorial","Análisis","Biblioteca de Métodos","Fundamentos","Normativa"],
        label_visibility="collapsed")
    st.divider()
    st.markdown(f"<p style='color:{MUTED};font-size:.68rem;line-height:1.6;'>Estimaciones colorimétricas digitales. No sustituyen métodos certificados.</p>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

if pagina=="Análisis":
    st.markdown("<h1>Análisis Colorimétrico Digital</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};margin-top:-6px;font-size:.85rem;'>Calibración, cuantificación y evaluación normativa por imágenes RGB.</p>",unsafe_allow_html=True)

    tab_cap, tab_proc, tab_cal, tab_rep = st.tabs(
        ["Captura", "Procesamiento", "Calibración", "Reporte"])

    # ── TAB 1: CAPTURA ──────────────────────────────────────────────────────
    with tab_cap:
        slbl("Paso 1 — Cargar imagen")
        c1,c2=st.columns(2)
        with c1:
            uf=st.file_uploader("Subir imagen",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uf:
                loaded=load_image(uf)
                if loaded is not None:
                    st.session_state["image"]=loaded; st.session_state["rois"]=[]
                    st.session_state["_asgn_fp"]=""
                    st.session_state["selected_channel"]="G_norm"
        with c2:
            cam=st.camera_input("Capturar con cámara",label_visibility="collapsed")
            if cam:
                loaded=load_image(cam)
                if loaded is not None:
                    st.session_state["image"]=loaded; st.session_state["rois"]=[]
                    st.session_state["_asgn_fp"]=""
                    st.session_state["selected_channel"]="G_norm"

        if st.session_state["image"] is None:
            ibox("Cargue o capture una imagen para comenzar.")
            footer(); st.stop()

        img=st.session_state["image"]; H,W=img.shape[:2]

        # ── Control de calidad de imagen ────────────────────────
        qc = check_image_quality(img)
        icons = {"ok":"✓","warn":"⚠","fail":"✗"}
        colors_qc = {"ok":SUCCESS,"warn":"#F59E0B","fail":DANGER}
        qc_parts = []
        for v in qc.values():
            gr    = v["grade"]
            icon  = icons[gr]
            col_c = colors_qc[gr]
            lbl   = v["label"]
            qc_parts.append(f'<span style="color:{col_c};font-size:.78rem;">{icon} {lbl}</span>')
        qc_html = " &nbsp; ".join(qc_parts)
        with st.expander("Control de calidad de imagen", expanded=any(v["grade"]!="ok" for v in qc.values())):
            st.markdown(qc_html, unsafe_allow_html=True)
            if any(v["grade"]=="fail" for v in qc.values()):
                wbox("Una o más métricas de calidad están fuera del rango recomendado. Se recomienda repetir la captura.")
            elif any(v["grade"]=="warn" for v in qc.values()):
                st.markdown(f'<div class="info-box">Algunas métricas requieren revisión. Los resultados pueden ser válidos pero verificar condiciones de captura.</div>',unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0;'>",unsafe_allow_html=True)
        slbl("Paso 2 — Definir regiones de interés (ROIs)")

        ctrl_col,img_col=st.columns([1,1],gap="large")

        with ctrl_col:
            dev=st.selectbox("Tipo de dispositivo",
                             ["Viales lineales","Microplaca de 96 pocillos","Personalizado"],
                             key="dev_sel")
            st.session_state["device_type"]=dev

            is_plate = (dev == "Microplaca de 96 pocillos")
            # Forzar circular en microplaca
            use_circular = is_plate  # Siempre circular en microplaca
            if not is_plate:
                use_circular = st.toggle("Usar ROIs circulares (reducen ruido de fondo)",
                                         value=st.session_state.get("use_circular",False), key="circ_tog")
            st.session_state["use_circular"] = use_circular

            # Diámetro global (solo si circular está activo)
            if use_circular:
                global_diam = st.slider("Diámetro global de pocillo (px)", 6, 80,
                                        st.session_state.get("global_diam",18), key="g_diam")
                st.session_state["global_diam"] = global_diam
                if is_plate:
                    st.markdown('<div class="info-box">En microplaca los ROIs son circulares por defecto.</div>',
                                unsafe_allow_html=True)
            else:
                global_diam = None

            # Mensajes de coordenadas recomendadas según dispositivo
            if is_plate:
                st.markdown(
                    '<div class="info-box">'
                    '<b>Coordenadas recomendadas para microplaca:</b><br>'
                    'X inicial: <b>318 px</b> &nbsp;|&nbsp; '
                    'Y inicial: <b>228 px</b> &nbsp;|&nbsp; '
                    'Diámetro pocillo: <b>60 px</b> &nbsp;|&nbsp; '
                    'Espaciado X: <b>170 px</b> &nbsp;|&nbsp; '
                    'Espaciado Y: <b>170 px</b>'
                    '</div>',
                    unsafe_allow_html=True)
            elif dev == "Viales lineales":
                st.markdown(
                    '<div class="info-box">'
                    '<b>Coordenadas recomendadas para viales:</b><br>'
                    'X inicial: <b>394 px</b> &nbsp;|&nbsp; '
                    'Y inicial: <b>497 px</b> &nbsp;|&nbsp; '
                    'Ancho ROI: <b>33 px</b> &nbsp;|&nbsp; '
                    'Alto ROI: <b>49 px</b> &nbsp;|&nbsp; '
                    'Espaciado Y: <b>108 px</b>'
                    '</div>',
                    unsafe_allow_html=True)

            freeze=st.toggle("Bloquear ROIs",value=st.session_state["freeze_rois"],key="frz")
            st.session_state["freeze_rois"]=freeze

            if not freeze:
                # Generar ROIs en tiempo real según dispositivo
                if dev=="Viales lineales":
                    n  = st.number_input("N de viales",2,24,6,1,key="vn")
                    x0 = st.slider("X inicial (px)",0,W-1,int(W*.05),key="vx0")
                    y0 = st.slider("Y inicial (px)",0,H-1,int(H*.25),key="vy0")
                    rw = st.slider("Ancho ROI (px)",5,200,40,key="vrw")
                    rh = st.slider("Alto ROI (px)", 5,300,60,key="vrh")
                    dx = st.slider("Espaciado X (px)",0,300,int(W*.08),key="vdx")
                    dy = st.slider("Espaciado Y (px)",0,300,0,key="vdy")
                    rois=gen_rois_linear(x0,y0,rw,rh,int(n),dx,dy)

                elif is_plate:
                    # Microplaca: ocultar ancho/alto, mostrar diámetro, usar w=h=global_diam
                    diam = st.session_state.get("global_diam", 60)
                    n_rows = st.number_input("Filas", 1,8, 8,1,key="prows")
                    n_cols = st.number_input("Columnas",1,12,12,1,key="pcols")
                    x0 = st.slider("X inicial (px)",0,W-1,318,key="px0")
                    y0 = st.slider("Y inicial (px)",0,H-1,228,key="py0")
                    dx = st.slider("Espaciado X (px)",10,300,170,key="pdx")
                    dy = st.slider("Espaciado Y (px)",10,300,170,key="pdy")
                    # w y h igual al diámetro (para bounding box del círculo)
                    rois=gen_rois_plate(x0,y0,diam,diam,dx,dy,int(n_rows),int(n_cols))

                else:  # Personalizado
                    n  = st.number_input("N de ROIs",2,50,6,1,key="cn")
                    x0 = st.slider("X inicial (px)",0,W-1,int(W*.05),key="cx0")
                    y0 = st.slider("Y inicial (px)",0,H-1,int(H*.1), key="cy0")
                    rw = st.slider("Ancho ROI (px)",5,200,30,key="crw")
                    rh = st.slider("Alto ROI (px)", 5,200,30,key="crh")
                    dx = st.slider("Espaciado X (px)",0,300,int(W*.08),key="cdx")
                    dy = st.slider("Espaciado Y (px)",0,300,int(H*.08),key="cdy")
                    rois=gen_rois_linear(x0,y0,rw,rh,int(n),dx,dy)

                st.session_state["rois"]=rois
            else:
                rois=st.session_state.get("rois",[])
                if rois: okbox(f"ROIs bloqueadas — {len(rois)} regiones. Desactive para reajustar.")
                else:    wbox("No hay ROIs definidas. Desactive el bloqueo para configurar.")

        with img_col:
            rois=st.session_state.get("rois",[])
            use_circ = st.session_state.get("use_circular",False)
            diam_map = {r["label"]: st.session_state.get("global_diam",18) for r in rois} if use_circ else None
            if rois:
                # Asegurar persistencia de asignación
                ensure_asgn(rois)
                tm=dict(zip(st.session_state["assignment_df"]["ROI"],
                            st.session_state["assignment_df"]["Tipo"]))
                ann=draw_rois(img,rois,tm,circular=use_circ,diam_map=diam_map)
                st.session_state["annotated_img"]=ann
                leg=" ".join(
                    f'<span style="background:{c};color:{TEXT};padding:1px 7px;'
                    f'border-radius:3px;font-size:.68rem;font-weight:700;margin-right:2px;">'
                    f'{TIPO_SHORT[t]}</span>'
                    for t,c in TIPO_COLORS.items() if t!="Sin asignar")
                st.markdown(leg,unsafe_allow_html=True)
                st.image(ann,
                         caption=f"Overlay en tiempo real — {'ROIs circulares' if use_circ else 'ROIs rectangulares'}",
                         use_container_width=True)
            else:
                st.image(img,caption="Imagen original — defina ROIs para ver el overlay",
                         use_container_width=True)

        if not rois:
            ibox("Configure las ROIs para continuar.")
            footer(); st.stop()
        footer()

    # ── TAB 2: PROCESAMIENTO ────────────────────────────────────────────────
    with tab_proc:
        rois=st.session_state.get("rois",[]); img=st.session_state.get("image")
        if not rois or img is None:
            wbox("Defina las ROIs en la pestaña Captura primero."); footer(); st.stop()
        # Leer ROIs desde session_state, no regenerar
        ensure_asgn(rois)
        fp=roi_fp(rois)
        slbl("Paso 3 — Asignar tipos y concentraciones")
        ibox("Asigne BLANK=Blanco, STD=Estándar, SMP=Muestra. Pocillos de la misma columna forman triplicados.")

        dev=st.session_state.get("device_type","")
        is_plate=(dev=="Microplaca de 96 pocillos")
        use_circ = st.session_state.get("use_circular",False)
        diam_map = {r["label"]: st.session_state.get("global_diam",18) for r in rois} if use_circ else None

        tbl_col,vis_col=st.columns([6,5] if is_plate else [1,1],gap="large")

        with tbl_col:
            edited=st.data_editor(
                st.session_state["assignment_df"],
                column_config={
                    "Tipo":    st.column_config.SelectboxColumn("Tipo",   options=TIPOS,   required=True),
                    "Unidad":  st.column_config.SelectboxColumn("Unidad", options=UNIDADES,required=True),
                    "Analito": st.column_config.SelectboxColumn("Analito",options=ANALITOS,required=True),
                    "Concentracion":st.column_config.NumberColumn("Conc.",min_value=0.0,step=0.001,format="%.4f"),
                    "Factor_dil":   st.column_config.NumberColumn("F.Dil",min_value=0.01,step=0.1,format="%.2f"),
                },
                num_rows="fixed",use_container_width=True,key=f"asgn_{fp}")
            st.session_state["assignment_df"]=edited
            blank_r=edited[edited["Tipo"]=="Blanco"]
            blank=blank_r["ROI"].iloc[0] if not blank_r.empty else None
            st.session_state["blank_label"]=blank
            if blank: okbox(f"Blanco: <b>{blank}</b>")
            else: wbox("Sin blanco asignado.")

        with vis_col:
            # Overlay siempre actualizado
            tm2=dict(zip(edited["ROI"],edited["Tipo"]))
            ann2=draw_rois(img,rois,tm2,circular=use_circ,diam_map=diam_map)
            st.session_state["annotated_img"]=ann2

            legend=" ".join(
                f'<span style="background:{c};color:{TEXT};padding:2px 8px;'
                f'border-radius:3px;font-size:.7rem;font-weight:700;margin-right:3px;">'
                f'{TIPO_SHORT[t]}</span>'
                for t,c in TIPO_COLORS.items() if t!="Sin asignar")
            st.markdown(f'<p class="slbl">Overlay en tiempo real</p>',unsafe_allow_html=True)
            st.markdown(legend,unsafe_allow_html=True)
            st.image(ann2, caption="Los colores se actualizan al instante al editar la tabla",
                     use_container_width=True)

            if is_plate:
                tri_grp=detect_triplicates(edited)
                st.session_state["tri_groups"]=tri_grp
                with st.expander("Grid 8x12 — distribución de pocillos",expanded=False):
                    st.plotly_chart(plot_plate(edited,tri_grp), use_container_width=True, key="plate_grid_exp")
                if tri_grp:
                    n_g=len(tri_grp); n_w=sum(len(v) for v in tri_grp.values())
                    ibox(f"<b>{n_g} grupos de triplicados</b> ({n_w} pocillos detectados).")
        footer()

    # ── TAB 3: CALIBRACIÓN ─────────────────────────────────────────────────
    with tab_cal:
        rois=st.session_state.get("rois",[]); img=st.session_state.get("image")
        adf=st.session_state.get("assignment_df")
        if not rois or img is None or adf is None:
            wbox("Complete Captura y Procesamiento primero."); footer(); st.stop()
        blank=st.session_state.get("blank_label")

        with st.expander("Fundamento — absorbancia digital",expanded=False):
            st.markdown("**A_dig = log₁₀(I_blanco / I_muestra)**\n\nDonde I es la intensidad normalizada del canal seleccionado (% del total R+G+B). El sistema evalúa los canales y permite elegir el óptimo.")

        if st.button("Extraer RGB y calibrar",key="btn_cal"):
            with st.spinner("Procesando canales de color..."):
                use_circ = st.session_state.get("use_circular",False)
                diam_map = {r["label"]: st.session_state.get("global_diam",18) for r in rois} if use_circ else None

                df_ext = extract_extended_channels(img, rois, circular=use_circ, diam_map=diam_map)
                df_norm = normalize_rgb(df_ext)
                df_abs = calc_absorbance(df_norm, blank)

                df_merged = df_abs.merge(
                    adf[["ROI","Tipo","Nombre","Concentracion","Unidad","Analito","Factor_dil"]],
                    on="ROI",how="left")
                bch,ch_res = best_channel(df_merged)
                st.session_state.update(dict(df_rgb=df_ext,df_norm=df_norm,df_abs=df_abs,
                                             df_merged=df_merged,best_ch=bch,all_ch=ch_res))
                # Inicialmente seleccionamos el mejor canal, pero permitiremos cambio manual
                st.session_state["selected_channel"] = bch
                okbox("Extracción completada. Revise el barrido de canales y seleccione el canal final.")

        ch_res = st.session_state.get("all_ch",{})
        df_merged = st.session_state.get("df_merged")
        if df_merged is not None and ch_res:
            st.markdown("### Barrido de canales")
            # Selector manual de canal
            available_channels = [c for c in ALL_CHANNELS if c in ch_res]
            if not available_channels:
                available_channels = list(ch_res.keys())
            if available_channels:
                sel_ch = st.selectbox(
                    "Canal para calibración final",
                    options=available_channels,
                    index=available_channels.index(st.session_state.get("selected_channel", available_channels[0])),
                    format_func=lambda x: CHANNEL_LABELS.get(x,x),
                    key="channel_selector"
                )
                st.session_state["selected_channel"] = sel_ch

                # Recalcular calibración para el canal seleccionado
                std = df_merged[df_merged["Tipo"]=="Estándar"]
                if len(std)>=2 and f"A_{sel_ch}" in std.columns:
                    concs = std["Concentracion"].values.astype(float)
                    sigs = std[f"A_{sel_ch}"].values.astype(float)
                    cal = fit_line(concs, sigs)
                    if cal:
                        bsigs = df_merged.loc[df_merged["Tipo"]=="Blanco", f"A_{sel_ch}"].dropna().values
                        ld, lq, proxy = calc_lod_loq(cal, bsigs if len(bsigs)>=2 else None)
                        cal.update({"LOD":ld,"LOQ":lq,"lod_proxy":proxy})
                        unit = std["Unidad"].iloc[0] if not std.empty else "mg/L"
                        an = std["Analito"].iloc[0]  if not std.empty else "Analito"
                        png = cal_to_png(cal, concs, sigs, sel_ch, an, unit, ld, lq)
                        cf = plot_cal(concs, sigs, cal, sel_ch, an, unit, ld, lq)
                        rf = plot_residuals(concs, cal, sel_ch)
                        tg = st.session_state.get("tri_groups",{}) or detect_triplicates(adf)
                        td = triplate_stats(df_merged, tg, f"A_{sel_ch}") if tg else None
                        st.session_state.update(dict(cal_result=cal, cal_fig=cf, res_fig=rf,
                                                     cal_concs=concs, cal_sigs=sigs, cal_unit=unit,
                                                     cal_analyte=an, cal_ch=sel_ch, cal_png=png,
                                                     tri_df=td))
                        st.success(f"Calibración actualizada para canal {sel_ch}")

            # Mostrar gráficas de barrido (si hay datos)
            if ch_res:
                # Mostrar comparativa de canales
                st.plotly_chart(plot_channels(ch_res), use_container_width=True)
                df_ch_tbl = plot_channel_table(ch_res)
                st.dataframe(df_ch_tbl, use_container_width=True, hide_index=True)

                # Mostrar gráficas individuales para los canales principales
                std = df_merged[df_merged["Tipo"]=="Estándar"]
                if len(std)>=2 and "Concentracion" in std.columns:
                    concs_all = std["Concentracion"].values.astype(float)
                    cols_to_plot = [c for c in ["R_norm","G_norm","B_norm"] if c in ch_res]
                    if cols_to_plot:
                        st.markdown("**Respuesta individual de canales RGB**")
                        tab_r, tab_g, tab_b = st.tabs(["Rojo", "Verde", "Azul"])
                        for ch, tab in zip(cols_to_plot, [tab_r, tab_g, tab_b]):
                            with tab:
                                if f"A_{ch}" in df_merged.columns:
                                    sigs_ch = std[f"A_{ch}"].dropna().values
                                    cal_ch = ch_res.get(ch)
                                    if cal_ch:
                                        fig_ch = plot_cal(concs_all, sigs_ch, cal_ch, ch,
                                                          std["Analito"].iloc[0], std["Unidad"].iloc[0],
                                                          cal_ch.get("LOD",np.nan), cal_ch.get("LOQ",np.nan))
                                        st.plotly_chart(fig_ch, use_container_width=True)
                                        r2_i, _ = interpret_r2(cal_ch["r2"])
                                        st.write(f"R²: {cal_ch['r2']:.5f} → {r2_i}")

            # Mensaje sobre pendiente negativa
            cal = st.session_state.get("cal_result")
            if cal:
                slope_msg, slope_col = interpret_slope(cal["m"])
                st.markdown(
                    f'<div style="background:{PRIMARY};border-left:3px solid {slope_col};'
                    f'border-radius:0 6px 6px 0;padding:10px 14px;font-size:.82rem;'
                    f'color:{TEXT};margin:8px 0;line-height:1.5;">'
                    f'<b>Pendiente {cal["m"]:+.4f}:</b> {slope_msg}</div>',
                    unsafe_allow_html=True)

        # Calibración final y cuantificación
        cal = st.session_state.get("cal_result")
        if cal:
            sel_ch = st.session_state.get("selected_channel","G_norm")
            unit = st.session_state.get("cal_unit","mg/L")
            ld = cal.get("LOD",float("nan")); lq = cal.get("LOQ",float("nan"))
            r2_int = interpret_r2(cal["r2"])

            st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0;'>",unsafe_allow_html=True)
            slbl("Métricas de calibración final")
            c1,c2,c3,c4=st.columns(4)
            mc("R²",f"{cal['r2']:.5f}",interpret=r2_int,explain=STAT_EXPL["R2"],col=c1)
            mc("Pendiente m",f"{cal['m']:.4f}",explain=STAT_EXPL["slope"],col=c2)
            mc("LOD",f"{ld:.3f}" if not math.isnan(ld) else "N/D",explain=STAT_EXPL["LOD"],col=c3)
            mc("LOQ",f"{lq:.3f}" if not math.isnan(lq) else "N/D",explain=STAT_EXPL["LOQ"],col=c4)

            if cal.get("lod_proxy"): ibox("LOD/LOQ calculados con error residual como proxy. Incluya réplicas del blanco para mayor rigor.")

            t1,t2 = st.tabs(["Curva de calibración","Residuos"])
            with t1:
                if st.session_state.get("cal_fig"): st.plotly_chart(st.session_state["cal_fig"],use_container_width=True)
            with t2:
                if st.session_state.get("res_fig"):
                    st.plotly_chart(st.session_state["res_fig"],use_container_width=True)
                    ibox("Residuos aleatorios alrededor de cero = buen ajuste. Patrón sistemático = no-linealidad.")

            if st.session_state.get("tri_df") is not None:
                td=st.session_state["tri_df"]
                if not td.empty:
                    with st.expander("Estadísticas de triplicados",expanded=True):
                        st.dataframe(td,use_container_width=True,hide_index=True)
                        bad=td[td["CV_%"]>10] if "CV_%" in td.columns else pd.DataFrame()
                        if not bad.empty: wbox(f"CV% > 10% en columnas: {', '.join(bad['Grupo'].tolist())}. Revisar técnica.")

            with st.expander("Tabla de absorbancias digitales"):
                dm=st.session_state.get("df_merged")
                if dm is not None:
                    cols=[c for c in ["ROI","Tipo","Concentracion","R_norm","G_norm","B_norm",f"A_{sel_ch}"] if c in dm.columns]
                    st.dataframe(dm[cols].round(4),use_container_width=True)
                    st.download_button("Descargar CSV",dm[cols].to_csv(index=False).encode(),"elementa_datos.csv","text/csv")

        st.markdown(f"<hr style='border-color:{BORDER};margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Cuantificación de muestras")
        meth=st.radio("Método",["Calibración externa","Adición de estándar"],horizontal=True)

        if meth=="Calibración externa":
            if st.button("Calcular concentraciones",key="btn_q"):
                cal = st.session_state.get("cal_result")
                dm = st.session_state.get("df_merged")
                sel_ch = st.session_state.get("selected_channel","G_norm")
                if cal is None or dm is None:
                    st.error("Seleccione un canal y calibre primero.")
                else:
                    m,b=cal["m"],cal["b"]
                    samples=dm[dm["Tipo"]=="Muestra"].copy(); res=[]
                    for _,row in samples.iterrows():
                        a=row.get(f"A_{sel_ch}",float("nan")); dil=float(row.get("Factor_dil",1) or 1)
                        c_r=(a-b)/m if not math.isnan(a) and abs(m)>1e-12 else float("nan")
                        c_c=c_r*dil if not math.isnan(c_r) else float("nan")
                        res.append({"Muestra":str(row.get("Nombre","")) or row["ROI"],"ROI":row["ROI"],
                                    "Canal":sel_ch,"A_digital":round(a,4) if not math.isnan(a) else None,
                                    "Conc_calc":round(c_r,3) if not math.isnan(c_r) else None,
                                    "Factor_dil":dil,"Conc_corregida":round(c_c,3) if not math.isnan(c_c) else None,
                                    "Unidad":str(row.get("Unidad","mg/L")),"Analito":str(row.get("Analito",""))})
                    df_res=pd.DataFrame(res); st.session_state["df_results"]=df_res
                    st.dataframe(df_res,use_container_width=True,hide_index=True)
                    st.download_button("Descargar resultados CSV",df_res.to_csv(index=False).encode(),"elementa_resultados.csv","text/csv")
        else:
            ibox("Ingrese la señal de la muestra (C_añadida=0) y las adiciones.")
            n_add=st.number_input("N adiciones",2,8,3); sa_data=[{"C_añadida":0.0,"Señal":0.0}]+[{"C_añadida":0.0,"Señal":0.0} for _ in range(int(n_add))]
            sa_ed=st.data_editor(pd.DataFrame(sa_data),column_config={
                "C_añadida":st.column_config.NumberColumn("C añadida",step=0.001,format="%.4f"),
                "Señal":st.column_config.NumberColumn("Señal",step=0.0001,format="%.5f")},
                num_rows="fixed",use_container_width=True,key="sa_ed")
            if st.button("Calcular por adición de estándar",key="btn_sa"):
                sa=std_addition(sa_ed["C_añadida"].values,sa_ed["Señal"].values)
                if sa is None: st.error("No fue posible ajustar la regresión.")
                else:
                    st.session_state["sa_result"]=sa
                    an2=st.session_state.get("cal_analyte","Analito"); unit2=st.session_state.get("cal_unit","mg/L")
                    sf=plot_sa(sa_ed["C_añadida"].values,sa_ed["Señal"].values,sa,an2,unit2)
                    st.session_state["sa_fig"]=sf
                    mc("Concentración estimada",f"{sa['c_sample']:.3f} {unit2}",
                       interpret=(f"R² = {sa['r2']:.4f}",SUCCESS))
                    st.plotly_chart(sf,use_container_width=True)
        footer()

    # ── TAB 4: REPORTE ──────────────────────────────────────────────────────
    with tab_rep:
        slbl("Evaluación normativa")
        wbox("Verificar siempre los límites en la versión oficial vigente (DOF). Valores mostrados son referenciales.")
        df_res=st.session_state.get("df_results"); sa_r=st.session_state.get("sa_result")
        if df_res is not None and not df_res.empty:
            for _,row in df_res.iterrows():
                try:
                    cv=float(row["Conc_corregida"]); an=str(row["Analito"])
                    st.markdown(f"**{row['Muestra']}** — {an}: `{cv:.3f} {row['Unidad']}`")
                    for ev in norm_eval(an,cv):
                        ls=f"{ev['limite']:.3g} mg/L" if ev["limite"] else "—"
                        st.markdown(f"&nbsp;&nbsp;<span class='badge-{ev['badge']}'>{ev['status']}</span> <span style='color:{MUTED};font-size:.82rem;'>{ev['norma']} | Límite: {ls}</span>",unsafe_allow_html=True)
                except: pass
        elif sa_r:
            an3=st.session_state.get("cal_analyte",""); unit3=st.session_state.get("cal_unit","mg/L")
            cv3=sa_r["c_sample"]
            st.markdown(f"**Adición de estándar** — {an3}: `{cv3:.3f} {unit3}`")
            for ev in norm_eval(an3,cv3):
                ls=f"{ev['limite']:.3g} mg/L" if ev["limite"] else "—"
                st.markdown(f"&nbsp;&nbsp;<span class='badge-{ev['badge']}'>{ev['status']}</span> <span style='color:{MUTED};font-size:.82rem;'>{ev['norma']} | Límite: {ls}</span>",unsafe_allow_html=True)
        else:
            ibox("Complete la cuantificación en la pestaña Calibración.")

        st.markdown(f"<hr style='border-color:{BORDER};margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Exportar reporte PDF")
        adf2=st.session_state.get("assignment_df")
        an_pdf=adf2["Analito"].iloc[0] if adf2 is not None and not adf2.empty else "N/D"
        unit_pdf=adf2["Unidad"].iloc[0]  if adf2 is not None and not adf2.empty else "mg/L"
        meth_pdf="Adición de estándar" if sa_r else "Calibración externa"
        sel_ch = st.session_state.get("selected_channel","G_norm")

        if st.button("Generar reporte PDF",key="btn_pdf"):
            cal_png = st.session_state.get("cal_png")
            # Si no hay cal_png, generarlo nuevamente
            if cal_png is None:
                cal = st.session_state.get("cal_result")
                if cal:
                    cal_png = cal_to_png(cal, st.session_state.get("cal_concs"),
                                         st.session_state.get("cal_sigs"),
                                         sel_ch, an_pdf, unit_pdf,
                                         cal.get("LOD",np.nan), cal.get("LOQ",np.nan))
            # Generar barrido de canales PNG
            sweep_pngs = []
            ch_res = st.session_state.get("all_ch",{})
            if ch_res and df_merged is not None:
                std = df_merged[df_merged["Tipo"]=="Estándar"]
                if len(std)>=2:
                    concs_all = std["Concentracion"].values.astype(float)
                    for ch in ["R_norm","G_norm","B_norm"]:
                        if ch in ch_res and f"A_{ch}" in df_merged.columns:
                            sigs = std[f"A_{ch}"].dropna().values
                            cal_ch = ch_res[ch]
                            if cal_ch:
                                png_ch = channel_sweep_to_png(ch, cal_ch, concs_all, sigs, unit_pdf)
                                if png_ch: sweep_pngs.append(png_ch)
            try:
                pdf_b = gen_pdf(an_pdf,meth_pdf,
                                st.session_state.get("df_norm"),df_res,
                                st.session_state.get("cal_result"),
                                st.session_state.get("annotated_img"),
                                st.session_state.get("tri_df"),
                                cal_png, sweep_pngs, sel_ch, unit_pdf)
                b64 = base64.b64encode(pdf_b).decode()
                safe_analyte = sanitize_filename(an_pdf)
                fname = f"Elementa_{safe_analyte}_PWA_{now_mx():%Y%m%d_%H%M}.pdf"
                href = (f'<a href="data:application/pdf;base64,{b64}" download="{fname}" '
                        f'style="background:{ACCENT};color:white;padding:10px 24px;'
                        f'border-radius:6px;text-decoration:none;font-weight:700;font-size:.85rem;'
                        f'display:inline-block;margin-top:8px;">Descargar reporte PDF</a>')
                st.markdown(href,unsafe_allow_html=True)
                okbox("Reporte generado exitosamente.")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")
        footer()

# ══════════════════════════════════════════════════════════════════════════════
#  TUTORIAL
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Tutorial":
    st.markdown("<h1>Guía de inicio rápido</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};margin-top:-6px;'>Siga estos pasos para realizar su primer análisis colorimétrico con Elementa.</p>",unsafe_allow_html=True)
    steps=[
        ("Paso 1 — Preparación de estándares y muestras",f"""
**Prepare su serie de calibración:**
- Prepare al menos **5 estándares** de concentración conocida que abarquen el rango esperado de las muestras.
- Incluya un **blanco de reactivos** (todos los reactivos sin analito).
- Prepare las muestras en las mismas condiciones que los estándares (mismo volumen, pH, tiempo de reacción).
- Para microplaca: use el patrón de distribución recomendado — columna 1 = Blanco (A1, B1, C1), columnas 2-6 = Estándares, columnas 7+ = Muestras.

**Consideraciones críticas:**
- Tiempo de reacción constante para todos los pozos.
- Temperatura controlada.
- Cubrir la placa durante la incubación para evitar evaporación y contaminación.
        """),
        ("Paso 2 — Captura correcta de imágenes",f"""
**Condiciones de captura recomendadas:**
- **Usar la caja de adquisición Elementa** con iluminación LED blanca difusa y fondo negro mate.
- **Evitar iluminación externa**: apagar luces del laboratorio o bloquear entrada de luz natural.
- **Distancia constante**: 20-25 cm entre cámara y placa (marcar la posición en la caja).
- **Cámara paralela**: la placa debe verse sin perspectiva ni inclinación.
- **Modo manual**: desactivar el ajuste automático de exposición y balance de blancos.
- **Formato PNG** en lugar de JPEG para evitar compresión con pérdida.
- **Esperar 2-3 segundos** antes de capturar para que la cámara estabilice la exposición.

**Indicadores de buena captura:**
- Todos los pozos visibles y sin reflejos.
- Colores uniformes dentro de cada pozo.
- Sin bordes saturados (completamente blancos) ni subexpuestos (completamente negros).
        """),
        ("Paso 3 — Verificación de detección de pozos",f"""
**En la pestaña Captura > Paso 2:**
1. Seleccione el tipo de dispositivo (Microplaca o Viales).
2. Ajuste los sliders de posición — la imagen se actualiza en tiempo real.
3. Use las coordenadas recomendadas como punto de partida.
4. Active **ROIs circulares** (microplaca) para mejor precisión — excluye píxeles de fondo y bordes.
5. Active **Bloquear ROIs** una vez que los contornos estén bien posicionados.

**Verificar:**
- Cada contorno debe cubrir el contenido del pozo sin incluir el borde plástico.
- Los colores de los contornos cambian automáticamente al asignar tipos en la siguiente pestaña.
- En la pestaña Procesamiento verá el Grid inteligente con la distribución completa.
        """),
        ("Paso 4 — Carga de concentraciones y tipos",f"""
**En la pestaña Procesamiento:**
1. Asigne el tipo a cada pocillo en la tabla: BLANK, STD, SMP, CTRL.
2. Ingrese la concentración conocida para cada Estándar (STD).
3. Seleccione el analito y las unidades.
4. El grid inteligente se actualiza en tiempo real mostrando la distribución.
5. Los triplicados (pocillos de la misma columna) se detectan automáticamente.

**Consejo:** Use la columna 'Nombre' para identificar cada muestra. Estos nombres aparecen en el reporte PDF.
        """),
        ("Paso 5 — Calibración e interpretación de resultados",f"""
**Calibración (pestaña Calibración):**
- Haga clic en **Extraer RGB y calibrar**.
- Revise el barrido de canales: R, G, B, y otros.
- Seleccione el canal que mejor rendimiento ofrezca (mayor R², buena sensibilidad).
- El sistema sugiere un canal, pero usted puede elegir otro.

**Estadísticas clave:**

| Parámetro | Descripción | Criterio de aceptación |
|---|---|---|
| **R²** | Linealidad del método | >= 0.995 para métodos de campo |
| **Pendiente m** | Sensibilidad analítica | Positiva (respuesta directa) o negativa (respuesta inversa) |
| **LOD** | Mínimo detectable | Reportar muestras < LOD como "no detectado" |
| **LOQ** | Mínimo cuantificable | Muestras entre LOD y LOQ son semicuantitativas |
| **CV%** | Reproducibilidad triplicados | < 5% excelente; < 10% aceptable |

**Interpretación de pendiente negativa:**
Es completamente válida en métodos como DPPH y ABTS (canal azul). El sistema lo indica automáticamente y calcula la concentración correctamente usando x = (A - b) / m.
        """),
    ]
    for title, content in steps:
        with st.expander(title, expanded=False):
            st.markdown(content)
    st.markdown("<hr style='border-color:#334155;margin:24px 0;'>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:.8rem;text-align:center;'>¿Listo para comenzar? Navegue a la sección <b>Análisis</b> en el menú lateral.</p>",unsafe_allow_html=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
#  BIBLIOTECA DE MÉTODOS
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Biblioteca de Métodos":
    st.markdown("<h1>Biblioteca de Métodos Analíticos</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};'>Protocolos colorimétricos preconfigurados para guiar la selección de canal y condiciones de análisis.</p>",unsafe_allow_html=True)
    cat=st.radio("Categoría",list(PROTOCOL_LIBRARY.keys()),horizontal=True)
    st.divider()
    for nombre, proto in PROTOCOL_LIBRARY[cat].items():
        with st.expander(f"{nombre}   |   λ = {proto['lambda_ref']} nm   |   Canal sugerido: {CHANNEL_LABELS.get(proto['canal'],proto['canal'])}", expanded=False):
            col_a, col_b = st.columns([2,1])
            with col_a:
                st.markdown(f"**Principio químico:**  \n{proto['principio']}")
                st.markdown(f"**Observaciones:** {proto['obs']}")
                st.markdown(f"**Referencias:** {proto['ref']}")
            with col_b:
                st.markdown(
                    f'<div class="mc">'
                    f'<p class="lbl">Analito</p><p class="val" style="font-size:1rem;">{proto["analito"]}</p>'
                    f'<p class="lbl" style="margin-top:8px;">Unidad</p><p class="val" style="font-size:1rem;">{proto["unidad"]}</p>'
                    f'<p class="lbl" style="margin-top:8px;">Color esperado</p><p class="val" style="font-size:.9rem;">{proto["color"]}</p>'
                    f'<p class="lbl" style="margin-top:8px;">Canal sugerido</p>'
                    f'<p class="val" style="font-size:.9rem;color:{SUCCESS};">{CHANNEL_LABELS.get(proto["canal"],proto["canal"])}</p>'
                    f'<p class="lbl" style="margin-top:8px;">λ referencia</p><p class="val" style="font-size:1rem;">{proto["lambda_ref"]} nm</p>'
                    f'</div>',
                    unsafe_allow_html=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTOS
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Fundamentos":
    st.markdown("<h1>Fundamentos del análisis colorimétrico digital</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};'>Base científica de los métodos implementados en Elementa.</p>",unsafe_allow_html=True)

    topics = {
        "¿Qué es una ROI?": (
            "**Región de Interés (ROI)** es la zona de la imagen que se analiza. "
            "En Elementa, cada ROI corresponde a un pocillo o vial. "
            "Las ROIs pueden ser **rectangulares** (viales) o **circulares** (microplacas), "
            "y definen el área de donde se extraen los valores de color."
        ),
        "¿Qué es la absorbancia digital?": (
            "Se calcula como: **A_dig = log₁₀(I_blanco / I_muestra)**. "
            "I_blanco es la intensidad del canal (p.ej., R_norm) en el pocillo blanco, "
            "e I_muestra la intensidad en el pocillo de interés. "
            "Esta transformación sigue la ley de Beer-Lambert y permite construir curvas de calibración lineales."
        ),
        "Barrido de canales": (
            "El sensor de la cámara captura tres canales: Rojo (R), Verde (G) y Azul (B). "
            "Elementa además convierte a espacios de color HSV y CIELAB, obteniendo 9 canales. "
            "El barrido evalúa la linealidad (R²) de cada canal para elegir el más sensible al analito. "
            "Por ejemplo, el Cr(VI) con DPC responde mejor en el canal verde."
        ),
        "Pendiente positiva y negativa": (
            "Una **pendiente positiva** indica que la absorbancia aumenta con la concentración (relación directa). "
            "Una **pendiente negativa** indica que la señal disminuye (relación inversa), lo cual es normal en ensayos como DPPH o ABTS. "
            "Ambas son válidas analíticamente; Elementa calcula la concentración correctamente en ambos casos usando x = (A - b)/m."
        ),
        "¿Qué significa R²?": (
            "El coeficiente de determinación R² mide qué tan bien los puntos experimentales se ajustan a una línea recta. "
            "R² cercano a 1 indica excelente linealidad; valores debajo de 0.99 sugieren revisar la técnica."
        ),
        "LOD y LOQ": (
            "**Límite de Detección (LOD)**: concentración mínima que puede distinguirse del ruido (3.3 σ / |m|). "
            "**Límite de Cuantificación (LOQ)**: concentración mínima que puede medirse con precisión (10 σ / |m|). "
            "Resultados entre LOD y LOQ son semicuantitativos."
        ),
        "Longitud de onda digital": (
            "En espectrofotometría se elige una λ específica (nm). "
            "En colorimetría digital, cada canal de color actúa como una banda ancha equivalente. "
            "Elementa selecciona la 'longitud de onda digital' que maximiza la sensibilidad para cada analito."
        ),
    }

    for title, content in topics.items():
        with st.expander(title, expanded=False):
            st.markdown(content)

    # Diagrama ilustrativo: ejemplo de ROIs y curva
    st.markdown("### Ejemplo visual")
    col1, col2 = st.columns(2)
    with col1:
        # Generar imagen de ejemplo con ROIs (usando matplotlib)
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots(figsize=(4,3), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')
        # Simular pocillos
        for i in range(3):
            for j in range(2):
                rect = patches.Rectangle((0.1+j*0.35, 0.1+i*0.35), 0.3, 0.3,
                                         linewidth=1, edgecolor='#4ade80', facecolor='none')
                ax.add_patch(rect)
                ax.text(0.25+j*0.35, 0.25+i*0.35, f"T{i*2+j}", color='white', ha='center', va='center', fontsize=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='#0f172a', bbox_inches='tight', dpi=120)
        buf.seek(0)
        st.image(buf, caption="Ejemplo de ROIs rectangulares", use_container_width=True)
        plt.close()
    with col2:
        # Curva de calibración de ejemplo
        x = np.linspace(0,1,6)
        y = 0.5*x + 0.02 + np.random.normal(0,0.02,6)
        cal = fit_line(x, y)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Datos'))
        fig2.add_trace(go.Scatter(x=x, y=cal['m']*x+cal['b'], mode='lines', name='Ajuste'))
        fig2.update_layout(template='plotly_dark', paper_bgcolor='#0B1120', plot_bgcolor='#0B1120',
                           margin=dict(l=40,r=20,t=40,b=40), height=280,
                           title="Curva de calibración de ejemplo",
                           xaxis_title="Concentración", yaxis_title="Absorbancia")
        st.plotly_chart(fig2, use_container_width=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
#  NORMATIVA
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Normativa":
    st.markdown("<h1>Normativa y fuentes de referencia</h1>",unsafe_allow_html=True)
    slbl("Límites permisibles de referencia")
    wbox("Verificar siempre en la versión oficial vigente del DOF (dof.gob.mx). Valores informativos.")
    norm_rows=[{"Analito":a,"Norma":n,"Límite (mg/L)":l} for a,ns in NORMATIVE_LIMITS.items() for n,l in ns.items()]
    st.dataframe(pd.DataFrame(norm_rows),use_container_width=True,hide_index=True)
    st.divider()
    slbl("Referencias")
    refs=[("NOM-127-SSA1-2021","Agua potable. Límites permisibles. DOF 2021.","https://www.dof.gob.mx"),
          ("NOM-001-SEMARNAT-2021","Descargas aguas residuales. DOF 2021.","https://www.dof.gob.mx"),
          ("IARC Monographs Vol. 49","Chromium, Nickel and Welding. [Cr(VI) Grupo 1].","https://monographs.iarc.who.int"),
          ("Miller & Miller (2010)","Statistics and Chemometrics for Analytical Chemistry. Pearson.",""),
          ("Brand-Williams et al. (1995)","DPPH free radical method. LWT 28(1).",""),
          ("Cardoso Steele et al. (2019)","Digital image colorimetry on smartphone. Trends Anal. Chem. 111.","")]
    for r,t,u in refs:
        if u: st.markdown(f"- **{r}**: {t} — [Ver]({u})")
        else: st.markdown(f"- **{r}**: {t}")
    st.divider()
    slbl("Editar límites normativos (sesión actual)")
    ibox("Cambios solo para esta sesión. Para permanentes editar NORMATIVE_LIMITS en el código fuente.")
    ed_rows=[{"Analito":a,"Norma":n,"Limite_mg_L":l} for a,ns in NORMATIVE_LIMITS.items() for n,l in ns.items()]
    ed_df=st.data_editor(pd.DataFrame(ed_rows),column_config={"Limite_mg_L":st.column_config.NumberColumn("Límite (mg/L)",min_value=0.0,step=0.001,format="%.4f")},num_rows="fixed",use_container_width=True,key="norm_ed")
    if st.button("Aplicar cambios"):
        NORMATIVE_LIMITS.clear()
        for _,row in ed_df.iterrows():
            NORMATIVE_LIMITS.setdefault(row["Analito"],{})[row["Norma"]]=row["Limite_mg_L"]
        okbox("Límites actualizados para esta sesión.")
    footer()
