"""
Elementa v1 — Sistema analítico colorimétrico digital
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
#  PALETA OSCURA ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════

BG       = "#020617"
PRIMARY  = "#0F172A"
CARD     = "#1E293B"
CARD2    = "#263546"
ACCENT   = "#2563EB"
SUCCESS  = "#059669"
DANGER   = "#DC2626"
WARNING  = "#F59E0B"
TEXT     = "#E2E8F0"
MUTED    = "#94A3B8"
BORDER   = "#334155"
PLOT_BG  = "#0B1120"

# Colores para el overlay (se usarán en draw_rois)
TIPO_COLORS = {
    "Blanco":           "#0EA5E9",
    "Estándar":         "#059669",
    "Muestra":          "#2563EB",
    "Control":          "#7C3AED",
    "Adición estándar": "#DB2777",
    "Sin asignar":      "#1E293B",
}
TIPO_BGR = {
    "Blanco":           (233, 165, 8),
    "Estándar":         (105, 150, 5),
    "Muestra":          (235, 99, 37),
    "Control":          (237, 58, 124),
    "Adición estándar": (119, 39, 219),
    "Sin asignar":      (59, 41, 30),
}
TIPO_SHORT = {
    "Blanco":"BLANK","Estándar":"STD","Muestra":"SMP",
    "Control":"CTRL","Adición estándar":"ADD","Sin asignar":"--",
}
TIPOS    = ["Sin asignar","Blanco","Estándar","Muestra","Control","Adición estándar"]
ANALITOS = ["Fenólicos totales","Cr(VI)","Pb","Cd","Cr total","DPPH","ABTS","FRAP","Nitrógeno amoniacal","Otro"]
UNIDADES = ["mg/L","µg/L","ppm","µM","mM","%","µg/mL","Otro"]

NORMATIVE_LIMITS = {
    "Pb":       {"NOM-127-SSA1-2021 agua potable":0.01,"NOM-001-SEMARNAT-2021 descarga A":0.2,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cd":       {"NOM-127-SSA1-2021 agua potable":0.003,"NOM-001-SEMARNAT-2021 descarga A":0.1,"NOM-001-SEMARNAT-2021 descarga B":0.2},
    "Cr total": {"NOM-127-SSA1-2021 agua potable":0.05,"NOM-001-SEMARNAT-2021 descarga A":0.5,"NOM-001-SEMARNAT-2021 descarga B":1.0},
    "Cr(VI)":   {"NOM-127-SSA1-2021 agua potable":0.05},
}

STAT_EXPL = {
    "R2":       "Coeficiente de determinación. R² ≥ 0.999 = linealidad excelente.",
    "slope":    "Pendiente (m). Sensibilidad analítica.",
    "intercept": "Intercepto (b). Idealmente cercano al blanco de reactivos.",
    "se":       "Error estándar de la pendiente.",
    "LOD":      "Límite de detección (3.3·σ/|m|).",
    "LOQ":      "Límite de cuantificación (10·σ/|m|).",
    "CV":       "Coeficiente de Variación. CV < 5% excelente, 5-10% aceptable, >10% revisar técnica.",
}

def interpret_r2(r2):
    if r2 >= 0.999: return "Linealidad excelente", SUCCESS
    if r2 >= 0.995: return "Muy buena", SUCCESS
    if r2 >= 0.990: return "Ligera dispersión experimental", WARNING
    return "Revisar calibración", DANGER

def interpret_slope(m):
    if m > 0:
        return "Relación directa: la señal aumenta con la concentración.", ACCENT
    else:
        return ("Relación inversa: la señal disminuye con la concentración. "
                "Esto puede ser normal dependiendo del canal o transformación digital utilizada.", WARNING)

# ─── Biblioteca de Protocolos Analíticos (actualizada a λ=760 para Folin) ─────
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
    },
    "Antioxidantes y bioactivos": {
        "Fenólicos totales — Folin-Ciocalteu": dict(
            analito="Fenólicos totales", unidad="mg GAE/L",
            principio="Los grupos fenólicos reducen el reactivo de Folin-Ciocalteu formando un complejo azul intenso.",
            lambda_ref=760, color="Azul", canal="R_norm",
            obs="pH alcalino con Na2CO3. Incubar 2 h a temperatura ambiente. λ referencia = 760 nm.",
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
#  STREAMLIT CONFIG + CSS (MODO OSCURO)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Elementa v1", page_icon=None, layout="wide",
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

def gen_rois_tubes(x0,y0,radius,h,ntubes,dx):
    rois = []
    for i in range(ntubes):
        x_center = x0 + i*dx
        y_positions = [
            y0 + int(h * 0.25),
            y0 + int(h * 0.5),
            y0 + int(h * 0.75)
        ]
        for j, yc in enumerate(y_positions):
            label = f"Tubo{i+1}"
            if j==0: suf = "_Sup"
            elif j==1: suf = "_Med"
            else: suf = "_Inf"
            full_label = label + suf
            rois.append({
                "x": x_center - radius,
                "y": yc - radius,
                "w": 2*radius,
                "h": 2*radius,
                "label": full_label,
                "cx": x_center,
                "cy": yc,
                "radius": radius,
                "tube_id": i+1
            })
    return rois

def draw_rois(img, rois, type_map=None, circular=False, diam_map=None):
    """
    Dibuja ROIs con colores vivos por tipo de muestra.
    El color se actualiza en tiempo real al cambiar la asignación.
    """
    out = img.copy()
    for roi in rois:
        tipo = (type_map or {}).get(roi["label"], "Sin asignar")
        rgb  = TIPO_BGR.get(tipo, (30, 41, 59))
        bgr  = (rgb[2], rgb[1], rgb[0])
        if "cx" in roi and "cy" in roi and "radius" in roi:
            cx, cy, r = roi["cx"], roi["cy"], roi["radius"]
            cv2.circle(out, (cx, cy), r, bgr, 2)
            cv2.circle(out, (cx, cy), 2, bgr, -1)
            x_text = cx - 10
            y_text = cy - r - 5
        else:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            if circular:
                diam = (diam_map or {}).get(roi["label"], min(w, h))
                cx, cy, r = x + w//2, y + h//2, diam//2
                cv2.circle(out, (cx, cy), r, bgr, 2)
                cv2.circle(out, (cx, cy), 2, bgr, -1)
            else:
                cv2.rectangle(out, (x, y), (x+w, y+h), bgr, 2)
            x_text, y_text = x, max(y-3, 10)
        short = TIPO_SHORT.get(tipo, "")
        cv2.putText(out, roi["label"], (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1, cv2.LINE_AA)
        if short and short != "--":
            if "cx" in roi:
                cx2, cy2 = roi["cx"], roi["cy"]
            else:
                cx2, cy2 = x + w//2, y + h//2
            cv2.putText(out, short, (cx2-10, cy2+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr, 1, cv2.LINE_AA)
    return out

def plot_plate_grid(asgn_df):
    """
    Genera un grid tipo Excel (8x12) con las etiquetas de los pocillos,
    coloreando cada celda según el tipo de muestra asignado.
    """
    rl = list("ABCDEFGH")
    all_rows = sorted(set(r for r in rl if any(r in roi for roi in asgn_df["ROI"])), key=lambda x: rl.index(x))
    all_cols = sorted(set(int("".join(c for c in roi if c.isdigit())) for roi in asgn_df["ROI"] if any(c.isdigit() for c in roi)))
    if not all_rows or not all_cols:
        return go.Figure()
    
    tipo_map = dict(zip(asgn_df["ROI"], asgn_df["Tipo"]))
    
    z = []
    txt = []
    hov = []
    for rw in all_rows:
        zr, tr, hr = [], [], []
        for cl in all_cols:
            roi = f"{rw}{cl}"
            tipo = tipo_map.get(roi, "Sin asignar")
            short = TIPO_SHORT.get(tipo, "--")
            idx = {"Blanco":1,"Estándar":2,"Muestra":3,"Control":4,"Adición estándar":5}.get(tipo, 0)
            zr.append(idx)
            tr.append(f"{short}")
            hr.append(f"<b>{roi}</b><br>{tipo}")
        z.append(zr)
        txt.append(tr)
        hov.append(hr)
    
    colorscale = [
        [0, "#1E293B"],
        [0.2, "#0C2540"],
        [0.4, "#052e16"],
        [0.6, "#0D2159"],
        [0.8, "#2D0A3E"],
        [1.0, "#1e293b"]
    ]
    
    fig = go.Figure(go.Heatmap(
        z=z, text=txt, texttemplate="%{text}",
        customdata=hov, hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale, showscale=False,
        xgap=2, ygap=2, zmin=0, zmax=5,
        textfont=dict(family="JetBrains Mono", size=8, color=TEXT)
    ))
    fig.update_xaxes(tickvals=list(range(len(all_cols))), ticktext=[str(c) for c in all_cols],
                     side="top", showgrid=False, tickfont=dict(size=9, color=MUTED))
    fig.update_yaxes(tickvals=list(range(len(all_rows))), ticktext=all_rows,
                     autorange="reversed", showgrid=False, tickfont=dict(size=9, color=MUTED))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter,sans-serif", color=TEXT, size=11),
        margin=dict(l=52, r=20, t=48, b=48),
        height=max(220, 50*len(all_rows)+80),
        title=dict(text="Mapa de placa (grid Excel)", font=dict(size=12, color=TEXT))
    )
    return fig

# ─── Extracción completa de señales ──────────────────────────────────────────
def extract_all_signals(img, rois, circular=True, diam_map=None):
    """Extrae RGB, HSV, CIELAB, combinaciones y deja ED para después."""
    H, W = img.shape[:2]
    rows = []
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)

    for roi in rois:
        if circular and "cx" in roi and "cy" in roi and "radius" in roi:
            cx, cy, r = roi["cx"], roi["cy"], roi["radius"]
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(W, cx+r), min(H, cy+r)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                rows.append(empty_row(roi["label"])); continue
            Yg, Xg = np.ogrid[:crop.shape[0], :crop.shape[1]]
            mask = ((Yg-(cy-y1))**2 + (Xg-(cx-x1))**2) <= r**2
            sl = (slice(y1,y2), slice(x1,x2))
        else:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            sl = (slice(max(0,y), min(H,y+h)), slice(max(0,x), min(W,x+w)))
            mask = np.ones((sl[0].stop-sl[0].start, sl[1].stop-sl[1].start), bool)

        def ch_stats(arr2d):
            v = arr2d[sl][mask].ravel() if arr2d[sl].shape[:2] == mask.shape else arr2d[sl].ravel()
            return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)

        rm,rs = ch_stats(img[:,:,0]); gm,gs = ch_stats(img[:,:,1]); bm,bs = ch_stats(img[:,:,2])
        hm,hs = ch_stats(img_hsv[:,:,0]); sm,ss = ch_stats(img_hsv[:,:,1]); vm,vs = ch_stats(img_hsv[:,:,2])
        lm,ls2= ch_stats(img_lab[:,:,0]); am,as2 = ch_stats(img_lab[:,:,1]); blm,bls = ch_stats(img_lab[:,:,2])

        eps=1e-9; tot=rm+gm+bm+eps
        rn = rm/tot; gn = gm/tot; bn = bm/tot

        row = {"ROI":roi["label"],
               "R":round(rm,2),"G":round(gm,2),"B":round(bm,2),
               "R_sd":round(rs,2),"G_sd":round(gs,2),"B_sd":round(bs,2),
               "R_norm":round(rn*100,3),"G_norm":round(gn*100,3),"B_norm":round(bn*100,3),
               "R+G":round(rm+gm,2),"R+B":round(rm+bm,2),"G+B":round(gm+bm,2),"R+G+B":round(tot,2),
               "R_norm+G_norm":round((rn+gn)*100,3),"R_norm+B_norm":round((rn+bn)*100,3),"G_norm+B_norm":round((gn+bn)*100,3),
               "H":round(hm,2),"S":round(sm,2),"V":round(vm,2),
               "L":round(lm,2),"a":round(am,2),"b_lab":round(blm,2),
               }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def empty_row(label):
    return {"ROI":label,"R":np.nan,"G":np.nan,"B":np.nan,"R_sd":np.nan,"G_sd":np.nan,"B_sd":np.nan,
            "R_norm":np.nan,"G_norm":np.nan,"B_norm":np.nan,
            "R+G":np.nan,"R+B":np.nan,"G+B":np.nan,"R+G+B":np.nan,
            "R_norm+G_norm":np.nan,"R_norm+B_norm":np.nan,"G_norm+B_norm":np.nan,
            "H":np.nan,"S":np.nan,"V":np.nan,"L":np.nan,"a":np.nan,"b_lab":np.nan}

def add_euclidean_distance(df, blank_row):
    if blank_row is None or blank_row.empty:
        df["ED"] = np.nan
        df["ED_norm"] = np.nan
        return df
    b = blank_row.iloc[0]
    df["ED"] = np.sqrt((df["R"] - b["R"])**2 + (df["G"] - b["G"])**2 + (df["B"] - b["B"])**2)
    df["ED_norm"] = np.sqrt((df["R_norm"] - b["R_norm"])**2 + (df["G_norm"] - b["G_norm"])**2 + (df["B_norm"] - b["B_norm"])**2)
    return df

# ─── Absorbancias (clásica e invertida) ──────────────────────────────────────
def compute_absorbances(df, blank_label, signal_columns):
    if blank_label is None or blank_label not in df["ROI"].values:
        for col in signal_columns:
            df[f"A_{col}"] = np.nan
            df[f"A_inv_{col}"] = np.nan
        return df
    blank = df[df["ROI"] == blank_label].iloc[0]
    for col in signal_columns:
        bv = blank[col]
        if np.isnan(bv) or bv == 0:
            df[f"A_{col}"] = np.nan
            df[f"A_inv_{col}"] = np.nan
            continue
        eps = 1e-9
        df[f"A_{col}"] = df[col].apply(lambda v: math.log10((bv+eps)/(v+eps)) if pd.notna(v) and v>0 else np.nan)
        df[f"A_inv_{col}"] = df[col].apply(lambda v: math.log10((v+eps)/(bv+eps)) if pd.notna(v) and v>0 else np.nan)
    return df

# ─── Análisis de regresión y métricas ────────────────────────────────────────
def fit_line(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return None
    m, b, r, _, se = stats.linregress(x, y)
    return {"m": m, "b": b, "r2": r**2, "se": se, "n": len(x),
            "res": y - (m*x + b), "x_fit": x, "y_fit": y}

def calc_sy_x(cal):
    if cal is None or len(cal["res"]) < 3:
        return np.nan
    return np.sqrt(np.sum(cal["res"]**2) / (len(cal["res"]) - 2))

def calc_lod_loq(cal, sy_x):
    if sy_x is None or np.isnan(sy_x):
        return np.nan, np.nan
    m = abs(cal["m"])
    if m < 1e-12:
        return np.nan, np.nan
    return 3.3 * sy_x / m, 10 * sy_x / m

def compute_ida(r2, sy_x, slope, lod, loq, cv=None, w=None):
    """Devuelve los valores sin normalizar; la normalización se hará entre todas las señales."""
    if w is None:
        w = {"r2":0.30, "sy_x":0.25, "slope":0.15, "lod":0.10, "loq":0.10, "cv":0.10}
    return {"r2": r2, "sy_x": sy_x, "slope": slope, "lod": lod, "loq": loq, "cv": cv if cv else np.nan}

def normalize_ida_params(ida_list):
    """Normaliza los parámetros del IDA entre todas las señales para calcular una puntuación 0-100."""
    if not ida_list:
        return ida_list
    df = pd.DataFrame(ida_list)
    for col in ["r2", "slope"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[f"{col}_norm"] = (df[col] - mn) / (mx - mn) * 100
            else:
                df[f"{col}_norm"] = 100
    for col in ["sy_x", "lod", "loq", "cv"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[f"{col}_norm"] = (1 - (df[col] - mn) / (mx - mn)) * 100
            else:
                df[f"{col}_norm"] = 100
    w = {"r2":0.30, "sy_x":0.25, "slope":0.15, "lod":0.10, "loq":0.10, "cv":0.10}
    df["IDA"] = 0
    for k, weight in w.items():
        norm_col = f"{k}_norm"
        if norm_col in df.columns:
            df["IDA"] += df[norm_col] * weight
    return df.to_dict(orient="records")

# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PLOTLY (MODO OSCURO)
# ══════════════════════════════════════════════════════════════════════════════

_PLT = dict(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(family="Inter,sans-serif",color=TEXT,size=11),
            margin=dict(l=52,r=20,t=48,b=48))

def plot_cal(concs, sigs, cal, ch, analyte, unit, lod, loq, ida=None):
    x0=max(0,float(concs.min())*0.85) if float(concs.min())>0 else 0.0
    x1=float(concs.max())*1.15; xl=np.linspace(x0,x1,300)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=concs,y=sigs,mode="markers",
        marker=dict(color=ACCENT,size=10,line=dict(color=PLOT_BG,width=1.5)),name="Estándares"))
    fig.add_trace(go.Scatter(x=xl,y=cal["m"]*xl+cal["b"],mode="lines",
        line=dict(color=SUCCESS,width=2.2),name="Regresión lineal"))
    if not np.isnan(lod): fig.add_vline(x=lod,line_dash="dot",line_color=DANGER,
        annotation_text=f"LOD={lod:.3f}",annotation_font_color=DANGER,annotation_font_size=9)
    if not np.isnan(loq): fig.add_vline(x=loq,line_dash="dot",line_color=WARNING,
        annotation_text=f"LOQ={loq:.3f}",annotation_font_color=WARNING,annotation_font_size=9)
    m,b,r2=cal["m"],cal["b"],cal["r2"]; sgn="+" if b>=0 else "-"
    eq=f"A = {m:.4f}·C {sgn} {abs(b):.4f}   |   R² = {r2:.5f}"
    if ida is not None: eq += f"   |   IDA = {ida:.1f}"
    fig.add_annotation(x=0.03,y=0.97,xref="paper",yref="paper",text=eq,showarrow=False,
        font=dict(color="#4ADE80",size=10,family="JetBrains Mono"),
        bgcolor="rgba(11,17,32,.85)",bordercolor=SUCCESS,borderwidth=1,borderpad=5)
    fig.update_layout(**_PLT,title=f"Curva de calibración — {analyte} | Señal: {ch}",
        xaxis_title=f"Concentración ({unit})",yaxis_title="Absorbancia digital")
    return fig

def plot_residuals(concs,cal,ch):
    yfit=cal["m"]*concs+cal["b"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=yfit,y=cal["res"],mode="markers",
        marker=dict(color=ACCENT,size=8,line=dict(color=PLOT_BG,width=1)),name="Residuos"))
    fig.add_hline(y=0,line_dash="dash",line_color=MUTED,line_width=1)
    fig.update_layout(**_PLT,height=260,title=f"Residuos — Señal: {ch}",
        xaxis_title="Señal predicha",yaxis_title="Residuo (obs - pred)")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICA MATPLOTLIB PARA PDF (CORREGIDO EL PARÁMETRO normalized)
# ══════════════════════════════════════════════════════════════════════════════

def cal_to_png(cal, concs, sigs, ch, analyte, unit, lod, loq, normalized=False):
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
        title_txt = f"y={m:.4f}x {sgn} {abs(b):.4f}  |  R²={r2:.5f}"
        if normalized:
            title_txt += " (señal normalizada)"
        ax.text(0.03,0.97,title_txt,
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

# ══════════════════════════════════════════════════════════════════════════════
#  REPORTE PDF (CORREGIDO EL LLAMADO A cal_to_png)
# ══════════════════════════════════════════════════════════════════════════════

def gen_pdf(analyte,method,df_signals,df_results,cal,annotated_img,tri_df,
            cal_png_bytes,selected_signal,unit="mg/L", ida=None, inversion=False):
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
    hdr=Table([[Paragraph("ELEMENTA v1",ts),
                Paragraph(f"<b>Reporte de análisis colorimétrico</b><br/>"
                          f"<font size='8'>{now}</font><br/>"
                          f"<font size='8'>Analito: {analyte} | Método: {method} | λ ref: 760 nm</font>",
                          ps("HR",textColor=C["txt"],fontSize=9,fontName="Helvetica",alignment=2))]],
              colWidths=[3.0*inch,4.3*inch])
    hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C["bg"]),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,0),(-1,0),2,C["grn"])]))
    story.append(hdr); story.append(Spacer(1,8))
    av=Table([[Paragraph("<b>AVISO:</b> Estimaciones colorimétricas digitales. No sustituyen métodos "
                         "instrumentales certificados.",ws)]], colWidths=[7.3*inch])
    av.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HexColor("#450a0a")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.append(av); story.append(Spacer(1,12))
    if annotated_img is not None:
        story.append(Paragraph("A) Imagen procesada — regiones de interés", h2s))
        pil=Image.fromarray(annotated_img); ib=BytesIO(); pil.save(ib,"PNG"); ib.seek(0)
        story.append(RLImage(ib,width=4.8*inch,height=3.0*inch,kind="proportional"))
        story.append(Spacer(1,8))
    if df_signals is not None and not df_signals.empty:
        story.append(Paragraph("B) Tabla de barrido de señales digitales", h2s))
        story.append(note("La selección se basó en el Índice de Desempeño Analítico (IDA), no únicamente en R²."))
        cols=list(df_signals.columns)
        td=[cols]+[[str(v) for v in row] for _,row in df_signals.iterrows()]
        cw=[1.0*inch]*len(cols)
        story.append(dtbl(td,cw,C["grn"]))
        story.append(Spacer(1,10))
    if cal and cal_png_bytes:
        story.append(Paragraph("C) Curva de calibración final", h2s))
        story.append(RLImage(BytesIO(cal_png_bytes),width=5.8*inch,height=3.2*inch))
        txt = f"Señal seleccionada: {selected_signal}."
        if inversion: txt += " Se aplicó inversión de señal para obtener pendiente positiva."
        story.append(note(txt))
        story.append(Spacer(1,8))
    if cal:
        story.append(Paragraph("D) Resumen analítico", h2s))
        lod=cal.get("LOD",float("nan")); loq=cal.get("LOQ",float("nan"))
        summary_data=[
            ["Parámetro","Valor"],
            ["Analito",analyte],["Método",method],["λ referencia","760 nm"],
            ["Señal seleccionada",selected_signal],
            ["Pendiente (m)",f"{cal['m']:.4f}"],["Intercepto (b)",f"{cal['b']:.4f}"],
            ["R²",f"{cal['r2']:.5f}"],["Sy/x",f"{cal.get('sy_x','N/D')}"],
            ["LOD",f"{lod:.3f}" if not math.isnan(lod) else "N/D"],
            ["LOQ",f"{loq:.3f}" if not math.isnan(loq) else "N/D"],
            ["IDA",f"{ida:.1f}" if ida else "N/D"],
            ["Nº estándares",str(cal.get("n",""))],
            ["Blanco usado",st.session_state.get("blank_label","No especificado")],
            ["Fecha/hora",now],
        ]
        story.append(dtbl(summary_data,[2.5*inch,4.8*inch],C["grn"]))
        story.append(Spacer(1,10))
    if tri_df is not None and not tri_df.empty:
        story.append(Paragraph("Estadísticas de triplicados", h2s))
        cols=list(tri_df.columns)
        td2=[cols]+[[str(v) if v is not None else "N/D" for v in row] for _,row in tri_df.iterrows()]
        cw2=[max(.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dtbl(td2,cw2,C["grn"])); story.append(Spacer(1,10))
    if df_results is not None and not df_results.empty:
        story.append(Paragraph("E) Resultados de cuantificación", h2s))
        cols=list(df_results.columns)
        td3=[cols]+[[f"{v:.3f}" if isinstance(v,float) else str(v) for v in row]
                    for _,row in df_results.iterrows()]
        cw3=[max(.7*inch,7.3*inch/len(cols))]*len(cols)
        story.append(dtbl(td3,cw3)); story.append(Spacer(1,10))
    story.append(HRFlowable(width="100%",thickness=0.5,color=C["brd"]))
    story.append(Spacer(1,5))
    story.append(note("<b>Nota científica:</b> La absorbancia digital se calcula sobre el complemento del color "
                      "cuando es necesario. La selección del modelo se realiza mediante IDA."))
    story.append(Spacer(1,8))
    story.append(Paragraph("Derechos reservados (Katyutzka Villarreal, 2026)  |  Elementa v1",fs))
    doc.build(story); buf.seek(0)
    return buf.read()

def sanitize_filename(name):
    name = name.replace("(", "").replace(")", "").replace(" ", "_")
    name = re.sub(r'[^\w\-_\.]', '', name)
    return name

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init():
    defs=dict(image=None,rois=[],freeze_rois=False,device_type="Viales lineales",
              use_circular=False,global_diam=18,
              assignment_df=None, blank_label=None,
              df_signals=None,df_abs=None,df_merged=None,
              cal_result=None,best_signal="G_norm",all_signals={},tri_groups={},tri_df=None,
              df_results=None,annotated_img=None,
              cal_fig=None,res_fig=None,
              cal_concs=None,cal_sigs=None,cal_unit="mg/L",cal_analyte="",cal_ch="",
              cal_png=None,selected_signal="G_norm",
              assignment_df_backup=None,rois_backup=None,
              original_image=None,
              assignment_editor_data=None,
              ida_df=None)
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
    st.markdown('<div class="footer">Derechos reservados (Katyutzka Villarreal, 2026) | Elementa v1 — Sistema Analítico Colorimétrico Digital</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT};margin:0;font-size:1.5rem;font-weight:700;letter-spacing:-.02em;'>Elementa v1</h2>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:.65rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;margin:2px 0 16px 0;'>Sistema Colorimétrico Digital</p>",unsafe_allow_html=True)
    st.divider()
    pagina=st.radio("Sección",
        ["Tutorial","Análisis","Biblioteca de Métodos","Fundamentos","Normativa"],
        label_visibility="collapsed")
    st.divider()
    st.markdown(f"<p style='color:{MUTED};font-size:.68rem;line-height:1.6;'>Estimaciones colorimétricas digitales. No sustituyen métodos certificados.</p>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA ANÁLISIS (CAPTURA, PROCESAMIENTO, CALIBRACIÓN, REPORTE)
# ══════════════════════════════════════════════════════════════════════════════

if pagina=="Análisis":
    st.markdown("<h1>Análisis Colorimétrico Digital</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};margin-top:-6px;font-size:.85rem;'>Calibración, cuantificación y evaluación normativa por imágenes RGB.</p>",unsafe_allow_html=True)

    tab_cap, tab_proc, tab_cal, tab_rep = st.tabs(["Captura", "Procesamiento", "Calibración", "Reporte"])

    # ── CAPTURA ────────────────────────────────────────────────────────────
    with tab_cap:
        slbl("Paso 1 — Cargar imagen")
        c1,c2=st.columns(2)
        with c1:
            uf=st.file_uploader("Subir imagen",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uf:
                loaded=load_image(uf)
                if loaded is not None:
                    st.session_state["original_image"]=loaded.copy()
                    st.session_state["image"]=loaded.copy()
                    st.session_state["rois"]=[]
                    st.session_state["assignment_editor_data"] = None
                    st.session_state["selected_signal"]="G_norm"
        with c2:
            cam=st.camera_input("Capturar con cámara",label_visibility="collapsed")
            if cam:
                loaded=load_image(cam)
                if loaded is not None:
                    st.session_state["original_image"]=loaded.copy()
                    st.session_state["image"]=loaded.copy()
                    st.session_state["rois"]=[]
                    st.session_state["assignment_editor_data"] = None
                    st.session_state["selected_signal"]="G_norm"

        if st.session_state["original_image"] is None:
            ibox("Cargue o capture una imagen para comenzar.")
            footer(); st.stop()

        # Rotación
        st.markdown("---")
        slbl("Rotación de imagen")
        rotation_angle = st.selectbox("Rotación de imagen", [0, 90, -90, 180, -180], key="rotation_angle")

        def rotate_image(img, angle):
            if angle == 0: return img
            if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if angle == -90: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if angle == 180 or angle == -180: return cv2.rotate(img, cv2.ROTATE_180)

        original = st.session_state["original_image"]
        rotated = rotate_image(original, rotation_angle)
        st.session_state["image"] = rotated
        if st.session_state.get("rois"):
            wbox("Al cambiar la rotación, reajuste las ROIs si es necesario.")

        img = rotated
        H, W = img.shape[:2]

        # Control de calidad
        qc = check_image_quality(img)
        icons = {"ok":"✓","warn":"⚠","fail":"✗"}
        colors_qc = {"ok":SUCCESS,"warn":WARNING,"fail":DANGER}
        qc_parts = []
        for v in qc.values():
            gr=v["grade"]; icon=icons[gr]; col_c=colors_qc[gr]; lbl=v["label"]
            qc_parts.append(f'<span style="color:{col_c};font-size:.78rem;">{icon} {lbl}</span>')
        with st.expander("Control de calidad de imagen", expanded=any(v["grade"]!="ok" for v in qc.values())):
            st.markdown(" &nbsp; ".join(qc_parts), unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0;'>",unsafe_allow_html=True)
        slbl("Paso 2 — Definir regiones de interés (ROIs)")
        ctrl_col,img_col=st.columns([1,1],gap="large")

        with ctrl_col:
            dev=st.selectbox("Tipo de dispositivo",
                             ["Viales lineales","Microplaca de 96 pocillos","Tubos de ensayo (3 ROIs circulares)","Personalizado"],
                             key="dev_sel")
            st.session_state["device_type"]=dev
            is_plate = (dev == "Microplaca de 96 pocillos")
            is_tubes = (dev == "Tubos de ensayo (3 ROIs circulares)")
            use_circular = is_plate or is_tubes
            if not is_plate and not is_tubes:
                use_circular = st.toggle("Usar ROIs circulares", value=st.session_state.get("use_circular",False), key="circ_tog")
            st.session_state["use_circular"] = use_circular
            if use_circular:
                global_diam = st.slider("Diámetro global (px)", 6, 80, st.session_state.get("global_diam",18), key="g_diam")
                st.session_state["global_diam"] = global_diam
            else:
                global_diam = None

            freeze=st.toggle("Bloquear ROIs", value=st.session_state["freeze_rois"], key="frz")
            st.session_state["freeze_rois"]=freeze

            rois = st.session_state.get("rois", [])
            if not freeze:
                if dev=="Viales lineales":
                    n=st.number_input("N de viales",2,24,6,1,key="vn"); x0=st.slider("X inicial",0,W-1,int(W*.05),key="vx0"); y0=st.slider("Y inicial",0,H-1,int(H*.25),key="vy0"); rw=st.slider("Ancho ROI",5,200,40,key="vrw"); rh=st.slider("Alto ROI",5,300,60,key="vrh"); dx=st.slider("Espaciado X",0,300,int(W*.08),key="vdx"); dy=st.slider("Espaciado Y",0,300,0,key="vdy")
                    rois=gen_rois_linear(x0,y0,rw,rh,int(n),dx,dy)
                elif is_plate:
                    diam = st.session_state.get("global_diam", 60); n_rows=st.number_input("Filas",1,8,8,1,key="prows"); n_cols=st.number_input("Columnas",1,12,12,1,key="pcols"); x0=st.slider("X inicial",0,W-1,318,key="px0"); y0=st.slider("Y inicial",0,H-1,228,key="py0"); dx=st.slider("Espaciado X",10,300,170,key="pdx"); dy=st.slider("Espaciado Y",10,300,170,key="pdy")
                    rois=gen_rois_plate(x0,y0,diam,diam,dx,dy,int(n_rows),int(n_cols))
                elif is_tubes:
                    radius = st.slider("Radio del círculo (px)", 5, 50, 15, key="t_radius"); h = st.slider("Altura del tubo (px)", 50, 500, 200, key="t_h"); ntubes = st.number_input("Número de tubos", 1, 12, 6, 1, key="t_ntubes"); x0 = st.slider("X inicial (centro primer tubo)", 0, W-1, 100, key="t_x0"); y0 = st.slider("Y inicial (borde superior)", 0, H-1, 100, key="t_y0"); dx = st.slider("Espaciado X entre tubos", 20, 300, 120, key="t_dx")
                    rois=gen_rois_tubes(x0,y0,radius,h,int(ntubes),dx)
                else:
                    n=st.number_input("N de ROIs",2,50,6,1,key="cn"); x0=st.slider("X inicial",0,W-1,int(W*.05),key="cx0"); y0=st.slider("Y inicial",0,H-1,int(H*.1), key="cy0"); rw=st.slider("Ancho ROI",5,200,30,key="crw"); rh=st.slider("Alto ROI",5,200,30,key="crh"); dx=st.slider("Espaciado X",0,300,int(W*.08),key="cdx"); dy=st.slider("Espaciado Y",0,300,int(H*.08),key="cdy")
                    rois=gen_rois_linear(x0,y0,rw,rh,int(n),dx,dy)

                st.session_state["rois"]=rois
                st.session_state["rois_backup"]=[r.copy() for r in rois]
                current_labels = [r["label"] for r in rois]
                if st.session_state.get("assignment_editor_data") is None or \
                   list(st.session_state["assignment_editor_data"]["ROI"]) != current_labels:
                    df = pd.DataFrame([{"ROI":label,"Tipo":"Sin asignar","Nombre":"",
                                        "Concentracion":0.0,"Unidad":"mg/L",
                                        "Factor_dil":1.0,"Analito":"Fenólicos totales","Observaciones":""}
                                       for label in current_labels])
                    st.session_state["assignment_editor_data"] = df
                    st.session_state["assignment_df"] = df.copy()
            else:
                if not rois and st.session_state.get("rois_backup"):
                    rois = st.session_state["rois_backup"]
                    st.session_state["rois"] = rois
                    wbox("ROIs recuperadas del respaldo.")
                if rois:
                    okbox(f"ROIs bloqueadas — {len(rois)} regiones.")
                else:
                    wbox("No hay ROIs definidas. Desactive el bloqueo para configurar.")

        with img_col:
            rois = st.session_state.get("rois", [])
            use_circ = st.session_state.get("use_circular", False)
            diam_map = {r["label"]: st.session_state.get("global_diam", 18) for r in rois} if not is_tubes and use_circ else None
            if rois:
                try:
                    tm = {}
                    if st.session_state.get("assignment_df") is not None:
                        tm = dict(zip(st.session_state["assignment_df"]["ROI"],
                                      st.session_state["assignment_df"]["Tipo"]))
                    ann = draw_rois(img, rois, tm, circular=use_circ, diam_map=diam_map)
                    st.session_state["annotated_img"] = ann
                    st.image(ann, caption=f"Overlay {'circular' if use_circ else 'rectangular'}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error dibujando ROIs: {e}")
                    st.image(img, use_container_width=True)
            else:
                st.image(img, caption="Imagen original", use_container_width=True)

        if not st.session_state.get("rois"):
            ibox("Configure las ROIs para continuar.")
            footer(); st.stop()
        footer()

    # ── PROCESAMIENTO ──────────────────────────────────────────────────────
    with tab_proc:
        rois = st.session_state.get("rois", [])
        if not rois and st.session_state.get("rois_backup"):
            rois = st.session_state["rois_backup"]
            st.session_state["rois"] = rois
            wbox("ROIs recuperadas del respaldo.")
        img = st.session_state.get("image")
        if not rois or img is None:
            wbox("Defina las ROIs en Captura primero.")
            footer(); st.stop()

        if st.session_state.get("assignment_editor_data") is None:
            current_labels = [r["label"] for r in rois]
            df = pd.DataFrame([{"ROI":label,"Tipo":"Sin asignar","Nombre":"",
                                "Concentracion":0.0,"Unidad":"mg/L",
                                "Factor_dil":1.0,"Analito":"Fenólicos totales","Observaciones":""}
                               for label in current_labels])
            st.session_state["assignment_editor_data"] = df
            st.session_state["assignment_df"] = df.copy()

        slbl("Paso 3 — Asignar tipos y concentraciones")
        ibox("Los datos se guardan automáticamente y nunca se pierden.")

        edited = st.data_editor(
            st.session_state["assignment_editor_data"],
            column_config={
                "Tipo":    st.column_config.SelectboxColumn("Tipo",   options=TIPOS,   required=True),
                "Unidad":  st.column_config.SelectboxColumn("Unidad", options=UNIDADES,required=True),
                "Analito": st.column_config.SelectboxColumn("Analito",options=ANALITOS,required=True),
                "Concentracion":st.column_config.NumberColumn("Conc.",min_value=0.0,step=0.001,format="%.4f"),
                "Factor_dil":   st.column_config.NumberColumn("F.Dil",min_value=0.01,step=0.1,format="%.2f"),
            },
            num_rows="fixed",
            use_container_width=True,
            key="asgn_editor"
        )

        st.session_state["assignment_editor_data"] = edited
        st.session_state["assignment_df"] = edited.copy()
        st.session_state["assignment_df_backup"] = edited.copy()

        blank_r = edited[edited["Tipo"]=="Blanco"]
        blank = blank_r["ROI"].iloc[0] if not blank_r.empty else None
        st.session_state["blank_label"] = blank
        if blank: okbox(f"Blanco: <b>{blank}</b>")
        else: wbox("Sin blanco asignado.")

        use_circ = st.session_state.get("use_circular", False)
        diam_map = {r["label"]: st.session_state.get("global_diam", 18) for r in rois} if use_circ else None
        tm2 = dict(zip(edited["ROI"], edited["Tipo"]))
        try:
            ann2 = draw_rois(img, rois, tm2, circular=use_circ, diam_map=diam_map)
            st.session_state["annotated_img"] = ann2
            st.image(ann2, caption="Overlay actualizado", use_container_width=True)
        except Exception as e:
            st.error(f"Error al dibujar ROIs: {e}")
            st.image(img, use_container_width=True)

        # Grid tipo Excel para microplaca
        dev = st.session_state.get("device_type", "")
        if dev == "Microplaca de 96 pocillos":
            st.markdown("### Mapa de placa")
            fig_grid = plot_plate_grid(edited)
            st.plotly_chart(fig_grid, use_container_width=True, key="plate_grid")

        footer()

    # ── CALIBRACIÓN (BARRIDO COMPLETO E IDA) ──────────────────────────────
    with tab_cal:
        rois=st.session_state.get("rois",[]); img=st.session_state.get("image")
        adf=st.session_state.get("assignment_df")
        if not rois or img is None or adf is None:
            wbox("Complete Captura y Procesamiento primero."); footer(); st.stop()
        blank=st.session_state.get("blank_label")
        device = st.session_state.get("device_type","")
        is_tubes = (device == "Tubos de ensayo (3 ROIs circulares)")

        with st.expander("Fundamento — absorbancia digital",expanded=False):
            st.markdown("**A = log₁₀(I_blanco / I_muestra)** y su versión invertida si es necesario.")

        if st.button("Extraer señales y barrer",key="btn_cal"):
            with st.spinner("Procesando todas las señales digitales..."):
                use_circ = st.session_state.get("use_circular",False)
                df_signals = extract_all_signals(img, rois, circular=use_circ)
                blank_row = df_signals[df_signals["ROI"]==blank] if blank else None
                df_signals = add_euclidean_distance(df_signals, blank_row)

                signal_columns = [
                    "R","G","B",
                    "R_norm","G_norm","B_norm",
                    "R+G","R+B","G+B","R+G+B",
                    "R_norm+G_norm","R_norm+B_norm","G_norm+B_norm",
                    "H","S","V","L","a","b_lab",
                    "ED","ED_norm"
                ]
                df_signals = compute_absorbances(df_signals, blank, signal_columns)

                df_merged = df_signals.merge(
                    adf[["ROI","Tipo","Nombre","Concentracion","Unidad","Analito","Factor_dil"]],
                    on="ROI",how="left")

                std = df_merged[df_merged["Tipo"]=="Estándar"]
                if len(std)>=2 and "Concentracion" in std.columns:
                    concs = std["Concentracion"].values.astype(float)
                    ida_list = []
                    all_signals = {}
                    for col in signal_columns:
                        ac = f"A_{col}"
                        ac_inv = f"A_inv_{col}"
                        if ac not in df_merged.columns:
                            continue
                        sigs = std[ac].dropna().values
                        if len(sigs)>=2:
                            cal = fit_line(concs, sigs)
                            if cal:
                                sy_x = calc_sy_x(cal)
                                lod, loq = calc_lod_loq(cal, sy_x)
                                ida_raw = compute_ida(cal["r2"], sy_x, abs(cal["m"]), lod, loq)
                                ida_raw["signal"] = col
                                ida_raw["type"] = "Absorbancia clásica"
                                ida_raw["m_orig"] = cal["m"]
                                ida_raw["m_final"] = cal["m"]
                                ida_raw["inverted"] = False
                                ida_list.append(ida_raw)
                                all_signals[col] = {"cal":cal, "sigs":sigs, "lod":lod, "loq":loq, "sy_x":sy_x, "inverted":False}
                        if cal and cal["m"] < 0 and ac_inv in df_merged.columns:
                            sigs_inv = std[ac_inv].dropna().values
                            if len(sigs_inv)>=2:
                                cal_inv = fit_line(concs, sigs_inv)
                                if cal_inv:
                                    sy_x_inv = calc_sy_x(cal_inv)
                                    lod_inv, loq_inv = calc_lod_loq(cal_inv, sy_x_inv)
                                    ida_inv = compute_ida(cal_inv["r2"], sy_x_inv, abs(cal_inv["m"]), lod_inv, loq_inv)
                                    ida_inv["signal"] = col
                                    ida_inv["type"] = "Absorbancia invertida"
                                    ida_inv["m_orig"] = cal["m"]
                                    ida_inv["m_final"] = cal_inv["m"]
                                    ida_inv["inverted"] = True
                                    ida_list.append(ida_inv)
                                    all_signals[col+"_inv"] = {"cal":cal_inv, "sigs":sigs_inv, "lod":lod_inv, "loq":loq_inv, "sy_x":sy_x_inv, "inverted":True}

                    if ida_list:
                        ida_normalized = normalize_ida_params(ida_list)
                        best = max(ida_normalized, key=lambda x: x["IDA"])
                        best_signal = best["signal"] + ("_inv" if best["inverted"] else "")
                        best_info = all_signals[best_signal]
                        st.session_state["ida_df"] = ida_normalized
                        st.session_state["all_signals"] = all_signals
                        st.session_state["best_signal"] = best_signal
                        st.session_state["selected_signal"] = best_signal
                        st.session_state["df_signals"] = df_signals
                        st.session_state["df_merged"] = df_merged
                        st.session_state["cal_result"] = best_info["cal"]
                        st.session_state["cal_concs"] = concs
                        st.session_state["cal_sigs"] = best_info["sigs"]
                        st.session_state["cal_lod"] = best_info["lod"]
                        st.session_state["cal_loq"] = best_info["loq"]
                        st.session_state["cal_sy_x"] = best_info["sy_x"]
                        st.session_state["cal_inverted"] = best_info["inverted"]
                        st.session_state["cal_ida"] = best["IDA"]
                        st.success(f"Barrido completado. Mejor señal: **{best_signal}** con IDA = {best['IDA']:.1f}")

        ida_df = st.session_state.get("ida_df")
        if ida_df is not None:
            st.markdown("### Tabla comparativa de señales digitales")
            df_show = pd.DataFrame(ida_df)
            display_cols = ["signal","type","m_orig","m_final","r2","sy_x","lod","loq","IDA","inverted"]
            st.dataframe(df_show[display_cols].round(4), use_container_width=True)

            all_signals = st.session_state.get("all_signals", {})
            signal_options = list(all_signals.keys())
            sel_manual = st.selectbox("Seleccionar manualmente otra señal", options=signal_options,
                                      index=signal_options.index(st.session_state.get("selected_signal", signal_options[0])))
            if sel_manual != st.session_state.get("selected_signal"):
                st.session_state["selected_signal"] = sel_manual
                info = all_signals[sel_manual]
                st.session_state["cal_result"] = info["cal"]
                st.session_state["cal_sigs"] = info["sigs"]
                st.session_state["cal_lod"] = info["lod"]
                st.session_state["cal_loq"] = info["loq"]
                st.session_state["cal_sy_x"] = info["sy_x"]
                st.session_state["cal_inverted"] = info["inverted"]

            cal = st.session_state.get("cal_result")
            if cal:
                concs = st.session_state["cal_concs"]
                sigs = st.session_state["cal_sigs"]
                ch = st.session_state["selected_signal"]
                unit = st.session_state.get("cal_unit","mg/L")
                lod = st.session_state.get("cal_lod", np.nan)
                loq = st.session_state.get("cal_loq", np.nan)
                ida_val = st.session_state.get("cal_ida", None)
                fig = plot_cal(concs, sigs, cal, ch, "Fenólicos totales", unit, lod, loq, ida_val)
                st.plotly_chart(fig, use_container_width=True)
                slope_msg, slope_col = interpret_slope(cal["m"])
                st.markdown(f'<div style="background:{PRIMARY};border-left:3px solid {slope_col};padding:10px;margin:8px 0;">{slope_msg}</div>', unsafe_allow_html=True)
                if st.session_state.get("cal_inverted"):
                    st.info("Se utilizó la absorbancia invertida para obtener una pendiente positiva.")

        st.markdown(f"<hr style='border-color:{BORDER};margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Cuantificación de muestras")
        if st.button("Calcular concentraciones",key="btn_q"):
            cal = st.session_state.get("cal_result")
            dm = st.session_state.get("df_merged")
            ch = st.session_state.get("selected_signal","G_norm")
            prefix = "A_inv_" if st.session_state.get("cal_inverted") else "A_"
            ac = f"{prefix}{ch.replace('_inv','')}"
            if cal is None or dm is None:
                st.error("Ejecute primero el barrido de señales.")
            else:
                m,b=cal["m"],cal["b"]
                samples=dm[dm["Tipo"]=="Muestra"].copy(); res=[]
                for _,row in samples.iterrows():
                    a=row.get(ac,float("nan")); dil=float(row.get("Factor_dil",1) or 1)
                    c_r=(a-b)/m if not math.isnan(a) and abs(m)>1e-12 else float("nan")
                    c_c=c_r*dil if not math.isnan(c_r) else float("nan")
                    res.append({"Muestra":str(row.get("Nombre","")) or row["ROI"],"ROI":row["ROI"],
                                "Señal":ch,"A_digital":round(a,4) if not math.isnan(a) else None,
                                "Conc_calc":round(c_r,3) if not math.isnan(c_r) else None,
                                "Factor_dil":dil,"Conc_corregida":round(c_c,3) if not math.isnan(c_c) else None,
                                "Unidad":str(row.get("Unidad","mg/L")),"Analito":str(row.get("Analito",""))})
                df_res=pd.DataFrame(res); st.session_state["df_results"]=df_res
                st.dataframe(df_res,use_container_width=True,hide_index=True)
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("Descargar resultados CSV", csv, "elementa_resultados.csv", "text/csv")
                full_csv = st.session_state.get("df_signals")
                if full_csv is not None:
                    full_csv = full_csv.merge(adf[["ROI","Tipo","Concentracion","Unidad"]], on="ROI", how="left")
                    csv_raw = full_csv.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar datos crudos completos", csv_raw, "elementa_datos_crudos.csv", "text/csv")
        footer()

    # ── REPORTE ────────────────────────────────────────────────────────────
    with tab_rep:
        slbl("Evaluación normativa")
        wbox("Verificar siempre los límites en la versión oficial vigente (DOF).")
        df_res=st.session_state.get("df_results")
        if df_res is not None and not df_res.empty:
            for _,row in df_res.iterrows():
                try:
                    cv=float(row["Conc_corregida"]); an=str(row["Analito"])
                    st.markdown(f"**{row['Muestra']}** — {an}: `{cv:.3f} {row['Unidad']}`")
                except: pass
        else:
            ibox("Complete la cuantificación en la pestaña Calibración.")

        st.markdown(f"<hr style='border-color:{BORDER};margin:20px 0;'>",unsafe_allow_html=True)
        slbl("Exportar reporte PDF")
        if st.button("Generar reporte PDF",key="btn_pdf"):
            cal = st.session_state.get("cal_result")
            ida_val = st.session_state.get("cal_ida")
            inverted = st.session_state.get("cal_inverted", False)
            if cal:
                concs = st.session_state["cal_concs"]
                sigs = st.session_state["cal_sigs"]
                ch = st.session_state["selected_signal"]
                unit = st.session_state.get("cal_unit","mg/L")
                lod = st.session_state.get("cal_lod", np.nan)
                loq = st.session_state.get("cal_loq", np.nan)
                # CORREGIDO: se agregó el argumento normalized=False
                cal_png = cal_to_png(cal, concs, sigs, ch, "Fenólicos totales", unit, lod, loq, normalized=False)
            else:
                cal_png = None
            try:
                pdf_b = gen_pdf(
                    analyte="Fenólicos totales",
                    method="Folin-Ciocalteu (760 nm)",
                    df_signals=pd.DataFrame(st.session_state.get("ida_df",[])),
                    df_results=df_res,
                    cal=cal,
                    annotated_img=st.session_state.get("annotated_img"),
                    tri_df=st.session_state.get("tri_df"),
                    cal_png_bytes=cal_png,
                    selected_signal=st.session_state.get("selected_signal",""),
                    unit=st.session_state.get("cal_unit","mg/L"),
                    ida=ida_val,
                    inversion=inverted
                )
                b64 = base64.b64encode(pdf_b).decode()
                safe_analyte = sanitize_filename("Fenólicos_totales")
                fname = f"Elementa_v1_{safe_analyte}_{now_mx():%Y%m%d_%H%M}.pdf"
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
    st.markdown(f"<p style='color:{MUTED};'>Siga estos pasos para realizar su primer análisis colorimétrico con Elementa v1.</p>",unsafe_allow_html=True)
    steps=[
        ("Paso 1 — Preparación de estándares y muestras",
         "Prepare estándares de fenólicos totales (ácido gálico) y un blanco de reactivos."),
        ("Paso 2 — Captura de imagen",
         "Use iluminación LED blanca difusa, fondo neutro, cámara paralela. Formato PNG."),
        ("Paso 3 — Definición de ROIs",
         "Ajuste las ROIs sobre los pocillos o tubos. Bloquéelas cuando estén alineadas."),
        ("Paso 4 — Asignación de tipos",
         "Asigne Blanco, Estándar y Muestra. Ingrese concentraciones conocidas."),
        ("Paso 5 — Barrido de señales",
         "Presione 'Extraer señales y barrer'. Revise la tabla comparativa y el IDA. Seleccione el mejor modelo."),
        ("Paso 6 — Cuantificación y reporte",
         "Calcule concentraciones y exporte el PDF con todos los detalles analíticos."),
    ]
    for title, content in steps:
        with st.expander(title, expanded=False):
            st.markdown(content)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
#  BIBLIOTECA DE MÉTODOS
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Biblioteca de Métodos":
    st.markdown("<h1>Biblioteca de Métodos Analíticos</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};'>Protocolos colorimétricos preconfigurados.</p>",unsafe_allow_html=True)
    cat=st.radio("Categoría",list(PROTOCOL_LIBRARY.keys()),horizontal=True)
    st.divider()
    for nombre, proto in PROTOCOL_LIBRARY[cat].items():
        with st.expander(f"{nombre}   |   λ = {proto['lambda_ref']} nm   |   Canal sugerido: {proto['canal']}", expanded=False):
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
                    f'<p class="val" style="font-size:.9rem;color:{SUCCESS};">{proto["canal"]}</p>'
                    f'<p class="lbl" style="margin-top:8px;">λ referencia</p><p class="val" style="font-size:1rem;">{proto["lambda_ref"]} nm</p>'
                    f'</div>',
                    unsafe_allow_html=True)
    footer()

# ══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTOS
# ══════════════════════════════════════════════════════════════════════════════

elif pagina=="Fundamentos":
    st.markdown("<h1>Fundamentos del análisis colorimétrico digital</h1>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};'>Base científica de Elementa v1.</p>",unsafe_allow_html=True)
    topics = {
        "Absorbancia digital": "A = log₁₀(I_blanco / I_muestra) sobre el complemento del color o la señal directa.",
        "Barrido de señales": "Se evalúan RGB, combinaciones, HSV, CIELAB y distancia euclidiana.",
        "Índice de Desempeño Analítico (IDA)": "Combina R², Sy/x, pendiente, LOD, LOQ y %CV para seleccionar el mejor modelo.",
        "Pendientes negativas": "Se interpretan sin eliminarlas; se puede usar la absorbancia invertida para visualización.",
        "Referencia UV-Vis": "Para fenólicos totales, λ = 760 nm (Folin-Ciocalteu).",
    }
    for title, content in topics.items():
        with st.expander(title, expanded=False):
            st.markdown(content)
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
          ("Singleton & Rossi (1965)","Folin-Ciocalteu method for total phenolics.",""),
          ("IARC Monographs Vol. 49","Chromium, Nickel and Welding.","https://monographs.iarc.who.int")]
    for r,t,u in refs:
        if u: st.markdown(f"- **{r}**: {t} — [Ver]({u})")
        else: st.markdown(f"- **{r}**: {t}")
    st.divider()
    slbl("Editar límites normativos (sesión actual)")
    ibox("Cambios solo para esta sesión.")
    ed_rows=[{"Analito":a,"Norma":n,"Limite_mg_L":l} for a,ns in NORMATIVE_LIMITS.items() for n,l in ns.items()]
    ed_df=st.data_editor(pd.DataFrame(ed_rows),column_config={"Limite_mg_L":st.column_config.NumberColumn("Límite (mg/L)",min_value=0.0,step=0.001,format="%.4f")},num_rows="fixed",use_container_width=True,key="norm_ed")
    if st.button("Aplicar cambios"):
        NORMATIVE_LIMITS.clear()
        for _,row in ed_df.iterrows():
            NORMATIVE_LIMITS.setdefault(row["Analito"],{})[row["Norma"]]=row["Limite_mg_L"]
        okbox("Límites actualizados para esta sesión.")
    footer()
