# Elementa - Digital Colorimetry App (Paper-Ready Version)
# Author: Katyutzka Villarreal (2026 💚)
# Refactored for Streamlit Cloud deployment

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os

# ── Lazy-loaded heavy imports (speeds up cold start) ──────────────────────────
@st.cache_resource
def _load_cv2():
    import cv2
    return cv2

@st.cache_resource
def _load_scipy_stats():
    from scipy import stats
    return stats

@st.cache_resource
def _load_plotly():
    import plotly.graph_objects as go
    return go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Elementa", layout="wide")
st.title("🧪 Elementa — Scientific Digital Colorimetry")

# ── Sidebar ───────────────────────────────────────────────────────────────────
mode   = st.sidebar.selectbox("Mode",   ["Calibration Curve", "Standard Addition"])
device = st.sidebar.selectbox("Device", ["Vials", "Microplate"])
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
csv_file      = st.sidebar.file_uploader("Upload CSV (Concentrations)", type=["csv"])

# ── Image utilities ───────────────────────────────────────────────────────────

def resize_for_processing(img, max_dim: int = 1200):
    """Downscale large images before heavy CV ops — prevents HoughCircles hangs."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        cv2 = _load_cv2()
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


@st.cache_data(show_spinner="Correcting perspective…")
def correct_perspective(img_bytes: bytes):
    """
    Detect the largest quadrilateral and apply a homography warp.
    Falls back to the original image if no quad is found.
    """
    cv2 = _load_cv2()
    from PIL import Image
    import io

    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    img = resize_for_processing(img)

    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            pts    = approx.reshape(4, 2).astype("float32")
            pts    = pts[np.argsort(pts[:, 1])]
            top    = pts[:2][np.argsort(pts[:2, 0])]
            bottom = pts[2:][np.argsort(pts[2:, 0])]
            ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
            w = int(max(np.linalg.norm(ordered[0] - ordered[1]),
                        np.linalg.norm(ordered[2] - ordered[3])))
            h = int(max(np.linalg.norm(ordered[0] - ordered[3]),
                        np.linalg.norm(ordered[1] - ordered[2])))
            dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")
            M   = cv2.getPerspectiveTransform(ordered, dst)
            return cv2.warpPerspective(img, M, (w, h))

    st.warning("⚠️ No quadrilateral found — skipping perspective correction.")
    return img


# ── ROI detection ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Detecting ROIs…")
def detect_vials(img_hash, img: np.ndarray):
    """Detect circular vials. img_hash is only used as a cache key."""
    cv2 = _load_cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=150
    )
    if circles is None:
        st.error("No circular vials detected. Try adjusting the image or switching to Microplate mode.")
        return []
    rois = []
    for (x, y, r) in np.round(circles[0]).astype(int):
        rois.append((int(x - r), int(y - r), int(2 * r), int(2 * r)))
    return sorted(rois, key=lambda b: b[0])


def detect_microplate(img: np.ndarray, rows: int = 8, cols: int = 12):
    h, w = img.shape[:2]
    sx, sy = w // cols, h // rows
    return [(j * sx, i * sy, sx, sy) for i in range(rows) for j in range(cols)]


# ── Feature extraction ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Extracting colour features…")
def extract_features(img_hash, img: np.ndarray, rois):
    cv2  = _load_cv2()
    rois = list(rois)
    lab_img  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    rgb_list, lab_list = [], []

    for (x, y, w, h) in rois:
        roi_rgb = img[y:y+h, x:x+w]
        roi_lab = lab_img[y:y+h, x:x+w]
        if roi_rgb.size == 0:
            rgb_list.append([1.0, 1.0, 1.0])
            lab_list.append([1.0, 1.0, 1.0])
        else:
            rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
            rgb = (rgb / (rgb.sum() + 1e-6)) * 100
            lab = np.mean(roi_lab.reshape(-1, 3), axis=0)
            rgb_list.append(rgb.tolist())
            lab_list.append(lab.tolist())

    return np.array(rgb_list), np.array(lab_list)


# ── Analytical functions ──────────────────────────────────────────────────────

def absorbance(blank: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute absorbance; guards against log(0) and negative values."""
    blank   = np.clip(blank,   1e-6, None)
    samples = np.clip(samples, 1e-6, None)
    return np.log10(blank / samples)


def compute_syx(x, y, m, b) -> float:
    n = len(x)
    if n <= 2:
        return float("nan")
    return float(np.sqrt(np.sum((y - (m * x + b)) ** 2) / (n - 2)))


def analyze(data: np.ndarray, conc: np.ndarray) -> dict:
    stats = _load_scipy_stats()
    res   = {}
    for i, ch in enumerate(["C1", "C2", "C3"]):
        y = data[:, i]
        m, b, r, _, _ = stats.linregress(conc, y)
        syx = compute_syx(conc, y, m, b)
        res[ch] = {"m": m, "b": b, "r2": r ** 2, "syx": syx}
    return res


def best_channel(res: dict) -> str:
    return sorted(res.items(), key=lambda x: (-x[1]["r2"], x[1]["syx"]))[0][0]


def validation_metrics(y, y_pred):
    residuals = y - y_pred
    return float(np.sqrt(np.mean(residuals ** 2))), float(np.mean(np.abs(residuals)))


# ── PDF export (cloud-safe) ───────────────────────────────────────────────────

def build_pdf(r2, lod, loq, rmse, mae, plot_fig) -> bytes:
    """Generate PDF in a temp dir and return bytes — works on read-only FS."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save plot as PNG using kaleido; fall back to omitting the image
        plot_path = os.path.join(tmpdir, "plot.png")
        pdf_path  = os.path.join(tmpdir, "report.pdf")

        include_plot = False
        try:
            plot_fig.write_image(plot_path)
            include_plot = True
        except Exception:
            st.warning("kaleido not available — PDF will be generated without the chart.")

        doc    = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        body   = [
            Paragraph("Elementa Analytical Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"R²:   {r2:.4f}",   styles["Normal"]),
            Paragraph(f"LOD:  {lod:.4f}",  styles["Normal"]),
            Paragraph(f"LOQ:  {loq:.4f}",  styles["Normal"]),
            Paragraph(f"RMSE: {rmse:.4f}", styles["Normal"]),
            Paragraph(f"MAE:  {mae:.4f}",  styles["Normal"]),
            Spacer(1, 12),
        ]
        if include_plot:
            body.append(RLImage(plot_path, width=400, height=300))

        doc.build(body)

        with open(pdf_path, "rb") as f:
            return f.read()


# ── Main ──────────────────────────────────────────────────────────────────────

if uploaded_file and csv_file:

    file_bytes = uploaded_file.read()   # read once; pass bytes to cached fns
    img = correct_perspective(file_bytes)

    df   = pd.read_csv(csv_file)
    conc = df.iloc[:, 0].values.astype(float)

    # ── ROI detection ─────────────────────────────────────────────────────────
    img_hash = hash(file_bytes)          # stable cache key
    if device == "Vials":
        rois = detect_vials(img_hash, img)
    else:
        rois = detect_microplate(img)

    if not rois:
        st.stop()

    st.write(f"Detected ROIs: **{len(rois)}**")

    # Draw bounding boxes
    cv2 = _load_cv2()
    img_draw = img.copy()
    for i, (x, y, w, h) in enumerate(rois):
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_draw, str(i), (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    st.image(img_draw, use_container_width=True)

    # ── Feature extraction ────────────────────────────────────────────────────
    _, lab_data = extract_features(img_hash, img, tuple(rois))

    n    = min(len(conc), len(lab_data))
    conc = conc[:n]
    data = lab_data[:n]

    blank_idx = st.selectbox("Select Blank ROI index", range(n))
    blank     = data[blank_idx]
    abs_data  = absorbance(blank, data)

    results = analyze(abs_data, conc)
    best    = best_channel(results)

    m, b = results[best]["m"], results[best]["b"]
    idx  = ["C1", "C2", "C3"].index(best)
    y    = abs_data[:, idx]

    # ── Standard Addition ─────────────────────────────────────────────────────
    if mode == "Standard Addition":
        x0 = -b / m if m != 0 else float("nan")
        st.subheader("Standard Addition — Unknown Concentration")
        st.metric("Cx", f"{abs(x0):.4f}" if not np.isnan(x0) else "N/A")

    # ── Plot ──────────────────────────────────────────────────────────────────
    go = _load_plotly()
    x_line = np.linspace(conc.min(), conc.max(), 100)
    y_line = m * x_line + b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc,   y=y,      mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",   name="Linear Fit"))
    fig.update_layout(
        xaxis_title="Concentration",
        yaxis_title="Absorbance",
        title=f"Best channel: {best}  |  R² = {results[best]['r2']:.4f}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Analytical figures of merit ───────────────────────────────────────────
    y_pred      = m * conc + b
    rmse, mae   = validation_metrics(y, y_pred)
    syx         = results[best]["syx"]
    LOD = 3.3 * syx / m  if m != 0 else float("nan")
    LOQ = 10.0 * syx / m if m != 0 else float("nan")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Best Channel", best)
    col2.metric("R²",   f"{results[best]['r2']:.4f}")
    col3.metric("LOD",  f"{LOD:.4f}"  if not np.isnan(LOD)  else "N/A")
    col4.metric("LOQ",  f"{LOQ:.4f}"  if not np.isnan(LOQ)  else "N/A")
    col5.metric("RMSE", f"{rmse:.4f}")

    # ── Residuals table ───────────────────────────────────────────────────────
    with st.expander("Residuals table"):
        resid_df = pd.DataFrame({
            "Conc": conc,
            "Abs (measured)": y,
            "Abs (fitted)": y_pred,
            "Residual": y - y_pred,
        })
        st.dataframe(resid_df.style.format("{:.5f}"), use_container_width=True)

    # ── Exports ───────────────────────────────────────────────────────────────
    st.divider()
    col_csv, col_pdf = st.columns(2)

    with col_csv:
        out = pd.DataFrame(abs_data, columns=["C1", "C2", "C3"])
        out.insert(0, "Conc", conc)
        st.download_button(
            "⬇️ Download CSV",
            data=out.to_csv(index=False),
            file_name="elementa_results.csv",
            mime="text/csv",
        )

    with col_pdf:
        if st.button("Generate PDF report"):
            with st.spinner("Building PDF…"):
                pdf_bytes = build_pdf(
                    results[best]["r2"], LOD, LOQ, rmse, mae, fig
                )
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name="elementa_report.pdf",
                mime="application/pdf",
            )

else:
    st.info("👈 Upload an image **and** a concentration CSV to get started.")

st.divider()
st.caption("All rights reserved · Katyutzka Villarreal 2026 💚")
