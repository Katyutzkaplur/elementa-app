# Elementa - Integrated Scientific Platform (Full UI + Analysis Engine)
# Author: Katyutzka Villarreal (2026)

import streamlit as st
import numpy as np
import pandas as pd

# Lazy imports
@st.cache_resource
def load_cv2():
    import cv2
    return cv2

@st.cache_resource
def load_stats():
    from scipy import stats
    return stats

@st.cache_resource
def load_plotly():
    import plotly.graph_objects as go
    return go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Elementa", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
h1, h2, h3 { font-weight: 700; letter-spacing: 1px; }
.stApp { background-color: #0b0f19; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
menu = st.sidebar.radio("Navigation", [
    "Analysis",
    "Microplate Designer",
    "Environmental Mode",
    "Theory",
    "Sources"
])

# ---------------- CORE FUNCTIONS ----------------

def absorbance(blank, samples):
    return np.log10((blank+1e-6)/(samples+1e-6))


def compute_syx(x,y,m,b):
    return np.sqrt(np.sum((y-(m*x+b))**2)/(len(x)-2))


def analyze(data, conc):
    stats = load_stats()
    res = {}
    for i,ch in enumerate(["C1","C2","C3"]):
        y = data[:,i]
        m,b,r,_,_ = stats.linregress(conc,y)
        syx = compute_syx(conc,y,m,b)
        res[ch] = {"m":m,"b":b,"r2":r**2,"syx":syx}
    return res


def best_channel(res):
    return sorted(res.items(), key=lambda x:(-x[1]['r2'],x[1]['syx']))[0][0]

# ---------------- ANALYSIS ----------------
if menu == "Analysis":

    st.title("Elementa")
    st.subheader("Digital Colorimetry Analysis")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg","png"])

    conc_input = st.text_area("Concentrations", "0,0.2,0.4,0.6,0.8,1.0")

    try:
        conc = np.array([float(x) for x in conc_input.split(",")])
    except:
        st.error("Invalid concentration format")
        conc = None

    if uploaded_image and conc is not None:
        from PIL import Image
        img = np.array(Image.open(uploaded_image))

        cv2 = load_cv2()
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Simple grid (fallback)
        h,w,_ = img.shape
        step = w//len(conc)

        data = []
        for i in range(len(conc)):
            roi = lab[:, i*step:(i+1)*step]
            val = np.mean(roi.reshape(-1,3), axis=0)
            data.append(val)

        data = np.array(data)

        blank_idx = st.selectbox("Blank index", range(len(conc)))
        blank = data[blank_idx]

        abs_data = absorbance(blank, data)

        results = analyze(abs_data, conc)
        best = best_channel(results)

        idx = ["C1","C2","C3"].index(best)
        y = abs_data[:, idx]

        go = load_plotly()
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=conc, y=y, mode='markers'))

        m = results[best]['m']
        b = results[best]['b']

        x_line = np.linspace(min(conc), max(conc),100)
        fig.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode='lines'))

        st.plotly_chart(fig, use_container_width=True)

        st.metric("R²", f"{results[best]['r2']:.4f}")

# ---------------- MICROPLATE ----------------
elif menu == "Microplate Designer":

    st.title("Microplate")

    plate = pd.DataFrame(np.zeros((8,12)))
    edited = st.data_editor(plate)

    st.write("Configured plate")
    st.dataframe(edited)

# ---------------- ENVIRONMENTAL ----------------
elif menu == "Environmental Mode":

    st.title("Standard Addition")

    x = st.text_area("Added concentration", "0,1,2,3")
    y = st.text_area("Signal", "0.1,0.2,0.3,0.4")

    try:
        x = np.array([float(i) for i in x.split(",")])
        y = np.array([float(i) for i in y.split(",")])

        m,b = np.polyfit(x,y,1)
        cx = -b/m

        st.metric("Cx", f"{abs(cx):.4f}")
    except:
        st.error("Invalid data")

# ---------------- THEORY ----------------
elif menu == "Theory":
    st.title("Theory")
    st.write("Beer-Lambert law and digital colorimetry principles.")

# ---------------- SOURCES ----------------
elif menu == "Sources":
    st.title("Sources")
    st.write("Scientific references.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Todos los derechos reservados. Katyutzka Villarreal (2026).")

