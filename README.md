# Elementa PWA

> A scientific Progressive Web Application (PWA) designed to transform a smartphone into a high-precision digital spectrophotometer for the detection of heavy metals such as Lead (Pb), Cadmium (Cd), and Chromium (Cr).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Development-emerald)

---

## Overview

**Elementa PWA** is an academic and scientific software platform developed to perform digital colorimetric analysis using smartphone imaging and computational optical processing.

Inspired by advanced digital analytical systems such as *PhotoMetrix*, this application integrates:

- Digital absorbance calculations
- Adaptive RGB channel analysis
- Beer–Lambert law implementation
- ROI-based optical processing
- Standard addition methodology
- Mexican regulatory evaluation (NOM standards)

The system aims to provide a low-cost, portable, and accessible alternative for environmental monitoring and analytical chemistry applications.

---

## Features

### Analytical Engine
- Smartphone-based spectrophotometric analysis
- Automatic ROI (Region of Interest) grid generation
- Real-time ROI adjustment
- Freeze ROI functionality
- White reference selection
- Illumination normalization correction
- Digital absorbance calculation

### Calibration and Quantification
- Automatic RGB channel optimization
- Linear regression analysis
- Adaptive channel selection based on highest \(R^2\)
- Calibration curve visualization
- LOD and LOQ estimation
- Standard addition method for environmental samples

### Data Visualization
- Interactive Plotly calibration graphs
- Dark-mode scientific interface
- Regression projection into the negative X-axis
- Professional analytical report generation

### Educational and Regulatory Module
- Scientific educational section
- Curated analytical chemistry facts
- Mexican NOM standards integration:
  - NOM-127-SSA1-2021
  - NOM-001-SEMARNAT-2021
- Automatic compliance evaluation:
  - CUMPLE
  - NO CUMPLE

---

## Mathematical Model

### Illumination Normalization

\[
I_{norm} = \left(\frac{C_{channel}}{R + G + B}\right) \times 100
\]

### Digital Absorbance

\[
A_{dig} = \log_{10}\left(\frac{I_{blank}}{I_{sample}}\right)
\]

### Standard Addition Method

\[
C_{sample} = \frac{b}{m}
\]

Where:
- \(m\) = slope
- \(b\) = intercept

---

## Application Structure

### Analysis
Core analytical engine for image acquisition, ROI processing, calibration, and quantification.

### Learn More
Educational scientific content related to:
- Heavy metal toxicity
- Chromium VI health effects
- Dithizone chemistry
- Bioaccumulation
- Digital colorimetry

### Sources and Information
Technical library including:
- Mexican NOM standards
- Comparative regulatory tables
- Scientific references
- Environmental thresholds

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Plotly
- SciPy
- ReportLab
- Pillow

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/elementa-pwa.git
cd elementa-pwa
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run elementa_app.py
```

---

## Project Structure

```bash
Elementa-PWA/
│
├── elementa_app.py
├── requirements.txt
├── assets/
├── reports/
├── images/
└── README.md
```

---

## Report Generation

The application automatically generates analytical PDF reports including:

- Elementa 2026 header
- Captured image with marked ROIs
- Calibration curves
- Results tables
- NOM compliance evaluation
- LOD and LOQ values

---

## Applications

- Environmental monitoring
- Water quality analysis
- Heavy metal detection
- Educational laboratories
- Portable analytical chemistry
- Citizen science

---

## Author

**Katyutzka Villarreal**  
Chemistry Student and Scientific Software Developer  
University of Guanajuato

---

## Academic Notice

This project was developed for academic, scientific, and educational purposes.

---

## License

Academic Use License — Elementa (2026)

All rights reserved.
