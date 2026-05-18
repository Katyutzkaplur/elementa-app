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

# Elementa PWA

> Aplicación Científica Progresiva (PWA) diseñada para transformar un smartphone en un espectrofotómetro digital de alta precisión para la detección de metales pesados como Plomo (Pb), Cadmio (Cd) y Cromo (Cr).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Licencia](https://img.shields.io/badge/Licencia-Académica-green)
![Estado](https://img.shields.io/badge/Estado-Desarrollo-emerald)

---

## Descripción General

**Elementa PWA** es una plataforma científica y académica desarrollada para realizar análisis colorimétricos digitales mediante procesamiento óptico computacional e imágenes obtenidas con smartphones.

Inspirada en sistemas avanzados de análisis digital como *PhotoMetrix*, la aplicación integra:

- Cálculo de absorbancia digital
- Análisis adaptativo de canales RGB
- Implementación de la Ley de Beer-Lambert
- Procesamiento óptico mediante ROIs
- Método de adición estándar
- Evaluación normativa mexicana (NOM)

El sistema busca proporcionar una alternativa portátil, accesible y de bajo costo para aplicaciones de monitoreo ambiental y química analítica.

---

## Características Principales

### Motor Analítico
- Análisis espectrofotométrico mediante smartphone
- Generación automática de ROIs
- Ajuste manual en tiempo real
- Función Freeze ROI
- Selección de blanco de reactivos
- Corrección de iluminación
- Cálculo de absorbancia digital

### Calibración y Cuantificación
- Optimización automática de canales RGB
- Regresión lineal
- Selección automática del mejor canal según \(R^2\)
- Curvas de calibración
- Estimación de LOD y LOQ
- Método de adición estándar para muestras ambientales

### Visualización de Datos
- Gráficas interactivas con Plotly
- Interfaz científica en modo oscuro
- Proyección de regresión hacia el eje X negativo
- Generación profesional de reportes

### Módulo Educativo y Normativo
- Sección educativa científica
- Datos de química analítica
- Integración de NOM mexicanas:
  - NOM-127-SSA1-2021
  - NOM-001-SEMARNAT-2021
- Evaluación automática de cumplimiento:
  - CUMPLE
  - NO CUMPLE

---

## Modelo Matemático

### Normalización de Iluminación

\[
I_{norm} = \left(\frac{C_{canal}}{R + G + B}\right) \times 100
\]

### Absorbancia Digital

\[
A_{dig} = \log_{10}\left(\frac{I_{blanco}}{I_{muestra}}\right)
\]

### Método de Adición Estándar

\[
C_{muestra} = \frac{b}{m}
\]

Donde:
- \(m\) = pendiente
- \(b\) = intercepto

---

## Estructura de la Aplicación

### Análisis
Motor principal para adquisición de imágenes, procesamiento de ROIs, calibración y cuantificación.

### Para Saber Más
Contenido educativo relacionado con:
- Toxicidad de metales pesados
- Efectos del Cromo VI
- Química de la ditizona
- Bioacumulación
- Colorimetría digital

### Fuentes e Información
Biblioteca técnica con:
- Normas Oficiales Mexicanas
- Tablas comparativas
- Referencias científicas
- Límites permisibles

---

## Tecnologías Utilizadas

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

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/tuusuario/elementa-pwa.git
cd elementa-pwa
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
streamlit run elementa_app.py
```

---

## Estructura del Proyecto

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

## Generación de Reportes

La aplicación genera automáticamente reportes PDF profesionales que incluyen:

- Encabezado de Elementa 2026
- Imagen analizada con ROIs
- Curvas de calibración
- Tabla de resultados
- Evaluación normativa NOM
- Valores de LOD y LOQ

---

## Aplicaciones

- Monitoreo ambiental
- Análisis de calidad del agua
- Detección de metales pesados
- Laboratorios educativos
- Química analítica portátil
- Ciencia ciudadana

---

## Autora

**Katyutzka Villarreal**  
Estudiante de Química y Desarrolladora de Software Científico  
Universidad de Guanajuato

---

## Aviso Académico

Este proyecto fue desarrollado con fines académicos, científicos y educativos.

---

## Licencia

Licencia de Uso Académico — Elementa (2026)

Todos los derechos reservados.
