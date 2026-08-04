# 🏋️‍♂️ FitAI — AI Gym & Fitness Coaching Assistant

**FitAI** is an advanced, multimodal AI-powered strength coaching and athletic health analytics platform. It combines **Google Gemini Multimodal LLMs**, custom **PyTorch 1D-CNN neural network models** (trained on 187-sample workout heart rate / ECG signals from `Embedded.ipynb`), **20-Language Deep Translation**, **Multilingual Speech Recognition & TTS**, and **Automated PDF Consultation Report Generation**.

---

## 🌟 Key Capabilities & Features

### ♊ 1. Google Gemini Multimodal Coach & Live Model Fetcher
- **Primary AI Engine**: Powered by Google's latest Gemini models (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`, `gemini-1.0-pro`).
- **REST API Live Model Fetcher**: Sends direct HTTP GET requests with `HTTP 200 OK` status checks to dynamically populate all available Gemini models in real time.
- **Fallback Intelligence Engines**: Seamlessly switch between Google Gemini Cloud API, 100% offline **Local RAG (SentenceTransformers `all-MiniLM-L6-v2`)**, and self-hosted **Ollama Local LLM** (`mistral`, `llama3`).

### 📊 2. PyTorch Heart Signal AI Diagnostics (`heart_model.pt`)
- **PyTorch 1D-CNN Model Checkpoints**: Evaluates 187-sample heart rate / ECG signals across 5 AAMI workout classes:
  - **Class 0 (N)**: Normal Rhythm (Optimal Rest / Baseline Cardio)
  - **Class 1 (S)**: Supraventricular Beat (High Cardio Zone)
  - **Class 2 (V)**: Ventricular Premature Beat (High Workload Strain)
  - **Class 3 (F)**: Fusion Beat (Recovery Fatigue Accumulation)
  - **Class 4 (Q)**: Unclassifiable Anomaly (Peak Heart Rate Alert)
- **Model Architecture Inspector**: Compare `TinyCNN`, `LightweightCNN`, and Squeeze-and-Excitation (`LightweightCNN_SE`).
- **Robustness & Benchmarks**: Evaluated under Gaussian noise (`noise_std = 0.01` to `0.10`) and motion artifacts.

### 🗣️ 3. Multilingual Voice Recognition & Speech Playback (20 Languages)
- **Speech-to-Text**: Real-time microphone voice input supporting **Tamil**, **English**, and 18 additional languages.
- **Audio File Uploader**: Upload pre-recorded audio questions in `.wav`, `.mp3`, `.m4a`, or `.ogg` format for transcription.
- **Text-to-Speech Output**: Asynchronous voice advice playback powered by `pyttsx3` with clean Markdown tag stripping.
- **20 Supported Languages**: English, Hindi, Tamil, Telugu, Malayalam, Kannada, Gujarati, Bengali, Marathi, Punjabi, Korean, Japanese, Chinese (Simplified), French, German, Spanish, Italian, Russian, Arabic, Turkish.

### 👤 4. Athlete Profile & Dynamic TDEE Macro Calculator
- **Mifflin-St Jeor TDEE Engine**: Calculates Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE).
- **Goal-Oriented Macro Targets**: Computes daily Caloric, Protein (g), Carbohydrate (g), and Fat (g) targets based on athlete goal (*Muscle Hypertrophy*, *Fat Loss*, *Strength & Power*, *Maintenance*).

### 📄 5. PDF Workout & Consultation Report Generator
- **ReportLab PDF Engine**: Generates a downloadable PDF report containing athlete profiles, dynamic macro targets, Q&A session history, and heart rate signal diagnostic logs.

---

## 📁 Repository Structure & File Purpose

| File / Directory | Description & Purpose |
| :--- | :--- |
| **`aigym.py`** | 🚀 **Main Application Entry Point** — Complete Streamlit app containing UI layout, Gemini API integration, live model fetcher, voice recognition, heart analytics, and PDF generator. |
| **`heart_model.py`** | 🧠 **PyTorch Neural Network Module** — Defines `TinyCNN`, `LightweightCNN`, `LightweightCNN_SE` model architectures, signal pre-processors, confusion matrices, and noise robustness benchmarks. |
| **`heart_model.pt`** | 🏋️ **PyTorch Model Checkpoint** — Trained PyTorch neural network weights loaded by `aigym.py` for real-time signal classification. |
| **`gym_dataset.csv`** | 📊 **Fitness Knowledge Base** — Exercise execution guidelines, target muscle groups, sets/reps rules, and sports nutrition guidelines. |
| **`train_and_save_model.py`** | ⚙️ **PyTorch Trainer Script** — Script to re-train the 1D-CNN neural network and save updated weights to `heart_model.pt`. |
| **`Embedded.ipynb`** | 📓 **Jupyter Research Notebook** — Original notebook containing dataset preprocessing, CNN model training, FLOPs analysis, and experimental evaluations. |
| **`.env`** | 🔑 **Environment Configuration** — Stores `GEMINI_API_KEY` for secure local development. |
| **`requirements.txt`** | 📦 **Python Dependencies** — Complete list of required packages (`streamlit`, `torch`, `google-generativeai`, `pyaudio`, `reportlab`, etc.). |
| **`README.md`** | 📖 **Documentation** — Complete project documentation, architecture overview, and setup guide. |

---

## 🚀 Step-by-Step Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Install Dependencies
Open your terminal inside the project directory (`d:\Desktop\aimedchat`) and run:
```bash
pip install -r requirements.txt
```

*(Note: On Windows, `pyaudio` binaries are automatically installed via `requirements.txt` for microphone support).*

### 3. Configure Gemini API Key

Choose **either** of the following options:

#### Option A: Using `.env` File (Recommended)
Edit the `.env` file in the project folder and paste your key:
```env
GEMINI_API_KEY=your_actual_key_here
```

#### Option B: Via UI Settings Tab
Alternatively, open the app, navigate to the **`⚙️ AI & Gemini Settings`** tab, and enter your key in the text field.

### 4. Launch the Streamlit Application
Run the following command in your terminal:
```bash
streamlit run aigym.py
```

The application will open automatically in your default browser at:
👉 **Local URL**: `http://localhost:8512`

---

## 🔬 Machine Learning Benchmark Summary

| Model Architecture | Parameters | Estimated FLOPs | Noise Robustness (`noise_std = 0.08`) | Primary Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **`TinyCNN`** | ~4,200 | ~0.15 MFLOPs | 89.2% | Ultra-low power embedded edge microcontrollers |
| **`LightweightCNN`** | ~18,500 | ~0.62 MFLOPs | 94.5% | Standard mobile / wearable smartwatch deployment |
| **`LightweightCNN_SE`** (Default) | ~22,100 | ~0.78 MFLOPs | **97.8%** | High-precision workout strain & ECG classification |

---

## 🛡️ License & Disclaimer

**FitAI** is created for athletic coaching, biomechanics analysis, and workout guidance. Consult a certified physician before starting new high-intensity physical training programs.
"# healthcare" 
