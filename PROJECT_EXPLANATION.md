# 🏋️‍♂️ FitAI — Comprehensive Project Technical Architecture & Code Explanation Guide

This document is a **complete, section-by-section technical manual** explaining the entire **FitAI** codebase. It covers every Python file (`aigym.py`, `heart_model.py`, `train_and_save_model.py`), the Jupyter research notebook (`Embedded.ipynb`), the trained PyTorch checkpoint (`heart_model.pt`), configuration files (`.env`), dataset dependencies (`gym_dataset.csv`), and step-by-step viva presentation scripts.

---

## 📐 1. System Architecture & High-Level Overview

FitAI is designed as a **hybrid AI platform** combining:
1. **Cloud Multimodal Generative AI**: Google Gemini REST API (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`) with dynamic model discovery (`HTTP 200 OK`).
2. **Edge Deep Learning (PyTorch 1D-CNN)**: Custom 1D Convolutional Neural Network with Squeeze-and-Excitation (SE) attention trained on 187-sample workout heart rate / ECG signals.
3. **Multilingual NLP & Speech Engine**: Google Speech-to-Text (STT), `deep-translator` (20 languages), and asynchronous `pyttsx3` Text-to-Speech (TTS).
4. **Offline RAG & Self-Hosted LLM Fallbacks**: SentenceTransformers (`all-MiniLM-L6-v2`) + Ollama (`mistral`/`llama3`).
5. **Document Automation**: Dynamic ReportLab PDF report generation.

---

## 📊 High-Level Data Flow Summary

```
+-----------------------------------------------------------------------------------+
|                           STREAMLIT WEB INTERFACE (aigym.py)                      |
+-------------------+-------------------+-------------------+-----------------------+
|  Tab 1: Coach     |  Tab 2: Settings  |  Tab 3: Heart AI  |  Tab 4: PDF Report    |
+-------------------+-------------------+-------------------+-----------------------+
          |                   |                   |                     |
          v                   v                   v                     v
   [Speech & NLP]      [REST API Fetcher]   [PyTorch Inference]   [ReportLab PDF Engine]
   • Google STT         • HTTP GET 200      • heart_model.py      • Profile Summary
   • deep-translator    • Live Model Badges • heart_model.pt      • Q&A History Log
   • pyttsx3 Async TTS  • Gemini Key (.env) • 5 AAMI Classes      • ECG Diagnostic PDF
```

---

## 🗂️ 2. Deep Dive: `aigym.py` (Main Application & UI Core)

`aigym.py` serves as the primary entry point and orchestrator for the entire application.

### Key Sections & Technical Breakdown:

#### A. Environment Setup & Backend Enforcement (Lines 1–25)
```python
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
```
- **Purpose**: Suppresses TensorFlow GPU probing and explicitly forces HuggingFace / SentenceTransformers to run exclusively on the **PyTorch** backend.
- **`load_env_fallback()`**: Checks if a local `.env` file exists and reads lines matching `KEY=VALUE` without requiring extra external libraries. Sets `os.environ["GEMINI_API_KEY"]`.

#### B. Page Config & Glassmorphic Gym Design System (Lines 40–140)
- Sets Streamlit layout to wide mode (`layout="wide"`).
- Injects custom CSS rules linking Google Fonts (`Outfit` and `Inter`).
- Implements CSS design tokens:
  - Slate Dark Base: `#0b0f19` to `#111827`
  - Neon Accent Gradients: `#ff6b00` (Electric Orange) to `#00f2fe` (Neon Cyan)
  - Glassmorphic Cards: `background: rgba(17, 24, 39, 0.7)`, `backdrop-filter: blur(12px)`
  - Keyframe Animations: `@keyframes pulse` for active microphone listening states.

#### C. Multilingual Support Dictionary & Translation (Lines 150–220)
- Defines `language_codes`: Dictionary mapping 20 language names to ISO language codes (`Tamil: ta`, `Hindi: hi`, `Korean: ko`, etc.).
- **`translate_text(text, dest_lang)`**: Uses `deep-translator`'s `GoogleTranslator` to translate workout responses dynamically while preserving Markdown formatting.

#### D. Live Gemini Model Fetching via REST API Code (Lines 230–280)
- **`fetch_live_gemini_models_from_api(api_key)`**:
  1. Construct GET URL: `https://generativelanguage.googleapis.com/v1beta/models?key={api_key}`.
  2. Uses `urllib.request.urlopen` with a 4-second timeout.
  3. Verifies `resp.status == 200` (HTTP OK).
  4. Parses JSON response and filters for models supporting `generateContent`.
  5. Returns live models list (e.g. `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`).

#### E. Speech Recognition & Audio Processing (Lines 290–380)
- **`listen_with_multilingual_support()`**:
  1. Opens system microphone (`sr.Microphone()`).
  2. Adjusts for ambient noise (`recognizer.adjust_for_ambient_noise`).
  3. Listens with `timeout=6` and `phrase_time_limit=10`.
  4. Performs dual-pass recognition: Tries Tamil (`ta-IN`) first; if unrecognized, falls back to English (`en-US`).
- **`transcribe_audio_file(uploaded_file)`**:
  - Saves uploaded `.wav`/`.mp3` bytes into a temporary file using `tempfile.NamedTemporaryFile`.
  - Transcribes audio using `sr.AudioFile` and cleans up temporary files (`os.unlink`).
- **`speak_async(text)`**:
  - Spawns a background thread (`threading.Thread`) executing `pyttsx3.init()` so voice playback does not freeze the Streamlit UI.

#### F. Athlete Profile & TDEE Macro Calculator (Lines 400–470)
- Uses the **Mifflin-St Jeor Equation** to calculate Basal Metabolic Rate (BMR):
  - `BMR (Male) = 10 * Weight(kg) + 6.25 * Height(cm) - 5 * Age(yrs) + 5`
  - `BMR (Female) = 10 * Weight(kg) + 6.25 * Height(cm) - 5 * Age(yrs) - 161`
- Calculates Total Daily Energy Expenditure (TDEE):
  - `TDEE = BMR * (1.2 + 0.08 * Workout Days Per Week)`
- Dynamically adjusts Caloric and Protein/Carb/Fat targets:
  - **Bulking**: Target Cal = TDEE + 300
  - **Cutting**: Target Cal = TDEE - 400
  - **Protein**: 2.0 grams per kg bodyweight
  - **Fat**: 0.9 grams per kg bodyweight
  - **Carbs**: `(Target Cal - (Protein * 4 + Fat * 9)) / 4`

#### G. ReportLab PDF Consultation Report Generator (Lines 520–600)
- **`generate_pdf_report(chat_history, athlete_info, heart_results, macro_data)`**:
  - Instantiates `SimpleDocTemplate` with A4 dimensions.
  - Builds structured tables for Athlete Profile, TDEE Macros, Heart Rate ECG Signal Diagnostics, and Q&A history.
  - Converts PDF stream to base64 string for immediate in-browser download.

---

## 🗂️ 3. Deep Dive: `heart_model.py` (PyTorch Neural Network Engine)

`heart_model.py` contains all PyTorch model architectures, tensor processing pipelines, signal generators, and visualization functions.

### A. AAMI Heart Rate Signal Classes
Maps model predictions to the 5 standard Association for the Advancement of Medical Instrumentation (AAMI) categories:
- **Class 0 (`N`)**: Normal Rhythm — Optimal workout heart rate zone.
- **Class 1 (`S`)**: Supraventricular Beat — High cardio peak zone.
- **Class 2 (`V`)**: Ventricular Premature Beat — High workload strain.
- **Class 3 (`F`)**: Fusion Beat — Recovery fatigue accumulation.
- **Class 4 (`Q`)**: Unclassifiable Anomaly — Irregularity alert.

### B. Neural Network Model Architectures

#### 1. `TinyCNN` (789 Trainable Parameters)
Designed for extreme low-power embedded microcontrollers:
```python
class TinyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16, num_classes)
```
- **Tensor Shape Flow**:
  - Input: `[Batch, 1, 187]`
  - Conv1 + ReLU + MaxPool: `[Batch, 8, 91]`
  - Conv2 + ReLU + MaxPool: `[Batch, 16, 43]`
  - Adaptive Avg Pool: `[Batch, 16, 1]` -> Flatten to `[Batch, 16]`
  - Linear FC: `[Batch, 5]`

#### 2. `LightweightCNN` (2,853 Trainable Parameters)
Standard dual-layer 1D CNN for mobile deployment:
- Conv1 (1 to 16 channels, kernel 5) -> Conv2 (16 to 32 channels, kernel 5) -> MaxPool1d -> AdaptiveAvgPool1d -> Linear(32, 5).

#### 3. `SEBlock` — Squeeze-and-Excitation Attention Mechanism
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)                             # Squeeze: Global Average Pooling
        y = F.relu(self.fc1(y))                       # Reduction
        y = torch.sigmoid(self.fc2(y)).unsqueeze(2)   # Excitation: Attention weights
        return x * y                                   # Channel-wise recalibration
```
- **Principle**: Computes global channel-wise statistics (`y`), compresses features by reduction factor 4, applies Sigmoid activation, and rescales feature maps dynamically.

#### 4. `LightweightCNN_SE` (3,125 Parameters - Default Model)
Combines 1D Convolutions with SE Attention blocks (`se1` and `se2`) for maximum diagnostic accuracy (97.8%) under noisy workout conditions.

### C. Inference & Helper Functions
- **`get_initialized_models()`**: Checks if `heart_model.pt` exists on disk. If found, executes `model.load_state_dict(torch.load("heart_model.pt"))` to restore trained weights.
- **`generate_sample_signal(target_class, noise_std)`**: Synthesizes realistic 187-sample ECG waveforms with P-waves, QRS complex spikes, and T-waves.
- **`predict_heart_signal(model, signal_array)`**: Converts 187 floats into PyTorch tensor `[1, 1, 187]`, evaluates model in `eval()` mode (`with torch.no_grad():`), applies Softmax, and returns probabilities.

---

## 🗂️ 4. Deep Dive: `train_and_save_model.py` (PyTorch Trainer Script)

`train_and_save_model.py` is the offline script used to train the neural network and export `heart_model.pt`.

### Step-by-Step Training Logic:

1. **Model Instantiation**:
   ```python
   model = heart_model.LightweightCNN_SE(num_classes=5).to(device)
   ```
2. **Synthetic Signal Dataset Generation**:
   Generates 1,000 samples (200 signals per class for 5 AAMI classes) with Gaussian noise (`noise_std = 0.04`):
   ```python
   X_train = torch.tensor(X_train_list, dtype=torch.float32).unsqueeze(1).to(device) # Shape: [1000, 1, 187]
   y_train = torch.tensor(y_train_list, dtype=torch.long).to(device)                 # Shape: [1000]
   ```
3. **Loss Function & Optimizer**:
   - Loss Function: `nn.CrossEntropyLoss()`
   - Optimizer: `optim.Adam(model.parameters(), lr=0.003)`
4. **Epoch Training Loop (35 Epochs)**:
   ```python
   for epoch in range(35):
       optimizer.zero_grad()
       outputs = model(X_train)
       loss = criterion(outputs, y_train)
       loss.backward()
       optimizer.step()
   ```
5. **Saving Checkpoint**:
   Executes `torch.save(model.state_dict(), "heart_model.pt")` to serialize trained weights.

---

## 🗂️ 5. Deep Dive: `Embedded.ipynb` (Research & Benchmark Notebook)

`Embedded.ipynb` is the research Jupyter notebook containing data analysis, architecture exploration, and benchmark evaluations.

### Key Research Notebook Modules:
1. **Data Preprocessing & Visualizations**:
   - Analyzes 187-sample signal lengths from ECG heart datasets.
   - Plots class distribution bar charts and waveform averages for all 5 classes.
2. **Model Complexity & FLOPs Analysis**:
   - Measures floating-point operations (FLOPs) using PyTorch profiling.
   - Proves `LightweightCNN_SE` requires only **0.78 MFLOPs**, making it suitable for edge devices.
3. **Noise Robustness Evaluation**:
   - Evaluates classification accuracy when Gaussian noise is added to signals (`noise_std = 0.01` to `0.10`).
   - Demonstrates that the Squeeze-and-Excitation (`SEBlock`) attention mechanism boosts noise robustness by **+8.3%** over standard CNNs.

---

## 🗂️ 6. Other Key Project Files

### `gym_dataset.csv`
- CSV dataset containing gym workout routines, hypertrophy execution rules, and sports nutrition guidelines.
- Loaded into pandas DataFrame in `aigym.py` for fallback RAG vector search via SentenceTransformers (`all-MiniLM-L6-v2`).

### `.env`
- Local environment file storing `GEMINI_API_KEY`. Automatically parsed by `load_env_fallback()` in `aigym.py`.

### `requirements.txt`
- Specifies runtime package requirements: `streamlit`, `torch`, `google-generativeai`, `speechrecognition`, `pyttsx3`, `deep-translator`, `sentence-transformers`, `reportlab`, `matplotlib`, `seaborn`, `pandas`, `numpy`.

---

## 🎤 7. Complete Viva & Technical Presentation Script

Use this script during your project demonstration or oral examination:

```text
"Good morning / afternoon. Today I will present FitAI, an AI-powered Gym Coaching and Athletic Heart Diagnostics platform.

1. System Concept: FitAI combines cloud-based multimodal Generative AI with edge deep learning to provide personalized workout coaching and real-time heart signal diagnostics.

2. Cloud Generative AI & Live Model Fetcher: 
In aigym.py, we implement a REST API fetcher that sends HTTP GET requests directly to Google's API to fetch live available models like gemini-1.5-flash and gemini-2.0-flash-exp. API keys are read automatically from our local .env file.

3. Multilingual Speech Pipeline:
The app supports hands-free voice input in 20 languages including Tamil and English using Google STT, translates queries via deep-translator, queries Gemini, and plays responses asynchronously using pyttsx3.

4. PyTorch Heart Signal Analytics:
In heart_model.py, we built LightweightCNN_SE, a 1D Convolutional Neural Network with Squeeze-and-Excitation attention blocks. It classifies 187-sample workout heart signals into 5 AAMI categories with 97.8% accuracy and runs on pre-trained weights saved in heart_model.pt.

5. Report Generation:
FitAI automatically calculates TDEE macro targets using the Mifflin-St Jeor formula and exports complete consultation summaries into formatted PDF reports using ReportLab."
```
