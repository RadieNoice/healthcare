import os
import sys
import json
import urllib.request

# Bypasses TensorFlow probing and forces PyTorch backend
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def load_env_fallback():
    if os.path.exists('.env'):
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

load_env_fallback()

import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import streamlit as st
from datetime import datetime
import io
import subprocess
import tempfile
import google.generativeai as genai
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import base64
import platform
import threading
import time
import matplotlib.pyplot as plt

# Import custom PyTorch embedded model helper
import heart_model

# Get API key from environment variable or .env file
def get_env_gemini_key():
    return os.getenv("GEMINI_API_KEY", "").strip()

# ---------------------------------------
# PAGE CONFIG & ULTRA-PREMIUM GYM STYLING
# ---------------------------------------
st.set_page_config(
    page_title="FitAI - Gym & Fitness Assistant",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark neon gym aesthetic with Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Dark Gym Base Theme */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0b0f19 100%);
        color: #f3f4f6;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 107, 0, 0.25);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 25px rgba(255, 107, 0, 0.15);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .header-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #ff6b00, #00f2fe, #ff6b00);
    }
    .header-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b00 0%, #ffa502 40%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.8px;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.25rem;
        font-weight: 500;
        margin: 0;
    }
    
    /* Glassmorphic Gym Cards */
    .gym-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    
    /* Custom Button Styling */
    .stButton > button {
        border-radius: 14px !important;
        background: linear-gradient(135deg, #ff6b00 0%, #00f2fe 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 18px rgba(255, 107, 0, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5) !important;
    }
    
    /* Metric & Badge styling */
    .metric-card {
        background: rgba(31, 41, 55, 0.8);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00f2fe;
    }
    .metric-lbl {
        color: #9ca3af;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Badges */
    .gym-badge {
        display: inline-block;
        padding: 5px 14px;
        margin: 3px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success { background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #4ade80; }
    .badge-warning { background: rgba(251, 146, 60, 0.2); color: #fb923c; border: 1px solid #fb923c; }
    .badge-danger { background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #f87171; }
    .badge-info { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
    
    /* Pulse effect */
    .voice-status {
        background: rgba(255, 107, 0, 0.2);
        border: 1.5px solid #ff6b00;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        color: #ffa502;
        font-weight: 700;
        font-size: 1.1rem;
        animation: pulse 1.4s infinite alternate;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.85; box-shadow: 0 0 10px rgba(255, 107, 0, 0.2); }
        100% { transform: scale(1.015); opacity: 1; box-shadow: 0 0 25px rgba(255, 107, 0, 0.5); }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to clean literal '\n' sequences into actual line breaks
def format_markdown_newlines(text: str) -> str:
    if not text:
        return ""
    clean = text.replace('\\n', '\n')
    return clean

# ---------------------------------------
# 20 SUPPORTED LANGUAGES DICTIONARY
# ---------------------------------------
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Russian": "ru",
    "Arabic": "ar",
    "Turkish": "tr"
}

# ---------------------------------------
# HELPER TESTING & API FETCH FUNCTIONS
# ---------------------------------------
def fetch_live_gemini_models_from_api(api_key):
    models_list = []
    if not api_key:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.0-pro"], "⚠️ Gemini API Key is empty! Enter key in Settings or add GEMINI_API_KEY to your .env file."
    
    # Method 1: Direct HTTP GET REST API with status code 200 response check
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=4) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                for m in data.get('models', []):
                    methods = m.get('supportedGenerationMethods', [])
                    if 'generateContent' in methods:
                        name = m.get('name', '').replace('models/', '')
                        if name and name not in models_list:
                            models_list.append(name)
                if models_list:
                    return models_list, f"🟢 Successfully fetched {len(models_list)} live Gemini models from Google API (HTTP 200 OK)!"
    except Exception:
        pass

    # Method 2: Google Generative AI Python SDK Fallback
    try:
        genai.configure(api_key=api_key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace('models/', '')
                if name and name not in models_list:
                    models_list.append(name)
        if models_list:
            return models_list, f"🟢 Successfully fetched {len(models_list)} live Gemini models!"
    except Exception as e:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.0-pro"], f"🔴 Error fetching models from API: {e}"

    return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.0-pro"], "⚠️ No models returned."

def test_gemini_connection(api_key, model_name="gemini-1.5-flash"):
    if not api_key:
        return False, "⚠️ API Key is empty. Please enter key in Settings or set GEMINI_API_KEY in .env."
    try:
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_name)
        res = m.generate_content("Ping! Respond with 'Pong'")
        if res.text:
            return True, f"🟢 Connected to Google Gemini ({model_name})! Response: {res.text.strip()}"
        return False, "Gemini API returned an empty response."
    except Exception as e:
        return False, f"🔴 Gemini API Connection Error: {e}"

# ---------------------------------------
# DATASET & EMBEDDINGS INITIALIZATION
# ---------------------------------------
@st.cache_data
def load_gym_dataset():
    try:
        data = pd.read_csv('gym_dataset.csv')
        data['cure'] = data['cure'].astype(str).str.replace('\\n', '\n')
        return data
    except Exception:
        return pd.DataFrame({
            'disease': ['Bench Press Chest Workout', 'Squats Leg Workout', 'Protein Intake & Nutrition', 'Muscle Hypertrophy Growth'],
            'cure': [
                "🎯 **Target Muscles**: Middle & Lower Pectoralis Major, Anterior Deltoids, Triceps.\n\n🏋️‍♂️ **Exercise Execution**:\n• Lie flat on bench with 5-point contact.\n• Grip barbell slightly wider than shoulder-width.\n• Lower bar with controlled 2-3 sec tempo to lower chest.\n• Drive feet into floor and press explosively.\n\n📊 **Sets & Reps**: 3-4 sets x 6-10 reps.\n\n⚡ **Pro Tip**: Keep shoulder blades retracted throughout.",
                "🎯 **Target Muscles**: Quadriceps, Gluteus Maximus, Hamstrings.\n\n🏋️‍♂️ **Exercise Execution**:\n• Bar on upper traps, feet shoulder-width.\n• Brace core, push hips back, lower hips parallel to ground.\n• Drive through mid-foot to stand.\n\n📊 **Sets & Reps**: 3-4 sets x 6-10 reps.",
                "🎯 **Protein Guidelines**: Consume 1.6-2.2g protein per kg of body weight daily.",
                "🎯 **Hypertrophy**: Train muscles 2x weekly with 10-20 total sets per week."
            ]
        })

df = load_gym_dataset()

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        return None

embedding_model = load_embedding_model()

# Universal Translator for 20 Languages (deep-translator)
def translate_text(text, dest_lang='en'):
    if not text or dest_lang == 'en':
        return format_markdown_newlines(text)
    try:
        translator = GoogleTranslator(source='auto', target=dest_lang)
        translated = translator.translate(text)
        return format_markdown_newlines(translated)
    except Exception:
        return format_markdown_newlines(text)

# Speech recognition initializer
recognizer = sr.Recognizer()

# Live Speech Microphone Recognition Function
def listen_with_multilingual_support():
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak your workout question now (Multilingual support)!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
        
        try:
            query = recognizer.recognize_google(audio, language="ta-IN")
            st.success(f"🎤 Tamil Detected: {query}")
            english_trans = translate_text(query, dest_lang='en')
            st.info(f"🌐 English Translation: {english_trans}")
            return english_trans.lower(), "tamil"
        except sr.UnknownValueError:
            try:
                query = recognizer.recognize_google(audio, language="en-US")
                st.success(f"🎤 English Detected: {query}")
                return query.lower(), "english"
            except sr.UnknownValueError:
                st.warning("❌ Could not understand speech. Try speaking louder or typing below.")
                return "", "unknown"
        except sr.RequestError:
            st.error("❌ Google Speech Recognition service unreachable.")
            return "", "error"
    except Exception as e:
        st.error(f"❌ Microphone device error: {e}")
        st.info("💡 Make sure your microphone is connected, or upload an audio file below!")
        return "", "error"

# Uploaded Audio File Transcription Function
def transcribe_audio_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            try:
                query = recognizer.recognize_google(audio_data, language="ta-IN")
                english_trans = translate_text(query, dest_lang='en')
                os.unlink(tmp_path)
                return english_trans.lower()
            except Exception:
                query = recognizer.recognize_google(audio_data, language="en-US")
                os.unlink(tmp_path)
                return query.lower()
    except Exception as e:
        st.error(f"Error parsing audio file: {e}")
        return ""

# Text-to-speech engine (Async)
def speak(text: str) -> bool:
    try:
        clean_speech_text = text.replace('*', '').replace('#', '').replace('•', '')
        if platform.system() == "Darwin":
            subprocess.run(["say", clean_speech_text], check=True)
            return True
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.say(clean_speech_text)
        engine.runAndWait()
        return True
    except Exception:
        return False

def speak_async(text: str, delay: float = 0.1):
    def _run():
        try:
            time.sleep(delay)
            speak(text)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

# ---------------------------------------
# RAG & AI ENGINE SEARCH LOGIC
# ---------------------------------------
fitness_keywords = {
    "bench": "For bench press: Maintain 5-point contact (head, upper back, glutes, feet flat). Lower bar smoothly to mid-chest, drive feet into floor, and press up in a slight arc.",
    "squat": "For squats: Bar on upper traps, brace core, push knees outward over toes, drop hips parallel or below, and push through mid-foot/heels.",
    "deadlift": "For deadlifts: Stand hip-width, grip bar outside legs, keep back flat, chest up, pull close to shins by extending hips and knees together.",
    "protein": "Target 1.6 - 2.2g protein per kg bodyweight daily. Distribute evenly across 3-5 meals with high leucine sources.",
    "weight loss": "To cut fat: Maintain a 300-500 kcal daily deficit, consume 2.0-2.4g/kg protein, keep training heavy to preserve lean muscle mass.",
    "creatine": "Creatine Monohydrate (3-5g daily): Increases muscle phosphocreatine stores, boosting strength output, power, and muscle cell volume."
}

def find_local_rag_match(user_input):
    if embedding_model is None:
        for kw, resp in fitness_keywords.items():
            if kw in user_input.lower():
                return format_markdown_newlines(resp)
        return "Focus on fundamental compound movements (Squat, Bench, Deadlift, OHP), progressive overload, structured recovery, and high protein intake."
    try:
        user_emb = embedding_model.encode(user_input, convert_to_tensor=True)
        ds_embs = embedding_model.encode(df['disease'].tolist(), convert_to_tensor=True)
        sims = util.pytorch_cos_sim(user_emb, ds_embs)[0]
        best_idx = sims.argmax().item()
        if sims[best_idx].item() < 0.35:
            for kw, resp in fitness_keywords.items():
                if kw in user_input.lower():
                    return format_markdown_newlines(resp)
            return "Focus on progressive overload, adding 1 rep or small weight increments weekly, taking 1.6-2.2g/kg protein, and getting 7-9 hours of sleep."
        return format_markdown_newlines(df.iloc[best_idx]['cure'])
    except Exception:
        return "Ensure strict exercise execution, progressive weight load, 1.6-2.2g/kg protein, and sufficient recovery."

# Enhanced Coach Prompt Template
fitness_prompt = PromptTemplate(
    input_variables=["chat_history", "question", "athlete_info"],
    template="""You are FitAI, an elite Certified Strength & Conditioning Coach and Sports Nutritionist.

Athlete Context & Daily Macro Targets: {athlete_info}
Previous Conversation: {chat_history}
Athlete Question: {question}

Provide an expert, structured, and inspiring fitness answer. Format your response into clear sections:
1. 🎯 **Athlete Target & Context**
2. 🏋️‍♂️ **Workout Blueprint & Exercise Execution**
3. 🥗 **Nutritional Alignment & Macros**
4. ⚡ **Pro Form Tip & Safety Warning**

FitAI Coach Response:"""
)

def run_ai_engine(question, macro_summary=""):
    engine_choice = st.session_state.get('ai_engine_choice', "♊ Google Gemini API (Cloud LLM)")
    athlete_info = f"Name: {st.session_state.get('athlete_name', 'Athlete')}, Goal: {st.session_state.get('fitness_goal', 'Hypertrophy')}, Weight: {st.session_state.get('athlete_weight', 75)}kg, Height: {st.session_state.get('athlete_height', 178)}cm | Daily Targets: {macro_summary}"

    # Gemini API Engine (Primary - Reads from Settings OR .env)
    if "Gemini" in engine_choice:
        api_key = st.session_state.get('gemini_api_key', get_env_gemini_key())
        model_name = st.session_state.get('gemini_model', "gemini-1.5-flash")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                full_prompt = fitness_prompt.format(chat_history="", question=question, athlete_info=athlete_info)
                res = model.generate_content(full_prompt)
                if res.text:
                    return format_markdown_newlines(res.text.strip())
            except Exception as e:
                st.warning(f"⚠️ Gemini API Error ({e}). Falling back to Local RAG.")
        else:
            st.warning("💡 Gemini API Key is empty. Please enter your API key in the Settings page or set GEMINI_API_KEY in your .env file!")

    # Ollama Engine
    elif "Ollama" in engine_choice:
        base_url = st.session_state.get('ollama_url', 'http://localhost:11434')
        model_name = st.session_state.get('ollama_model', 'mistral')
        try:
            llm = OllamaLLM(model=model_name, base_url=base_url)
            res = llm.invoke(fitness_prompt.format(chat_history="", question=question, athlete_info=athlete_info))
            return format_markdown_newlines(res)
        except Exception as e:
            st.warning(f"⚠️ Ollama Error ({e}). Is Ollama running on {base_url}? Using Local RAG fallback.")

    # Local RAG Fallback
    base_match = find_local_rag_match(question)
    return f"📌 **Personalized Guidance for {st.session_state.get('athlete_name', 'Athlete')}** ({st.session_state.get('fitness_goal', 'Hypertrophy')})\n\n{base_match}"

# PDF Generation Function
def generate_pdf_report(chat_history_list, athlete_info, heart_results=None, macro_data=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_path = temp_file.name
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, alignment=TA_CENTER, textColor=colors.HexColor('#ff6b00'), spaceAfter=15)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#00f2fe'), spaceBefore=12, spaceAfter=8)
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#1e293b'), spaceAfter=5)
        
        story.append(Paragraph("🏋️‍♂️ FitAI - Athletic Fitness & Consultation Report", title_style))
        story.append(Spacer(1, 8))
        
        # Athlete Profile Table
        story.append(Paragraph("👤 Athlete Profile & Nutrition Metrics", heading_style))
        athlete_table_data = [
            ["Athlete Name", athlete_info.get('name', 'Athlete'), "Primary Goal", athlete_info.get('goal', 'Hypertrophy')],
            ["Age / Gender", f"{athlete_info.get('age', 24)} yrs / {athlete_info.get('gender', 'Male')}", "Weight / Height", f"{athlete_info.get('weight', 75)} kg / {athlete_info.get('height', 178)} cm"],
            ["BMR / TDEE", f"{macro_data.get('bmr', 0)} / {macro_data.get('tdee', 0)} kcal", "Target Calories", f"{macro_data.get('target_cal', 0)} kcal / day"],
            ["Daily Macros", f"Protein: {macro_data.get('protein', 0)}g | Carbs: {macro_data.get('carbs', 0)}g | Fats: {macro_data.get('fats', 0)}g", "Training Frequency", f"{athlete_info.get('workout_days', 5)} Days / Week"]
        ]
        
        t = Table(athlete_table_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch, 2.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('PADDING', (0,0), (-1,-1), 5)
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        
        # Diagnostics
        if heart_results:
            story.append(Paragraph("📊 Heart Rate Signal Diagnostics (PyTorch Model)", heading_style))
            story.append(Paragraph(f"<b>Model Checkpoint:</b> {heart_results.get('model_name')}", body_style))
            story.append(Paragraph(f"<b>Signal State:</b> {heart_results.get('title')} ({heart_results.get('code')})", body_style))
            story.append(Paragraph(f"<b>Diagnostic Notes:</b> {heart_results.get('description')}", body_style))
            story.append(Spacer(1, 10))
        
        # Conversation Summary
        story.append(Paragraph("💬 Coaching Q&A History", heading_style))
        for msg in chat_history_list:
            prefix = "Athlete" if msg['type'] == 'user' else "FitAI Coach"
            content = format_markdown_newlines(msg['content']).replace('📝 ', '').replace('🎤 ', '').replace('⚡ ', '')
            story.append(Paragraph(f"<b>{prefix}:</b> {content}", body_style))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("⚠ Disclaimer: FitAI provides exercise & nutrition coaching. Consult a physician before starting new high-intensity programs.", ParagraphStyle('Disc', parent=body_style, fontSize=7.5, textColor=colors.gray)))
        
        doc.build(story)
        with open(pdf_path, 'rb') as f:
            data = f.read()
        os.unlink(pdf_path)
        return data
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None

# Session State Init & Dynamic .env / Settings API Key Lookup
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'last_heart_diagnostic' not in st.session_state:
    st.session_state.last_heart_diagnostic = None
if 'quick_prompt' not in st.session_state:
    st.session_state.quick_prompt = ""
if 'ai_engine_choice' not in st.session_state:
    st.session_state.ai_engine_choice = "♊ Google Gemini API (Cloud LLM)"
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = get_env_gemini_key()
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = "gemini-1.5-flash"

# Automatically fetch available models from API if key is present
if 'available_gemini_models' not in st.session_state:
    key_to_use = st.session_state.gemini_api_key or get_env_gemini_key()
    if key_to_use:
        fetched_models, _ = fetch_live_gemini_models_from_api(key_to_use)
        st.session_state.available_gemini_models = fetched_models
    else:
        st.session_state.available_gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.0-pro"]

if 'ollama_url' not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = "mistral"

# ---------------------------------------
# HEADER
# ---------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🏋️‍♂️ FitAI - Gym & Fitness Assistant</h1>
    <p class="header-subtitle">Google Gemini Multimodal AI • PyTorch Heart Signal Analytics • 20-Language Support</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# SIDEBAR CONTROLS (CLEAN & UNCLUTTERED)
# ---------------------------------------
with st.sidebar:
    st.markdown("### 🌐 20 Supported Languages Choice")
    language_choice = st.selectbox(
        "Select Response Language:",
        list(language_codes.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Workout Stats")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(st.session_state.chat_history)}</div><div class="metric-lbl">Messages</div></div>', unsafe_allow_html=True)
    with col_sb2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{"YES" if st.session_state.last_heart_diagnostic else "NO"}</div><div class="metric-lbl">ECG Tested</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_heart_diagnostic = None
        st.rerun()

# ---------------------------------------
# ATHLETE PROFILE & TDEE MACRO CALCULATOR
# ---------------------------------------
with st.expander("👤 Athlete Profile & Interactive Macro Calculator", expanded=True):
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        athlete_name = st.text_input("Athlete Name", value="Alex Gymnast", key="athlete_name")
    with p_col2:
        athlete_age = st.number_input("Age (yrs)", min_value=12, max_value=90, value=24, key="athlete_age")
    with p_col3:
        athlete_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="athlete_gender")
    with p_col4:
        fitness_goal = st.selectbox("Primary Goal", ["Muscle Hypertrophy (Bulking)", "Fat Loss & Cutting", "Strength & Power", "General Maintenance"], key="fitness_goal")
        
    p_col5, p_col6, p_col7 = st.columns(3)
    with p_col5:
        athlete_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, key="athlete_weight")
    with p_col6:
        athlete_height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=178.0, key="athlete_height")
    with p_col7:
        workout_days = st.slider("Workout Days / Week", min_value=1, max_value=7, value=5, key="workout_days")

    # Dynamic TDEE & Macro Calculations
    if athlete_gender == "Male":
        bmr = 10 * athlete_weight + 6.25 * athlete_height - 5 * athlete_age + 5
    else:
        bmr = 10 * athlete_weight + 6.25 * athlete_height - 5 * athlete_age - 161
        
    activity_multiplier = 1.2 + (workout_days * 0.08)
    tdee = int(bmr * activity_multiplier)
    
    if "Bulking" in fitness_goal:
        target_calories = tdee + 300
    elif "Cutting" in fitness_goal:
        target_calories = tdee - 400
    else:
        target_calories = tdee
        
    protein_g = int(athlete_weight * 2.0)
    fat_g = int(athlete_weight * 0.9)
    carb_g = int((target_calories - (protein_g * 4 + fat_g * 9)) / 4)
    if carb_g < 50: carb_g = 50
    
    macro_data = {
        'bmr': int(bmr), 'tdee': tdee, 'target_cal': target_calories,
        'protein': protein_g, 'fats': fat_g, 'carbs': carb_g
    }
    
    st.markdown("#### ⚡ Dynamic Daily Nutrition Targets:")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{target_calories}</div><div class="metric-lbl">Target Kcal</div></div>', unsafe_allow_html=True)
    with m_c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{protein_g}g</div><div class="metric-lbl">Protein</div></div>', unsafe_allow_html=True)
    with m_c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{carb_g}g</div><div class="metric-lbl">Carbs</div></div>', unsafe_allow_html=True)
    with m_c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{fat_g}g</div><div class="metric-lbl">Fats</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------
# MAIN APPLICATION TABS
# ---------------------------------------
tab_chat, tab_settings, tab_analytics, tab_report = st.tabs([
    "💬 FitAI Workout Coach", 
    "⚙️ AI & Gemini Settings",
    "📊 Heart Rate AI Signal Analytics (Embedded PyTorch Model)", 
    "📄 Consultation Report & PDF"
])

# =========================================================
# TAB 1: WORKOUT COACH & CHAT
# =========================================================
with tab_chat:
    st.markdown(f"### 💬 Chat with FitAI Gemini Strength Coach ({language_choice})")
    
    # Quick Prompt Chips (One-click questions)
    st.markdown("##### ⚡ Quick Prompt Chips (Click to Ask):")
    chip_c1, chip_c2, chip_c3, chip_c4, chip_c5, chip_c6 = st.columns(6)
    
    if chip_c1.button("🏋️ PPL Routine", use_container_width=True):
        st.session_state.quick_prompt = "Push Pull Legs PPL Workout Split routine"
    if chip_c2.button("💪 Chest Growth", use_container_width=True):
        st.session_state.quick_prompt = "Bench Press Chest Workout"
    if chip_c3.button("🥗 Protein Guide", use_container_width=True):
        st.session_state.quick_prompt = "Protein Intake & Nutrition"
    if chip_c4.button("⚡ Bench Plateau", use_container_width=True):
        st.session_state.quick_prompt = "How to increase bench press max weight?"
    if chip_c5.button("🔥 Fat Loss Plan", use_container_width=True):
        st.session_state.quick_prompt = "Fat Loss Caloric Deficit"
    if chip_c6.button("🦵 Squat Safety", use_container_width=True):
        st.session_state.quick_prompt = "Squats Leg Workout form"

    # Native Streamlit Scrollable Chat Container
    with st.container(height=480):
        if not st.session_state.chat_history:
            st.markdown(f"""
            <div style="text-align: center; color: #9ca3af; padding: 40px 20px;">
                <h3 style="color: #ff6b00;">🏋️‍♂️ Welcome to FitAI Strength Coaching!</h3>
                <p style="font-size: 1.1rem;">Active Model: <strong style="color: #00f2fe;">{st.session_state.gemini_model}</strong> | Language: <strong style="color: #00f2fe;">{language_choice}</strong></p>
                <p>Ask any question about workout splits, exercise biomechanics, hypertrophy, or sports nutrition!</p>
                <p>💡 API key automatically loaded from <strong>.env</strong> or <strong>⚙️ AI Settings</strong> tab!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, message in enumerate(st.session_state.chat_history):
                clean_text = format_markdown_newlines(message["content"])
                if message['type'] == 'user':
                    with st.chat_message("user", avatar="🏋️‍♂️"):
                        st.markdown(clean_text)
                else:
                    with st.chat_message("assistant", avatar="♊"):
                        st.markdown(f"**FitAI Gemini Coach ({st.session_state.gemini_model}):**\n\n{clean_text}")
                        if st.button(f"🔊 Listen Advice #{idx+1}", key=f"tts_{idx}"):
                            speak_async(clean_text)
    
    # Determine default text if quick prompt was clicked
    default_text = st.session_state.quick_prompt if st.session_state.quick_prompt else ""
    user_input = st.text_input("💬 Ask FitAI about workouts, form, or diet...", value=default_text, placeholder="e.g. Best chest workout split? How to do barbell squats safely?", key="user_text_input")
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        if st.button("📤 Send Question", use_container_width=True) or (st.session_state.quick_prompt and user_input):
            if user_input:
                st.session_state.chat_history.append({'type': 'user', 'content': f"📝 {user_input}", 'timestamp': datetime.now()})
                
                macro_summary_str = f"{target_calories} Kcal, {protein_g}g Protein, {carb_g}g Carbs"
                raw_response = run_ai_engine(user_input, macro_summary_str)
                final_response = translate_text(raw_response, dest_lang=language_codes[language_choice])
                
                st.session_state.chat_history.append({'type': 'bot', 'content': final_response, 'timestamp': datetime.now()})
                speak_async(final_response)
                
                st.session_state.quick_prompt = ""
                st.rerun()
                
    with col_s2:
        if st.button("🎤 Live Microphone", key="mic_btn", use_container_width=True):
            st.session_state.listening = True
            st.rerun()

    # Voice & Audio Input Options
    st.markdown("---")
    audio_col1, audio_col2 = st.columns([1, 1])
    with audio_col1:
        st.markdown("##### 📁 Alternative: Upload Audio Recording (WAV/MP3/M4A):")
        uploaded_audio = st.file_uploader("Upload audio question:", type=["wav", "mp3", "m4a", "ogg"], key="audio_uploader")
        if uploaded_audio is not None:
            if st.button("🎙️ Process Uploaded Audio", key="process_audio"):
                with st.spinner("Transcribing audio file..."):
                    transcribed_text = transcribe_audio_file(uploaded_audio)
                    if transcribed_text:
                        st.session_state.chat_history.append({'type': 'user', 'content': f"🎤 [Audio File]: {transcribed_text}", 'timestamp': datetime.now()})
                        macro_summary_str = f"{target_calories} Kcal, {protein_g}g Protein"
                        raw_response = run_ai_engine(transcribed_text, macro_summary_str)
                        final_response = translate_text(raw_response, dest_lang=language_codes[language_choice])
                        st.session_state.chat_history.append({'type': 'bot', 'content': final_response, 'timestamp': datetime.now()})
                        speak_async(final_response)
                        st.rerun()

    with audio_col2:
        if st.session_state.listening:
            st.markdown('<div class="voice-status">🎤 Listening to Live Microphone... Speak in any of 20 languages!</div>', unsafe_allow_html=True)
            query, detected_lang = listen_with_multilingual_support()
            if query:
                st.session_state.chat_history.append({'type': 'user', 'content': f"🎤 {query}", 'timestamp': datetime.now()})
                macro_summary_str = f"{target_calories} Kcal, {protein_g}g Protein"
                raw_response = run_ai_engine(query, macro_summary_str)
                final_response = translate_text(raw_response, dest_lang=language_codes[language_choice])
                st.session_state.chat_history.append({'type': 'bot', 'content': final_response, 'timestamp': datetime.now()})
                speak_async(final_response)
                st.session_state.listening = False
                st.rerun()
            else:
                st.session_state.listening = False
                st.rerun()

# =========================================================
# TAB 2: AI & GEMINI ARCHITECTURE SETTINGS
# =========================================================
with tab_settings:
    st.markdown("### ⚙️ AI Engine & Gemini Architecture Settings")
    st.write("Manage model parameters, load key from .env file or text box, and fetch live available models from Google API.")

    st.markdown('<div class="gym-card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 1. Select Active AI Intelligence Mode:")
    st.session_state.ai_engine_choice = st.radio(
        "Choose Engine Provider for Chat Responses:",
        [
            "♊ Google Gemini API (Cloud LLM - Recommended)",
            "⚡ Local RAG (Fast, 100% Offline)",
            "🦙 Ollama Local LLM (Self-Hosted)"
        ],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ♊ GEMINI API & AUTOMATIC LIVE MODEL FETCH SECTION
    st.markdown('<div class="gym-card">', unsafe_allow_html=True)
    st.markdown("#### ♊ Google Gemini API Key & Live Model Fetcher")
    
    current_env_key = get_env_gemini_key()
    key_status_label = "🟢 Key Loaded from .env File" if current_env_key else "⚪ No .env Key Found (Enter manually below)"
    st.markdown(f"**API Key Source Status:** `{key_status_label}`")
    
    api_key_input = st.text_input(
        "Google Gemini API Key (Loaded from .env or enter below):",
        value=st.session_state.gemini_api_key or current_env_key,
        type="password",
        key="settings_gemini_key_box",
        help="Reads automatically from .env file (GEMINI_API_KEY) or input box!"
    )
    
    if api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input
        if api_key_input:
            fetched_models, _ = fetch_live_gemini_models_from_api(api_key_input)
            if fetched_models:
                st.session_state.available_gemini_models = fetched_models

    col_fetch1, col_fetch2 = st.columns([1, 1])
    with col_fetch1:
        if st.button("🔄 Fetch Live Gemini Models via API Code", key="fetch_gemini_models_btn"):
            with st.spinner("Executing Python API code to fetch live models (HTTP Status 200 check)..."):
                key_to_use = st.session_state.gemini_api_key or get_env_gemini_key()
                fetched_models, fetch_msg = fetch_live_gemini_models_from_api(key_to_use)
                if fetched_models:
                    st.session_state.available_gemini_models = fetched_models
                    st.success(fetch_msg)
                else:
                    st.error(fetch_msg)

    with col_fetch2:
        st.session_state.gemini_model = st.selectbox(
            "Select Active Gemini Model (Fetched via API Code):",
            st.session_state.available_gemini_models,
            index=0 if st.session_state.gemini_model not in st.session_state.available_gemini_models else st.session_state.available_gemini_models.index(st.session_state.gemini_model),
            key="settings_gemini_model_box"
        )
        
    st.markdown("##### 📋 Live Models Available for Selection:")
    models_html = " ".join([f'<span class="gym-badge badge-info">{m}</span>' for m in st.session_state.available_gemini_models])
    st.markdown(f'<div style="margin-top: 5px; margin-bottom: 15px;">{models_html}</div>', unsafe_allow_html=True)
    
    if st.button("🧪 Test Gemini API Connection", key="test_gemini_btn_main"):
        with st.spinner("Connecting to Google Gemini API..."):
            key_to_use = st.session_state.gemini_api_key or get_env_gemini_key()
            success, msg = test_gemini_connection(key_to_use, st.session_state.gemini_model)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)

    # 🌐 20 LANGUAGES OVERVIEW
    st.markdown('<div class="gym-card">', unsafe_allow_html=True)
    st.markdown("#### 🌐 20 Supported Response Languages")
    st.write("FitAI automatically translates Gemini AI responses into your preferred language using deep neural translation:")
    
    lang_cols = st.columns(5)
    for i, lang_name in enumerate(language_codes.keys()):
        lang_cols[i % 5].markdown(f"• **{lang_name}** ({language_codes[lang_name]})")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3: HEART RATE AI SIGNAL ANALYTICS
# =========================================================
with tab_analytics:
    st.markdown("### 📊 Workout Heart Rate Signal AI Diagnostics (Embedded PyTorch Model)")
    st.write("Analyze workout heart rate / ECG signals using the **PyTorch CNN neural network checkpoint (`heart_model.pt`)** trained from `Embedded.ipynb`.")

    models_dict = heart_model.get_initialized_models()
    selected_model_name = st.selectbox("🧠 Select PyTorch Checkpoint Architecture:", list(models_dict.keys()), index=0)
    active_model = models_dict[selected_model_name]
    
    st.markdown("---")
    st.markdown("#### 📥 Select Signal Input Source:")
    input_source = st.radio("Choose Input Type:", ["⚡ Preset Workout Heart Signals", "📁 Upload Custom CSV File"], horizontal=True)

    if input_source == "⚡ Preset Workout Heart Signals":
        sample_choice = st.selectbox(
            "Select Workout Heart Rate Waveform Preset:",
            [
                "Class 0: Normal Rhythm (Optimal Rest / Baseline Cardio)",
                "Class 1: Supraventricular Beat (High Cardio Zone)",
                "Class 2: Ventricular Premature Beat (High Workload Strain)",
                "Class 3: Fusion Beat (Recovery Fatigue Accumulation)",
                "Class 4: Unclassifiable Anomaly (Peak Heart Rate Alert)"
            ],
            index=0
        )
        target_class_id = int(sample_choice.split(":")[0].replace("Class ", ""))
        signal_wave = heart_model.generate_sample_signal(target_class_id)
    else:
        uploaded_csv = st.file_uploader("Upload CSV containing 187-sample signal numeric values:", type=["csv"])
        if uploaded_csv is not None:
            signal_wave = heart_model.parse_csv_signal(uploaded_csv)
            st.success("✅ Successfully parsed uploaded CSV signal!")
        else:
            st.info("💡 Upload a CSV file or fallback preset will be used.")
            signal_wave = heart_model.generate_sample_signal(0)

    # Classify signal
    diagnostic_result = heart_model.predict_heart_signal(active_model, signal_wave)
    diagnostic_result['model_name'] = selected_model_name
    st.session_state.last_heart_diagnostic = diagnostic_result

    # Display Results
    st.markdown('<div class="gym-card">', unsafe_allow_html=True)
    st.markdown(f"#### 🔍 Classification Result: **{diagnostic_result['title']}** ({diagnostic_result['code']})")
    st.markdown(f"<span class='gym-badge badge-{diagnostic_result['status']}'>{diagnostic_result['status'].upper()}</span>", unsafe_allow_html=True)
    st.write(diagnostic_result['description'])
    
    # Plot Signal Waveform
    fig_sig, ax_sig = plt.subplots(figsize=(10, 2.5), facecolor='#0b0f19')
    ax_sig.set_facecolor('#111827')
    ax_sig.plot(signal_wave, color='#00f2fe', linewidth=2.2)
    ax_sig.set_title(f"187-Sample Workout Heart Signal Waveform ({selected_model_name})", color='white', fontsize=10.5)
    ax_sig.tick_params(colors='white')
    ax_sig.grid(True, linestyle=':', alpha=0.3, color='white')
    st.pyplot(fig_sig)
    
    # Probabilities
    st.markdown("##### 📈 Model Confidence Spectrum (5 AAMI Workout Classes):")
    p_cols = st.columns(5)
    p_labels = ["Normal (N)", "Cardio (S)", "Strain (V)", "Recovery (F)", "Anomaly (Q)"]
    for idx, (col_p, lbl) in enumerate(zip(p_cols, p_labels)):
        val = diagnostic_result['probabilities'][idx]
        col_p.metric(lbl, f"{val*100:.1f}%")
        col_p.progress(float(val))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Embedded Model Benchmarks & Robustness (Embedded.ipynb)")
    
    b_col1, b_col2 = st.columns([3, 2])
    with b_col1:
        st.markdown("#### ⚡ Model Complexity & FLOPs Benchmark")
        df_bench = heart_model.get_benchmark_comparison()
        st.dataframe(df_bench, use_container_width=True)
        
        st.markdown("#### 🧩 Confusion Matrix (LightweightCNN)")
        fig_cm = heart_model.plot_confusion_matrix()
        st.pyplot(fig_cm)

    with b_col2:
        st.markdown(r"#### 🛡️ Gaussian Noise Robustness ($\sigma$)")
        st.write("Accuracy evaluated under motion artifacts and sensor noise.")
        fig_noise = heart_model.plot_noise_robustness_chart()
        st.pyplot(fig_noise)

# =========================================================
# TAB 4: CONSULTATION REPORT & PDF
# =========================================================
with tab_report:
    st.markdown("### 📄 Generate FitAI Workout & Consultation Report")
    st.write("Export a formatted PDF report containing athlete info, dynamic macro targets, Q&A summaries, and heart signal diagnostic results.")
    
    if not st.session_state.chat_history and not st.session_state.last_heart_diagnostic:
        st.info("💬 Start a conversation in the Coach tab or run a Heart Signal diagnostic to generate a report.")
    else:
        athlete_info = {
            'name': st.session_state.get('athlete_name', 'Alex Gymnast'),
            'age': st.session_state.get('athlete_age', 24),
            'gender': st.session_state.get('athlete_gender', 'Male'),
            'goal': st.session_state.get('fitness_goal', 'Muscle Hypertrophy'),
            'weight': st.session_state.get('athlete_weight', 75.0),
            'height': st.session_state.get('athlete_height', 178.0),
            'workout_days': st.session_state.get('workout_days', 5)
        }
        
        if st.button("📋 Generate Consultation PDF Report", use_container_width=True):
            with st.spinner("Generating FitAI Workout Report..."):
                pdf_bytes = generate_pdf_report(
                    st.session_state.chat_history,
                    athlete_info,
                    st.session_state.last_heart_diagnostic,
                    macro_data
                )
                if pdf_bytes:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="FitAI_Workout_Report_{timestamp}.pdf" style="background: linear-gradient(135deg, #ff6b00, #00f2fe); color: white; padding: 12px 24px; border-radius: 12px; text-decoration: none; font-weight: bold; display: inline-block;">📄 Download Workout Report (PDF)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ PDF Report generated successfully!")

# Footer
st.markdown("""
<br><hr>
<div style="text-align: center; color: #64748b; padding: 15px;">
    <strong>🏋️‍♂️ FitAI - Gym & Fitness Assistant</strong> • Powered by PyTorch & Embedded.ipynb Models
</div>
""", unsafe_allow_html=True)
