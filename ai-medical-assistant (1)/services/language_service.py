"""
Language Detection Service
Handles language detection and analysis
"""

import asyncio
from typing import Dict, Any, Optional
from langdetect import detect, detect_langs, DetectorFactory
import re

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0

class LanguageService:
    """
    Service for language detection and analysis
    """
    
    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil", 
            "te": "Telugu",
            "gu": "Gujarati",
            "ko": "Korean",
            "tr": "Turkish",
            "de": "German",
            "fr": "French",
            "ar": "Arabic",
            "ur": "Urdu",
            "zh-cn": "Chinese",
            "ja": "Japanese"
        }
        
        # Medical keywords in different languages
        self.medical_keywords = {
            "en": [
                'pain', 'fever', 'headache', 'cough', 'cold', 'symptom', 'doctor',
                'medicine', 'treatment', 'health', 'medical', 'disease', 'infection'
            ],
            "ta": [
                'வலி', 'காய்ச்சல்', 'தலைவலி', 'இருமல்', 'சளி', 'அறிகுறி', 'மருத்துவர்',
                'மருந்து', 'சிகிச்சை', 'உடல்நலம்', 'மருத்துவ', 'நோய்', 'தொற்று'
            ],
            "hi": [
                'दर्द', 'बुखार', 'सिरदर्द', 'खांसी', 'सर्दी', 'लक्षण', 'डॉक्टर',
                'दवा', 'इलाज', 'स्वास्थ्य', 'चिकित्सा', 'बीमारी', 'संक्रमण'
            ]
        }

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text with confidence scoring
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language detection results
        """
        try:
            if not text or len(text.strip()) < 3:
                return {
                    "language": "unknown",
                    "confidence": 0.0,
                    "is_medical_text": False,
                    "supported": False,
                    "alternatives": []
                }
            
            # Clean text for better detection
            cleaned_text = self._clean_text(text)
            
            # Detect primary language
            primary_lang = detect(cleaned_text)
            
            # Get alternative languages with probabilities
            lang_probs = detect_langs(cleaned_text)
            alternatives = [
                {"language": lang.lang, "confidence": lang.prob}
                for lang in lang_probs[:3]
            ]
            
            # Get confidence for primary language
            primary_confidence = next(
                (lang.prob for lang in lang_probs if lang.lang == primary_lang),
                0.5
            )
            
            # Check if text is medical
            is_medical = await self._is_medical_text(cleaned_text, primary_lang)
            
            # Check if language is supported
            is_supported = primary_lang in self.supported_languages
            
            return {
                "language": primary_lang,
                "confidence": primary_confidence,
                "is_medical_text": is_medical,
                "supported": is_supported,
                "alternatives": alternatives,
                "language_name": self.supported_languages.get(primary_lang, "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_medical_text": False,
                "supported": False,
                "alternatives": [],
                "error": str(e)
            }

    def _clean_text(self, text: str) -> str:
        """
        Clean text for better language detection
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    async def _is_medical_text(self, text: str, language: str) -> bool:
        """
        Check if text contains medical content
        
        Args:
            text: Text to analyze
            language: Detected language
            
        Returns:
            Boolean indicating if text is medical
        """
        text_lower = text.lower()
        
        # Check for medical keywords in detected language
        medical_keywords = self.medical_keywords.get(language, self.medical_keywords["en"])
        
        medical_word_count = sum(1 for keyword in medical_keywords if keyword in text_lower)
        
        # Also check English medical keywords as fallback
        if language != "en":
            english_keywords = self.medical_keywords["en"]
            medical_word_count += sum(1 for keyword in english_keywords if keyword in text_lower)
        
        return medical_word_count > 0

    async def get_language_info(self, lang_code: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a language
        
        Args:
            lang_code: Language code
            
        Returns:
            Language information dictionary
        """
        if lang_code not in self.supported_languages:
            return None
        
        return {
            "code": lang_code,
            "name": self.supported_languages[lang_code],
            "supported": True,
            "has_medical_keywords": lang_code in self.medical_keywords,
            "medical_keyword_count": len(self.medical_keywords.get(lang_code, []))
        }

    async def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.supported_languages

    def is_healthy(self) -> bool:
        """Check if language service is healthy"""
        return True

file="services/report_service.py"
"""
Report Generation Service
Handles PDF report generation from chat history
"""

import asyncio
import io
import tempfile
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from models.schemas import PatientInfo, ChatMessage
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ReportService:
    """
    Service for generating comprehensive medical reports from chat history
    """
    
    def __init__(self):
        self.is_initialized = True

    async def generate_comprehensive_report(
        self,
        messages: List[Dict[str, Any]],
        patient_info: Optional[PatientInfo] = None,
        include_insights: bool = True,
        include_recommendations: bool = True
    ) -> io.BytesIO:
        """
        Generate a comprehensive PDF medical report
        
        Args:
            messages: Chat history messages
            patient_info: Patient information
            include_insights: Whether to include AI insights
            include_recommendations: Whether to include recommendations
            
        Returns:
            PDF file as BytesIO buffer
        """
        try:
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create PDF document
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_LEFT,
                textColor=colors.darkgreen
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12,
                alignment=TA_JUSTIFY
            )
            
            # Header
            story.append(Paragraph("🏥 Dr. AI Medical Consultation Report", title_style))
            story.append(Spacer(1, 20))
            
            # Patient Information Section
            if patient_info:
                story.append(Paragraph("👤 Patient Information", subtitle_style))
                
                patient_data = [
                    ["Name", patient_info.name or 'Not provided'],
                    ["Age", str(patient_info.age) if patient_info.age else 'Not provided'],
                    ["Gender", patient_info.gender or 'Not provided'],
                    ["Weight", f"{patient_info.weight} kg" if patient_info.weight else 'Not provided'],
                    ["Height", f"{patient_info.height} cm" if patient_info.height else 'Not provided'],
                    ["Blood Group", patient_info.blood_group or 'Not provided'],
                    ["Emergency Contact", patient_info.emergency_contact or 'Not provided']
                ]
                
                patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(patient_table)
                story.append(Spacer(1, 20))
            
            # Report metadata
            current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"<b>Report Generated:</b> {current_time}", normal_style))
            story.append(Paragraph(f"<b>Total Messages:</b> {len(messages)}", normal_style))
            story.append(Spacer(1, 20))
            
            # Process messages
            user_concerns, ai_recommendations = self._process_messages(messages)
            
            # Summary section
            story.append(Paragraph("📋 Consultation Summary", subtitle_style))
            story.append(Paragraph(f"<b>Number of Health Concerns Discussed:</b> {len(user_concerns)}", normal_style))
            story.append(Paragraph(f"<b>Number of AI Recommendations Provided:</b> {len(ai_recommendations)}", normal_style))
            story.append(Spacer(1, 20))
            
            # User concerns section
            if user_concerns:
                story.append(Paragraph("🤔 Patient Concerns & Symptoms", subtitle_style))
                for i, concern in enumerate(user_concerns, 1):
                    story.append(Paragraph(f"<b>{i}. [{concern['timestamp']}]</b> {concern['concern']}", normal_style))
                story.append(Spacer(1, 20))
            
            # AI recommendations section
            if ai_recommendations:
                story.append(Paragraph("💡 Dr. AI Recommendations", subtitle_style))
                for i, rec in enumerate(ai_recommendations, 1):
                    story.append(Paragraph(f"<b>{i}. [{rec['timestamp']}]</b> {rec['recommendation']}", normal_style))
                story.append(Spacer(1, 20))
            
            # Insights section
            if include_insights:
                story.append(Paragraph("🔍 Key Medical Insights", subtitle_style))
                insights = self._generate_insights(messages, patient_info)
                for insight in insights:
                    story.append(Paragraph(f"• {insight}", normal_style))
                story.append(Spacer(1, 20))
            
            # Recommendations section
            if include_recommendations:
                story.append(Paragraph("💡 Personalized Recommendations", subtitle_style))
                recommendations = self._generate_recommendations(messages, patient_info)
                for rec in recommendations:
                    story.append(Paragraph(f"• {rec}", normal_style))
                story.append(Spacer(1, 20))
            
            # Medical disclaimer
            story.append(Paragraph("⚠ Important Medical Disclaimer", subtitle_style))
            disclaimer_text = """
            This report is generated by Dr. AI, an artificial intelligence medical assistant. 
            The information provided is for educational and informational purposes only and 
            should not be considered as medical advice, diagnosis, or treatment. 
            
            Always consult with qualified healthcare professionals for proper medical 
            diagnosis, treatment, and care. This report is not a substitute for 
            professional medical consultation.
            
            If you are experiencing a medical emergency, please contact emergency 
            services immediately.
            """
            story.append(Paragraph(disclaimer_text, normal_style))
            
            # Build PDF
            doc.build(story)
            
            # Reset buffer position
            buffer.seek(0)
            
            logger.info("Medical report generated successfully")
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating medical report: {e}")
            raise

    def _process_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Process messages to extract user concerns and AI recommendations"""
        user_concerns = []
        ai_recommendations = []
        
        for message in messages:
            if message.get('type') == 'user':
                content = self._clean_message_content(message.get('content', ''))
                if content and self._is_english_content(content):
                    timestamp = self._format_timestamp(message.get('timestamp'))
                    user_concerns.append({
                        'timestamp': timestamp,
                        'concern': content
                    })
            elif message.get('type') == 'bot':
                content = self._clean_message_content(message.get('content', ''))
                if content and self._is_english_content(content):
                    timestamp = self._format_timestamp(message.get('timestamp'))
                    ai_recommendations.append({
                        'timestamp': timestamp,
                        'recommendation': content
                    })
        
        return user_concerns, ai_recommendations

    def _clean_message_content(self, content: str) -> str:
        """Clean message content by removing emoji prefixes"""
        if content.startswith('📝 '):
            content = content[3:]
        elif content.startswith('🎤 '):
            content = content[3:]
        elif content.startswith('🏥 '):
            content = content[3:]
        
        return content.strip()

    def _is_english_content(self, content: str) -> bool:
        """Check if content is in English (exclude Tamil characters)"""
        tamil_chars = ['த', 'ம', 'ன', 'க', 'ப', 'ர', 'ல', 'ட', 'ண', 'ச', 'ய', 'வ', 'ழ', 'ள', 'ற', 'ந']
        return not any(char in content for char in tamil_chars)

    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for display"""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime("%I:%M %p")
            except:
                return "Unknown time"
        elif hasattr(timestamp, 'strftime'):
            return timestamp.strftime("%I:%M %p")
        else:
            return "Unknown time"

    def _generate_insights(self, messages: List[Dict[str, Any]], patient_info: Optional[PatientInfo]) -> List[str]:
        """Generate medical insights from conversation"""
        insights = []
        
        user_messages = [msg for msg in messages if msg.get('type') == 'user']
        
        if len(user_messages) > 5:
            insights.append("Extended consultation indicates thorough discussion of health concerns")
        
        # Analyze message content for patterns
        all_content = ' '.join([msg.get('content', '').lower() for msg in user_messages])
        
        if 'pain' in all_content:
            insights.append("Pain-related symptoms discussed - consider pain management strategies")
        
        if any(word in all_content for word in ['stress', 'anxiety', 'worried']):
            insights.append("Mental health aspects identified - holistic care approach recommended")
        
        if any(word in all_content for word in ['sleep', 'tired', 'fatigue']):
            insights.append("Sleep and energy concerns noted - lifestyle factors may be relevant")
        
        if patient_info and patient_info.age:
            if patient_info.age > 65:
                insights.append("Age-related health considerations important for comprehensive care")
            elif patient_info.age &lt< 18:
                insights.append("Pediatric health factors considered in recommendations")
        
        # Default insights if none generated
        if not insights:
            insights.extend([
                "Medical consultation completed successfully",
                "Patient concerns were addressed with appropriate guidance",
                "Professional medical consultation recommended for serious concerns"
            ])
        
        return insights

    def _generate_recommendations(self, messages: List[Dict[str, Any]], patient_info: Optional[PatientInfo]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        user_messages = [msg for msg in messages if msg.get('type') == 'user']
        all_content = ' '.join([msg.get('content', '').lower() for msg in user_messages])
        
        # Content-based recommendations
        if 'pain' in all_content:
            recommendations.append("Document pain patterns and discuss with healthcare provider")
        
        if any(word in all_content for word in ['stress', 'anxiety']):
            recommendations.append("Consider stress management techniques such as meditation or counseling")
        
        if any(word in all_content for word in ['sleep', 'insomnia']):
            recommendations.append("Focus on sleep hygiene and consider consulting a sleep specialist")
        
        # General recommendations
        recommendations.extend([
            "Continue monitoring symptoms and seek professional help if they worsen",
            "Maintain a healthy lifestyle with proper diet, exercise, and sleep",
            "Schedule regular check-ups with your healthcare provider"
        ])
        
        # Patient-specific recommendations
        if patient_info:
            if patient_info.age and patient_info.age > 65:
                recommendations.append("Consider additional preventive screenings appropriate for your age group")
            
            if patient_info.medical_history:
                recommendations.append("Ensure all healthcare providers are aware of your complete medical history")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def is_healthy(self) -> bool:
        """Check if report service is healthy"""
        return self.is_initialized

file="config/settings.py"
"""
Configuration settings for the FastAPI application
"""

import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # AI Service Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_MODEL: str = "mistral"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    
    # Translation Configuration
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: List[str] = [
        "en", "hi", "ta", "te", "gu", "ko", "tr", 
        "de", "fr", "ar", "ur", "zh", "ja"
    ]
    
    # Medical Configuration
    MEDICAL_SIMILARITY_THRESHOLD: float = 0.5
    MAX_CHAT_HISTORY: int = 100
    
    # Report Configuration
    REPORT_TITLE: str = "Dr. AI Medical Consultation Report"
    INCLUDE_DISCLAIMER: bool = True
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

file="utils/logger.py"
"""
Logging utilities for the application
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = getattr(logging, (level or "INFO").upper())
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


