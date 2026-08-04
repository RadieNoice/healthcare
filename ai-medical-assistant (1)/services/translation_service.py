"""
Translation Service
Handles text translation with medical context awareness
"""

import asyncio
from typing import Optional, Dict, Any
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import google.generativeai as genai

from config.settings import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# Set seed for consistent language detection
DetectorFactory.seed = 0

class TranslationService:
    """
    Service for handling text translation with medical context awareness
    """
    
    def __init__(self):
        self.is_initialized = False
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
            "zh": "Chinese",
            "ja": "Japanese"
        }
        
        # Medical keywords for context detection
        self.medical_keywords = [
            'pain', 'fever', 'headache', 'cough', 'cold', 'symptom', 'doctor', 
            'medicine', 'treatment', 'health', 'medical', 'disease', 'infection',
            'blood', 'heart', 'lung', 'stomach', 'bone', 'muscle', 'joint'
        ]
        
        # Initialize Google Generative AI if available
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Google Generative AI configured for translation")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini for translation: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        self.is_initialized = True

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of input text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected language, confidence, and medical context
        """
        try:
            # Detect language
            detected_lang = detect(text)
            
            # Check if text appears to be medical
            is_medical = any(keyword in text.lower() for keyword in self.medical_keywords)
            
            # Simple confidence calculation (langdetect doesn't provide confidence)
            confidence = 0.9 if len(text) > 20 else 0.7
            
            return {
                "language": detected_lang,
                "confidence": confidence,
                "is_medical_text": is_medical,
                "supported": detected_lang in self.supported_languages
            }
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_medical_text": False,
                "supported": False
            }

    async def translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        medical_context: bool = True
    ) -> str:
        """
        Translate text between languages
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            medical_context: Whether to use medical translation context
            
        Returns:
            Translated text
        """
        try:
            # If source and target are the same, return original text
            if source_lang == target_lang:
                return text
            
            # Use Google Translator for general translation
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(text)
            
            logger.info(f"Translated text from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text

    async def translate_tamil_medical(self, tamil_text: str) -> str:
        """
        Specialized Tamil to English medical translation using Gemini AI
        
        Args:
            tamil_text: Tamil text to translate
            
        Returns:
            English translation with medical context preserved
        """
        if not self.gemini_model:
            # Fallback to regular translation
            return await self.translate_text(tamil_text, "ta", "en")
        
        try:
            prompt = f"""
            You are a professional bilingual translator specializing in Tamil ↔ English medical translations.

            Task:
            1. Translate the following Tamil text into natural, fluent English.
            2. If the text contains medical terms, ensure accurate medical context is preserved without oversimplification.
            3. If the text is unrelated to medicine, translate it normally without adding any medical interpretation.
            4. Maintain meaning, tone, and clarity exactly as intended in the original.

            Tamil:
            {tamil_text}

            English:
            """
            
            response = self.gemini_model.generate_content(prompt)
            translated_text = response.text.strip()
            
            # Validate translation contains medical context if original was medical
            if any(keyword in translated_text.lower() for keyword in self.medical_keywords):
                logger.info("Medical context preserved in Tamil translation")
                return translated_text
            else:
                # If no medical context detected, provide medical consultation prompt
                return "Medical consultation required. Please describe your health symptoms or concerns."
                
        except Exception as e:
            logger.error(f"Tamil medical translation error: {e}")
            # Fallback to regular translation
            return await self.translate_text(tamil_text, "ta", "en")

    async def batch_translate(
        self, 
        texts: list, 
        source_lang: str, 
        target_lang: str
    ) -> list:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        try:
            translated_texts = []
            
            for text in texts:
                translated = await self.translate_text(text, source_lang, target_lang)
                translated_texts.append(translated)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            return translated_texts
            
        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            return texts  # Return original texts if translation fails

    async def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages
        
        Returns:
            Dictionary of language codes and names
        """
        return self.supported_languages

    async def validate_language_code(self, lang_code: str) -> bool:
        """
        Validate if a language code is supported
        
        Args:
            lang_code: Language code to validate
            
        Returns:
            Boolean indicating if language is supported
        """
        return lang_code in self.supported_languages

    def is_healthy(self) -> bool:
        """Check if the translation service is healthy"""
        return self.is_initialized
