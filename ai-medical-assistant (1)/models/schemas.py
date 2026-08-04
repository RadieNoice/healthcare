"""
Pydantic models for request/response schemas
Defines the data structures used throughout the API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class LanguageCode(str, Enum):
    """Supported language codes"""
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    GUJARATI = "gu"
    KOREAN = "ko"
    TURKISH = "tr"
    GERMAN = "de"
    FRENCH = "fr"
    ARABIC = "ar"
    URDU = "ur"
    CHINESE = "zh"
    JAPANESE = "ja"

class PatientInfo(BaseModel):
    """Patient information model"""
    name: Optional[str] = Field(None, description="Patient's full name")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient's age")
    gender: Optional[str] = Field(None, description="Patient's gender")
    weight: Optional[float] = Field(None, ge=0, le=500, description="Patient's weight in kg")
    height: Optional[float] = Field(None, ge=0, le=300, description="Patient's height in cm")
    blood_group: Optional[str] = Field(None, description="Patient's blood group")
    emergency_contact: Optional[str] = Field(None, description="Emergency contact information")
    medical_history: Optional[List[str]] = Field(default_factory=list, description="Previous medical conditions")
    current_medications: Optional[List[str]] = Field(default_factory=list, description="Current medications")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies")

class ChatMessage(BaseModel):
    """Individual chat message model"""
    id: str = Field(..., description="Unique message identifier")
    type: str = Field(..., description="Message type: 'user' or 'bot'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    language: Optional[str] = Field(None, description="Message language code")
    confidence_score: Optional[float] = Field(None, description="AI confidence score")

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, description="User's medical question")
    language: str = Field(default="en", description="Preferred response language")
    patient_info: Optional[PatientInfo] = Field(None, description="Patient information")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Previous chat messages")
    is_voice: bool = Field(default=False, description="Whether input was from voice")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI-generated medical response")
    is_medical: bool = Field(..., description="Whether question was medical-related")
    confidence_score: float = Field(..., description="AI confidence in response")
    health_tip: Optional[str] = Field(None, description="Additional health tip")
    medical_keywords: List[str] = Field(default_factory=list, description="Detected medical keywords")
    recommendations: List[str] = Field(default_factory=list, description="Medical recommendations")
    urgency_level: str = Field(default="low", description="Urgency level: low, medium, high, emergency")
    follow_up_needed: bool = Field(default=False, description="Whether follow-up is recommended")

class TranslationRequest(BaseModel):
    """Translation request model"""
    text: str = Field(..., min_length=1, description="Text to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    medical_context: bool = Field(default=True, description="Whether to use medical translation context")

class TranslationResponse(BaseModel):
    """Translation response model"""
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Detected/specified source language")
    target_language: str = Field(..., description="Target language")
    is_medical_context: bool = Field(..., description="Whether medical context was used")
    confidence_score: float = Field(default=1.0, description="Translation confidence")

class ReportRequest(BaseModel):
    """Medical report generation request"""
    messages: List[ChatMessage] = Field(..., description="Chat history for report")
    patient_info: Optional[PatientInfo] = Field(None, description="Patient information")
    include_insights: bool = Field(default=True, description="Include AI-generated insights")
    include_recommendations: bool = Field(default=True, description="Include personalized recommendations")
    report_type: str = Field(default="comprehensive", description="Type of report to generate")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, bool] = Field(..., description="Individual service statuses")

class MedicalInsight(BaseModel):
    """Medical insight model"""
    category: str = Field(..., description="Insight category")
    insight: str = Field(..., description="Generated insight")
    confidence: float = Field(..., description="Confidence in insight")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence")

class ConversationAnalysis(BaseModel):
    """Conversation analysis model"""
    total_messages: int = Field(..., description="Total number of messages")
    medical_topics: List[str] = Field(..., description="Identified medical topics")
    sentiment_analysis: Dict[str, float] = Field(..., description="Sentiment scores")
    urgency_indicators: List[str] = Field(default_factory=list, description="Urgency indicators found")
    insights: List[MedicalInsight] = Field(default_factory=list, description="Generated insights")
    recommendations: List[str] = Field(default_factory=list, description="Personalized recommendations")

class LanguageDetectionResult(BaseModel):
    """Language detection result model"""
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Detection confidence")
    is_medical_text: bool = Field(..., description="Whether text appears medical")

# Validation functions
@validator('age')
def validate_age(cls, v):
    if v is not None and (v < 0 or v > 120):
        raise ValueError('Age must be between 0 and 120')
    return v

@validator('weight')
def validate_weight(cls, v):
    if v is not None and (v < 0 or v > 500):
        raise ValueError('Weight must be between 0 and 500 kg')
    return v

@validator('height')
def validate_height(cls, v):
    if v is not None and (v < 0 or v > 300):
        raise ValueError('Height must be between 0 and 300 cm')
    return v
