"""
FastAPI Medical Assistant Backend
Main application entry point with API routes and configuration
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from datetime import datetime
import logging
from typing import List, Optional

# Import our custom modules
from models.schemas import (
    ChatRequest, 
    ChatResponse, 
    TranslationRequest, 
    TranslationResponse,
    ReportRequest,
    PatientInfo,
    HealthCheckResponse
)
from services.medical_service import MedicalService
from services.translation_service import TranslationService
from services.report_service import ReportService
from services.language_service import LanguageService
from config.settings import get_settings
from utils.logger import setup_logger

# Initialize settings and logger
settings = get_settings()
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Medical Assistant Backend",
    description="A comprehensive medical AI assistant with multilingual support and PDF reporting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
medical_service = MedicalService()
translation_service = TranslationService()
report_service = ReportService()
language_service = LanguageService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("Starting AI Medical Assistant Backend...")
    await medical_service.initialize()
    logger.info("Medical service initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down AI Medical Assistant Backend...")

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify service status
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services={
            "medical_ai": medical_service.is_healthy(),
            "translation": translation_service.is_healthy(),
            "report_generator": report_service.is_healthy()
        }
    )

@app.get("/")
async def root():
    """
    Root endpoint with basic API information
    """
    return {
        "message": "AI Medical Assistant Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Language detection endpoint
@app.post("/api/detect-language")
async def detect_language(text: str):
    """
    Detect the language of input text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Detected language code and confidence score
    """
    try:
        result = await language_service.detect_language(text)
        return result
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Language detection failed")

# Translation endpoint
@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between languages with medical context awareness
    
    Args:
        request: Translation request containing text, source and target languages
        
    Returns:
        Translated text with medical context preservation
    """
    try:
        # Check if text is medical-related for specialized translation
        is_medical = await medical_service.is_medical_question(request.text)
        
        if is_medical and request.source_language == "ta" and request.target_language == "en":
            # Use specialized Tamil-English medical translation
            translated_text = await translation_service.translate_tamil_medical(request.text)
        else:
            # Use general translation service
            translated_text = await translation_service.translate_text(
                request.text, 
                request.source_language, 
                request.target_language
            )
        
        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            is_medical_context=is_medical
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Translation failed")

# Main chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Process medical chat messages and return AI responses
    
    Args:
        request: Chat request containing message, patient info, and chat history
        
    Returns:
        AI-generated medical response with personalized advice
    """
    try:
        # Validate that the question is medical-related
        if not await medical_service.is_medical_question(request.message):
            return ChatResponse(
                response=await medical_service.get_medical_focus_response(),
                is_medical=False,
                confidence_score=0.0,
                health_tip=None
            )
        
        # Process the medical question
        response_data = await medical_service.process_medical_question(
            question=request.message,
            patient_info=request.patient_info,
            chat_history=request.chat_history,
            language=request.language
        )
        
        # Translate response if needed
        if request.language != "en":
            response_data["response"] = await translation_service.translate_text(
                response_data["response"], 
                "en", 
                request.language
            )
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

# Conversation analysis endpoint
@app.post("/api/analyze-conversation")
async def analyze_conversation(chat_history: List[dict], patient_info: Optional[PatientInfo] = None):
    """
    Analyze conversation patterns and generate medical insights
    
    Args:
        chat_history: List of chat messages
        patient_info: Optional patient information
        
    Returns:
        Medical insights and conversation analysis
    """
    try:
        analysis = await medical_service.analyze_conversation(chat_history, patient_info)
        return analysis
    except Exception as e:
        logger.error(f"Conversation analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Conversation analysis failed")

# PDF report generation endpoint
@app.post("/api/generate-report")
async def generate_medical_report(request: ReportRequest):
    """
    Generate comprehensive PDF medical report from chat history
    
    Args:
        request: Report request containing messages and patient information
        
    Returns:
        PDF file as streaming response
    """
    try:
        # Generate the PDF report
        pdf_buffer = await report_service.generate_comprehensive_report(
            messages=request.messages,
            patient_info=request.patient_info,
            include_insights=True,
            include_recommendations=True
        )
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_report_{timestamp}.pdf"
        
        # Return PDF as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate medical report")

# Medical knowledge search endpoint
@app.post("/api/search-medical-knowledge")
async def search_medical_knowledge(query: str, limit: int = 5):
    """
    Search medical knowledge base for relevant information
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        Relevant medical information and advice
    """
    try:
        results = await medical_service.search_knowledge_base(query, limit)
        return {"results": results}
    except Exception as e:
        logger.error(f"Medical knowledge search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Medical knowledge search failed")

# Get health tips endpoint
@app.get("/api/health-tips")
async def get_health_tips(category: Optional[str] = None, count: int = 3):
    """
    Get personalized health tips
    
    Args:
        category: Optional category filter (sleep, stress, nutrition, etc.)
        count: Number of tips to return
        
    Returns:
        List of relevant health tips
    """
    try:
        tips = await medical_service.get_health_tips(category, count)
        return {"health_tips": tips}
    except Exception as e:
        logger.error(f"Health tips error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get health tips")

# Medical validation endpoint
@app.post("/api/validate-medical")
async def validate_medical_question(text: str):
    """
    Validate if a question is medical-related
    
    Args:
        text: Text to validate
        
    Returns:
        Validation result with confidence score
    """
    try:
        is_medical = await medical_service.is_medical_question(text)
        confidence = await medical_service.get_medical_confidence(text)
        
        return {
            "is_medical": is_medical,
            "confidence_score": confidence,
            "medical_keywords_found": await medical_service.extract_medical_keywords(text)
        }
    except Exception as e:
        logger.error(f"Medical validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Medical validation failed")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
