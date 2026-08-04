from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
from datetime import datetime
import random

app = FastAPI(title="AI Medical Assistant Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PatientInfo(BaseModel):
    name: str
    age: int
    gender: str
    weight: float
    height: float
    bloodGroup: str
    emergencyContact: str

class Message(BaseModel):
    id: str
    type: str
    content: str
    timestamp: str
    language: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    language: str
    patientInfo: PatientInfo
    chatHistory: List[Message]
    isVoice: bool = False

class ReportRequest(BaseModel):
    messages: List[Message]
    patientInfo: PatientInfo

# Medical knowledge base
MEDICAL_RESPONSES = {
    'headache': 'For headaches, try resting in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers like acetaminophen or ibuprofen. If headaches are severe, frequent, or accompanied by other symptoms like fever, vision changes, or neck stiffness, please consult a healthcare professional.',
    
    'fever': 'For fever, rest and stay hydrated with plenty of fluids. You can take acetaminophen or ibuprofen to reduce fever. Monitor your temperature regularly. Seek medical attention if fever is over 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms.',
    
    'cough': 'For cough, stay hydrated and try warm liquids like tea with honey. Avoid irritants like smoke. Over-the-counter cough suppressants may help. See a doctor if cough persists for more than 2 weeks, produces blood, or is accompanied by fever and difficulty breathing.',
    
    'cold': 'For common cold, get plenty of rest and drink fluids. Use saline nasal drops for congestion. Over-the-counter medications can help with symptoms. Most colds resolve in 7-10 days. See a doctor if symptoms worsen or last longer than 10 days.',
    
    'stomach': 'For stomach issues, try eating bland foods like rice, bananas, and toast. Stay hydrated with clear fluids. Avoid spicy, fatty, or dairy foods. Consider over-the-counter antacids for heartburn. Seek medical care if you have severe pain, persistent vomiting, or signs of dehydration.',
    
    'stress': 'For stress management, try deep breathing exercises, regular physical activity, and adequate sleep. Consider meditation or yoga. Maintain social connections and don\'t hesitate to seek professional help if stress becomes overwhelming.',
    
    'sleep': 'For better sleep, maintain a regular sleep schedule, create a comfortable sleep environment, and avoid screens before bedtime. Limit caffeine and large meals before sleep. If sleep problems persist, consult a healthcare provider.',
    
    'anxiety': 'For anxiety, practice relaxation techniques like deep breathing or progressive muscle relaxation. Regular exercise and adequate sleep can help. Consider talking to a mental health professional if anxiety interferes with daily activities.',
    
    'pain': 'For pain management, identify the source and type of pain. Apply ice for acute injuries or heat for muscle tension. Over-the-counter pain relievers can help. If pain is severe, persistent, or interferes with daily activities, consult a healthcare provider.',
    
    'diabetes': 'For diabetes management, monitor blood sugar levels regularly, follow a balanced diet, exercise regularly, and take medications as prescribed. Regular check-ups with your healthcare provider are essential.',
    
    'blood pressure': 'For blood pressure management, maintain a healthy diet low in sodium, exercise regularly, manage stress, and take medications as prescribed. Regular monitoring is important.',
    
    'heart': 'For heart health, maintain a balanced diet, exercise regularly, avoid smoking, limit alcohol, and manage stress. Regular check-ups and following prescribed medications are crucial.',
    
    'weight': 'For healthy weight management, focus on a balanced diet with portion control, regular physical activity, adequate sleep, and stress management. Consult a healthcare provider for personalized advice.'
}

MEDICAL_KEYWORDS = [
    'pain', 'hurt', 'ache', 'fever', 'cough', 'cold', 'flu', 'headache', 'stomach',
    'nausea', 'vomit', 'diarrhea', 'constipation', 'rash', 'itch', 'swelling',
    'bruise', 'cut', 'burn', 'bleeding', 'dizzy', 'tired', 'fatigue', 'weak',
    'sick', 'ill', 'disease', 'infection', 'virus', 'bacteria', 'medicine',
    'medication', 'pill', 'tablet', 'symptom', 'diagnosis', 'treatment',
    'doctor', 'hospital', 'clinic', 'emergency', 'surgery', 'therapy',
    'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 'pressure',
    'diabetes', 'cancer', 'asthma', 'allergy', 'arthritis', 'sleep',
    'insomnia', 'stress', 'anxiety', 'depression', 'mental', 'diet',
    'nutrition', 'vitamin', 'exercise', 'fitness', 'weight', 'health'
]

HEALTH_TIPS = [
    "Remember to stay hydrated throughout the day.",
    "Regular exercise can help improve overall health and mood.",
    "Adequate sleep (7-9 hours) is crucial for good health.",
    "A balanced diet rich in fruits and vegetables supports your immune system.",
    "Don't forget to wash your hands regularly to prevent infections.",
    "Take breaks from screen time to rest your eyes.",
    "Practice stress management techniques like deep breathing.",
    "Regular health check-ups can help prevent serious conditions.",
    "Maintain good posture to prevent back and neck problems.",
    "Limit processed foods and choose whole foods when possible."
]

def is_medical_question(message: str) -> bool:
    """Check if the message is medical-related"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in MEDICAL_KEYWORDS)

def get_medical_response(message: str, patient_info: PatientInfo) -> str:
    """Generate medical response based on message content"""
    message_lower = message.lower()
    
    # Find relevant response
    response = "I understand your concern. While I can provide general health information, it's important to consult a healthcare professional for proper diagnosis and treatment. Your health and safety are the top priority."
    
    for condition, advice in MEDICAL_RESPONSES.items():
        if condition in message_lower:
            response = advice
            break
    
    # Add personalized touch
    if patient_info.name:
        response = f"Hello {patient_info.name}, {response}"
    
    # Add age-specific advice
    if patient_info.age > 65:
        response += "\n\nAs someone over 65, it's especially important to monitor your symptoms closely and don't hesitate to contact your healthcare provider if you have any concerns."
    elif patient_info.age < 18:
        response += "\n\nFor pediatric concerns, it's always best to consult with a pediatrician or your child's healthcare provider."
    
    # Add health tip
    random_tip = random.choice(HEALTH_TIPS)
    response += f"\n\n💡 Health Tip: {random_tip}"
    
    # Add disclaimer
    response += "\n\n⚠️ This information is for educational purposes only and should not replace professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
    
    return response

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Process chat message and return AI response"""
    try:
        # Check if message is medical-related
        if not is_medical_question(request.message):
            return {
                "response": "I'm Dr. AI, your specialized medical assistant. I'm designed to help with health-related questions and medical concerns. For general questions, I'd recommend consulting other AI assistants. How can I help you with your health today?"
            }
        
        # Generate medical response
        response = get_medical_response(request.message, request.patientInfo)
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/generate-report")
async def generate_report(request: ReportRequest):
    """Generate medical report from chat history"""
    try:
        current_date = datetime.now().strftime("%B %d, %Y")
        user_messages = [msg for msg in request.messages if msg.type == 'user']
        bot_messages = [msg for msg in request.messages if msg.type == 'bot']
        
        report_content = f"""
🏥 DR. AI MEDICAL CONSULTATION REPORT
Generated on: {current_date}

👤 PATIENT INFORMATION
Name: {request.patientInfo.name or 'Not provided'}
Age: {request.patientInfo.age or 'Not provided'}
Gender: {request.patientInfo.gender or 'Not provided'}
Weight: {request.patientInfo.weight or 'Not provided'} kg
Height: {request.patientInfo.height or 'Not provided'} cm
Blood Group: {request.patientInfo.bloodGroup or 'Not provided'}
Emergency Contact: {request.patientInfo.emergencyContact or 'Not provided'}

📊 CONSULTATION SUMMARY
Total Messages: {len(request.messages)}
Patient Concerns: {len(user_messages)}
AI Recommendations: {len(bot_messages)}

🤔 PATIENT CONCERNS
{chr(10).join([f"{i+1}. {msg.content}" for i, msg in enumerate(user_messages)])}

💡 AI RECOMMENDATIONS
{chr(10).join([f"{i+1}. {msg.content[:200]}..." if len(msg.content) > 200 else f"{i+1}. {msg.content}" for i, msg in enumerate(bot_messages)])}

🔍 KEY MEDICAL INSIGHTS
• Medical consultation completed successfully
• Patient concerns were addressed with appropriate guidance
• Professional medical consultation recommended for serious concerns
• Health education and preventive care information provided

💡 PERSONALIZED RECOMMENDATIONS
• Continue monitoring symptoms and seek professional help if they worsen
• Maintain a healthy lifestyle with proper diet, exercise, and sleep
• Schedule regular check-ups with healthcare provider
• Follow the specific advice provided during the consultation

⚠️ MEDICAL DISCLAIMER
This report is generated by Dr. AI, an artificial intelligence medical assistant. 
The information provided is for educational and informational purposes only and 
should not be considered as medical advice, diagnosis, or treatment.

Always consult with qualified healthcare professionals for proper medical 
diagnosis, treatment, and care. This report is not a substitute for 
professional medical consultation.

If you are experiencing a medical emergency, please contact emergency 
services immediately.
"""
        
        return {"report": report_content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Medical Assistant Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("🏥 Starting AI Medical Assistant Backend...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
