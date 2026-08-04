"""
Medical AI Service
Handles medical question processing, validation, and response generation
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import google.generativeai as genai

from models.schemas import PatientInfo, ChatMessage, MedicalInsight, ConversationAnalysis
from config.settings import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

class MedicalService:
    """
    Core medical AI service for processing medical questions and generating responses
    """
    
    def __init__(self):
        self.model = None
        self.llm = None
        self.medical_dataset = None
        self.chat_history = None
        self.is_initialized = False
        
        # Medical keywords for validation
        self.medical_keywords = [
            'pain', 'hurt', 'ache', 'fever', 'cough', 'cold', 'flu', 'headache', 'stomach',
            'nausea', 'vomit', 'diarrhea', 'constipation', 'rash', 'itch', 'swelling',
            'bruise', 'cut', 'burn', 'bleeding', 'dizzy', 'tired', 'fatigue', 'weak',
            'sick', 'ill', 'disease', 'infection', 'virus', 'bacteria', 'medicine',
            'medication', 'pill', 'tablet', 'symptom', 'diagnosis', 'treatment',
            'doctor', 'hospital', 'clinic', 'emergency', 'ambulance', 'surgery',
            'operation', 'therapy', 'recovery', 'healing', 'wound', 'injury',
            'bone', 'muscle', 'joint', 'back', 'neck', 'shoulder', 'knee', 'ankle',
            'heart', 'lung', 'liver', 'kidney', 'brain', 'blood', 'pressure',
            'diabetes', 'cancer', 'asthma', 'allergy', 'arthritis', 'sleep',
            'insomnia', 'stress', 'anxiety', 'depression', 'mental', 'diet',
            'nutrition', 'vitamin', 'supplement', 'exercise', 'fitness',
            'weight', 'obesity', 'cholesterol', 'blood sugar', 'hypertension'
        ]
        
        # Medical responses database
        self.medical_responses = {
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
        
        # Health tips categorized by keywords
        self.health_tips = {
            "sleep": [
                "Try to get at least 7-8 hours of sleep each night.",
                "Establish a regular sleep routine to improve sleep quality.",
                "Avoid screens before bed to help your mind relax.",
                "Keep your bedroom cool, dark, and quiet for better sleep.",
                "Avoid caffeine and large meals close to bedtime."
            ],
            "energy": [
                "Make sure you're eating a balanced diet to maintain energy.",
                "Exercise regularly to boost your energy levels.",
                "Stay hydrated throughout the day to avoid fatigue.",
                "Take short breaks during work to prevent burnout.",
                "Consider iron-rich foods if you feel constantly tired."
            ],
            "stress": [
                "Take short breaks throughout the day to reduce stress.",
                "Practice mindfulness or meditation to help manage stress.",
                "Engage in physical activity to reduce anxiety and stress.",
                "Connect with friends and family for emotional support.",
                "Try deep breathing exercises when feeling overwhelmed."
            ],
            "nutrition": [
                "Eat a variety of colorful fruits and vegetables daily.",
                "Choose whole grains over refined grains when possible.",
                "Include lean proteins in your meals for sustained energy.",
                "Limit processed foods and added sugars in your diet.",
                "Stay hydrated with water throughout the day."
            ],
            "general": [
                "Drink plenty of water throughout the day.",
                "Get at least 30 minutes of exercise every day.",
                "Eat a balanced diet rich in fruits and vegetables.",
                "Wash your hands regularly to prevent infections.",
                "Schedule regular check-ups with your healthcare provider."
            ]
        }
        
        # Medical prompt template
        self.medical_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "patient_info"],
            template="""You are Dr. AI, a specialized medical AI assistant. You focus ONLY on health-related questions and medical concerns. 

If the user asks non-medical questions, politely redirect them to your medical focus.

Patient Information: {patient_info}

Previous Conversation: {chat_history}

User Question: {question}

Remember: You are a medical AI doctor, not a general chatbot. Always maintain your medical focus and provide helpful, accurate medical advice while recommending professional consultation for serious concerns. Consider the patient's age, gender, and other relevant factors when providing personalized medical guidance.

Dr. AI Response:"""
        )

    async def initialize(self):
        """Initialize the medical service with AI models and datasets"""
        try:
            logger.info("Initializing Medical Service...")
            
            # Initialize Sentence Transformer model for semantic similarity
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Sentence Transformer: {e}")
                self.model = None
            
            # Initialize LangChain LLM
            try:
                self.llm = OllamaLLM(model="mistral")
                self.chat_history = ChatMessageHistory()
                logger.info("LangChain LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain LLM: {e}")
                self.llm = None
                self.chat_history = None
            
            # Initialize Google Generative AI if API key is available
            if settings.GEMINI_API_KEY:
                try:
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    logger.info("Google Generative AI configured successfully")
                except Exception as e:
                    logger.warning(f"Failed to configure Google Generative AI: {e}")
            
            # Load medical dataset
            self.medical_dataset = self._create_medical_dataset()
            logger.info("Medical dataset loaded successfully")
            
            self.is_initialized = True
            logger.info("Medical Service initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Medical Service: {e}")
            raise

    def _create_medical_dataset(self) -> pd.DataFrame:
        """Create a comprehensive medical dataset"""
        data = {
            'disease': [
                'headache', 'fever', 'cough', 'cold', 'stomach ache', 'back pain',
                'sore throat', 'runny nose', 'fatigue', 'insomnia', 'stress',
                'anxiety', 'depression', 'high blood pressure', 'diabetes',
                'asthma', 'allergies', 'migraine', 'dizziness', 'nausea',
                'joint pain', 'muscle pain', 'chest pain', 'shortness of breath',
                'skin rash', 'constipation', 'diarrhea', 'heartburn', 'bloating'
            ],
            'cure': [
                'Rest in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers. If severe or persistent, consult a doctor.',
                'Rest, stay hydrated, and monitor your temperature. Take acetaminophen if needed. Seek medical attention if fever is high or persistent.',
                'Stay hydrated, use honey for soothing, and rest. Consider over-the-counter cough suppressants. See a doctor if cough persists.',
                'Rest, drink plenty of fluids, and use saline nasal drops. Over-the-counter cold medications may help. Symptoms usually resolve in 7-10 days.',
                'Try gentle abdominal massage, peppermint tea, or over-the-counter antacids. Avoid spicy foods. See a doctor if pain is severe.',
                'Rest, apply heat or ice, and consider over-the-counter pain relievers. Gentle stretching may help. Consult a doctor for persistent pain.',
                'Gargle with warm salt water, use throat lozenges, and stay hydrated. Rest your voice. See a doctor if symptoms worsen.',
                'Use saline nasal spray, stay hydrated, and rest. Over-the-counter decongestants may help. Symptoms usually improve within a week.',
                'Ensure adequate sleep, eat a balanced diet, and exercise regularly. Consider stress management techniques. See a doctor if persistent.',
                'Establish a regular sleep schedule, avoid screens before bed, and create a relaxing bedtime routine. Consider consulting a sleep specialist.',
                'Practice relaxation techniques, exercise regularly, and maintain a healthy work-life balance. Consider therapy if stress is overwhelming.',
                'Practice deep breathing, mindfulness, and consider therapy. Regular exercise and adequate sleep can help. Seek professional help if needed.',
                'Consider therapy, medication if prescribed, and maintain social connections. Exercise and healthy lifestyle habits are important.',
                'Reduce salt intake, exercise regularly, and maintain a healthy weight. Monitor blood pressure regularly and follow doctor\'s advice.',
                'Follow a balanced diet, exercise regularly, and monitor blood sugar levels. Take medications as prescribed and see your doctor regularly.',
                'Avoid triggers, use prescribed inhalers, and maintain good air quality. Have an action plan and see your doctor regularly.',
                'Avoid known allergens, use air purifiers, and consider over-the-counter antihistamines. See an allergist for severe allergies.',
                'Rest in a dark, quiet room. Avoid triggers like certain foods or stress. Consider preventive medications if prescribed.',
                'Sit or lie down to prevent falls. Stay hydrated and avoid sudden movements. See a doctor if dizziness is severe or persistent.',
                'Eat small, bland meals and stay hydrated. Avoid strong odors and fatty foods. See a doctor if nausea is severe or persistent.',
                'Apply ice or heat, rest the affected joint, and consider over-the-counter pain relievers. See a doctor if pain persists.',
                'Rest, apply heat or ice, and gentle stretching. Over-the-counter pain relievers may help. Consult a doctor for severe pain.',
                'Seek immediate medical attention for chest pain. While waiting, sit upright and try to stay calm. Call emergency services if severe.',
                'Sit upright, practice slow deep breathing, and stay calm. If severe or persistent, seek immediate medical attention.',
                'Keep the area clean and dry, avoid irritants, and consider over-the-counter treatments. See a doctor if rash spreads or worsens.',
                'Increase fiber intake, drink more water, and exercise regularly. Consider over-the-counter laxatives if needed. See a doctor if persistent.',
                'Stay hydrated with clear fluids, eat bland foods, and rest. Avoid dairy and fatty foods. See a doctor if severe or persistent.',
                'Avoid trigger foods, eat smaller meals, and consider over-the-counter antacids. Elevate your head when sleeping.',
                'Eat slowly, avoid gas-producing foods, and consider over-the-counter remedies. See a doctor if bloating is severe or persistent.'
            ]
        }
        return pd.DataFrame(data)

    async def is_medical_question(self, question: str) -> bool:
        """
        Check if a question is medical-related using keyword matching
        
        Args:
            question: The question to validate
            
        Returns:
            Boolean indicating if question is medical-related
        """
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.medical_keywords)

    async def get_medical_confidence(self, question: str) -> float:
        """
        Get confidence score for medical question classification
        
        Args:
            question: The question to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        question_lower = question.lower()
        medical_word_count = sum(1 for keyword in self.medical_keywords if keyword in question_lower)
        total_words = len(question_lower.split())
        
        if total_words == 0:
            return 0.0
        
        # Calculate confidence based on medical keyword density
        confidence = min(medical_word_count / total_words * 2, 1.0)
        return confidence

    async def extract_medical_keywords(self, text: str) -> List[str]:
        """
        Extract medical keywords from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of found medical keywords
        """
        text_lower = text.lower()
        found_keywords = [keyword for keyword in self.medical_keywords if keyword in text_lower]
        return found_keywords

    async def get_medical_focus_response(self) -> str:
        """Get a response redirecting to medical focus"""
        responses = [
            "I'm Dr. AI, your specialized medical assistant. I'm designed to help with health-related questions and medical concerns. For general questions, I'd recommend consulting other AI assistants. How can I help you with your health today?",
            "As your AI medical doctor, I focus specifically on health and medical issues. I can help you with symptoms, health advice, medication questions, and wellness tips. What health concern would you like to discuss?",
            "I'm a medical AI assistant, not a general chatbot. I'm here to help with your health questions, symptoms, medical advice, and wellness guidance. What medical issue would you like to discuss?",
            "Hello! I'm your AI medical doctor. I specialize in health-related questions and medical concerns. For non-medical topics, other AI assistants would be better suited. How can I assist you with your health today?"
        ]
        return random.choice(responses)

    async def find_best_medical_response(self, user_input: str) -> str:
        """
        Find the best medical response using semantic similarity or keyword matching
        
        Args:
            user_input: User's medical question
            
        Returns:
            Appropriate medical response
        """
        if self.model is not None and self.medical_dataset is not None:
            try:
                # Use semantic similarity matching
                user_input_embedding = self.model.encode(user_input, convert_to_tensor=True)
                disease_embeddings = self.model.encode(self.medical_dataset['disease'].tolist(), convert_to_tensor=True)
                
                similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
                best_match_idx = similarities.argmax().item()
                best_match_score = similarities[best_match_idx].item()
                
                SIMILARITY_THRESHOLD = 0.5
                
                if best_match_score >= SIMILARITY_THRESHOLD:
                    return self.medical_dataset.iloc[best_match_idx]['cure']
                    
            except Exception as e:
                logger.warning(f"Semantic similarity matching failed: {e}")
        
        # Fallback to keyword matching
        user_input_lower = user_input.lower()
        for condition, response in self.medical_responses.items():
            if condition in user_input_lower:
                return response
        
        # Default response
        return "I understand your concern. While I can provide general health information, it's important to consult a healthcare professional for proper diagnosis and treatment. Your health and safety are the top priority."

    async def get_personalized_health_tip(self, user_input: str, patient_info: Optional[PatientInfo] = None) -> str:
        """
        Get a personalized health tip based on user input and patient information
        
        Args:
            user_input: User's question or concern
            patient_info: Optional patient information
            
        Returns:
            Relevant health tip
        """
        user_input_lower = user_input.lower()
        
        # Determine category based on input
        if any(word in user_input_lower for word in ["tired", "fatigue", "energy", "exhausted"]):
            category = "energy"
        elif any(word in user_input_lower for word in ["sleep", "insomnia", "rest", "bed"]):
            category = "sleep"
        elif any(word in user_input_lower for word in ["stress", "anxious", "worry", "tension"]):
            category = "stress"
        elif any(word in user_input_lower for word in ["diet", "food", "nutrition", "eat"]):
            category = "nutrition"
        else:
            category = "general"
        
        tips = self.health_tips.get(category, self.health_tips["general"])
        selected_tip = random.choice(tips)
        
        # Personalize based on patient info
        if patient_info:
            if patient_info.age and patient_info.age > 65:
                selected_tip += " This is especially important as we age."
            elif patient_info.age and patient_info.age < 18:
                selected_tip += " This is important for healthy growth and development."
        
        return selected_tip

    async def process_medical_question(
        self, 
        question: str, 
        patient_info: Optional[PatientInfo] = None,
        chat_history: List[ChatMessage] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process a medical question and generate a comprehensive response
        
        Args:
            question: The medical question
            patient_info: Optional patient information
            chat_history: Previous chat messages
            language: Response language
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Get base medical response
            base_response = await self.find_best_medical_response(question)
            
            # Personalize response based on patient info
            if patient_info and patient_info.name:
                base_response = f"Hello {patient_info.name}, {base_response}"
            
            # Add age-specific advice
            if patient_info and patient_info.age:
                if patient_info.age > 65:
                    base_response += "\n\nAs someone over 65, it's especially important to monitor your symptoms closely and don't hesitate to contact your healthcare provider if you have any concerns."
                elif patient_info.age < 18:
                    base_response += "\n\nFor pediatric concerns, it's always best to consult with a pediatrician or your child's healthcare provider."
            
            # Get health tip
            health_tip = await self.get_personalized_health_tip(question, patient_info)
            
            # Determine urgency level
            urgency_level = await self._assess_urgency(question)
            
            # Extract medical keywords
            medical_keywords = await self.extract_medical_keywords(question)
            
            # Get confidence score
            confidence_score = await self.get_medical_confidence(question)
            
            # Add medical disclaimer
            base_response += "\n\n⚠️ This information is for educational purposes only and should not replace professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
            
            return {
                "response": base_response,
                "is_medical": True,
                "confidence_score": confidence_score,
                "health_tip": health_tip,
                "medical_keywords": medical_keywords,
                "recommendations": await self._generate_recommendations(question, patient_info),
                "urgency_level": urgency_level,
                "follow_up_needed": urgency_level in ["high", "emergency"]
            }
            
        except Exception as e:
            logger.error(f"Error processing medical question: {e}")
            raise

    async def _assess_urgency(self, question: str) -> str:
        """
        Assess the urgency level of a medical question
        
        Args:
            question: The medical question
            
        Returns:
            Urgency level: low, medium, high, emergency
        """
        question_lower = question.lower()
        
        # Emergency keywords
        emergency_keywords = [
            "chest pain", "heart attack", "stroke", "can't breathe", "choking",
            "severe bleeding", "unconscious", "overdose", "suicide", "emergency"
        ]
        
        # High urgency keywords
        high_urgency_keywords = [
            "severe pain", "high fever", "difficulty breathing", "blood",
            "severe headache", "vision loss", "paralysis"
        ]
        
        # Medium urgency keywords
        medium_urgency_keywords = [
            "persistent", "worsening", "fever", "pain", "infection", "swelling"
        ]
        
        if any(keyword in question_lower for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in question_lower for keyword in high_urgency_keywords):
            return "high"
        elif any(keyword in question_lower for keyword in medium_urgency_keywords):
            return "medium"
        else:
            return "low"

    async def _generate_recommendations(self, question: str, patient_info: Optional[PatientInfo] = None) -> List[str]:
        """
        Generate personalized medical recommendations
        
        Args:
            question: The medical question
            patient_info: Optional patient information
            
        Returns:
            List of recommendations
        """
        recommendations = []
        question_lower = question.lower()
        
        # General recommendations based on symptoms
        if "headache" in question_lower:
            recommendations.extend([
                "Stay hydrated by drinking plenty of water",
                "Rest in a quiet, dark room",
                "Consider over-the-counter pain relievers if appropriate"
            ])
        elif "fever" in question_lower:
            recommendations.extend([
                "Monitor your temperature regularly",
                "Stay hydrated with fluids",
                "Get plenty of rest"
            ])
        elif "stress" in question_lower or "anxiety" in question_lower:
            recommendations.extend([
                "Practice deep breathing exercises",
                "Consider meditation or mindfulness techniques",
                "Maintain regular physical activity"
            ])
        
        # Age-specific recommendations
        if patient_info and patient_info.age:
            if patient_info.age > 65:
                recommendations.append("Schedule regular check-ups with your healthcare provider")
            elif patient_info.age < 18:
                recommendations.append("Discuss symptoms with a pediatrician")
        
        # General health recommendations
        recommendations.extend([
            "Maintain a balanced diet rich in fruits and vegetables",
            "Get adequate sleep (7-9 hours per night)",
            "Stay physically active as appropriate for your condition"
        ])
        
        return recommendations[:5]  # Limit to 5 recommendations

    async def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the medical knowledge base for relevant information
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant medical information
        """
        results = []
        query_lower = query.lower()
        
        # Search through medical responses
        for condition, response in self.medical_responses.items():
            if condition in query_lower or any(word in response.lower() for word in query_lower.split()):
                results.append({
                    "condition": condition,
                    "advice": response,
                    "relevance_score": self._calculate_relevance(query_lower, condition, response)
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]

    def _calculate_relevance(self, query: str, condition: str, response: str) -> float:
        """Calculate relevance score for search results"""
        query_words = set(query.lower().split())
        condition_words = set(condition.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        condition_overlap = len(query_words.intersection(condition_words))
        response_overlap = len(query_words.intersection(response_words))
        
        # Weight condition matches higher than response matches
        relevance = (condition_overlap * 2 + response_overlap) / len(query_words)
        return min(relevance, 1.0)

    async def get_health_tips(self, category: Optional[str] = None, count: int = 3) -> List[str]:
        """
        Get health tips, optionally filtered by category
        
        Args:
            category: Optional category filter
            count: Number of tips to return
            
        Returns:
            List of health tips
        """
        if category and category in self.health_tips:
            tips = self.health_tips[category]
        else:
            # Combine all tips if no category specified
            all_tips = []
            for category_tips in self.health_tips.values():
                all_tips.extend(category_tips)
            tips = all_tips
        
        # Return random selection of tips
        return random.sample(tips, min(count, len(tips)))

    async def analyze_conversation(
        self, 
        chat_history: List[Dict[str, Any]], 
        patient_info: Optional[PatientInfo] = None
    ) -> ConversationAnalysis:
        """
        Analyze conversation patterns and generate insights
        
        Args:
            chat_history: List of chat messages
            patient_info: Optional patient information
            
        Returns:
            Conversation analysis with insights and recommendations
        """
        try:
            # Extract user messages for analysis
            user_messages = [msg for msg in chat_history if msg.get('type') == 'user']
            
            # Identify medical topics
            medical_topics = set()
            for msg in user_messages:
                content = msg.get('content', '').lower()
                for keyword in self.medical_keywords:
                    if keyword in content:
                        medical_topics.add(keyword)
            
            # Simple sentiment analysis (positive/negative/neutral)
            sentiment_scores = await self._analyze_sentiment(user_messages)
            
            # Identify urgency indicators
            urgency_indicators = []
            for msg in user_messages:
                content = msg.get('content', '').lower()
                if any(urgent in content for urgent in ["severe", "emergency", "urgent", "help"]):
                    urgency_indicators.append(content[:50] + "...")
            
            # Generate insights
            insights = await self._generate_conversation_insights(user_messages, medical_topics)
            
            # Generate recommendations
            recommendations = await self._generate_conversation_recommendations(
                user_messages, patient_info, medical_topics
            )
            
            return ConversationAnalysis(
                total_messages=len(chat_history),
                medical_topics=list(medical_topics),
                sentiment_analysis=sentiment_scores,
                urgency_indicators=urgency_indicators,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            raise

    async def _analyze_sentiment(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Simple sentiment analysis of messages"""
        positive_words = ["good", "better", "fine", "well", "healthy", "improved"]
        negative_words = ["bad", "worse", "terrible", "awful", "sick", "pain", "hurt"]
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for msg in messages:
            content = msg.get('content', '').lower()
            words = content.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_words:
                    positive_count += 1
                elif word in negative_words:
                    negative_count += 1
        
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": max(neutral_ratio, 0.0)
        }

    async def _generate_conversation_insights(
        self, 
        messages: List[Dict[str, Any]], 
        medical_topics: set
    ) -> List[MedicalInsight]:
        """Generate insights from conversation analysis"""
        insights = []
        
        if len(medical_topics) > 3:
            insights.append(MedicalInsight(
                category="symptom_complexity",
                insight="Multiple medical topics discussed, suggesting complex health concerns",
                confidence=0.8,
                supporting_evidence=list(medical_topics)[:3]
            ))
        
        if any("pain" in topic for topic in medical_topics):
            insights.append(MedicalInsight(
                category="pain_management",
                insight="Pain-related concerns identified, consider pain management strategies",
                confidence=0.9,
                supporting_evidence=["pain-related keywords found"]
            ))
        
        if len(messages) > 10:
            insights.append(MedicalInsight(
                category="engagement",
                insight="Extended conversation indicates patient engagement and detailed health discussion",
                confidence=0.7,
                supporting_evidence=[f"{len(messages)} messages exchanged"]
            ))
        
        return insights

    async def _generate_conversation_recommendations(
        self, 
        messages: List[Dict[str, Any]], 
        patient_info: Optional[PatientInfo],
        medical_topics: set
    ) -> List[str]:
        """Generate recommendations based on conversation analysis"""
        recommendations = []
        
        # Topic-based recommendations
        if "stress" in medical_topics or "anxiety" in medical_topics:
            recommendations.append("Consider stress management techniques such as meditation or counseling")
        
        if "sleep" in medical_topics or "insomnia" in medical_topics:
            recommendations.append("Focus on sleep hygiene and consider consulting a sleep specialist")
        
        if "pain" in medical_topics:
            recommendations.append("Document pain patterns and discuss with healthcare provider")
        
        # General recommendations
        recommendations.extend([
            "Schedule regular follow-up appointments with your healthcare provider",
            "Keep a health diary to track symptoms and improvements",
            "Maintain open communication with your medical team"
        ])
        
        # Patient-specific recommendations
        if patient_info:
            if patient_info.age and patient_info.age > 65:
                recommendations.append("Consider additional preventive screenings appropriate for your age group")
            
            if patient_info.medical_history:
                recommendations.append("Ensure all healthcare providers are aware of your complete medical history")
        
        return recommendations[:5]  # Limit to 5 recommendations

    def is_healthy(self) -> bool:
        """Check if the medical service is healthy and operational"""
        return self.is_initialized and self.medical_dataset is not None
