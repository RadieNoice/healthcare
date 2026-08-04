export async function POST(request: Request) {
  try {
    const { message, language, patientInfo, chatHistory, isVoice } = await request.json()

    // Medical keywords for validation
    const medicalKeywords = [
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

    // Check if message is medical-related
    const isMedical = medicalKeywords.some(keyword => 
      message.toLowerCase().includes(keyword)
    )

    if (!isMedical) {
      return Response.json({
        response: "I'm Dr. AI, your specialized medical assistant. I'm designed to help with health-related questions and medical concerns. For general questions, I'd recommend consulting other AI assistants. How can I help you with your health today?"
      })
    }

    // Medical knowledge base
    const medicalResponses = {
      'headache': 'For headaches, try resting in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers like acetaminophen or ibuprofen. If headaches are severe, frequent, or accompanied by other symptoms like fever, vision changes, or neck stiffness, please consult a healthcare professional.',
      
      'fever': 'For fever, rest and stay hydrated with plenty of fluids. You can take acetaminophen or ibuprofen to reduce fever. Monitor your temperature regularly. Seek medical attention if fever is over 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms.',
      
      'cough': 'For cough, stay hydrated and try warm liquids like tea with honey. Avoid irritants like smoke. Over-the-counter cough suppressants may help. See a doctor if cough persists for more than 2 weeks, produces blood, or is accompanied by fever and difficulty breathing.',
      
      'cold': 'For common cold, get plenty of rest and drink fluids. Use saline nasal drops for congestion. Over-the-counter medications can help with symptoms. Most colds resolve in 7-10 days. See a doctor if symptoms worsen or last longer than 10 days.',
      
      'stomach': 'For stomach issues, try eating bland foods like rice, bananas, and toast. Stay hydrated with clear fluids. Avoid spicy, fatty, or dairy foods. Consider over-the-counter antacids for heartburn. Seek medical care if you have severe pain, persistent vomiting, or signs of dehydration.',
      
      'stress': 'For stress management, try deep breathing exercises, regular physical activity, and adequate sleep. Consider meditation or yoga. Maintain social connections and don\'t hesitate to seek professional help if stress becomes overwhelming.',
      
      'sleep': 'For better sleep, maintain a regular sleep schedule, create a comfortable sleep environment, and avoid screens before bedtime. Limit caffeine and large meals before sleep. If sleep problems persist, consult a healthcare provider.',
      
      'anxiety': 'For anxiety, practice relaxation techniques like deep breathing or progressive muscle relaxation. Regular exercise and adequate sleep can help. Consider talking to a mental health professional if anxiety interferes with daily activities.'
    }

    // Find relevant response
    let response = "I understand your concern. While I can provide general health information, it's important to consult a healthcare professional for proper diagnosis and treatment. Your health and safety are the top priority."

    for (const [condition, advice] of Object.entries(medicalResponses)) {
      if (message.toLowerCase().includes(condition)) {
        response = advice
        break
      }
    }

    // Add personalized touch based on patient info
    if (patientInfo.name) {
      response = `Hello ${patientInfo.name}, ${response}`
    }

    // Add age-specific advice if relevant
    if (patientInfo.age > 65) {
      response += "\n\nAs someone over 65, it's especially important to monitor your symptoms closely and don't hesitate to contact your healthcare provider if you have any concerns."
    } else if (patientInfo.age < 18) {
      response += "\n\nFor pediatric concerns, it's always best to consult with a pediatrician or your child's healthcare provider."
    }

    // Add general health tip
    const healthTips = [
      "Remember to stay hydrated throughout the day.",
      "Regular exercise can help improve overall health and mood.",
      "Adequate sleep (7-9 hours) is crucial for good health.",
      "A balanced diet rich in fruits and vegetables supports your immune system.",
      "Don't forget to wash your hands regularly to prevent infections."
    ]
    
    const randomTip = healthTips[Math.floor(Math.random() * healthTips.length)]
    response += `\n\n💡 Health Tip: ${randomTip}`

    // Add medical disclaimer
    response += "\n\n⚠️ This information is for educational purposes only and should not replace professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."

    return Response.json({ response })

  } catch (error) {
    console.error('Chat API error:', error)
    return Response.json(
      { error: 'Failed to process message' },
      { status: 500 }
    )
  }
}
