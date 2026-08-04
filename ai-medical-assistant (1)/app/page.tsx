'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Mic, MicOff, Send, Download, User, MessageCircle, FileText } from 'lucide-react'
import { PatientInfoForm } from '@/components/patient-info-form'
import { ChatMessage } from '@/components/chat-message'
import { VoiceRecorder } from '@/components/voice-recorder'

interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
  language?: string
}

interface PatientInfo {
  name: string
  age: number
  gender: string
  weight: number
  height: number
  bloodGroup: string
  emergencyContact: string
}

const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi' },
  { code: 'ta', name: 'Tamil' },
  { code: 'te', name: 'Telugu' },
  { code: 'gu', name: 'Gujarati' },
  { code: 'ko', name: 'Korean' },
  { code: 'tr', name: 'Turkish' },
  { code: 'de', name: 'German' },
  { code: 'fr', name: 'French' },
  { code: 'ar', name: 'Arabic' },
  { code: 'ur', name: 'Urdu' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' }
]

export default function MedicalAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputText, setInputText] = useState('')
  const [selectedLanguage, setSelectedLanguage] = useState('en')
  const [isLoading, setIsLoading] = useState(false)
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    name: '',
    age: 25,
    gender: 'Male',
    weight: 70,
    height: 170,
    bloodGroup: 'O+',
    emergencyContact: ''
  })
  const [isListening, setIsListening] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (text: string, isVoice = false) => {
    if (!text.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: text,
      timestamp: new Date(),
      language: selectedLanguage
    }

    setMessages(prev => [...prev, userMessage])
    setInputText('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: text,
          language: selectedLanguage,
          patientInfo,
          chatHistory: messages,
          isVoice
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: data.response,
        timestamp: new Date(),
        language: selectedLanguage
      }

      setMessages(prev => [...prev, botMessage])

      // Text-to-speech for bot response
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(data.response)
        utterance.lang = selectedLanguage === 'en' ? 'en-US' : selectedLanguage
        utterance.rate = 0.8
        speechSynthesis.speak(utterance)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleVoiceResult = (transcript: string) => {
    sendMessage(transcript, true)
  }

  const generateReport = async () => {
    if (messages.length === 0) return

    try {
      const response = await fetch('/api/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages,
          patientInfo
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate report')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = `medical-report-${new Date().toISOString().split('T')[0]}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Error generating report:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <MessageCircle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Dr. AI Medical Assistant
                </h1>
                <p className="text-sm text-gray-600">Your specialized AI medical doctor</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {languages.map((lang) => (
                    <SelectItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {messages.length > 0 && (
                <Button onClick={generateReport} variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Report
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Patient Info Sidebar */}
          <div className="lg:col-span-1">
            <PatientInfoForm 
              patientInfo={patientInfo} 
              onPatientInfoChange={setPatientInfo} 
            />
            
            {/* Chat Statistics */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center">
                  <FileText className="w-5 h-5 mr-2" />
                  Chat Statistics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Messages:</span>
                    <span className="font-medium">{messages.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">User Messages:</span>
                    <span className="font-medium">{messages.filter(m => m.type === 'user').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">AI Responses:</span>
                    <span className="font-medium">{messages.filter(m => m.type === 'bot').length}</span>
                  </div>
                </div>
                {messages.length > 0 && (
                  <Button 
                    onClick={() => setMessages([])} 
                    variant="outline" 
                    size="sm" 
                    className="w-full mt-4"
                  >
                    Clear Chat
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-2">
            <Card className="h-[600px] flex flex-col">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <MessageCircle className="w-5 h-5 mr-2" />
                  Medical Consultation
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                  {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-center">
                      <div className="space-y-4">
                        <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto">
                          <MessageCircle className="w-8 h-8 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-gray-800">Welcome to Dr. AI!</h3>
                          <p className="text-gray-600 max-w-md">
                            I'm your specialized AI medical doctor. I can help with health symptoms, 
                            medical advice, medication questions, and wellness tips.
                          </p>
                          <p className="text-sm text-blue-600 mt-2">
                            🎤 Click the microphone to speak or type your medical question below.
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    messages.map((message) => (
                      <ChatMessage key={message.id} message={message} />
                    ))
                  )}
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3 max-w-xs">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="border-t pt-4">
                  <div className="flex space-x-2">
                    <div className="flex-1">
                      <Textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Ask Dr. AI about your health concerns..."
                        className="min-h-[60px] resize-none"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            sendMessage(inputText)
                          }
                        }}
                      />
                    </div>
                    <div className="flex flex-col space-y-2">
                      <VoiceRecorder
                        onResult={handleVoiceResult}
                        language={selectedLanguage}
                        isListening={isListening}
                        setIsListening={setIsListening}
                      />
                      <Button
                        onClick={() => sendMessage(inputText)}
                        disabled={!inputText.trim() || isLoading}
                        size="sm"
                      >
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-white/80 backdrop-blur-sm border-t border-gray-200 mt-8">
        <div className="max-w-6xl mx-auto px-4 py-6 text-center">
          <p className="text-sm text-gray-600">
            <strong>⚠️ Medical Disclaimer:</strong> This AI assistant provides general health information only. 
            Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.
          </p>
        </div>
      </div>
    </div>
  )
}
