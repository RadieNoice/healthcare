'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Mic, MicOff } from 'lucide-react'

interface VoiceRecorderProps {
  onResult: (transcript: string) => void
  language: string
  isListening: boolean
  setIsListening: (listening: boolean) => void
}

export function VoiceRecorder({ onResult, language, isListening, setIsListening }: VoiceRecorderProps) {
  const [isSupported, setIsSupported] = useState(false)
  const recognitionRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition
      if (SpeechRecognition) {
        setIsSupported(true)
        recognitionRef.current = new SpeechRecognition()
        recognitionRef.current.continuous = false
        recognitionRef.current.interimResults = false
        recognitionRef.current.maxAlternatives = 1

        recognitionRef.current.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript
          onResult(transcript)
          setIsListening(false)
        }

        recognitionRef.current.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error)
          setIsListening(false)
        }

        recognitionRef.current.onend = () => {
          setIsListening(false)
        }
      }
    }
  }, [onResult, setIsListening])

  useEffect(() => {
    if (recognitionRef.current) {
      recognitionRef.current.lang = language === 'en' ? 'en-US' : 
                                   language === 'ta' ? 'ta-IN' :
                                   language === 'hi' ? 'hi-IN' :
                                   language === 'te' ? 'te-IN' :
                                   language === 'gu' ? 'gu-IN' :
                                   `${language}-${language.toUpperCase()}`
    }
  }, [language])

  const toggleListening = () => {
    if (!isSupported || !recognitionRef.current) return

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    } else {
      recognitionRef.current.start()
      setIsListening(true)
    }
  }

  if (!isSupported) {
    return (
      <Button variant="outline" size="sm" disabled>
        <MicOff className="w-4 h-4" />
      </Button>
    )
  }

  return (
    <Button
      onClick={toggleListening}
      variant={isListening ? "destructive" : "outline"}
      size="sm"
      className={isListening ? "animate-pulse" : ""}
    >
      {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
    </Button>
  )
}
