'use client'

import { format } from 'date-fns'

interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
  language?: string
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.type === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
          isUser
            ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-br-sm'
            : 'bg-gray-100 text-gray-800 rounded-bl-sm'
        }`}
      >
        <div className="text-sm leading-relaxed">
          {message.content}
        </div>
        <div
          className={`text-xs mt-2 ${
            isUser ? 'text-blue-100' : 'text-gray-500'
          }`}
        >
          {format(message.timestamp, 'HH:mm')}
        </div>
      </div>
    </div>
  )
}
