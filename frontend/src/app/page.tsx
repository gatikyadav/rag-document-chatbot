'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, MessageSquare, FileText, Clock, Loader2 } from 'lucide-react';
import axios from 'axios';

interface Source {
  filename: string;
  file_type: string;
  url: string;
  relevance_score: number;
  snippet: string;
  page?: number;
  slide_number?: number;
  sheet?: string;
  section?: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence?: number;
  processing_time?: number;
  timestamp: Date;
}

export default function ChatUSG() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('question', inputValue);
      formData.append('max_sources', '5');

      const response = await axios.post('http://127.0.0.1:8000/api/v1/ask', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        confidence: response.data.confidence,
        processing_time: response.data.processing_time,
        timestamp: new Date(response.data.timestamp),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to get response. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const formatProcessingTime = (time: number) => {
    return `${time.toFixed(2)}s`;
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Header - Red Background */}
      <header className="bg-red-600 p-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* USG Logo placeholder */}
            <div className="w-12 h-12 bg-white rounded-lg flex items-center justify-center">
              <MessageSquare className="w-6 h-6 text-red-600" />
            </div>
            <h1 className="text-2xl font-bold text-white">
              chat<span className="text-black">USG</span>
            </h1>
          </div>
          <div className="text-white font-medium">
            Your personal USG Assistant
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 max-w-4xl mx-auto w-full p-4 flex flex-col">
        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto mb-4 space-y-6">
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <MessageSquare className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">Welcome to chatUSG</h2>
              <p className="text-gray-400 max-w-md mx-auto">
                Ask questions about your documents and get intelligent answers with source citations.
                I can help you find information from PDFs, Word documents, presentations, and more.
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className="space-y-4">
                {/* User Message - Red Bubble */}
                {message.type === 'user' && (
                  <div className="flex justify-end">
                    <div className="bg-red-600 text-white p-4 rounded-lg max-w-2xl">
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>
                  </div>
                )}

                {/* Assistant Message - No Bubble, Just Text */}
                {message.type === 'assistant' && (
                  <div className="flex justify-start">
                    <div className="max-w-2xl w-full">
                      <p className="text-white whitespace-pre-wrap mb-4">{message.content}</p>
                      
                      {/* Sources */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="border-t border-gray-700 pt-4 mt-4">
                          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
                            <FileText className="w-4 h-4 mr-2" />
                            Sources ({message.sources.length})
                          </h4>
                          <div className="space-y-2">
                            {message.sources.map((source, index) => (
                              <div key={index} className="bg-gray-900 border border-gray-700 p-3 rounded text-sm">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="font-medium text-red-400">
                                    {source.filename}
                                  </span>
                                  <span className="text-gray-400">
                                    {(source.relevance_score * 100).toFixed(1)}% match
                                  </span>
                                </div>
                                {source.snippet && (
                                  <p className="text-gray-300 text-xs">
                                    {source.snippet}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Metadata */}
                      {(message.confidence !== undefined || message.processing_time !== undefined) && (
                        <div className="border-t border-gray-700 pt-3 mt-4 flex items-center space-x-4 text-xs text-gray-400">
                          {message.confidence !== undefined && (
                            <span className="flex items-center">
                              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                              Confidence: {formatConfidence(message.confidence)}
                            </span>
                          )}
                          {message.processing_time !== undefined && (
                            <span className="flex items-center">
                              <Clock className="w-3 h-3 mr-1" />
                              {formatProcessingTime(message.processing_time)}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
          
          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="flex items-center space-x-2 text-gray-400">
                <Loader2 className="w-4 h-4 animate-spin text-red-500" />
                <span>Thinking...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900 border border-red-700 text-red-200 p-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {/* Input Area */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <div className="flex space-x-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-red-500"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-2 rounded-lg transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black border-t border-red-800 p-4 text-center text-gray-400 text-sm">
        Developed by Gatik Yadav and Russell Erfan
      </footer>
    </div>
  );
}