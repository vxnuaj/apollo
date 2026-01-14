'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, ChevronDown } from 'lucide-react';
import { ChatSession, Message } from '../types';
import styles from './ChatInterface.module.css';

interface ChatInterfaceProps {
  session: ChatSession | undefined;
  onAddMessage: (message: Message) => void;
  isDarkMode: boolean;
  onNewSession: () => void;
}

export default function ChatInterface({ session, onAddMessage, isDarkMode, onNewSession }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'deepseek-r1' | 'gpt-oss'>('deepseek-r1');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const modelDropdownRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [session?.messages]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(event.target as Node)) {
        setShowModelDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !session || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    onAddMessage(userMessage);
    setInput('');
    setIsLoading(true);

    // TODO: Replace this with actual API call to your inference engine
    // For now, simulating a response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'apollo',
        content: 'This is a placeholder response. Connect this to your LLM inference engine to get real responses.',
        timestamp: new Date().toISOString(),
      };
      onAddMessage(assistantMessage);
      setIsLoading(false);
    }, 1000);
  };

  if (!session) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyState}>
          <h2>No session selected</h2>
          <p>Create a new chat to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles.container} ${!isDarkMode ? styles.lightMode : ''}`}>
      {session.messages.length === 0 ? (
        <div className={styles.welcomeContainer}>
          <img src="/Apollo.svg" alt="Apollo" className={styles.welcomeLogo} />
          <form onSubmit={handleSubmit} className={styles.welcomeInputForm}>
            <div className={styles.inputWrapper}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Chat with APOLLO"
                className={styles.welcomeInput}
                disabled={isLoading}
              />
              <div className={styles.modelSelectorContainer} ref={modelDropdownRef}>
                <button
                  type="button"
                  className={styles.modelSelector}
                  onClick={() => setShowModelDropdown(!showModelDropdown)}
                >
                  {selectedModel}
                  <ChevronDown size={16} />
                </button>
                {showModelDropdown && (
                  <div className={styles.modelDropdown}>
                    <button
                      type="button"
                      className={`${styles.modelOption} ${selectedModel === 'deepseek-r1' ? styles.modelOptionActive : ''}`}
                      onClick={() => {
                        setSelectedModel('deepseek-r1');
                        setShowModelDropdown(false);
                      }}
                    >
                      deepseek-r1
                    </button>
                    <button
                      type="button"
                      className={`${styles.modelOption} ${selectedModel === 'gpt-oss' ? styles.modelOptionActive : ''}`}
                      onClick={() => {
                        setSelectedModel('gpt-oss');
                        setShowModelDropdown(false);
                      }}
                    >
                      gpt-oss
                    </button>
                  </div>
                )}
              </div>
            </div>
            <button
              type="submit"
              className={styles.sendButton}
              disabled={!input.trim() || isLoading}
            >
              <ArrowUp size={20} />
            </button>
          </form>
        </div>
      ) : (
        <>
          <div className={styles.logoTopLeft} onClick={onNewSession} style={{ cursor: 'pointer' }}>
            <img src="/Apollo.svg" alt="Apollo" className={styles.logoSmall} />
          </div>
          <div className={styles.messages}>
            <div className={styles.messagesInner}>
              {session.messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.role === 'user' ? styles.userMessage : styles.assistantMessage
                  }`}
                >
                  <div className={styles.messageRole}>
                    {message.role === 'user' ? 'You' : 'APOLLO'}
                  </div>
                  <div className={styles.messageContent}>{message.content}</div>
                </div>
              ))}
              {isLoading && (
                <div className={`${styles.message} ${styles.assistantMessage}`}>
                  <div className={styles.messageRole}>Apollo</div>
                  <div className={styles.messageContent}>
                    <div className={styles.loadingDots}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className={styles.inputContainer}>
            <form onSubmit={handleSubmit} className={styles.inputForm}>
              <div className={styles.inputWrapper}>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message..."
                  className={styles.input}
                  disabled={isLoading}
                />
                <div className={styles.modelSelectorContainer} ref={modelDropdownRef}>
                  <button
                    type="button"
                    className={styles.modelSelector}
                    onClick={() => setShowModelDropdown(!showModelDropdown)}
                  >
                    {selectedModel}
                    <ChevronDown size={16} />
                  </button>
                  {showModelDropdown && (
                    <div className={styles.modelDropdown}>
                      <button
                        type="button"
                        className={`${styles.modelOption} ${selectedModel === 'deepseek-r1' ? styles.modelOptionActive : ''}`}
                        onClick={() => {
                          setSelectedModel('deepseek-r1');
                          setShowModelDropdown(false);
                        }}
                      >
                        deepseek-r1
                      </button>
                      <button
                        type="button"
                        className={`${styles.modelOption} ${selectedModel === 'gpt-oss' ? styles.modelOptionActive : ''}`}
                        onClick={() => {
                          setSelectedModel('gpt-oss');
                          setShowModelDropdown(false);
                        }}
                      >
                        gpt-oss
                      </button>
                    </div>
                  )}
                </div>
              </div>
              <button
                type="submit"
                className={styles.sendButton}
                disabled={!input.trim() || isLoading}
              >
                <ArrowUp size={20} />
              </button>
            </form>
          </div>
        </>
      )}
    </div>
  );
}
