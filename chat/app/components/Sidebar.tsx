'use client';

import React from 'react';
import { MessageSquarePlus, Trash2, Clock, Sun, Moon, Settings } from 'lucide-react';
import { ChatSession } from '../types';
import styles from './Sidebar.module.css';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  isWelcomeScreen?: boolean;
  isDarkMode: boolean;
  onToggleTheme: () => void;
  onSettings?: () => void;
}

export default function Sidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  isOpen,
  onToggle,
  isWelcomeScreen = false,
  isDarkMode,
  onToggleTheme,
  onSettings,
}: SidebarProps) {
  const [previewSessionId, setPreviewSessionId] = React.useState<string | null>(null);
  const previewSession = sessions.find(s => s.id === previewSessionId);
  if (!isOpen) {
    return (
      <>
        <div className={isWelcomeScreen ? styles.newChatWelcome : styles.newChatCollapsed}>
          <button
            className={`${styles.toggleButton} ${!isDarkMode ? styles.lightModeButton : ''}`}
            onClick={onNewSession}
            title="New Chat"
          >
            <MessageSquarePlus size={16} />
          </button>
        </div>
        <div className={isWelcomeScreen ? styles.sidebarWelcome : styles.sidebarCollapsed}>
          <button
            className={`${styles.toggleButton} ${!isDarkMode ? styles.lightModeButton : ''}`}
            onClick={onToggle}
            title="History"
          >
            <Clock size={16} />
          </button>
        </div>
        <div className={isWelcomeScreen ? styles.themeWelcome : styles.themeCollapsed}>
          <button
            className={`${styles.toggleButton} ${!isDarkMode ? styles.lightModeButton : ''}`}
            onClick={onToggleTheme}
            title={isDarkMode ? "Light Mode" : "Dark Mode"}
          >
            {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
        {onSettings && (
          <div className={isWelcomeScreen ? styles.settingsWelcome : styles.settingsCollapsed}>
            <button
              className={`${styles.toggleButton} ${!isDarkMode ? styles.lightModeButton : ''}`}
              onClick={onSettings}
              title="Settings"
            >
              <Settings size={16} />
            </button>
          </div>
        )}
      </>
    );
  }

  return (
    <>
      <div className={isWelcomeScreen ? styles.newChatWelcome : styles.newChatCollapsed}>
        <button
          className={`${styles.toggleButton} ${!isDarkMode ? styles.lightModeButton : ''}`}
          onClick={onNewSession}
          title="New Chat"
        >
          <MessageSquarePlus size={16} />
        </button>
      </div>
      <div className={styles.backdrop} onClick={onToggle}></div>
      <div className={isWelcomeScreen ? styles.sidebarContainerWelcome : styles.sidebarContainer}>
        <div className={`${styles.sidebar} ${!isDarkMode ? styles.lightModePanel : ''}`}>
          <div className={styles.header}>
            <h1 className={styles.title}>History</h1>
            <button
              className={styles.closeButton}
              onClick={onToggle}
              title="Close"
            >
              ×
            </button>
          </div>

          <div className={styles.mainContent}>
            <div className={styles.sessionList}>
              {sessions.map(session => (
                <div
                  key={session.id}
                  className={`${styles.sessionItem} ${
                    session.id === currentSessionId ? styles.active : ''
                  }`}
                  onClick={() => {
                    onSelectSession(session.id);
                    onToggle();
                  }}
                  onMouseEnter={() => setPreviewSessionId(session.id)}
                >
                  <div className={styles.sessionTitle}>{session.title}</div>
                  <button
                    className={styles.deleteButton}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    title="Delete Chat"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>

            <div className={styles.previewPanel}>
              {previewSession ? (
                <div className={styles.previewContent}>
                  <h3 className={styles.previewTitle}>{previewSession.title}</h3>
                  <div className={styles.previewMessages}>
                    {previewSession.messages.map((message) => (
                      <div key={message.id} className={styles.previewMessage}>
                        <div className={styles.previewMessageRole}>
                          {message.role === 'user' ? 'You' : 'APOLLO'}
                        </div>
                        <div className={styles.previewMessageContent}>
                          {message.content}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className={styles.previewEmpty}>
                  Hover over a chat to preview
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
