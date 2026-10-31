import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import {
  Send,
  Sparkles,
  User,
  Bot,
  PlusCircle,
  HelpCircle,
  Clock,
  Layers,
  ArrowRight
} from 'lucide-react';

export default function ChatArea({
  messages,
  streamingMessage,
  currentStage,
  onSendMessage,
  onNewChat,
  onSelectCitation,
  selectedSourceId,
  isGenerating
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage, currentStage]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const samplePrompts = [
    'What is the guaranteed response time for Severity 1 incidents?',
    'What are the penalties or SLA service credit schedules?',
    'What is the minimum password length requirement?'
  ];

  // Helper to format [Source X] into clickable badges
  const renderMessageContent = (content) => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          // Intercept text to replace [Source X] with clickable badge
          p: ({ children }) => {
            return (
              <p className="mb-2 leading-relaxed text-sm">
                {React.Children.map(children, (child) => {
                  if (typeof child === 'string') {
                    const parts = child.split(/(\[Source \d+\])/g);
                    return parts.map((part, i) => {
                      const match = part.match(/\[Source (\d+)\]/);
                      if (match) {
                        const sourceId = `Source ${match[1]}`;
                        const isSelected = selectedSourceId === sourceId;
                        return (
                          <button
                            key={i}
                            onClick={() => onSelectCitation(sourceId)}
                            className={`inline-flex items-center px-1.5 py-0.5 mx-1 rounded text-[11px] font-mono font-bold transition-all ${
                              isSelected
                                ? 'bg-indigo-500 text-white shadow-md shadow-indigo-500/50 scale-105'
                                : 'bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30 border border-indigo-500/30'
                            }`}
                            title={`Inspect ${sourceId}`}
                          >
                            {part}
                          </button>
                        );
                      }
                      return part;
                    });
                  }
                  return child;
                })}
              </p>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-lg border border-slate-700/80 bg-slate-900/60">
              <table className="min-w-full divide-y divide-slate-700/80 text-xs">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-indigo-300 bg-slate-800/80 uppercase tracking-wider text-[11px]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-slate-300 border-t border-slate-800">
              {children}
            </td>
          ),
          ul: ({ children }) => <ul className="list-disc pl-5 my-2 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 my-2 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="text-sm text-slate-200">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <main className="flex-1 flex flex-col h-[calc(100vh-4rem)] bg-slate-950">
      {/* Top Controls Bar */}
      <div className="h-12 border-b border-slate-800/80 px-6 flex items-center justify-between bg-slate-900/30">
        <div className="flex items-center space-x-2 text-xs text-slate-400">
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
          <span>Interactive Hybrid RAG Chat</span>
        </div>
        <button
          onClick={onNewChat}
          className="flex items-center space-x-1.5 text-xs text-slate-300 hover:text-white px-2.5 py-1 rounded-md hover:bg-slate-800 transition-colors"
        >
          <PlusCircle className="w-3.5 h-3.5 text-indigo-400" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && !streamingMessage ? (
          <div className="max-w-xl mx-auto py-16 text-center">
            <div className="w-12 h-12 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 flex items-center justify-center mx-auto mb-4 text-indigo-400 shadow-xl shadow-indigo-950">
              <Sparkles className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">
              DocuMind Hybrid RAG Intelligence
            </h3>
            <p className="text-xs text-slate-400 mb-6">
              Ask questions across your uploaded documents. Powered by Gemini dense embeddings, BM25 sparse keyword vectors, and Cross-Encoder reranking.
            </p>

            {/* Prompt Suggestions */}
            <div className="space-y-2 text-left">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
                Suggested Prompts
              </p>
              {samplePrompts.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInput(prompt);
                    inputRef.current?.focus();
                  }}
                  className="w-full text-xs text-slate-300 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 rounded-xl p-3 flex items-center justify-between group transition-all"
                >
                  <span>{prompt}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((m, idx) => (
              <div
                key={m.id || idx}
                className={`flex items-start space-x-3.5 ${
                  m.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {m.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-2xl rounded-2xl p-4 shadow-sm ${
                    m.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-none'
                      : 'bg-slate-900/90 border border-slate-800/90 text-slate-200 rounded-tl-none'
                  }`}
                >
                  {m.role === 'user' ? (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</p>
                  ) : (
                    renderMessageContent(m.content)
                  )}

                  {/* Render Citation Badges under message if available */}
                  {m.retrieved_chunks && m.retrieved_chunks.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap gap-1.5 items-center">
                      <span className="text-[10px] text-slate-500 font-medium">Sources:</span>
                      {m.retrieved_chunks.map((c, cIdx) => (
                        <button
                          key={cIdx}
                          onClick={() => onSelectCitation(c.source_id)}
                          className={`text-[11px] font-mono px-2 py-0.5 rounded font-semibold transition-all ${
                            selectedSourceId === c.source_id
                              ? 'bg-indigo-500 text-white shadow-sm'
                              : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                          }`}
                        >
                          {c.source_id}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {m.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            ))}

            {/* Live Streaming Message Container */}
            {isGenerating && (
              <div className="flex items-start space-x-3.5 justify-start">
                <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
                  <Bot className="w-4 h-4 animate-pulse" />
                </div>
                <div className="max-w-2xl rounded-2xl rounded-tl-none p-4 bg-slate-900/90 border border-slate-800 text-slate-200 shadow-sm">
                  {/* Status Progress Pill */}
                  {currentStage && (
                    <div className="inline-flex items-center space-x-2 text-xs text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded-full mb-3">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping"></span>
                      <span>{currentStage}</span>
                    </div>
                  )}

                  {streamingMessage ? (
                    renderMessageContent(streamingMessage)
                  ) : (
                    <div className="flex items-center space-x-2 text-xs text-slate-400 py-1">
                      <span>Reasoning and synthesizing answer...</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Box */}
      <div className="p-4 border-t border-slate-800 bg-slate-900/40">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative flex items-center">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your indexed documents... (Enter to send)"
            disabled={isGenerating}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-3 pr-12 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 resize-none shadow-lg shadow-black/20"
          />
          <button
            type="submit"
            disabled={!input.trim() || isGenerating}
            className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors disabled:opacity-40 disabled:hover:bg-indigo-600"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </main>
  );
}
