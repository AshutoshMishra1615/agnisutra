"use client";

import { useState, useRef, useEffect } from "react";
import api from "../services/api";
import { Send, Bot, User, Sparkles } from "lucide-react";
import { toast } from "sonner";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function ChatWidget() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I am your Krishi Saathi AI advisor. Ask me anything about your crops, soil, or weather.",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!query.trim()) return;

    const userMsg = query;
    setQuery("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      // Get yield context from localStorage
      let yieldContext = null;
      if (typeof window !== "undefined") {
        const stored = localStorage.getItem("lastYieldPrediction");
        if (stored) {
          yieldContext = JSON.parse(stored);
        }
      }

      const response = await api.post("/krishi-saathi/chat", {
        query: userMsg,
        session_id: "demo-session", // In a real app, manage sessions
        language: "en", // Could come from context
        yield_context: yieldContext, // Pass the context
      });

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.data.answer },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      toast.error("Failed to get response");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#1a2e1a]/20 backdrop-blur-md rounded-xl overflow-hidden">
      <div className="bg-[#0E1A0E]/80 p-4 border-b border-[#879d7b]/20 flex items-center gap-3">
        <div className="p-2 bg-[#4ade80]/10 rounded-full">
          <Bot className="text-[#4ade80]" size={20} />
        </div>
        <div>
          <h3 className="font-bold text-white flex items-center gap-2">
            Krishi Saathi AI
            <span className="px-2 py-0.5 rounded-full bg-[#4ade80]/20 text-[#4ade80] text-[10px] font-bold">
              BETA
            </span>
          </h3>
          <p className="text-xs text-gray-400 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
            Online
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-[#4ade80]/10 flex items-center justify-center shrink-0 mt-1">
                <Sparkles size={14} className="text-[#4ade80]" />
              </div>
            )}

            <div
              className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${
                msg.role === "user"
                  ? "bg-[#4ade80] text-[#050b05] rounded-tr-none font-medium"
                  : "bg-[#0E1A0E]/80 text-gray-200 border border-[#879d7b]/20 rounded-tl-none"
              }`}
            >
              {msg.role === "user" ? (
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </p>
              ) : (
                <div className="text-sm leading-relaxed markdown-content">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      ul: ({ ...props }) => (
                        <ul
                          className="list-disc pl-4 space-y-1 my-2"
                          {...props}
                        />
                      ),
                      ol: ({ ...props }) => (
                        <ol
                          className="list-decimal pl-4 space-y-1 my-2"
                          {...props}
                        />
                      ),
                      li: ({ ...props }) => <li className="mb-1" {...props} />,
                      h1: ({ ...props }) => (
                        <h1
                          className="text-lg font-bold mt-4 mb-2 text-[#4ade80]"
                          {...props}
                        />
                      ),
                      h2: ({ ...props }) => (
                        <h2
                          className="text-base font-bold mt-3 mb-2 text-[#4ade80]"
                          {...props}
                        />
                      ),
                      h3: ({ ...props }) => (
                        <h3
                          className="text-sm font-bold mt-2 mb-1 text-[#4ade80]"
                          {...props}
                        />
                      ),
                      strong: ({ ...props }) => (
                        <strong className="font-bold text-white" {...props} />
                      ),
                      p: ({ ...props }) => (
                        <p className="mb-2 last:mb-0" {...props} />
                      ),
                      table: ({ ...props }) => (
                        <div className="overflow-x-auto my-2">
                          <table
                            className="min-w-full border-collapse border border-[#879d7b]/30"
                            {...props}
                          />
                        </div>
                      ),
                      th: ({ ...props }) => (
                        <th
                          className="border border-[#879d7b]/30 px-2 py-1 bg-[#1a2e1a]/50 text-left text-xs font-bold text-[#4ade80]"
                          {...props}
                        />
                      ),
                      td: ({ ...props }) => (
                        <td
                          className="border border-[#879d7b]/30 px-2 py-1 text-xs"
                          {...props}
                        />
                      ),
                      code: ({ ...props }) => (
                        <code
                          className="bg-black/30 px-1 py-0.5 rounded text-[#4ade80] font-mono text-xs"
                          {...props}
                        />
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1">
                <User size={14} className="text-gray-300" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex justify-start gap-3">
            <div className="w-8 h-8 rounded-full bg-[#4ade80]/10 flex items-center justify-center shrink-0 mt-1">
              <Sparkles size={14} className="text-[#4ade80]" />
            </div>
            <div className="bg-[#0E1A0E]/80 p-4 rounded-2xl rounded-tl-none border border-[#879d7b]/20 flex items-center gap-2">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-[#4ade80] rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                <span className="w-2 h-2 bg-[#4ade80] rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                <span className="w-2 h-2 bg-[#4ade80] rounded-full animate-bounce"></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-[#0E1A0E]/80 border-t border-[#879d7b]/20">
        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about crop diseases, soil health, or weather..."
            className="w-full bg-[#1a2e1a]/50 text-white placeholder-gray-500 border border-[#879d7b]/30 rounded-xl py-3 pl-4 pr-12 focus:outline-none focus:border-[#4ade80] focus:ring-1 focus:ring-[#4ade80] transition-all"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!query.trim() || loading}
            className="absolute right-2 p-2 bg-[#4ade80] hover:bg-[#22c55e] text-[#050b05] rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={18} />
          </button>
        </div>
        <p className="text-[10px] text-center text-gray-500 mt-2">
          AI can make mistakes. Please verify important agricultural decisions.
        </p>
      </div>
    </div>
  );
}
