"use client";

import ChatWidget from "../components/ChatWidget";
import Header from "../components/HeaderDashboard";
import { Sparkles } from "lucide-react";

export default function SoilReportsPage() {
  return (
    <div className="min-h-screen bg-[#050b05] text-white">
      <Header userName="User" showIcons={true} />
      <main className="max-w-5xl mx-auto px-4 py-8">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#4ade80]/10 text-[#4ade80] text-sm font-medium mb-4 border border-[#4ade80]/20">
            <Sparkles size={14} />
            <span>Powered by Krishi Saathi AI</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
            AI Soil Advisor
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Get instant, expert advice on soil health, crop suitability, and
            fertilizer management. Our AI analyzes your soil data to provide
            personalized recommendations.
          </p>
        </div>

        <div className="glass-panel rounded-2xl overflow-hidden shadow-2xl border border-[#879d7b]/20 h-[600px]">
          <ChatWidget />
        </div>
      </main>
    </div>
  );
}
