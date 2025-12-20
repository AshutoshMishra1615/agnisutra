"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import LanguageSelector from "./components/LanguageSelector";
import YieldPredictionWidget from "./components/YieldPredictionWidget";
import DiseaseDetectionWidget from "./components/DiseaseDetectionWidget";
import { useTranslations } from "next-intl";
import {
  ArrowRight,
  Lock,
  Sprout,
  Satellite,
  Bot,
  BarChart3,
} from "lucide-react";

export default function Home() {
  const t = useTranslations("landing");
  const [isLanguageSelected, setIsLanguageSelected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if language is already selected
    const storedLang = localStorage.getItem("app_language");
    if (storedLang) {
      setIsLanguageSelected(true);
    }
    // Simulate initial app load
    const timer = setTimeout(() => setIsLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#050b05] flex flex-col items-center justify-center">
        <div className="relative w-24 h-24 mb-4">
          <div className="absolute inset-0 border-4 border-[#4ade80]/30 rounded-full animate-ping"></div>
          <div className="absolute inset-0 border-4 border-[#4ade80] rounded-full animate-spin border-t-transparent"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <Sprout className="text-[#4ade80]" size={32} />
          </div>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-widest">
          AGNISUTRA
        </h1>
        <p className="text-[#4ade80] text-sm mt-2 font-medium">
          {t("empowering")}
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050b05] bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-[#1a2e1a] via-[#050b05] to-[#050b05] text-white overflow-x-hidden">
      <LanguageSelector onComplete={() => setIsLanguageSelected(true)} />

      <div
        className={`transition-opacity duration-700 ${
          isLanguageSelected
            ? "opacity-100"
            : "opacity-0 h-screen overflow-hidden"
        }`}
      >
        {/* Navbar */}
        <nav className="border-b border-[#879d7b]/20 bg-[#050b05]/80 backdrop-blur-md sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2 group cursor-pointer">
              <div className="p-2 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
                <Sprout className="text-[#4ade80]" size={24} />
              </div>
              <span className="text-xl font-bold tracking-wide group-hover:text-[#4ade80] transition-colors">
                AGNISUTRA
              </span>
            </div>
            <div className="flex items-center gap-4">
              <Link
                href="/login"
                className="text-sm font-medium text-gray-300 hover:text-white transition-colors"
              >
                {t("login")}
              </Link>
              <Link
                href="/register"
                className="bg-[#4ade80] hover:bg-[#22c55e] text-[#050b05] px-4 py-2 rounded-xl text-sm font-bold transition-all hover:shadow-lg hover:shadow-[#4ade80]/20"
              >
                {t("getStarted")}
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="relative">
          <div className="absolute inset-0 bg-linear-to-b from-[#4ade80]/5 to-transparent pointer-events-none" />
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-24 text-center">
            <h1 className="text-4xl md:text-7xl font-bold mb-6 bg-clip-text text-transparent bg-linear-to-r from-white via-[#4ade80] to-[#22c55e] leading-tight">
              {t.rich("heroTitle", {
                br: () => <br />,
              })}
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              {t("heroDesc")}
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/dashboard"
                className="flex items-center gap-2 bg-[#4ade80] hover:bg-[#22c55e] text-[#050b05] px-8 py-4 rounded-xl font-bold text-lg transition-all hover:scale-105 hover:shadow-xl hover:shadow-[#4ade80]/20"
              >
                {t("goToDashboard")} <ArrowRight size={20} />
              </Link>
              <p className="text-sm text-gray-500 flex items-center gap-1 bg-[#1a2e1a]/50 px-4 py-2 rounded-full border border-[#879d7b]/20">
                <Lock size={14} /> {t("loginRequired")}
              </p>
            </div>
          </div>
        </div>

        {/* Public Tools Section */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <span className="w-1 h-8 bg-[#4ade80] rounded-full"></span>
              {t("freeTools")}
            </h2>
            <span className="text-sm text-[#4ade80] bg-[#4ade80]/10 px-3 py-1 rounded-full border border-[#4ade80]/20 font-medium">
              {t("noLogin")}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Yield Prediction */}
            <div className="relative group">
              <div className="absolute -inset-1 bg-linear-to-r from-[#4ade80]/20 to-blue-500/20 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
              <YieldPredictionWidget />
            </div>

            {/* Disease Detection */}
            <div className="relative group">
              <div className="absolute -inset-1 bg-linear-to-r from-red-500/20 to-[#4ade80]/20 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
              <DiseaseDetectionWidget />
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="bg-[#1a2e1a]/20 border-y border-[#879d7b]/10 py-20 mt-12 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold mb-4">{t("whyChoose")}</h2>
              <p className="text-gray-400">
                {t("comprehensive")}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: Satellite,
                  title: t("satellite.title"),
                  desc: t("satellite.desc"),
                  color: "text-blue-400",
                  bg: "bg-blue-400/10",
                },
                {
                  icon: Bot,
                  title: t("features.advisory.title"),
                  desc: t("chatbot.desc"),
                  color: "text-[#4ade80]",
                  bg: "bg-[#4ade80]/10",
                },
                {
                  icon: BarChart3,
                  title: t("analytics.title"),
                  desc: t("analytics.desc"),
                  color: "text-purple-400",
                  bg: "bg-purple-400/10",
                },
              ].map((feature, i) => (
                <div
                  key={i}
                  className="bg-[#050b05]/50 p-8 rounded-2xl border border-[#879d7b]/20 hover:border-[#4ade80]/50 transition-all hover:-translate-y-1 group"
                >
                  <div
                    className={`w-14 h-14 rounded-xl ${feature.bg} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}
                  >
                    <feature.icon className={feature.color} size={28} />
                  </div>
                  <h3 className="text-xl font-bold mb-3 text-white">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400 leading-relaxed">
                    {feature.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-[#879d7b]/20 py-12 mt-12 bg-[#050b05]">
          <div className="max-w-7xl mx-auto px-4 text-center">
            <div className="flex items-center justify-center gap-2 mb-4 opacity-50">
              <Sprout size={20} />
              <span className="font-bold">AGNISUTRA</span>
            </div>
            <p className="text-gray-500 text-sm">
              {t("copyright")}
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
