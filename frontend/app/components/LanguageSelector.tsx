"use client";

import { useState, useEffect } from "react";
import { Check } from "lucide-react";

const languages = [
  { code: "en", name: "English", native: "English" },
  { code: "hi", name: "Hindi", native: "हिन्दी" },
  { code: "mr", name: "Marathi", native: "मराठी" },
  { code: "gu", name: "Gujarati", native: "ગુજરાતી" },
  { code: "ta", name: "Tamil", native: "தமிழ்" },
];

export default function LanguageSelector({
  onComplete,
}: {
  onComplete: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedLang, setSelectedLang] = useState("en");
  const [pendingLang, setPendingLang] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      const storedLang = localStorage.getItem("app_language");
      if (!storedLang) {
        setIsOpen(true);
      } else {
        setSelectedLang(storedLang);
        onComplete();
      }
    }, 0);
    return () => clearTimeout(timer);
  }, [onComplete]);

  useEffect(() => {
    if (pendingLang) {
      document.cookie = `locale=${pendingLang}; path=/; max-age=31536000`;
      window.location.reload();
    }
  }, [pendingLang]);

  const handleSelect = (langCode: string) => {
    setSelectedLang(langCode);
    localStorage.setItem("app_language", langCode);
    setIsOpen(false);
    onComplete();
    setPendingLang(langCode);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#050b05]/90 backdrop-blur-md">
      <div className="bg-[#1a2e1a]/20 border border-[#879d7b]/20 rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl animate-in fade-in zoom-in duration-300 backdrop-blur-xl">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">
            Welcome to AgniSutra
          </h2>
          <p className="text-gray-400">Please select your preferred language</p>
          <p className="text-[#4ade80] text-sm mt-1 font-medium">
            कृपया अपनी पसंदीदा भाषा चुनें
          </p>
        </div>

        <div className="space-y-3">
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => handleSelect(lang.code)}
              className={`w-full flex items-center justify-between p-4 rounded-xl border transition-all duration-200 group ${
                selectedLang === lang.code
                  ? "bg-[#4ade80] border-[#4ade80] text-[#050b05]"
                  : "bg-[#050b05]/50 border-[#879d7b]/20 text-gray-300 hover:border-[#4ade80]/50 hover:bg-[#4ade80]/10"
              }`}
            >
              <div className="flex flex-col items-start">
                <span className="font-bold text-lg">{lang.native}</span>
                <span
                  className={`text-xs ${
                    selectedLang === lang.code
                      ? "text-[#050b05]/80"
                      : "text-gray-500 group-hover:text-[#4ade80]/80"
                  }`}
                >
                  {lang.name}
                </span>
              </div>
              {selectedLang === lang.code && <Check size={20} />}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
