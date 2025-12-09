import LanguageSwitcher from "./LanguageSwitcher";
import Link from "next/link";
import { Bell, User, Menu } from "lucide-react";
import Image from "next/image";
import { useTranslations } from "next-intl";
import LogOut from "./LogOut";
import { useWebSocket } from "../hooks/useWebSocket";
import { useState } from "react";

interface HeaderProps {
  userName?: string;
  userId?: number;
  showIcons?: boolean;
}

export default function Header({
  userName,
  userId,
  showIcons = true,
}: HeaderProps) {
  const t = useTranslations();

  // Dynamic WebSocket URL based on current host
  const [wsUrl] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const host = window.location.hostname;
      return `${protocol}://${host}:8000/ws/alerts`;
    }
    return "";
  });

  useWebSocket(wsUrl, userId);

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-[#879d7b]/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo Section */}
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="flex items-center gap-2 group">
              <div className="relative w-10 h-10 transition-transform group-hover:scale-105">
                <Image
                  src="/images/logo-1.png"
                  alt="AgniSutra Logo"
                  fill
                  className="object-contain drop-shadow-[0_0_10px_rgba(74,222,128,0.5)]"
                />
              </div>
              <span className="text-white font-bold text-xl tracking-tight group-hover:text-[#4ade80] transition-colors">
                {t("name")}
              </span>
            </Link>

            {/* Desktop Navigation Links (Optional - can be added here) */}
            <nav className="hidden md:flex ml-10 space-x-8">
              <Link
                href="/dashboard"
                className="text-gray-300 hover:text-[#4ade80] px-3 py-2 rounded-md text-sm font-medium transition-colors"
              >
                Dashboard
              </Link>
              <Link
                href="/disease-detection"
                className="text-gray-300 hover:text-[#4ade80] px-3 py-2 rounded-md text-sm font-medium transition-colors"
              >
                Disease AI
              </Link>
              <Link
                href="/soil-reports"
                className="text-gray-300 hover:text-[#4ade80] px-3 py-2 rounded-md text-sm font-medium transition-colors"
              >
                Soil Health
              </Link>
            </nav>
          </div>

          {/* Right Actions */}
          {showIcons && (
            <div className="flex items-center gap-4">
              <button className="p-2 text-gray-400 hover:text-[#4ade80] hover:bg-[#4ade80]/10 rounded-full transition-all relative">
                <Bell size={20} />
                <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
              </button>

              <div className="h-6 w-px bg-gray-700/50 mx-1"></div>

              <LanguageSwitcher />

              <div className="flex items-center gap-3 pl-2">
                <div className="hidden md:block text-right">
                  <p className="text-sm font-medium text-white">
                    {userName || "Farmer"}
                  </p>
                  <p className="text-xs text-gray-400">Pro Plan</p>
                </div>
                <div className="h-8 w-8 rounded-full bg-linear-to-br from-[#4ade80] to-[#22c55e] p-px">
                  <div className="h-full w-full rounded-full bg-[#050b05] flex items-center justify-center">
                    <User size={16} className="text-[#4ade80]" />
                  </div>
                </div>
                <LogOut />
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
