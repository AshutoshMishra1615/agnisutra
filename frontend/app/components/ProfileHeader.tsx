// app/profile/components/Header.tsx
"use client";
import Link from "next/link";
import { Sprout } from "lucide-react";

export default function ProfileHeader() {
  return (
    <header className="bg-[#1a2e1a]/20 backdrop-blur-md border-b border-[#879d7b]/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
        <Link href="/dashboard" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-full bg-[#4ade80]/10 flex items-center justify-center group-hover:bg-[#4ade80]/20 transition-colors">
            <Sprout className="text-[#4ade80]" size={20} />
          </div>
          <span className="text-xl font-bold text-white group-hover:text-[#4ade80] transition-colors">
            AgniSutra
          </span>
        </Link>
      </div>
    </header>
  );
}
