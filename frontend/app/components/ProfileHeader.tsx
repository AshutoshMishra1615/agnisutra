// app/profile/components/Header.tsx
"use client"
import Link from "next/link"

export default function ProfileHeader() {
  return (
    <header className="bg-[#2f4136]">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
            {/* replace with logo img */}
            <span className="text-green-200 font-bold">🌱</span>
          </div>
          <span className="text-xl font-semibold text-white">AgniSutra</span>
        </Link>
      </div>
    </header>
  )
}
