// app/profile/ProfileLayout.tsx
"use client"

import type { User } from "../types/user"
import ProfileHeader from "../components/ProfileHeader"
import CoverImage from "../components/CoverImage"
import AvatarSection from "../components/AvatarSection"
import InfoSection from "../components/InfoSection"
import DangerZone from "../components/DangerSection"

export default function ProfileLayout({ user }: { user: User }) {
  return (
    <div className="min-h-screen bg-[#06120a] text-white">
      <ProfileHeader />
      <main className="max-w-7xl mx-auto px-4 md:px-8">
        <CoverImage src={user.cover} alt={`${user.name} cover`} />
        <div className="relative -mt-20 md:-mt-28">
          <div className="md:flex md:items-start md:justify-between">
            <div className="md:flex md:items-end md:gap-6">
              <AvatarSection user={user} />
              {/* Name & joined for mobile if needed; AvatarSection handles it */}
            </div>

            {/* Action buttons on desktop aligned right */}
            <div className="mt-6 md:mt-0 md:flex md:items-center">
              {/* Buttons rendered inside AvatarSection for mobile, duplicated on desktop here */}
            </div>
          </div>
        </div>

        <section className="mt-8">
          <InfoSection user={user} />
        </section>

        <section className="mt-12 mb-20">
          <DangerZone />
        </section>
      </main>
    </div>
  )
}
