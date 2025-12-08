// app/profile/components/AvatarSection.tsx
"use client"
import React from "react"
import { User } from "../types/user"
import ActionButtons from "./ActionButtons"

export default function AvatarSection({ user }: { user: User }) {
  const avatar = user.avatar ?? "/images/default-avatar.jpg"

  return (
    <div className="flex flex-col md:flex-row md:items-end md:gap-6 w-full">
      {/* Avatar */}
      <div className="shrink-0">
        <div className="relative">
          <img
            src={avatar}
            alt={`${user.name} avatar`}
            className="w-28 h-28 md:w-40 md:h-40 rounded-full border-8 border-[#08170f] object-cover shadow-lg"
          />
        </div>
      </div>

      {/* Name + joined + buttons */}
      <div className="mt-4 md:mt-0 flex-1">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold">{user.name}</h1>
            <p className="text-sm text-gray-300 mt-1">
              Joined on {new Date(user.joinedOn).toLocaleDateString()}
            </p>
          </div>

          {/* Buttons show stacked on mobile, horizontal on desktop */}
          <div className="mt-4 md:mt-0">
            <ActionButtons />
          </div>
        </div>
      </div>
    </div>
  )
}
