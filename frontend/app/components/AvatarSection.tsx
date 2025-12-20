// app/profile/components/AvatarSection.tsx
"use client";
import React from "react";
import { User } from "../types/user";
import ActionButtons from "./ActionButtons";
import { Calendar } from "lucide-react";

export default function AvatarSection({ user }: { user: User }) {
  const avatar = user.avatar ?? "/images/default-avatar.jpg";

  return (
    <div className="flex flex-col md:flex-row md:items-end md:gap-6 w-full px-4">
      {/* Avatar */}
      <div className="shrink-0 relative group">
        <div className="absolute inset-0 bg-[#4ade80] rounded-full blur-md opacity-20 group-hover:opacity-40 transition-opacity"></div>
        <img
          src={avatar}
          alt={`${user.name} avatar`}
          className="relative w-32 h-32 md:w-40 md:h-40 rounded-full border-4 border-[#050b05] object-cover shadow-2xl"
        />
      </div>

      {/* Name + joined + buttons */}
      <div className="mt-4 md:mt-0 flex-1 pb-2">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight">
              {user.name}
            </h1>
            <div className="flex items-center gap-2 text-sm text-gray-400 mt-2">
              <Calendar size={14} />
              <span>
                Joined on {new Date(user.joinedOn).toLocaleDateString()}
              </span>
            </div>
          </div>

          {/* Buttons show stacked on mobile, horizontal on desktop */}
          <div className="w-full md:w-auto">
            <ActionButtons />
          </div>
        </div>
      </div>
    </div>
  );
}
