// app/profile/components/InfoSection.tsx
"use client"
import React from "react"
import type { User } from "../types/user"
import InfoRow from "./InfoRow"

export default function InfoSection({ user }: { user: User }) {
  return (
    <div className="bg-[#07130d] rounded-md p-4 md:p-8 mt-4">
      <div className="grid gap-0">
        <InfoRow label="NAME" value={user.name} />
        <InfoRow label="EMAIL ID" value={user.email} />
        <InfoRow label="MOBILE" value={user.mobile} />
        <InfoRow label="USERNAME" value={user.username} />
        <InfoRow label="USER ID" value={user.id} />
      </div>

      {/* Buttons on small screens: keep them accessible near info */}
      <div className="mt-6 flex gap-4">
        <button
          className="px-6 py-2 border border-red-400 text-red-400 rounded-md text-sm hover:bg-red-900/30 transition"
          onClick={() => {
            // logout logic
            console.log("logout")
          }}
        >
          Logout
        </button>

        <button
          className="px-6 py-2 bg-red-500 text-white rounded-md text-sm hover:brightness-95 transition"
          onClick={() => {
            // delete account logic
            console.log("delete")
          }}
        >
          Delete Account
        </button>
      </div>
    </div>
  )
}
