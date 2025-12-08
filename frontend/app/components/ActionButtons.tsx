// app/profile/components/ActionButtons.tsx
"use client"
import React from "react"
import { useRouter } from "next/navigation"

export default function ActionButtons() {
  const router = useRouter()

  const handleChangePassword = () => {
    // implement modal or navigation
    router.push("/profile/change-password")
  }

  const handleEditProfile = () => {
    router.push("/profile/edit")
  }

  return (
    <div className="flex gap-3 flex-wrap">
      <button
        onClick={handleChangePassword}
        className="px-4 py-2 border border-green-400 text-green-200 rounded-md text-sm hover:bg-green-900/40 transition"
      >
        Change Password
      </button>

      <button
        onClick={handleEditProfile}
        className="px-4 py-2 bg-green-400 text-black rounded-md text-sm hover:brightness-95 transition"
      >
        Edit Profile
      </button>
    </div>
  )
}
