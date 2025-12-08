// app/profile/components/DangerZone.tsx
"use client"
import React from "react"

export default function DangerZone() {
  return (
    <div className="bg-[#07130d] rounded-md p-6 md:p-8">
      <h3 className="text-lg font-semibold text-white">Danger Zone</h3>
      <p className="text-sm text-gray-300 mt-2">
        Actions here are irreversible. Use with caution.
      </p>

      <div className="mt-6 flex gap-4">
        <button
          className="px-6 py-2 border border-red-400 text-red-400 rounded-md text-sm hover:bg-red-900/30 transition"
          onClick={() => {
            console.log("logout")
          }}
        >
          Logout
        </button>

        <button
          className="px-6 py-2 bg-red-500 text-white rounded-md text-sm hover:brightness-95 transition"
          onClick={() => {
            if (
              confirm(
                "Are you sure you want to delete your account? This action cannot be undone."
              )
            ) {
              // call API to delete
              console.log("deleted")
            }
          }}
        >
          Delete Account
        </button>
      </div>
    </div>
  )
}
