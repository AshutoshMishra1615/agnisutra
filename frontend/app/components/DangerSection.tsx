// app/profile/components/DangerZone.tsx
"use client";
import React from "react";
import { AlertTriangle, LogOut, Trash2 } from "lucide-react";

export default function DangerZone() {
  return (
    <div className="bg-[#1a2e1a]/20 backdrop-blur-md border border-red-500/20 rounded-xl p-6 md:p-8">
      <div className="flex items-center gap-2 mb-2">
        <AlertTriangle className="text-red-500" size={20} />
        <h3 className="text-lg font-bold text-white">Danger Zone</h3>
      </div>
      <p className="text-sm text-gray-400">
        Actions here are irreversible. Use with caution.
      </p>

      <div className="mt-6 flex flex-wrap gap-4">
        <button
          className="flex items-center gap-2 px-6 py-2.5 border border-red-500/30 text-red-400 rounded-xl text-sm font-medium hover:bg-red-500/10 transition-all"
          onClick={() => {
            console.log("logout");
          }}
        >
          <LogOut size={16} />
          Logout
        </button>

        <button
          className="flex items-center gap-2 px-6 py-2.5 bg-red-500/10 text-red-400 border border-red-500/30 rounded-xl text-sm font-medium hover:bg-red-500/20 transition-all"
          onClick={() => {
            if (
              confirm(
                "Are you sure you want to delete your account? This action cannot be undone."
              )
            ) {
              // call API to delete
              console.log("deleted");
            }
          }}
        >
          <Trash2 size={16} />
          Delete Account
        </button>
      </div>
    </div>
  );
}
