// app/profile/components/InfoSection.tsx
"use client";
import React from "react";
import type { User } from "../types/user";
import InfoRow from "./InfoRow";
import { LogOut, Trash2 } from "lucide-react";

export default function InfoSection({ user }: { user: User }) {
  return (
    <div className="bg-[#1a2e1a]/20 backdrop-blur-md border border-[#879d7b]/20 rounded-xl p-4 md:p-8 mt-4">
      <div className="grid gap-0 divide-y divide-[#879d7b]/10">
        <InfoRow label="NAME" value={user.name} />
        <InfoRow label="EMAIL ID" value={user.email} />
        <InfoRow label="MOBILE" value={user.mobile} />
        <InfoRow label="USERNAME" value={user.username} />
        <InfoRow label="USER ID" value={user.id} />
      </div>

      {/* Buttons on small screens: keep them accessible near info */}
      <div className="mt-8 flex flex-wrap gap-4">
        <button
          className="flex items-center gap-2 px-6 py-2.5 border border-red-500/30 text-red-400 rounded-xl text-sm font-medium hover:bg-red-500/10 transition-all"
          onClick={() => {
            // logout logic
            console.log("logout");
          }}
        >
          <LogOut size={16} />
          Logout
        </button>

        <button
          className="flex items-center gap-2 px-6 py-2.5 bg-red-500/10 text-red-400 border border-red-500/30 rounded-xl text-sm font-medium hover:bg-red-500/20 transition-all"
          onClick={() => {
            // delete account logic
            console.log("delete");
          }}
        >
          <Trash2 size={16} />
          Delete Account
        </button>
      </div>
    </div>
  );
}
