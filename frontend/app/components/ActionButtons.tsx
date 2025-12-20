// app/profile/components/ActionButtons.tsx
"use client";
import React from "react";
import { useRouter } from "next/navigation";
import { KeyRound, Pencil } from "lucide-react";

export default function ActionButtons() {
  const router = useRouter();

  const handleChangePassword = () => {
    // implement modal or navigation
    router.push("/profile/change-password");
  };

  const handleEditProfile = () => {
    router.push("/profile/edit");
  };

  return (
    <div className="flex gap-3 flex-wrap">
      <button
        onClick={handleChangePassword}
        className="flex items-center gap-2 px-4 py-2 border border-[#4ade80]/30 text-[#4ade80] rounded-xl text-sm font-medium hover:bg-[#4ade80]/10 transition-all"
      >
        <KeyRound size={16} />
        Change Password
      </button>

      <button
        onClick={handleEditProfile}
        className="flex items-center gap-2 px-4 py-2 bg-[#4ade80] text-[#050b05] rounded-xl text-sm font-bold hover:bg-[#22c55e] transition-all shadow-lg shadow-[#4ade80]/20"
      >
        <Pencil size={16} />
        Edit Profile
      </button>
    </div>
  );
}
