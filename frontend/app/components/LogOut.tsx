"use client";

import { useAuth } from "../hooks/useAuth";

export default function LogOut() {
  const { logout } = useAuth();

  return (
    <button
      onClick={logout}
      className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
    >
      Log Out
    </button>
  );
}