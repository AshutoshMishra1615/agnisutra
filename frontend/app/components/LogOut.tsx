"use client";

import { useAuth } from "../hooks/useAuth";
import api from "../services/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

export default function LogOut() {
  const { clearUser } = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    try {
      // The api instance automatically adds the Authorization header from localStorage
      await api.post("/auth/logout");

      clearUser(); // Clear user state in context and localStorage
      toast.success("Logout Successful");
      router.push("/login"); // Redirect to login page
    } catch (error) {
      // Even if the API call fails (e.g. token expired), we should still clear the local session
      console.error("Logout failed:", error);
      clearUser();
      toast.success("Logged out locally");
      router.push("/login");
    }
  };

  return (
    <button
      onClick={handleLogout}
      className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
    >
      Log Out
    </button>
  );
}
