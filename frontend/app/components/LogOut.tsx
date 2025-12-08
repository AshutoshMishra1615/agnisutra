"use client";

import { useAuth } from "../hooks/useAuth";
import axios from "axios";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

export default function LogOut() {
  const { user, clearUser } = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    try {
      await axios.post("/auth/logout", {}, {
        headers: {
          Authorization: `Bearer ${user?.access_Token}`,
        },
      });
      clearUser(); // Clear user state in context
      delete axios.defaults.headers.common["Authorization"]; // Remove Authorization header
      toast.success("Logout Successful");
      router.push("/login"); // Redirect to login page
    } catch (error) {
      toast.error("Logout Unsuccessful");
      console.error("Logout failed:", error);
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