"use client";

import { createContext, useContext, useState, ReactNode } from "react";
import { useRouter } from "next/navigation";

type User = {
  accessToken: string;
  name: string;
  email: string;
};

type AuthContextType = {
  user: User | null;
  login: (userData: User) => void;
  logout: () => void;
};

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();

  const login = (userData: User) => {
    setUser(userData);
    // Optionally, save user data to localStorage/sessionStorage
    localStorage.setItem("user", JSON.stringify(userData));
    router.push("/dashboard"); // Redirect after login
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("user");
    router.push("/login"); // Redirect after logout
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

