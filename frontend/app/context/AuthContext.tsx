"use client";

import { createContext, useState, useEffect, ReactNode } from "react";

type User = {
  access_Token: string;
  email: string;
};

type AuthContextType = {
  user: User | null;
  setUser: (userData: User) => void;
  clearUser: () => void;
};

export const AuthContext = createContext<AuthContextType | undefined>(
  undefined
);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUserState] = useState<User | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      try {
        setUserState(JSON.parse(storedUser));
      } catch (e) {
        console.error("Failed to parse user from localStorage", e);
        localStorage.removeItem("user");
      }
    }
  }, []);

  const setUser = (userData: User) => {
    console.log(userData);
    setUserState(userData);
    localStorage.setItem("user", JSON.stringify(userData)); // Save user data in localStorage
  };

  const clearUser = () => {
    setUserState(null);
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider value={{ user, setUser, clearUser }}>
      {children}
    </AuthContext.Provider>
  );
};
