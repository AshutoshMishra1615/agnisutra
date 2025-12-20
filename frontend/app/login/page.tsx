"use client";

import LoginCard from "../components/LoginCard";
import { useTranslations } from "next-intl";
import { Sprout, Droplets, Calendar, Bot, Activity } from "lucide-react";

export default function LoginPage() {
  const t = useTranslations();

  const benefits = [
    {
      icon: Sprout,
      text: t("register.benefits.diary"),
      color: "text-green-400",
    },
    {
      icon: Droplets,
      text: t("register.benefits.moisture"),
      color: "text-blue-400",
    },
    {
      icon: Calendar,
      text: t("register.benefits.timeline"),
      color: "text-yellow-400",
    },
    {
      icon: Bot,
      text: t("register.benefits.advisor"),
      color: "text-purple-400",
    },
    {
      icon: Activity,
      text: t("register.benefits.health"),
      color: "text-red-400",
    },
  ];

  return (
    <div className="min-h-screen w-full bg-[#050b05] bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-[#1a2e1a] via-[#050b05] to-[#050b05] flex items-center justify-center p-4 lg:p-8">
      <div className="w-full max-w-6xl flex flex-col lg:flex-row items-center justify-between gap-12 lg:gap-20">
        {/* Left Side - Benefits */}
        <div className="hidden lg:flex flex-col flex-1 space-y-8">
          <div className="space-y-4">
            <h1 className="text-5xl font-bold text-white leading-tight">
              Smart Farming <br />
              <span className="text-transparent bg-clip-text bg-linear-to-r from-[#4ade80] to-[#22c55e]">
                Revolutionized
              </span>
            </h1>
            <p className="text-gray-400 text-lg max-w-md">
              Join thousands of farmers using AI to optimize their yield and
              protect their crops.
            </p>
          </div>

          <div className="grid gap-4">
            {benefits.map((benefit, index) => (
              <div
                key={index}
                className="flex items-center gap-4 p-4 rounded-xl bg-[#1a2e1a]/20 backdrop-blur-sm border border-[#879d7b]/10 hover:bg-[#1a2e1a]/40 transition-all transform hover:translate-x-2"
              >
                <div className={`p-2 rounded-lg bg-white/5 ${benefit.color}`}>
                  <benefit.icon size={24} />
                </div>
                <span className="text-gray-200 font-medium">
                  {benefit.text}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="hidden lg:block w-px h-[600px] bg-linear-to-b from-transparent via-[#879d7b]/20 to-transparent"></div>

        {/* Right Side - Login Card */}
        <div className="w-full lg:flex-1 flex justify-center">
          <LoginCard />
        </div>
      </div>
    </div>
  );
}
