"use client";

import { useState, useEffect } from "react";
import Header from "../components/HeaderDashboard";
import Link from "next/link";
import { useTranslations } from "next-intl";
import AddField from "../components/AddField";
import SensorDataWidget from "../components/SensorDataWidget";
import dynamic from "next/dynamic";
import YieldPredictionWidget from "../components/YieldPredictionWidget";
import { useAuth } from "../hooks/useAuth";
import api from "../services/api";
import { AlertTriangle, ArrowRight, Sprout } from "lucide-react";

const NDVIWidget = dynamic(() => import("../components/NDVIWidget"), {
  ssr: false,
  loading: () => <div className="h-64 bg-[#1a2e1a] rounded-xl animate-pulse" />,
});

export default function DashboardPage() {
  const [simulationMode, setSimulationMode] = useState(false);
  const [userProfile, setUserProfile] = useState<any>(null);
  const { user } = useAuth();
  const t = useTranslations("dashboard");

  // Location State (Default: Kolkata, India)
  const [location, setLocation] = useState<{ lat: number; lon: number }>({
    lat: 22.5726,
    lon: 88.3639,
  });

  useEffect(() => {
    // Try to get user's real location
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          });
        },
        (error) => {
          console.warn("Geolocation failed, using default:", error);
        }
      );
    }
  }, []);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await api.get("/auth/me");
        setUserProfile(res.data);
      } catch (error) {
        console.error("Failed to fetch profile", error);
      }
    };
    if (user) {
      fetchProfile();
    }
  }, [user]);

  const userName = userProfile?.name || "Farmer";

  return (
    <div className="min-h-screen bg-[#050b05] text-white">
      <Header userName={userName} userId={userProfile?.id} showIcons={true} />

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-8">
        {/* Welcome Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              {t("title")}
            </h1>
            <h2 className="text-[#4ade80] text-xl font-medium flex items-center gap-2">
              <Sprout size={20} />
              {t("welcome")}, {userName}!
            </h2>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setSimulationMode(!simulationMode)}
              className={`px-4 py-2 rounded-full border transition-all flex items-center gap-2 text-sm font-medium ${
                simulationMode
                  ? "bg-yellow-500/20 border-yellow-500 text-yellow-200 shadow-[0_0_15px_rgba(234,179,8,0.3)]"
                  : "bg-white/5 border-white/10 text-gray-400 hover:bg-white/10"
              }`}
            >
              <AlertTriangle size={16} />
              {simulationMode ? t("simulation.active") : t("simulation.start")}
            </button>

            <Link
              href="/soil-reports"
              className="hidden md:flex rounded-full bg-[#4ade80] text-[#050b05] py-2 px-6 font-bold hover:bg-[#22c55e] transition-all shadow-[0_0_20px_rgba(74,222,128,0.3)] hover:shadow-[0_0_30px_rgba(74,222,128,0.5)] items-center gap-2"
            >
              <span>{t("actions.ai")}</span>
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>

        {/* Simulation Alert Banner */}
        {simulationMode && (
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl animate-pulse flex items-center gap-4 backdrop-blur-sm">
            <div className="p-2 bg-red-500/20 rounded-full">
              <AlertTriangle className="text-red-500" size={24} />
            </div>
            <div>
              <h3 className="text-red-400 font-bold text-lg">
                {t("simulation.alert.title")}
              </h3>
              <p className="text-red-200/80 text-sm">
                {t("simulation.alert.desc")}
              </p>
            </div>
          </div>
        )}

        {/* Real-time Monitoring Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* IoT Sensors */}
          <div className="lg:col-span-1 h-full">
            <SensorDataWidget />
          </div>

          {/* Satellite View */}
          <div className="lg:col-span-2 h-full">
            <NDVIWidget lat={location.lat} lon={location.lon} />
          </div>
        </div>

        {/* Yield Prediction & Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <YieldPredictionWidget />
          </div>

          <div className="lg:col-span-2 space-y-6">
            {/* Quick Actions */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Link
                href="/my-fields"
                className="glass-card rounded-xl p-6 flex flex-col items-center justify-center gap-3 group cursor-pointer"
              >
                <div className="p-3 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
                  <span className="text-2xl">🌾</span>
                </div>
                <span className="text-gray-300 font-medium group-hover:text-white transition-colors">
                  {t("actions.my")}
                </span>
              </Link>

              <div className="h-full w-full">
                <AddField />
              </div>

              <Link
                href="/soil-reports"
                className="glass-card rounded-xl p-6 flex flex-col items-center justify-center gap-3 group cursor-pointer"
              >
                <div className="p-3 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
                  <span className="text-2xl">📋</span>
                </div>
                <span className="text-gray-300 font-medium group-hover:text-white transition-colors">
                  {t("actions.soil")}
                </span>
              </Link>

              <Link
                href="/disease-detection"
                className="glass-card rounded-xl p-6 flex flex-col items-center justify-center gap-3 group cursor-pointer"
              >
                <div className="p-3 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
                  <span className="text-2xl">🔍</span>
                </div>
                <span className="text-gray-300 font-medium group-hover:text-white transition-colors">
                  {t("actions.disease")}
                </span>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
