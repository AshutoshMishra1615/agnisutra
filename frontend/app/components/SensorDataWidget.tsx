"use client";

import { useEffect, useState } from "react";
import api from "../services/api";
import { useTranslations } from "next-intl";
import { Loader2, Wind, Activity, Signal } from "lucide-react";

type SensorData = {
  temperature: number;
  humidity: number;
  moisture: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  timestamp: string;
};

export default function SensorDataWidget() {
  const t = useTranslations("sensor");
  const [data, setData] = useState<SensorData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const response = await api.get("/iot/latest");
      setData(response.data);
    } catch (error) {
      console.error("Error fetching sensor data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading)
    return (
      <div className="h-full flex items-center justify-center p-4 bg-[#1a2e1a]/20 rounded-xl border border-[#879d7b]/10">
        <Loader2 className="animate-spin text-[#4ade80]" />
      </div>
    );

  if (!data)
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 bg-[#1a2e1a]/20 rounded-xl border border-[#879d7b]/10 text-center">
        <Signal className="text-gray-500 mb-2" size={32} />
        <p className="text-gray-400">No sensor data available.</p>
        <button
          onClick={fetchData}
          className="mt-2 text-[#4ade80] text-sm hover:underline"
        >
          Retry Connection
        </button>
      </div>
    );

  return (
    <div className="h-full bg-[#1a2e1a]/40 backdrop-blur-md rounded-xl p-6 border border-[#879d7b]/20 flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <Activity className="text-[#4ade80]" size={24} />
          {t("title")}
        </h3>
        <div className="flex items-center gap-2">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
          </span>
          <span className="text-xs text-green-400 font-mono">
            {t("online")}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 flex-1">
        {/* Soil Moisture */}
        <div className="bg-[#0E1A0E]/60 p-4 rounded-xl border border-[#879d7b]/10 hover:border-[#4ade80]/30 transition-colors group col-span-2">
          <div className="flex items-center gap-2 text-gray-400 mb-2 group-hover:text-[#4ade80] transition-colors">
            <Wind size={18} />
            <span className="text-xs uppercase tracking-wider font-medium">
              {t("moisture")}
            </span>
          </div>
          <p className="text-3xl font-bold text-white">
            {data.moisture?.toFixed(1)}
            <span className="text-lg text-gray-500">%</span>
          </p>
        </div>

        {/* NPK Nutrients */}
        <div className="bg-[#0E1A0E]/60 p-4 rounded-xl border border-[#879d7b]/10 hover:border-[#4ade80]/30 transition-colors group col-span-2">
          <div className="flex items-center gap-2 text-gray-400 mb-2 group-hover:text-[#4ade80] transition-colors">
            <span className="text-xs uppercase tracking-wider font-bold">
              {t("npk")}
            </span>
            <span className="text-[10px] text-gray-500">mg/kg</span>
          </div>
          <div className="flex justify-between items-end h-8 mt-1 gap-1">
            <div className="flex flex-col items-center w-1/3">
              <div className="w-full bg-blue-500/20 h-1 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${Math.min(data.nitrogen, 100)}%` }}
                ></div>
              </div>
              <span className="text-xs text-blue-400 font-mono mt-1">
                N:{data.nitrogen}
              </span>
            </div>
            <div className="flex flex-col items-center w-1/3">
              <div className="w-full bg-purple-500/20 h-1 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500"
                  style={{ width: `${Math.min(data.phosphorus, 100)}%` }}
                ></div>
              </div>
              <span className="text-xs text-purple-400 font-mono mt-1">
                P:{data.phosphorus}
              </span>
            </div>
            <div className="flex flex-col items-center w-1/3">
              <div className="w-full bg-yellow-500/20 h-1 rounded-full overflow-hidden">
                <div
                  className="h-full bg-yellow-500"
                  style={{ width: `${Math.min(data.potassium, 100)}%` }}
                ></div>
              </div>
              <span className="text-xs text-yellow-400 font-mono mt-1">
                K:{data.potassium}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 text-right">
        <span className="text-[10px] text-gray-500">
          {t("last_updated")}: {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}
