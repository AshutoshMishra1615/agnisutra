"use client";

import { useState, useEffect } from "react";
import { fieldService, FieldAnalysis } from "../services/fieldService";
import {
  X,
  Loader2,
  Sprout,
  CloudRain,
  Activity,
  MessageSquare,
} from "lucide-react";
import { toast } from "react-hot-toast";
import Link from "next/link";

interface FieldAnalysisModalProps {
  fieldId: number;
  fieldName: string;
  onClose: () => void;
}

export default function FieldAnalysisModal({
  fieldId,
  fieldName,
  onClose,
}: FieldAnalysisModalProps) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<FieldAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<"ndvi" | "weather" | "yield">(
    "ndvi"
  );

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await fieldService.getFieldAnalysis(fieldId);
        setData(result);
      } catch (error) {
        console.error("Analysis fetch error:", error);
        toast.error("Failed to load analysis data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [fieldId]);

  if (!fieldId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0a120a] border border-[#879d7b]/30 rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl relative">
        {/* Header */}
        <div className="p-6 border-b border-[#879d7b]/20 flex items-center justify-between bg-[#1a2e1a]/50 rounded-t-2xl">
          <div>
            <h2 className="text-xl font-bold text-white">Field Analysis</h2>
            <p className="text-sm text-gray-400">
              Report for: <span className="text-[#4ade80]">{fieldName}</span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4">
              <Loader2 className="animate-spin text-[#4ade80]" size={40} />
              <p className="text-gray-400">Analyzing satellite data...</p>
            </div>
          ) : data ? (
            <div className="space-y-6">
              {/* Tabs */}
              <div className="flex p-1 bg-white/5 rounded-lg">
                <button
                  onClick={() => setActiveTab("ndvi")}
                  className={`flex-1 py-2 text-sm font-medium rounded-md transition-all flex items-center justify-center gap-2 ${
                    activeTab === "ndvi"
                      ? "bg-[#4ade80] text-[#050b05] shadow-lg"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  <Activity size={16} /> NDVI Health
                </button>
                <button
                  onClick={() => setActiveTab("weather")}
                  className={`flex-1 py-2 text-sm font-medium rounded-md transition-all flex items-center justify-center gap-2 ${
                    activeTab === "weather"
                      ? "bg-[#4ade80] text-[#050b05] shadow-lg"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  <CloudRain size={16} /> Weather
                </button>
                <button
                  onClick={() => setActiveTab("yield")}
                  className={`flex-1 py-2 text-sm font-medium rounded-md transition-all flex items-center justify-center gap-2 ${
                    activeTab === "yield"
                      ? "bg-[#4ade80] text-[#050b05] shadow-lg"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  <Sprout size={16} /> Yield Forecast
                </button>
              </div>

              {/* Tab Content */}
              <div className="min-h-[300px]">
                {activeTab === "ndvi" && (
                  <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Current Health
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {(data.ndvi.ndvi_flowering || 0).toFixed(2)}
                        </p>
                        <div className="w-full bg-gray-700 h-2 rounded-full mt-2 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                            style={{
                              width: `${
                                (data.ndvi.ndvi_flowering || 0) * 100
                              }%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Peak Vegetation
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {(data.ndvi.ndvi_peak || 0).toFixed(2)}
                        </p>
                      </div>
                    </div>

                    {data.ndvi.ndvi_image ? (
                      <div className="rounded-xl overflow-hidden border border-white/10 relative group">
                        <img
                          src={data.ndvi.ndvi_image}
                          alt="NDVI Map"
                          className="w-full h-48 object-cover"
                        />
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <span className="text-white font-medium">
                            Satellite Imagery
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="h-48 bg-white/5 rounded-xl flex items-center justify-center border border-white/10 border-dashed">
                        <p className="text-gray-500 text-sm">
                          No satellite imagery available
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "weather" && (
                  <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Temperature
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {data.weather.temperature}°C
                        </p>
                      </div>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Humidity
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {data.weather.humidity}%
                        </p>
                      </div>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Rainfall (1h)
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {data.weather.rainfall} mm
                        </p>
                      </div>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                        <p className="text-gray-400 text-xs uppercase tracking-wider mb-1">
                          Seasonal Rain
                        </p>
                        <p className="text-2xl font-bold text-white">
                          {data.weather.stats?.seasonal_rain_mm || 0} mm
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === "yield" && (
                  <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="bg-gradient-to-br from-[#1a2e1a] to-[#0a120a] p-6 rounded-xl border border-[#4ade80]/30 text-center">
                      <p className="text-[#4ade80] text-sm font-bold uppercase tracking-widest mb-2">
                        Predicted Yield
                      </p>
                      <div className="flex items-baseline justify-center gap-2">
                        <span className="text-5xl font-bold text-white">
                          {data.yield_forecast.predicted_yield.toFixed(2)}
                        </span>
                        <span className="text-xl text-gray-400">
                          {data.yield_forecast.unit}
                        </span>
                      </div>
                    </div>

                    {data.yield_forecast.alerts &&
                      data.yield_forecast.alerts.length > 0 && (
                        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
                          <h4 className="text-yellow-400 font-bold mb-2 flex items-center gap-2">
                            <Activity size={16} /> Insights
                          </h4>
                          <ul className="space-y-2">
                            {data.yield_forecast.alerts.map(
                              (alert: string, i: number) => (
                                <li
                                  key={i}
                                  className="text-sm text-yellow-200/80 flex items-start gap-2"
                                >
                                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-yellow-500 shrink-0" />
                                  {alert}
                                </li>
                              )
                            )}
                          </ul>
                        </div>
                      )}
                  </div>
                )}
              </div>

              {/* Ask Krishi Saathi Button */}
              <div className="pt-4 border-t border-white/10">
                <Link
                  href={`/disease-detection?context=field_analysis&field_id=${fieldId}`}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-[#4ade80] to-[#22c55e] hover:from-[#22c55e] hover:to-[#16a34a] text-[#050b05] font-bold flex items-center justify-center gap-2 transition-all shadow-[0_0_20px_rgba(74,222,128,0.2)] hover:shadow-[0_0_30px_rgba(74,222,128,0.4)]"
                >
                  <MessageSquare size={20} />
                  Ask Krishi Saathi about this Field
                </Link>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-12">
              Failed to load data. Please try again.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
