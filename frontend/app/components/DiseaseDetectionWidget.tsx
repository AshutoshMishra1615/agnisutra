"use client";

import { useState } from "react";
import api from "../services/api";
import {
  Loader2,
  Upload,
  ScanSearch,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useTranslations, useLocale } from "next-intl";

type DiseaseResult = {
  predicted_class: string;
  confidence: number;
  preventive_measures: string;
};

export default function DiseaseDetectionWidget() {
  const t = useTranslations("disease");
  const locale = useLocale();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DiseaseResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("language", locale); // Pass current language to backend

    try {
      const response = await api.post("/disease/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(response.data);
      toast.success(t("success"));
    } catch (error) {
      console.error("Disease detection error:", error);
      toast.error(t("error"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#1a2e1a]/40 backdrop-blur-md rounded-xl p-6 border border-[#879d7b]/20 h-full flex flex-col shadow-xl">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <ScanSearch className="text-[#4ade80]" size={24} />
          {t("title")}
        </h3>
        {result && (
          <span className="px-3 py-1 rounded-full bg-[#4ade80]/20 text-[#4ade80] text-xs font-bold flex items-center gap-1">
            <CheckCircle2 size={12} />
            {t("analyzed")}
          </span>
        )}
      </div>

      <div className="flex-1 flex flex-col gap-6">
        {/* Upload Area */}
        <div
          className={`relative border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center min-h-[240px] transition-all duration-300 group ${
            previewUrl
              ? "border-[#4ade80]/50 bg-[#0E1A0E]/80"
              : "border-[#879d7b]/30 bg-[#0E1A0E]/40 hover:bg-[#0E1A0E]/60 hover:border-[#4ade80]/50"
          }`}
        >
          {previewUrl ? (
            <div className="relative w-full h-full min-h-[200px] flex items-center justify-center">
              <Image
                src={previewUrl}
                alt="Preview"
                fill
                className="object-contain rounded-lg"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setResult(null);
                  }}
                  className="bg-red-500/80 text-white p-2 rounded-full hover:bg-red-600 transition-colors transform hover:scale-110"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Scanning Effect Overlay */}
              {loading && (
                <div className="absolute inset-0 overflow-hidden rounded-lg">
                  <div className="absolute top-0 left-0 w-full h-1 bg-[#4ade80] shadow-[0_0_15px_#4ade80] animate-[scan_2s_linear_infinite]"></div>
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="w-16 h-16 rounded-full bg-[#4ade80]/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <Upload className="text-[#4ade80]" size={32} />
              </div>
              <p className="text-white font-medium mb-1">{t("upload_title")}</p>
              <p className="text-gray-400 text-xs text-center max-w-[200px]">
                {t("upload_desc")}
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
            </>
          )}
        </div>

        {/* Action Button */}
        {!result && (
          <button
            onClick={handleUpload}
            disabled={!selectedFile || loading}
            className="w-full bg-gradient-to-r from-[#4ade80] to-[#22c55e] hover:from-[#22c55e] hover:to-[#16a34a] text-[#050b05] font-bold py-3 rounded-xl transition-all flex justify-center items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(74,222,128,0.2)] hover:shadow-[0_0_30px_rgba(74,222,128,0.4)] hover:scale-[1.02]"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                {t("analyzing")}
              </>
            ) : (
              <>
                <ScanSearch size={20} />
                {t("diagnose_btn")}
              </>
            )}
          </button>
        )}

        {/* Results Section */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-4 space-y-4">
            {/* Removed Diagnosis Box as requested */}

            <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl">
              <h4 className="text-blue-400 font-bold text-sm mb-2 flex items-center gap-2">
                <ShieldCheck size={16} />
                {t("treatment_title")}
              </h4>
              <div className="text-sm text-blue-100/80 leading-relaxed markdown-content">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    ul: ({ ...props }) => (
                      <ul
                        className="list-disc pl-4 space-y-1 my-2"
                        {...props}
                      />
                    ),
                    ol: ({ ...props }) => (
                      <ol
                        className="list-decimal pl-4 space-y-1 my-2"
                        {...props}
                      />
                    ),
                    li: ({ ...props }) => <li className="mb-1" {...props} />,
                    strong: ({ ...props }) => (
                      <strong className="font-bold text-white" {...props} />
                    ),
                    p: ({ ...props }) => (
                      <p className="mb-2 last:mb-0" {...props} />
                    ),
                  }}
                >
                  {result.preventive_measures}
                </ReactMarkdown>
              </div>
            </div>

            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
                setResult(null);
              }}
              className="w-full py-3 text-gray-400 hover:text-white text-sm hover:bg-white/5 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Upload size={14} />
              {t("analyze_another")}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
