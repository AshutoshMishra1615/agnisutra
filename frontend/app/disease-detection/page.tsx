"use client";

import DiseaseDetectionWidget from "../components/DiseaseDetectionWidget";
import Header from "../components/HeaderDashboard";

export default function DiseaseDetectionPage() {
  return (
    <div className="min-h-screen bg-[#050b05] bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-[#1a2e1a] via-[#050b05] to-[#050b05]">
      <Header userName="User" showIcons={true} />
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            Disease Detection
          </h1>
          <p className="text-gray-400">
            Upload a photo of your crop to identify diseases and get treatment
            recommendations.
          </p>
        </div>
        <div className="h-[600px]">
          <DiseaseDetectionWidget />
        </div>
      </main>
    </div>
  );
}
