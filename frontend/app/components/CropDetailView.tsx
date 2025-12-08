"use client";

import { CropData } from "../types/crop";
import GrowthChart from "./GrowthChart";
import { useTranslations } from 'next-intl';
import Image from "next/image";

interface CropDetailViewProps {
  crop: CropData;
}

export default function CropDetailView({ crop }: CropDetailViewProps) {
  const statusColor = {
    good: "bg-green-500",
    warning: "bg-yellow-500",
    bad: "bg-red-500",
  };

 const t = useTranslations();

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Column - Crop Info */}
      <div className="lg:col-span-4">
        <div className="bg-[#495643] rounded-xl py-2 px-3 relative">
           <div className="flex flex-col mb-4">
        <div className="flex items-center justify-between bg-[#1f2e1e] p-3 rounded-md mb-4">
          <h3 className="text-white text-xl font-semibold">{crop.name}</h3>
          <div className={`w-5 h-5 rounded-full ${statusColor[crop.status]}`} />
        </div>
        <div className='flex justify-end'>
        </div>
      </div>

          <div className="mb-6">
            <div>
              <div className="flex justify-center items-center">
                <h4 className="text-white font-bold  text-lg mb-3">
                 {t('dashboard.cropcard.growth')}
                </h4>
              </div>
            </div>
            <div className="flex justify-center">
              <GrowthChart percentage={crop.growthPercentage} />
            </div>
          </div>

          <div className="mb-3">
            <div className="flex items-baseline space-x-2">
              <span className="text-white text-lg">
                {t('dashboard.cropcard.soil')} : {crop.soilMoisture}%
              </span>
            </div>
            <span className="text-white text-sm">
              {t('dashboard.cropcard.needed')} : {crop.soilMoistureNeeded}%
            </span>
          </div>

          <div className="mb-6">
            <div className="flex items-baseline space-x-2">
              <span className="text-white text-lg">
                {t('dashboard.cropcard.temp')} : {crop.avgTemperature}°C
              </span>
            </div>
            <span className="text-white text-sm">
              {t('dashboard.cropcard.needed')} : {crop.temperatureNeeded}
            </span>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-white">{t('dashboard.cropcard.dates.sown')}</span>
              <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px] text-sm font-medium">
                {crop.sownDate}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white">{t('dashboard.cropcard.dates.irrigate')}</span>
              <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px] text-sm font-medium">
                {crop.lastIrrigation}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white">{t('dashboard.cropcard.dates.pest')}</span>
              <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px]text-sm font-medium">
                {crop.lastPesticide}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-white">{t('dashboard.cropcard.dates.expect')}</span>
              <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px] text-sm font-medium">
                {crop.expectedYield}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column - Health Analysis */}
      <div className="lg:col-span-8 space-y-6 bg-[#495643]">
        {/* Health Images */}
        <div className="grid grid-cols-1 px-4 py-4 md:grid-cols-2 gap-6">
          <div className="flex flex-col gap-4">
            <div>
              <h3 className="inline bg-[#1f2e1e] text-white px-4 py-2 rounded-lg font-medium">
                {t('dashboard.details.health.crop')}
              </h3>
            </div>
            <div className="bg-gray-300 rounded-b-xl h-48"></div>
          </div>
          <div className="flex flex-col gap-4">
            <div>
              <h3 className="inline bg-[#1f2e1e] text-white px-4 py-2 rounded-lg font-medium">
                {t('dashboard.details.health.soil')}
              </h3>
            </div>

            <div className="bg-gray-300 rounded-b-xl h-48"></div>
          </div>
          <div className="flex flex-col gap-4">
            <div>
              <h3 className="inline bg-[#1f2e1e] text-white px-4 py-2 rounded-lg font-medium">
                {t('dashboard.details.health.irrigate')}
              </h3>
            </div>

            <div className="bg-gray-300 rounded-b-xl h-48"></div>
          </div>
          <div className="bg-teal-700 rounded-xl p-4 relative">
            <h3 className="text-white font-semibold mb-2">{t('dashboard.details.health.analyse')}</h3>
            <p className="text-white text-xs mb-4">
              NDVI (For Rice - use for early stage growth)
            </p>
            <div className="flex justify-around items-start mb-3">
              <div className="text-center">
                <div className="w-12 h-12 bg-green-400 rounded-full flex items-center justify-center mb-1">
                  ✓
                </div>
                <span className="text-white text-xs">{t('dashboard.details.rating.good')}</span>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-yellow-400 rounded-full flex items-center justify-center mb-1">
                  ⚠
                </div>
                <span className="text-white text-xs">{t('dashboard.details.rating.medium')}</span>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center mb-1">
                  ✗
                </div>
                <span className="text-white text-xs">{t('dashboard.details.rating.bad')}</span>
              </div>
            </div>
            <div className="h-12 rounded-lg overflow-hidden flex">
              <div className="flex-1 bg-linear-to-r from-green-600 via-yellow-500 to-red-600"></div>
            </div>
            <div className="flex justify-between text-white text-xs mt-1">
              <span>-1.5</span>
              <span>-0.2</span>
              <span>0.2</span>
              <span>0.6-0.9</span>
              <span>+1</span>
            </div>
            <p className="text-white text-xs mt-3">
              Use when vegetable is for Small Plants
            </p>
          </div>
        </div>

        {/* Irrigation Health & Analysis Scale */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4"></div>

        {/* Irrigation Management */}
        <div className="rounded-xl p-4">
          <div className=" flex items-center justify-between mb-4 gap-4">
            <div className="flex-1 bg-[#1f2e1e] px-2 py-2 rounded-md">
              <h3 className="text-white font-semibold">{t('dashboard.details.management.irrigate')}</h3>
            </div>
            
            <button className="border border-white text-white px-4 py-2 rounded-md text-sm bg-[#1f2e1e] hover:bg-white hover:text-gray-900 transition-colors">
              {t('dashboard.details.management.add')} +
            </button>
          </div>
          <div className="grid grid-cols-4 gap-4 text-white font-bold text-sm">
            <div>{t('dashboard.details.management.params.date')}</div>
            <div>{t('dashboard.details.management.params.quantity')}</div>
            <div>{t('dashboard.details.management.params.evaporation')}</div>
            <div>{t('dashboard.details.management.params.Rainfall')}</div>
          </div>
        </div>

        {/* Pesticide Management */}
        <div className="rounded-xl p-4">
          <div className=" flex items-center justify-between mb-4 gap-4">
            <div className="flex-1 bg-[#1f2e1e] px-2 py-2 rounded-md">
              <h3 className="text-white font-semibold">{t('dashboard.details.management.pest')}</h3>
            </div>
            <button className="border border-white text-white px-4 py-2 rounded-md text-sm bg-[#1f2e1e] hover:bg-white hover:text-gray-900 transition-colors">
              {t('dashboard.details.management.add')} +
            </button>
          </div>
          <div className="grid grid-cols-4 gap-4 text-white font-bold text-sm">
            <div>{t('dashboard.details.management.params.date')}</div>
            <div>{t('dashboard.details.management.params.quantity')}</div>
            <div>{t('dashboard.details.management.params.evaporation')}</div>
            <div>{t('dashboard.details.management.params.Rainfall')}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
