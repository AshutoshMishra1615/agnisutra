'use client';

import { useState } from 'react';
import Header from '../components/HeaderDashboard';
import CropCard from '../components/CropCard';
import CropDetailView from '../components/CropDetailView';
import WeatherContainer from '../components/WeatherContainer';
import { mockCrops, mockWeather } from '../lib/mockData';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import AddField from '../components/AddField';

export default function DashboardPage() {
  const [expandedCropId, setExpandedCropId] = useState<string | null>(null);

  const expandedCrop = mockCrops.find(crop => crop.id === expandedCropId);

  const t = useTranslations('dashboard');

  return (
    <div className="min-h-screen bg-[#0E1A0E]">
      <Header userName="Priyanshu" showIcons={true} />

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-6">
        {/* Welcome Section */}
        <div className="hidden  lg:flex items-center justify-between">
          <div>
            <h1 className="tex(t-white text-3xl md:text-4xl font-bold mb-2">{t('title')}</h1>
            <h2 className="text-white text-3xl md:text-4xl font-bold">Priyanshu !</h2>
          </div>
         <Link
            href="/soil-reports"
            className="rounded-[7px] border-[0.56px] flex justify-center items-center bg-[#879d7b] border-white py-2 px-4"
          >
            <span>📋</span>
            <span>{t('actions.ai')}</span>
          </Link>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Link
            href="/my-fields"
            className="rounded-[7px] border-[0.56px] flex justify-center items-center bg-[#879d7b] border-white py-4"
          >
            <span>🌾</span>
            <span>{t('actions.my')}</span>
          </Link>
          <AddField/>
          <Link
            href="/soil-reports"
            className="rounded-[7px] border-[0.56px] flex justify-center items-center bg-[#879d7b] border-white py-4"
          >
            <span>📋</span>
            <span>{t('actions.soil')}</span>
          </Link>
          <Link
            href="/soil-reports"
            className="rounded-[7px] border-[0.56px] flex justify-center items-center bg-[#879d7b] border-white py-4 lg:hidden"
          >
            <span>📋</span>
            <span>{t('actions.ai')}</span>
          </Link>
        </div>

        {/* Weather Widget */}
        <WeatherContainer/>

        {/* Crop Health Section */}
        <div className="bg-[#d6d9b4] rounded-md px-1.5 py-0.5 shadow-[0_0_3.54_rgba(0,0,0,0.5)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🌱</span>
              <h2 className="text-gray-900 text-xl font-bold">{t('crop')}</h2>
            </div>
            <button className="bg-gray-800 hover:bg-gray-700 text-white px-6 py-2 rounded-md font-medium transition-colors">
              {t('actions.ai')}
            </button>
          </div>
        </div>

        {/* Crop Cards or Detail View */}
        {expandedCrop ? (
          <div>
            <button
              onClick={() => setExpandedCropId(null)}
              className="mb-4 text-white hover:text-[#495643] transition-colors"
            >
              ← {t('back')}
            </button>
            <CropDetailView crop={expandedCrop} />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {mockCrops.map((crop) => (
              <CropCard
                key={crop.id}
                crop={crop}
                isExpanded={false}
                onToggle={() => setExpandedCropId(crop.id)}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}