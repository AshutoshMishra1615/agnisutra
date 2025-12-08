

import { CropData } from '../types/crop';
import GrowthChart from './GrowthChart';
import { useTranslations } from 'next-intl';
interface CropCardProps {
  crop: CropData;
  isExpanded: boolean;
  onToggle: () => void;
}

export default function CropCard({ crop, isExpanded, onToggle }: CropCardProps) {
  const statusColor = {
    good: 'bg-green-500',
    warning: 'bg-yellow-500',
    bad: 'bg-red-500'
  };

  const t = useTranslations();

  return (
    <div className="bg-[#495643] rounded-xl py-2 px-3 relative">
      {/* Header */}
      <div className="flex flex-col mb-4">
        <div className="flex items-center justify-between bg-[#1f2e1e] p-3 rounded-md mb-4">
          <h3 className="text-white text-xl font-semibold">{crop.name}</h3>
          <div className={`w-5 h-5 rounded-full ${statusColor[crop.status]}`} />
        </div>
        <div className='flex justify-end'>
          <button
          onClick={onToggle}
          className="bg-white text-gray-900 px-4 py-1.5 rounded-sm text-sm font-medium hover:bg-gray-100 transition-colors"
        >
          Details
        </button>
        </div>
      </div>

      {/* Growth Chart */}
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

      {/* Soil Moisture */}
      <div className="mb-3">
        <div className="flex items-baseline space-x-2">
          <span className="text-white text-lg">{t('dashboard.cropcard.soil')} : {crop.soilMoisture}%</span>
        </div>
        <span className="text-white text-sm">{t('dashboard.cropcard.needed')} : {crop.soilMoistureNeeded}%</span>
      </div>

      {/* Temperature */}
      <div className="mb-6">
        <div className="flex items-baseline space-x-2">
          <span className="text-white text-lg">{t('dashboard.cropcard.temp')} : {crop.avgTemperature}°C</span>
        </div>
        <span className="text-white text-sm">{t('dashboard.cropcard.needed')} : {crop.temperatureNeeded}</span>
      </div>

      {/* Date Information */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-white">{t('dashboard.cropcard.dates.sown')}</span>
          <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px]text-sm font-medium">
            {crop.sownDate}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-white">{t('dashboard.cropcard.dates.irrigate')}</span>
          <span className="bg-white text-gray-900 px-4 py-1 rounded-[3px]text-sm font-medium">
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
  );
}
