
import { CropData, WeatherData } from "../types/crop";

export const mockCrops: CropData[] = [
  {
    id: '1',
    name: 'Rice',
    status: 'good',
    growthPercentage: 72,
    soilMoisture: 13,
    soilMoistureNeeded: 13,
    avgTemperature: 28,
    temperatureNeeded: '30°C',
    sownDate: '24 Nov',
    lastIrrigation: '28 Nov',
    lastPesticide: '29 Nov',
    expectedYield: '25 Dec'
  },
  {
    id: '2',
    name: 'Wheat',
    status: 'bad',
    growthPercentage: 60,
    soilMoisture: 3,
    soilMoistureNeeded: 14,
    avgTemperature: 28,
    temperatureNeeded: '15-20°C',
    sownDate: '24 Nov',
    lastIrrigation: '28 Nov',
    lastPesticide: '29 Nov',
    expectedYield: '25 Dec'
  },
  {
    id: '3',
    name: 'Onion',
    status: 'warning',
    growthPercentage: 45,
    soilMoisture: 3,
    soilMoistureNeeded: 14,
    avgTemperature: 28,
    temperatureNeeded: '15-20°C',
    sownDate: '24 Nov',
    lastIrrigation: '28 Nov',
    lastPesticide: '29 Nov',
    expectedYield: '25 Dec'
  }
];

export const mockWeather: WeatherData[] = [
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' },
  { day: 'SUN', date: 'Dec 7', condition: 'cloudy', icon: '⛅' }
];