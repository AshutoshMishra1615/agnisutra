export interface CropData {
  id: string;
  name: string;
  status: 'good' | 'warning' | 'bad';
  growthPercentage: number;
  soilMoisture: number;
  soilMoistureNeeded: number;
  avgTemperature: number;
  temperatureNeeded: string;
  sownDate: string;
  lastIrrigation: string;
  lastPesticide: string;
  expectedYield: string;
}

export interface WeatherData {
  day: string;
  date: string;
  condition: string;
  icon: string;
}

export interface IrrigationRecord {
  date: string;
  quantity: string;
  evaporation: string;
  rainfall: string;
}

export interface PesticideRecord {
  date: string;
  quantity: string;
  evaporation: string;
  rainfall: string;
}