import axios from "axios";
import { WeatherData } from "../types/crop";

const getWeatherForecast = async (lat: number, lon: number): Promise<WeatherData[]> => {
  try {
    const response = await axios.get("https://api.open-meteo.com/v1/forecast", {
      params: {
        latitude: lat,
        longitude: lon,
        daily: "weather_code",
        timezone: "auto",
        forecast_days: 7,
      },
    });

    // Map API response to WeatherData format
    const dailyData = response.data.daily;
    const weatherData: WeatherData[] = dailyData.time.map((date: string, index: number) => ({
      day: new Date(date).toLocaleDateString("en-US", { weekday: "short" }),
      date: new Date(date).toLocaleDateString("en-US"),
      icon: getWeatherIcon(dailyData.weather_code[index]), // Map weather code to an icon
    }));

    return weatherData;
  } catch (error) {
    console.error("Error fetching weather forecast:", error);
    throw error;
  }
};

// Helper function to map weather codes to icons
const getWeatherIcon = (code: number): string => {
  const weatherIcons: { [key: number]: string } = {
    0: "☀️", // Clear sky
    1: "🌤️", // Partly cloudy
    2: "☁️", // Cloudy
    3: "🌧️", // Rain
    // Add more mappings as needed
  };
  return weatherIcons[code] || "❓"; // Default icon for unknown codes
};

export { getWeatherForecast };