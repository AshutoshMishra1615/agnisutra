"use client";

import { useEffect, useState } from "react";
import { getWeatherForecast } from "../services/WeatherService";
import WeatherWidget from "./WeatherWidget";
import { WeatherData } from "../types/crop";

export default function WeatherContainer() {
  const [weather, setWeather] = useState<WeatherData[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const lat = 37.7749; // Example latitude
        const lon = -122.4194; // Example longitude
        const weatherData = await getWeatherForecast(lat, lon);
        setWeather(weatherData);
      } catch (err) {
        setError("Failed to fetch weather data.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchWeather();
  }, []);

  if (loading) return <div>Loading weather data...</div>;
  if (error) return <div>{error}</div>;

  return weather ? <WeatherWidget weather={weather} /> : null;
}