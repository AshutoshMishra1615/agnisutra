import { WeatherData } from '../types/crop';

interface WeatherWidgetProps {
  weather: WeatherData[];
}

export default function WeatherWidget({ weather }: WeatherWidgetProps) {
  return (
    <div className="">
      <div className="bg-[#364031] inline px-3 py-7 rounded-md">
        <div className='inline-flex gap-2 px-4 py-1 bg-[#d6d9b4] rounded-[5px]'>
          <span className="text-2xl">☀️</span>
          <h3 className="text-black font-semibold">Weather</h3>
        </div>
      </div>
      <div className="bg-[#364031] mb-4 flex gap-4 justify-between p-4 rounded-xl"
       style={{
    overflow: "auto",
    scrollbarWidth: "none", // Firefox
    msOverflowStyle: "none", // IE and Edge
  }}>
        {weather.map((day, index) => (
          <div key={index} className="text-center rounded-lg border-[0.2px] border-black  shadow-[0_2.03_3.29_rgba(0,0,0,0.59)] hover:bg-[#1f2e1e] py-2 px-4">
            <div className="text-3xl mb-1">{day.icon}</div>
            <div className="text-white text-xs font-medium">{day.day}</div>
            <div className="text-gray-400 text-xs">{day.date}</div>
          </div>
        ))}
      </div>
    </div>
  );
}