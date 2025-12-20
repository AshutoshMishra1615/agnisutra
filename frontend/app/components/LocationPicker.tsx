"use client";

import { useState } from "react";
import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for default marker icon
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface LocationPickerProps {
  onLocationSelect: (lat: number, lon: number) => void;
  initialLat?: number;
  initialLon?: number;
}

function LocationMarker({
  onLocationSelect,
  position,
}: {
  onLocationSelect: (lat: number, lon: number) => void;
  position: [number, number] | null;
}) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng.lat, e.latlng.lng);
    },
  });

  return position === null ? null : <Marker position={position}></Marker>;
}

export default function LocationPicker({
  onLocationSelect,
  initialLat,
  initialLon,
}: LocationPickerProps) {
  const [position, setPosition] = useState<[number, number] | null>(
    initialLat && initialLon ? [initialLat, initialLon] : null
  );

  // Default center (India)
  const defaultCenter: [number, number] = [20.5937, 78.9629];

  const handleLocationSelect = (lat: number, lon: number) => {
    setPosition([lat, lon]);
    onLocationSelect(lat, lon);
  };

  return (
    <div className="h-[300px] w-full rounded-lg overflow-hidden border border-white/10 relative z-0">
      <MapContainer
        center={position || defaultCenter}
        zoom={5}
        scrollWheelZoom={true}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <LocationMarker
          onLocationSelect={handleLocationSelect}
          position={position}
        />
      </MapContainer>

      {position && (
        <div className="absolute bottom-2 left-2 bg-black/80 text-white text-xs p-2 rounded z-1000">
          Selected: {position[0].toFixed(4)}, {position[1].toFixed(4)}
        </div>
      )}
    </div>
  );
}
