"use client";

import { MapContainer, TileLayer, FeatureGroup } from "react-leaflet";
import {EditControl} from 'react-leaflet-draw'
import L, { LeafletEvent, LatLng } from "leaflet";
import { useState } from "react";

// Declare module for react-leaflet-draw to fix missing types
// declare module "react-leaflet-draw" {
//   export const EditControl: any;
// }

interface MapProps {
  onLocationSelect?: (lat: number, lng: number) => void; // Callback for single point selection
  onFieldSelect?: (coords: LatLng[]) => void; // Callback for polygon/field selection
}

export default function Map({ onLocationSelect, onFieldSelect }: MapProps) {
  const [coords, setCoords] = useState<LatLng[]>([]);

  const handleCreated = (e: LeafletEvent) => {
    const layer = (e as any).layer;

    if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
      const latlngs = layer.getLatLngs();

      // Type narrowing to ensure latlngs is LatLng[]
      if (Array.isArray(latlngs) && Array.isArray(latlngs[0])) {
        const vertices: LatLng[] = latlngs[0] as LatLng[];
        console.log("BOUNDARY COORDINATES:", vertices);
        setCoords(vertices);

        if (onFieldSelect) {
          onFieldSelect(vertices); // Pass coordinates to parent component
        }
      }

      // Handle single point selection if onLocationSelect is provided
      if (onLocationSelect && layer instanceof L.Marker) {
        const { lat, lng } = layer.getLatLng();
        onLocationSelect(lat, lng);
      }
    }
  };

  return (
    <div className="w-full h-[600px]">
      <MapContainer
        center={[20.5937, 78.9629]} // India center
        zoom={6}
        className="w-full h-full rounded-md"
        style={{ height: "100%", width: "100%" }} // Ensure proper dimensions
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        <FeatureGroup>
          <EditControl
            position="topright"
            draw={{
              rectangle: true,
              polygon: true,
              circle: false,
              polyline: false,
              marker: false,
              circlemarker: false,
            }}
            edit={{
              edit: true,
              remove: true,
            }}
            onCreated={handleCreated}
          />
        </FeatureGroup>
      </MapContainer>

      {/* Display coordinates */}
      <div className="mt-4 p-4 bg-gray-900 text-white rounded-md">
        <h2 className="font-semibold">Selected Field Coordinates</h2>
        {coords.length === 0 && <p>No field selected.</p>}
        {coords.map((pt, index) => (
          <p key={index}>
            {index + 1}. Lat: {pt.lat.toFixed(6)}, Lng: {pt.lng.toFixed(6)}
          </p>
        ))}
      </div>
    </div>
  );
}