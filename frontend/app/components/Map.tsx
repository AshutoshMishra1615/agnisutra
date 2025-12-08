import { MapContainer, TileLayer, Marker, useMapEvents, Popup } from "react-leaflet";
import { useState } from "react";

interface MapProps {
  onLocationSelect: (lat: number, lng: number) => void;
}

export default function Map({ onLocationSelect }: MapProps) {
  const [position, setPosition] = useState<[number, number] | null>(null);

  const LocationMarker = () => {
    useMapEvents({
      click(e) {
        const { lat, lng } = e.latlng;
        setPosition([lat, lng]);
        onLocationSelect(lat, lng); // Pass the selected coordinates to the parent
      },
    });

    return position === null ? null : (
      <Marker position={position}>
        <Popup>
          Selected Location: <br />
          Latitude: {position[0]}, Longitude: {position[1]}
        </Popup>
      </Marker>
    );
  };

  return (
    <MapContainer center={[51.505, -0.09]} zoom={13} scrollWheelZoom={false} style={{ height: "500px", width: "100%" }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <LocationMarker />
    </MapContainer>
  );
}
