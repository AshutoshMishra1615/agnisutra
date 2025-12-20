import {
  MapContainer,
  TileLayer,
  Marker,
  useMapEvents,
  Popup,
  useMap,
} from "react-leaflet";
import { useState, useEffect } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for default marker icon missing in Leaflet with Next.js
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface MapProps {
  onLocationSelect: (lat: number, lng: number) => void;
}

function LocationController() {
  const map = useMap();

  useEffect(() => {
    map.locate().on("locationfound", function (e) {
      map.flyTo(e.latlng, map.getZoom());
    });
  }, [map]);

  return null;
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
          Latitude: {position[0].toFixed(4)}, Longitude:{" "}
          {position[1].toFixed(4)}
        </Popup>
      </Marker>
    );
  };

  return (
    <MapContainer
      center={[20.5937, 78.9629]}
      zoom={5}
      scrollWheelZoom={true}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <LocationController />
      <LocationMarker />
    </MapContainer>
  );
}
