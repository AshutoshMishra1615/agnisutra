import { useState } from "react";
import Map from "../components/Map";

export default function AddField() {
  const [showMap, setShowMap] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lng: number } | null>(null);

  const handleAddFieldClick = () => {
    setShowMap(true); // Show the map when the button is clicked
  };

  const handleLocationSelect = (lat: number, lng: number) => {
    setSelectedLocation({ lat, lng });
    setShowMap(false); // Hide the map after selecting a location
  };

  return (
    <div>
      <button
        onClick={handleAddFieldClick}
        className="rounded-[7px] border-[0.56px] flex justify-center items-center bg-[#879d7b] border-white py-4"
      >
        Add Field
      </button>

      {showMap && (
        <div className="mt-4">
          <Map onLocationSelect={handleLocationSelect} />
        </div>
      )}

      {selectedLocation && (
        <div className="mt-4">
          <p>Selected Location:</p>
          <p>Latitude: {selectedLocation.lat}</p>
          <p>Longitude: {selectedLocation.lng}</p>
        </div>
      )}
    </div>
  );
}