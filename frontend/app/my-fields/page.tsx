"use client";

import { useState, useEffect } from "react";
import Header from "../components/HeaderDashboard";
import { useAuth } from "../hooks/useAuth";
import { MapPin, Sprout, ArrowLeft, Plus, X, Loader2 } from "lucide-react";
import Link from "next/link";
import { fieldService, Field } from "../services/fieldService";
import { toast } from "react-hot-toast";
import dynamic from "next/dynamic";
import FieldAnalysisModal from "../components/FieldAnalysisModal";

// Dynamically import LocationPicker to avoid SSR issues with Leaflet
const LocationPicker = dynamic(() => import("../components/LocationPicker"), {
  ssr: false,
  loading: () => (
    <div className="h-[300px] w-full bg-white/5 rounded-lg flex items-center justify-center text-gray-400">
      <Loader2 className="animate-spin mr-2" /> Loading Map...
    </div>
  ),
});

export default function MyFieldsPage() {
  const { user } = useAuth();
  const userName = user?.name || "Farmer";

  const [fields, setFields] = useState<Field[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [selectedField, setSelectedField] = useState<Field | null>(null);

  const [formData, setFormData] = useState({
    name: "",
    crop: "",
    area_acres: "",
    lat: "",
    lon: "",
  });

  useEffect(() => {
    fetchFields();
  }, []);

  const fetchFields = async () => {
    try {
      const data = await fieldService.getFields();
      setFields(data);
    } catch (error) {
      console.error("Failed to fetch fields:", error);
      toast.error("Failed to load fields");
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      await fieldService.createField({
        name: formData.name,
        crop: formData.crop,
        area_acres: parseFloat(formData.area_acres),
        lat: parseFloat(formData.lat),
        lon: parseFloat(formData.lon),
      });

      toast.success("Field added successfully!");
      setIsModalOpen(false);
      setFormData({ name: "", crop: "", area_acres: "", lat: "", lon: "" });
      fetchFields(); // Refresh list
    } catch (error) {
      console.error("Failed to create field:", error);
      toast.error("Failed to add field");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050b05] text-white">
      <Header userName={userName} showIcons={true} />

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/dashboard"
              className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors"
            >
              <ArrowLeft size={20} />
            </Link>
            <h1 className="text-3xl font-bold bg-linear-to-r from-white to-gray-400 bg-clip-text text-transparent">
              My Fields
            </h1>
          </div>

          <button
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2 bg-[#4ade80] text-[#050b05] rounded-lg font-medium hover:bg-[#4ade80]/90 transition-colors"
          >
            <Plus size={20} />
            Add Field
          </button>
        </div>

        {loading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="animate-spin text-[#4ade80]" size={40} />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {fields.map((field) => (
              <div
                key={field.id}
                className="glass-card p-6 rounded-xl border border-[#879d7b]/20 hover:border-[#4ade80]/50 transition-all group"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
                    <Sprout className="text-[#4ade80]" size={24} />
                  </div>
                  <span className="text-xs font-mono text-gray-500 bg-white/5 px-2 py-1 rounded">
                    ID: {field.id}
                  </span>
                </div>

                <h3 className="text-xl font-bold text-white mb-2">
                  {field.name}
                </h3>

                <div className="space-y-2 text-sm text-gray-400">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
                    <span>
                      Crop: <span className="text-white">{field.crop}</span>
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    <span>
                      Area:{" "}
                      <span className="text-white">
                        {field.area_acres} acres
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <MapPin size={14} />
                    <span className="truncate">
                      {field.lat.toFixed(4)}, {field.lon.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-white/5 flex gap-2">
                  <button
                    onClick={() => setSelectedField(field)}
                    className="flex-1 py-2 rounded-lg bg-[#4ade80]/10 hover:bg-[#4ade80]/20 text-[#4ade80] text-sm font-medium transition-colors"
                  >
                    Analysis
                  </button>
                </div>
              </div>
            ))}

            {/* Add New Field Card (Clickable) */}
            <button
              onClick={() => setIsModalOpen(true)}
              className="glass-card p-6 rounded-xl border border-[#879d7b]/20 border-dashed hover:border-[#4ade80] transition-all flex flex-col items-center justify-center gap-4 group min-h-[250px]"
            >
              <div className="p-4 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80] transition-colors">
                <MapPin
                  className="text-[#4ade80] group-hover:text-[#050b05]"
                  size={32}
                />
              </div>
              <span className="text-gray-400 group-hover:text-white font-medium">
                Add New Field
              </span>
            </button>
          </div>
        )}
      </main>

      {/* Add Field Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#0a120a] border border-[#879d7b]/30 rounded-2xl w-full max-w-md p-6 relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-white"
            >
              <X size={24} />
            </button>

            <h2 className="text-2xl font-bold text-white mb-6">
              Add New Field
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Field Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-[#4ade80]"
                  placeholder="e.g. North Plot"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Crop Type
                </label>
                <select
                  name="crop"
                  value={formData.crop}
                  onChange={handleInputChange}
                  required
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-[#4ade80]"
                >
                  <option value="" className="bg-[#0a120a]">
                    Select Crop
                  </option>
                  <option value="Soybean" className="bg-[#0a120a]">
                    Soybean
                  </option>
                  <option value="Wheat" className="bg-[#0a120a]">
                    Wheat
                  </option>
                  <option value="Rice" className="bg-[#0a120a]">
                    Rice
                  </option>
                  <option value="Corn" className="bg-[#0a120a]">
                    Corn
                  </option>
                  <option value="Cotton" className="bg-[#0a120a]">
                    Cotton
                  </option>
                  <option value="Sugarcane" className="bg-[#0a120a]">
                    Sugarcane
                  </option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Expected Yield (t/ha)
                </label>
                <input
                  type="number"
                  name="area_acres"
                  value={formData.area_acres}
                  onChange={handleInputChange}
                  required
                  step="0.1"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-[#4ade80]"
                  placeholder="e.g. 2.5"
                />
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-400">
                  Field Location (Click on map)
                </label>
                <LocationPicker
                  onLocationSelect={(lat, lon) => {
                    setFormData((prev) => ({
                      ...prev,
                      lat: lat.toString(),
                      lon: lon.toString(),
                    }));
                  }}
                />
                {formData.lat && formData.lon && (
                  <p className="text-xs text-[#4ade80]">
                    Selected: {parseFloat(formData.lat).toFixed(4)},{" "}
                    {parseFloat(formData.lon).toFixed(4)}
                  </p>
                )}
                <input type="hidden" name="lat" value={formData.lat} required />
                <input type="hidden" name="lon" value={formData.lon} required />
              </div>

              <button
                type="submit"
                disabled={submitting || !formData.lat || !formData.lon}
                className="w-full py-3 mt-4 bg-[#4ade80] text-[#050b05] rounded-lg font-bold hover:bg-[#4ade80]/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Adding...
                  </>
                ) : (
                  "Add Field"
                )}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Analysis Modal */}
      {selectedField && (
        <FieldAnalysisModal
          fieldId={selectedField.id}
          fieldName={selectedField.name}
          onClose={() => setSelectedField(null)}
        />
      )}
    </div>
  );
}
