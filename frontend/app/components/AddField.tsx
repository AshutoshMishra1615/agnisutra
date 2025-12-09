"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Plus, MapPin, X, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { fieldService } from "../services/fieldService";
import { toast } from "react-hot-toast";

const LocationPicker = dynamic(() => import("../components/LocationPicker"), {
  ssr: false,
  loading: () => (
    <div className="h-[300px] w-full bg-[#1a2e1a] animate-pulse rounded-lg flex items-center justify-center text-gray-500">
      Loading Map...
    </div>
  ),
});

export default function AddField() {
  const [showModal, setShowModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const t = useTranslations("dashboard.actions");

  const [formData, setFormData] = useState({
    name: "",
    crop: "",
    area_acres: "",
    lat: "",
    lon: "",
  });

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
      setShowModal(false);
      setFormData({ name: "", crop: "", area_acres: "", lat: "", lon: "" });
      // Ideally we should refresh the dashboard or field list here
      // For now, just close the modal
    } catch (error) {
      console.error("Failed to create field:", error);
      toast.error("Failed to add field");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="glass-card rounded-xl p-6 flex flex-col items-center justify-center gap-3 group cursor-pointer w-full h-full hover:bg-[#4ade80]/5 transition-all"
      >
        <div className="p-3 rounded-full bg-[#4ade80]/10 group-hover:bg-[#4ade80]/20 transition-colors">
          <Plus className="text-[#4ade80]" size={24} />
        </div>
        <span className="text-gray-300 font-medium group-hover:text-white transition-colors">
          {t("add")}
        </span>
      </button>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#0E1A0E] border border-[#879d7b]/30 rounded-2xl w-full max-w-md p-6 relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowModal(false)}
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
                  <>
                    <Plus size={20} />
                    Add Field
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
