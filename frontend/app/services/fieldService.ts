import api from "./api";

export interface Field {
  id: number;
  name: string;
  crop: string;
  area_acres: number;
  lat: number;
  lon: number;
  created_at: string;
}

export interface FieldCreateData {
  name: string;
  crop: string;
  area_acres: number;
  lat: number;
  lon: number;
}

export interface FieldAnalysis {
  field_id: number;
  ndvi: any;
  weather: any;
  yield_forecast: any;
}

export const fieldService = {
  getFields: async (): Promise<Field[]> => {
    const response = await api.get<Field[]>("/krishi-saathi/fields");
    return response.data;
  },

  createField: async (
    data: FieldCreateData
  ): Promise<{ field: Field; analysis: any }> => {
    const response = await api.post("/krishi-saathi/fields", data);
    return response.data;
  },

  getFieldAnalysis: async (id: number): Promise<FieldAnalysis> => {
    const response = await api.get<FieldAnalysis>(
      `/krishi-saathi/fields/${id}/analysis`
    );
    return response.data;
  },
};
