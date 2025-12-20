import axios from "axios";

// Create axios instance
const api = axios.create({
  baseURL: "/api", // Proxied by Next.js to backend
  headers: {
    "Content-Type": "application/json",
  },
});

// Add a request interceptor to add the auth token to every request
api.interceptors.request.use(
  (config) => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("token");
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle 401 errors (unauthorized)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear token and redirect to login if unauthorized
      if (typeof window !== "undefined") {
        localStorage.removeItem("token");
        // Optional: Redirect to login page
        // window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
