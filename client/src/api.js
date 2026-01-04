import axios from "axios";

const BASE_URL =
  process.env.REACT_APP_BACKEND_API || "http://localhost:4000";

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
