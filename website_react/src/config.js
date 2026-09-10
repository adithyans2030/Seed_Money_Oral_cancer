// Backend API base URL. Set VITE_API_BASE_URL in a .env file (or the
// deployment environment, e.g. Netlify) to point at the deployed backend —
// it must not stay hardcoded to localhost outside local development.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
