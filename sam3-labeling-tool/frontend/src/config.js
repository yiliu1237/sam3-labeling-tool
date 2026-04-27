const viteApiUrl =
  typeof import.meta !== 'undefined' &&
  import.meta &&
  import.meta.env &&
  import.meta.env.VITE_API_URL
    ? import.meta.env.VITE_API_URL
    : undefined;

const windowApiUrl =
  typeof window !== 'undefined' && window.__APP_API_URL__
    ? window.__APP_API_URL__
    : undefined;

export const API_BASE_URL = viteApiUrl || windowApiUrl || 'http://127.0.0.1:8000';
