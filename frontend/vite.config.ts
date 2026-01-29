import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite configuration for PDF Character Graph frontend.
 *
 * Environment variables:
 * - VITE_API_PROXY_TARGET: Backend URL for development proxy (default: http://localhost:8000)
 * - VITE_API_BASE_URL: API base URL for production builds (optional)
 */
const proxyTarget = process.env.VITE_API_PROXY_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
      },
      "/health": {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
