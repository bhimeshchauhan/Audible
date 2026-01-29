import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Optional dev proxy:
 * - Set VITE_API_PROXY_TARGET=http://localhost:8000 (or wherever your backend runs)
 * - Then requests to /api/* from the dev server will be proxied to that target.
 */
const proxyTarget = process.env.VITE_API_PROXY_TARGET;

export default defineConfig({
  plugins: [react()],
  server: proxyTarget
    ? {
        proxy: {
          "/api": {
            target: proxyTarget,
            changeOrigin: true
          }
        }
      }
    : undefined
});

