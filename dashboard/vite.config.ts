import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: {
    rollupOptions: { output: { manualChunks: { charts: ["recharts"] } } },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
      "/wm-states/dashboard/api": {
        target: "http://127.0.0.1:8000",
        ws: true,
      },
      "/wm-states/docs": { target: "http://127.0.0.1:8000" },
      "/docs": { target: "http://127.0.0.1:8000" },
    },
  },
});
