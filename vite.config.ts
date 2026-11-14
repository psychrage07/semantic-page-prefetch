import path from "path";
import { fileURLToPath } from "url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const apiPort = process.env.PAGEWISE_API_PORT || "8765";
const apiTarget = `http://127.0.0.1:${apiPort}`;

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  preview: { proxy: { "/api": apiTarget } },
  server: {
    proxy: { "/api": apiTarget },
    watch: { ignored: ["**/backend/runs/**", "**/backend/.venv/**"] },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
