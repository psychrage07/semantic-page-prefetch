import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backend = path.join(root, "backend");
const bundledPython = process.platform === "win32"
  ? path.join(backend, ".venv", "Scripts", "python.exe")
  : path.join(backend, ".venv", "bin", "python");
const python = process.env.PAGEWISE_PYTHON || (existsSync(bundledPython) ? bundledPython : "python");
const vite = path.join(root, "node_modules", "vite", "bin", "vite.js");

const api = spawn(python, [path.join(backend, "local_lab.py")], { cwd: root, stdio: "inherit" });
const viteArgs = process.argv.includes("--preview") ? [vite, "preview"] : [vite];
const frontend = spawn(process.execPath, viteArgs, { cwd: root, stdio: "inherit" });
let stopping = false;

function stop() {
  if (stopping) return;
  stopping = true;
  frontend.kill();
  api.kill();
}

for (const child of [api, frontend]) {
  child.on("error", error => { console.error(error); stop(); process.exitCode = 1; });
  child.on("exit", code => { if (!stopping) { stop(); process.exitCode = code || 1; } });
}
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
