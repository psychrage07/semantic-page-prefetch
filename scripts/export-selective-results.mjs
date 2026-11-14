import { copyFile, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "backend/selective-prefetch-results.json");
const study = JSON.parse(await readFile(source, "utf8"));
for (const scenario of ["zero", "controlled-delay"]) {
  const policies = study.scenarios?.[scenario]?.runtime?.policies;
  if (!policies || !["none", "model_ranker", "combined"].every(name =>
    Number.isFinite(policies[name]?.mean_ms) && policies[name]?.queries > 0)) {
    throw new Error(`Incomplete measured selective results: ${scenario}`);
  }
  if (!["vs_none", "vs_model_ranker"].every(name =>
    ["mean", "ci95_low", "ci95_high"].every(field => Number.isFinite(policies.combined[name]?.[field])))) {
    throw new Error(`Missing paired interval: ${scenario}`);
  }
}
await copyFile(source, path.join(root, "public/pagewise-selective-prefetch.json"));
await copyFile(path.join(root, "backend/SELECTIVE_PREFETCH_RESULTS.md"), path.join(root, "public/selective-prefetch-results.md"));
console.log("Exported measured selective-policy results for the dashboard.");
