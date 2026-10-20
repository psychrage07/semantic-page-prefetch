// Browser smoke check using Chrome's DevTools protocol; no npm dependency.
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const out = resolve(root, `backend/.test-temp/neural-browser-${Date.now()}`);
await mkdir(out, { recursive: true });
await mkdir(resolve(out, "downloads"), { recursive: true });
const wait = ms => new Promise(done => setTimeout(done, ms));
const server = spawn(process.execPath, [resolve(root, "node_modules/vite/bin/vite.js"), "--host", "127.0.0.1", "--port", "5181", "--strictPort"], { cwd: root, windowsHide: true, stdio: "ignore" });
const chrome = spawn("C:/Program Files/Google/Chrome/Application/chrome.exe", ["--headless=new", "--remote-debugging-port=9331", `--user-data-dir=${resolve(out, "profile")}`, "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-gpu", "--disable-background-networking", "about:blank"], { windowsHide: true, stdio: "ignore" });
let ws;
try {
  for (let i = 0; i < 50; i++) {
    try { if ((await fetch("http://127.0.0.1:5181")).ok && (await fetch("http://127.0.0.1:9331/json/version")).ok) break; } catch {}
    await wait(200);
  }
  const target = await (await fetch("http://127.0.0.1:9331/json/new?about:blank", { method: "PUT" })).json();
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
  let next = 0;
  const pending = new Map();
  const errors = [];
  const downloads = [];
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails);
    if (message.method?.startsWith("Browser.download")) downloads.push(message);
    if (message.id && pending.has(message.id)) {
      const { ok, bad, timer } = pending.get(message.id);
      clearTimeout(timer); pending.delete(message.id);
      message.error ? bad(new Error(JSON.stringify(message.error))) : ok(message.result);
    }
  };
  const send = (method, params = {}) => new Promise((ok, bad) => {
    const id = ++next;
    const timer = setTimeout(() => { pending.delete(id); bad(new Error(`Timeout: ${method}`)); }, 15000);
    pending.set(id, { ok, bad, timer }); ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true, userGesture: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: "http://127.0.0.1:5181/?view=neural" });
  await wait(500);
  await evaluate(`new Promise((ok,bad)=>{let n=0;const t=setInterval(()=>{if(document.body.innerText.includes('47.74')){clearInterval(t);ok(true)}else if(++n>50){clearInterval(t);bad(new Error(document.body.innerText))}},100)})`);
  assert.equal(await evaluate("document.querySelectorAll('.neural-study tbody tr').length"), 29);
  assert.equal(await evaluate("document.querySelector('.neural-study').innerText.includes('21.63%')"), true);
  const desktop = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(resolve(out, "desktop.png"), Buffer.from(desktop.data, "base64"));
  await evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='Unseen templates').click()");
  await wait(200);
  assert.equal(await evaluate("document.querySelector('.neural-cards').innerText.includes('2.90')"), true);
  await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: resolve(out, "downloads"), eventsEnabled: true });
  const button = await evaluate("(()=>{const r=[...document.querySelectorAll('button')].find(b=>b.textContent.includes('Export study JSON')).getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
  await send("Input.dispatchMouseEvent", { type: "mousePressed", ...button, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", ...button, button: "left", clickCount: 1 });
  for (let i = 0; i < 100; i++) {
    try { if ((await readdir(resolve(out, "downloads"))).includes("pagewise-neural-improvement.json")) break; } catch {}
    await wait(100);
  }
  await writeFile(resolve(out, "download-events.json"), JSON.stringify(downloads, null, 2));
  const exported = JSON.parse(await readFile(resolve(out, "downloads/pagewise-neural-improvement.json"), "utf8"));
  assert.equal(exported.schema_version, 1);
  assert.equal(exported.seeds.length, 5);
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await wait(200);
  assert.equal(await evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), true);
  const mobile = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(resolve(out, "mobile.png"), Buffer.from(mobile.data, "base64"));
  await evaluate("[...document.querySelectorAll('.nav-link')].find(b=>b.textContent.includes('Overview')).click()");
  await wait(200);
  assert.equal(await evaluate("document.body.innerText.includes('Can semantics predict the next page?')"), true);
  assert.equal(errors.length, 0);
  console.log(JSON.stringify({ passed: true, checks: ["fresh measured values", "29 result rows", "split switching", "JSON download", "mobile overflow", "existing overview navigation", "no browser exceptions"], screenshots: out }));
} finally {
  if (ws) ws.close();
  chrome.kill();
  server.kill();
}
