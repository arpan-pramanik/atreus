/**
 * Unified Process Stopper for Atreus
 * Terminates all running backend (port 8000) and frontend (port 3000) processes cleanly.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = path.resolve(__dirname, "..");
const PID_FILE = path.join(ROOT_DIR, ".atreus.pids");
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 3000;

console.log("\n[Stop] Stopping Atreus streaming platform services...");

function killPort(port) {
  try {
    execSync(`fuser -k ${port}/tcp 2>/dev/null || true`);
    console.log(`[Stop] Freed port ${port}`);
  } catch {}
}

if (fs.existsSync(PID_FILE)) {
  try {
    const pids = JSON.parse(fs.readFileSync(PID_FILE, "utf8"));
    pids.forEach((pid) => {
      try {
        process.kill(pid, "SIGTERM");
      } catch {}
    });
    fs.unlinkSync(PID_FILE);
  } catch {}
}

killPort(BACKEND_PORT);
killPort(FRONTEND_PORT);

console.log("[Stop] Backend (port 8000) and Frontend (port 3000) have been stopped.\n");
