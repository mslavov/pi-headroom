/**
 * Headroom proxy lifecycle manager.
 *
 * Handles: Python detection, venv creation, pip install, background proxy spawn,
 * health polling, graceful shutdown, and crash recovery.
 *
 * Uses a dedicated venv (~/.pi/headroom-venv/) to avoid PEP 668 issues on
 * macOS/Homebrew and to keep the system Python clean.
 */

import { execFile, spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const IS_WINDOWS = process.platform === "win32";
const VENV_DIR = join(homedir(), ".pi", "headroom-venv");
const VENV_BIN = IS_WINDOWS ? join(VENV_DIR, "Scripts") : join(VENV_DIR, "bin");
const VENV_PYTHON = join(VENV_BIN, IS_WINDOWS ? "python.exe" : "python");
const VENV_HEADROOM = join(VENV_BIN, IS_WINDOWS ? "headroom.exe" : "headroom");

// ─── Python detection ─────────────────────────────────────────────────

/**
 * Find a Python >=3.10 interpreter. Tries python3 then python.
 * Returns the command string or null if not found.
 */
export async function findPython(): Promise<string | null> {
  for (const cmd of ["python3", "python"]) {
    const version = await getPythonVersion(cmd);
    if (version && version.major >= 3 && version.minor >= 10) {
      return cmd;
    }
  }
  return null;
}

async function getPythonVersion(
  cmd: string,
): Promise<{ major: number; minor: number } | null> {
  try {
    const output = await execAsync(cmd, ["--version"]);
    // "Python 3.12.4"
    const match = output.match(/Python (\d+)\.(\d+)/);
    if (match) {
      return { major: parseInt(match[1], 10), minor: parseInt(match[2], 10) };
    }
  } catch {
    // Command not found or errored
  }
  return null;
}

// ─── Venv + install ───────────────────────────────────────────────────

/**
 * Ensure the headroom venv exists and headroom-ai[proxy] is installed.
 * Returns the path to the headroom CLI in the venv, or null on failure.
 */
async function ensureVenv(
  onStatus: (msg: string) => void,
): Promise<string | null> {
  // 1. If venv already has headroom, we're done
  if (existsSync(VENV_HEADROOM)) {
    return VENV_HEADROOM;
  }

  // 2. Find system Python
  const python = await findPython();
  if (!python) {
    onStatus("Python >=3.10 not found — cannot install Headroom");
    return null;
  }

  // 3. Create venv if it doesn't exist
  if (!existsSync(VENV_PYTHON)) {
    onStatus("Creating Headroom venv...");
    try {
      await execAsync(python, ["-m", "venv", VENV_DIR], 60_000);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onStatus(`Failed to create venv: ${msg}`);
      return null;
    }
  }

  // 4. Install headroom-ai[proxy] into the venv
  onStatus("Installing headroom-ai (this may take a minute)...");
  try {
    await execAsync(
      VENV_PYTHON,
      ["-m", "pip", "install", "headroom-ai[proxy]", "--quiet", "--disable-pip-version-check"],
      180_000, // 3 minute timeout — first install downloads many deps
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    onStatus(`pip install failed: ${msg}`);
    return null;
  }

  // 5. Verify
  if (existsSync(VENV_HEADROOM)) {
    return VENV_HEADROOM;
  }

  // Fallback: try via python -m
  try {
    await execAsync(VENV_PYTHON, ["-m", "headroom.cli", "--help"], 10_000);
    return null; // CLI binary doesn't exist, but module works — handled separately
  } catch {
    // pass
  }

  onStatus("headroom installed but CLI not found in venv");
  return null;
}

/**
 * Ensure headroom is available. Checks system PATH first, then venv.
 * Returns the invocation method: { cmd, args } to spawn the proxy,
 * or null if installation failed.
 */
export async function ensureInstalled(
  onStatus: (msg: string) => void,
): Promise<{ cmd: string; args: string[] } | null> {
  // 1. Check if `headroom` CLI is already on system PATH
  if (await isCommandAvailable("headroom", ["--help"])) {
    return { cmd: "headroom", args: [] };
  }

  // 2. Check if venv already has headroom
  if (existsSync(VENV_HEADROOM) && await isCommandAvailable(VENV_HEADROOM, ["--help"])) {
    return { cmd: VENV_HEADROOM, args: [] };
  }

  // 3. Create venv and install
  const headroomPath = await ensureVenv(onStatus);
  if (headroomPath) {
    return { cmd: headroomPath, args: [] };
  }

  // 4. Fallback: try module invocation in venv
  if (existsSync(VENV_PYTHON) && await isCommandAvailable(VENV_PYTHON, ["-m", "headroom.cli", "--help"])) {
    return { cmd: VENV_PYTHON, args: ["-m", "headroom.cli"] };
  }

  return null;
}

async function isCommandAvailable(cmd: string, args: string[]): Promise<boolean> {
  try {
    await execAsync(cmd, args, 10_000);
    return true;
  } catch {
    return false;
  }
}

// ─── ProxyManager ─────────────────────────────────────────────────────

export class ProxyManager {
  private proc: ChildProcess | null = null;
  private weStartedIt = false;
  private stopping = false;
  private port: number;
  private host: string;
  /** Stored invocation method from ensureInstalled */
  private invocation: { cmd: string; args: string[] } | null = null;

  constructor(options?: { port?: number; host?: string }) {
    this.port = options?.port ?? 8787;
    this.host = options?.host ?? "127.0.0.1";
  }

  get baseUrl(): string {
    return `http://${this.host}:${this.port}`;
  }

  get isManaged(): boolean {
    return this.weStartedIt;
  }

  // ── Full lifecycle: detect → install → start → health-check ───────

  async ensureRunning(onStatus: (msg: string) => void): Promise<boolean> {
    if (this.stopping) return false;

    // 1. Already running? (external or our own)
    onStatus("Checking for running proxy...");
    if (await this.healthCheck()) {
      return true; // Don't touch it — someone else's proxy or our still-alive one
    }

    // 2. Ensure headroom is installed (venv-based)
    const invocation = await ensureInstalled(onStatus);
    if (!invocation) return false;
    this.invocation = invocation;

    // 3. Spawn proxy
    this.startProxy(onStatus);

    // 4. Poll for health with backoff
    const delays = [500, 1000, 1000, 2000, 2000, 2000, 2000, 2000];
    for (const delay of delays) {
      if (this.stopping) return false;
      await sleep(delay);

      // If process exited already, bail early
      if (this.proc && this.proc.exitCode !== null) {
        onStatus("Headroom proxy exited unexpectedly");
        this.proc = null;
        return false;
      }

      if (await this.healthCheck()) {
        this.weStartedIt = true;
        return true;
      }
    }

    // 5. Timed out — kill and report failure
    onStatus("Headroom proxy failed to start (health check timeout)");
    this.killProcess();
    return false;
  }

  // ── Health check ──────────────────────────────────────────────────

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  // ── Stop proxy (if we started it) ─────────────────────────────────

  async stop(): Promise<void> {
    this.stopping = true;

    if (!this.proc || !this.weStartedIt) {
      this.proc = null;
      this.weStartedIt = false;
      return;
    }

    const proc = this.proc;
    this.proc = null;
    this.weStartedIt = false;

    // Send SIGTERM (or hard-kill on Windows)
    try {
      if (IS_WINDOWS) {
        proc.kill();
      } else {
        proc.kill("SIGTERM");

        // Wait up to 3s for graceful exit
        const exited = await Promise.race([
          new Promise<boolean>((resolve) => {
            proc.on("exit", () => resolve(true));
          }),
          sleep(3000).then(() => false),
        ]);

        if (!exited && proc.exitCode === null) {
          proc.kill("SIGKILL");
        }
      }
    } catch {
      // Process may already be dead
    }
  }

  // ── Crash recovery ────────────────────────────────────────────────

  /**
   * Try to restart the proxy once if it crashed.
   * Returns true if recovered.
   */
  async tryRestart(onStatus: (msg: string) => void): Promise<boolean> {
    if (!this.weStartedIt) return false;
    if (this.proc && this.proc.exitCode === null) return false; // still running

    onStatus("Headroom proxy crashed, restarting...");
    this.proc = null;
    this.weStartedIt = false;
    this.stopping = false;
    return this.ensureRunning(onStatus);
  }

  // ── Private: spawn the proxy ──────────────────────────────────────

  private startProxy(onStatus: (msg: string) => void): void {
    onStatus("Starting Headroom proxy...");

    const inv = this.invocation;
    if (!inv) return;

    // Build args: e.g. ["proxy", "--port", "8787", "--host", "127.0.0.1"]
    // or ["-m", "headroom.cli", "proxy", "--port", "8787", ...]
    const spawnArgs = [...inv.args, "proxy", "--port", String(this.port), "--host", this.host];

    const proc = spawn(inv.cmd, spawnArgs, {
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
      env: { ...process.env },
    });

    proc.on("error", () => {
      // spawn error (e.g. command not found) — don't crash
    });

    proc.on("exit", () => {
      if (this.proc === proc) {
        this.proc = null;
        this.weStartedIt = false;
      }
    });

    // Unref streams so they don't keep the event loop alive on shutdown
    (proc.stdout as any)?.unref?.();
    (proc.stderr as any)?.unref?.();

    this.proc = proc;
  }

  private killProcess(): void {
    if (this.proc) {
      try {
        this.proc.kill(IS_WINDOWS ? undefined : "SIGKILL");
      } catch {
        // Already dead
      }
      this.proc = null;
    }
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function execAsync(
  cmd: string,
  args: string[],
  timeoutMs = 15_000,
): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout: timeoutMs }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr || error.message));
      } else {
        resolve((stdout || "") + (stderr || ""));
      }
    });
  });
}
