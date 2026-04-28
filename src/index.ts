/**
 * pi-headroom — Transparent LLM context compression for Pi using Headroom.
 *
 * Hooks into Pi's `context` event to compress messages before every LLM call.
 * Automatically installs and manages the Headroom proxy (zero-config).
 *
 * Set HEADROOM_URL to skip auto-management and use your own proxy.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { convertToLlm } from "@mariozechner/pi-coding-agent";
import { HeadroomClient, compress } from "headroom-ai";
import type { CompressResult } from "headroom-ai";
import { piToOpenAI, openAIToPi } from "./format-bridge.js";
import { ProxyManager } from "./proxy-manager.js";

export default function headroomExtension(pi: ExtensionAPI) {
  // ─── State ──────────────────────────────────────────────────────────

  let enabled = true;
  let proxyAvailable: boolean | null = null;
  let proxyWarningShown = false;
  let restartAttempted = false;

  let lastStats: {
    tokensBefore: number;
    tokensAfter: number;
    tokensSaved: number;
    ratio: number;
    transforms: string[];
  } = { tokensBefore: 0, tokensAfter: 0, tokensSaved: 0, ratio: 1.0, transforms: [] };

  let sessionTotals = { calls: 0, tokensSaved: 0 };

  // ─── Configuration ──────────────────────────────────────────────────

  const userUrl = process.env.HEADROOM_URL;
  const autoManage = !userUrl;
  const port = parseInt(process.env.HEADROOM_PORT || "8787", 10);
  const proxyManager = autoManage ? new ProxyManager({ port }) : null;
  const baseUrl = userUrl ? `${userUrl}:${port}` : `http://127.0.0.1:${port}`;
  const client = new HeadroomClient({ baseUrl, fallback: true, timeout: 15_000 });

  /** Simple health check — the SDK doesn't expose one, so we hit the proxy directly. */
  async function checkProxyHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(5_000) });
      return res.ok;
    } catch {
      return false;
    }
  }

  // ─── Session start: install/start proxy or health-check ─────────────

  pi.on("session_start", async (_event, ctx) => {
    proxyWarningShown = false;
    restartAttempted = false;
    sessionTotals = { calls: 0, tokensSaved: 0 };

    if (proxyManager) {
      // Auto-manage mode
      ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", "⏳ Headroom starting..."));

      const ok = await proxyManager.ensureRunning((msg) => {
        ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", `⏳ ${msg}`));
      });

      if (ok) {
        proxyAvailable = true;
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
        );
      } else {
        proxyAvailable = false;
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
        );
        ctx.ui.notify(
          "Headroom proxy could not be started. Context compression disabled.\nRun /headroom-health for details.",
          "warning",
        );
      }
    } else {
      // User-managed mode: just health-check
      const healthy = await checkProxyHealth();
      if (healthy) {
        proxyAvailable = true;
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
        );
      } else {
        proxyAvailable = false;
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
        );
      }
    }
  });

  // ─── Session shutdown: stop proxy if we started it ──────────────────

  pi.on("session_shutdown", async () => {
    if (proxyManager) {
      await proxyManager.stop();
    }
  });

  // ─── Core: compress context before every LLM call ───────────────────

  pi.on("context", async (event, ctx) => {
    if (!enabled || proxyAvailable === false) return;

    // Convert AgentMessage[] → Pi-AI Message[] → OpenAI format
    const piMessages = convertToLlm(event.messages);
    if (piMessages.length === 0) return;

    const openaiMessages = piToOpenAI(piMessages);
    if (openaiMessages.length === 0) return;

    try {
      const result: CompressResult = await compress(openaiMessages, {
        client,
        model: ctx.model?.id ?? "gpt-4o",
        fallback: true,
      });

      if (!result.compressed || result.tokensSaved <= 0) {
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("success", "✓") +
            ctx.ui.theme.fg("dim", ` Headroom (${openaiMessages.length} msgs, no compression needed)`),
        );
        return;
      }

      // Convert compressed OpenAI → Pi-AI Message[]
      const compressedPiMessages = openAIToPi(result.messages, piMessages);

      // Update stats
      lastStats = {
        tokensBefore: result.tokensBefore,
        tokensAfter: result.tokensAfter,
        tokensSaved: result.tokensSaved,
        ratio: result.compressionRatio,
        transforms: result.transformsApplied,
      };
      sessionTotals.calls++;
      sessionTotals.tokensSaved += result.tokensSaved;

      // Update status bar
      const saved = result.tokensSaved.toLocaleString();
      const pct = Math.round((1 - result.compressionRatio) * 100);
      const theme = ctx.ui.theme;
      ctx.ui.setStatus(
        "headroom",
        theme.fg("success", "✓") + theme.fg("dim", ` Headroom -${pct}% (${saved} saved)`),
      );

      return { messages: compressedPiMessages as any };
    } catch (error) {
      if (!proxyWarningShown) {
        proxyWarningShown = true;
        proxyAvailable = false;

        const errMsg = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`Headroom proxy unavailable: ${errMsg}`, "warning");
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
        );
      }

      // Mid-session crash recovery (one attempt per session)
      if (proxyManager && !restartAttempted) {
        restartAttempted = true;
        const recovered = await proxyManager.tryRestart((msg) => {
          ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", `⏳ ${msg}`));
        });
        if (recovered) {
          proxyAvailable = true;
          proxyWarningShown = false;
          ctx.ui.setStatus(
            "headroom",
            ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
          );
          // Don't retry compression this call — next context event will use it
        }
      }

      return;
    }
  });

  // ─── /headroom command — toggle and status ──────────────────────────

  pi.registerCommand("headroom", {
    description: "Toggle Headroom compression or show status. Usage: /headroom [on|off|status]",
    handler: async (args, ctx) => {
      const arg = args.trim().toLowerCase();

      if (arg === "on") {
        enabled = true;
        proxyWarningShown = false;
        restartAttempted = false;

        if (proxyManager) {
          // Try to start the proxy
          ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", "⏳ Starting..."));
          const ok = await proxyManager.ensureRunning((msg) => {
            ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", `⏳ ${msg}`));
          });
          if (ok) {
            proxyAvailable = true;
            ctx.ui.notify("Headroom compression enabled", "info");
            ctx.ui.setStatus(
              "headroom",
              ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
            );
          } else {
            proxyAvailable = false;
            ctx.ui.notify("Headroom enabled but proxy could not be started", "warning");
            ctx.ui.setStatus(
              "headroom",
              ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
            );
          }
        } else {
          // User-managed: just health-check
          const ok2 = await checkProxyHealth();
          if (ok2) {
            proxyAvailable = true;
            ctx.ui.notify("Headroom compression enabled", "info");
            ctx.ui.setStatus(
              "headroom",
              ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
            );
          } else {
            proxyAvailable = false;
            ctx.ui.notify("Headroom enabled but proxy is offline", "warning");
            ctx.ui.setStatus(
              "headroom",
              ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
            );
          }
        }
        return;
      }

      if (arg === "off") {
        enabled = false;
        ctx.ui.notify("Headroom compression disabled", "info");
        ctx.ui.setStatus("headroom", ctx.ui.theme.fg("dim", "○ Headroom off"));
        return;
      }

      // Status (default)
      const managedStr = proxyManager
        ? proxyManager.isManaged
          ? "auto (managed by extension)"
          : "auto (external proxy detected)"
        : "manual (HEADROOM_URL set)";

      const lines = [
        `Headroom Context Compression`,
        `  Enabled: ${enabled ? "yes" : "no"}`,
        `  Proxy:   ${baseUrl} (${proxyAvailable === true ? "online" : proxyAvailable === false ? "offline" : "unknown"})`,
        `  Mode:    ${managedStr}`,
        ``,
        `Session stats:`,
        `  Compressions: ${sessionTotals.calls}`,
        `  Tokens saved: ${sessionTotals.tokensSaved.toLocaleString()}`,
      ];

      if (lastStats.tokensBefore > 0) {
        const pct = Math.round((1 - lastStats.ratio) * 100);
        lines.push(
          ``,
          `Last compression:`,
          `  ${lastStats.tokensBefore.toLocaleString()} → ${lastStats.tokensAfter.toLocaleString()} tokens (-${pct}%)`,
          `  Transforms: ${lastStats.transforms.join(", ") || "none"}`,
        );
      }

      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  // ─── /headroom-health command — proxy diagnostics ───────────────────

  pi.registerCommand("headroom-health", {
    description: "Check Headroom proxy health and show diagnostics",
    handler: async (_args, ctx) => {
      ctx.ui.notify(`Checking Headroom proxy at ${baseUrl}...`, "info");

      const isHealthy = await checkProxyHealth();
      if (isHealthy) {
        proxyAvailable = true;

        const lines = [
          `Headroom proxy: online`,
          `  URL: ${baseUrl}`,
        ];

        if (proxyManager) {
          lines.push(`  Managed: ${proxyManager.isManaged ? "yes (started by extension)" : "no (external)"}`);
        }

        ctx.ui.notify(lines.join("\n"), "info");
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("success", "✓") + ctx.ui.theme.fg("dim", " Headroom"),
        );
      } else {
        proxyAvailable = false;
        const errMsg = "proxy did not respond";
        const helpLines = [
          `Headroom proxy offline`,
          `  URL: ${baseUrl}`,
          `  Error: ${errMsg}`,
        ];

        if (proxyManager) {
          helpLines.push(``, `The extension will auto-start the proxy on next session.`, `Or run: /headroom on`);
        } else {
          helpLines.push(``, `Start the proxy manually:`, `  headroom proxy`, `  # or`, `  pip install "headroom-ai[proxy]" && headroom proxy`);
        }

        ctx.ui.notify(helpLines.join("\n"), "error");
        ctx.ui.setStatus(
          "headroom",
          ctx.ui.theme.fg("warning", "⚠") + ctx.ui.theme.fg("dim", " Headroom offline"),
        );
      }
    },
  });
}
