# pi-headroom

Transparent LLM context compression for [Pi](https://github.com/mariozechner/pi-coding-agent) using [Headroom](https://github.com/chopratejas/headroom). Automatically compresses conversation context before every LLM call, saving 70–95% of tokens without changing your workflow.

**Zero-config:** The extension automatically installs the Headroom proxy (`pip install headroom-ai[proxy]`), starts it on session start, and stops it on exit. You don't need to touch the proxy manually.

## How It Works

```
Session start → auto-install headroom-ai[proxy] → spawn proxy on :8787
                                                          ↓
User prompt → Pi builds context → pi-headroom compresses → LLM receives compressed context
                                                          ↓
Session exit → proxy stopped automatically
```

1. **`session_start`**: Checks if proxy is running. If not, installs `headroom-ai[proxy]` via pip (if needed), spawns it as a background process, and polls until healthy.
2. **`context` event**: Before every LLM call, converts Pi messages to OpenAI format, sends them to the proxy for compression, converts back, and returns compressed messages.
3. **`session_shutdown`**: Gracefully stops the proxy (only if the extension started it).

## Prerequisites

- **Python ≥ 3.10** — needed to run the Headroom proxy (the extension auto-installs it via pip)

That's it. The extension handles everything else.

## Installation

```bash
# From local path (development)
pi install ./pi-headroom

# From npm (once published)
pi install npm:pi-headroom

# Quick test without installing
pi -e ./pi-headroom
```

## Configuration

| Env Variable     | Default                 | Description                                                |
|------------------|-------------------------|------------------------------------------------------------|
| `HEADROOM_URL`   | _(none)_                | Set to use your own proxy. **Disables auto-management.**   |
| `HEADROOM_PORT`  | `8787`                  | Port for the auto-managed proxy                            |

### Auto-management vs. manual mode

- **No env vars set** (default): The extension auto-installs, auto-starts, and auto-stops the proxy. Zero-config.
- **`HEADROOM_URL` set**: The extension skips auto-management and health-checks the URL you provide. You manage the proxy yourself.
- **`HEADROOM_PORT` set**: The auto-managed proxy starts on your chosen port instead of 8787.

## Commands

### `/headroom [on|off|status]`

Toggle compression or show status.

- `/headroom` or `/headroom status` — Show current state, proxy mode, and session compression stats
- `/headroom on` — Enable compression (auto-starts proxy if needed)
- `/headroom off` — Disable compression (passthrough mode)

### `/headroom-health`

Check proxy health and show diagnostics. Shows whether the proxy is managed by the extension or external.

## Status Bar

The extension shows progress and compression status in Pi's footer:

- `⏳ Installing headroom-ai...` — Auto-installing the proxy
- `⏳ Starting Headroom proxy...` — Spawning the proxy
- `✓ Headroom` — Proxy online, ready to compress
- `✓ Headroom -42% (1,234 saved)` — Last compression result
- `⚠ Headroom offline` — Proxy unavailable, using uncompressed context
- `○ Headroom off` — Compression disabled by user

## Behavior

- **Zero-config**: Installs and starts the proxy automatically on first use
- **Smart detection**: Won't reinstall or restart if already running (e.g., you started it manually)
- **Graceful fallback**: If anything fails, Pi continues with uncompressed context
- **Crash recovery**: If the proxy crashes mid-session, one automatic restart is attempted
- **Clean shutdown**: The proxy is stopped on session exit (only if the extension started it)
- **Cross-platform**: Works on macOS, Linux, and Windows

## Architecture

```
pi-headroom/
├── package.json          # Pi package manifest
├── tsconfig.json
├── src/
│   ├── index.ts          # Extension: context hook, lifecycle, commands
│   ├── format-bridge.ts  # Pi-AI ↔ OpenAI message format conversion
│   └── proxy-manager.ts  # Auto-install, start, stop, health check
└── README.md
```

## License

MIT
