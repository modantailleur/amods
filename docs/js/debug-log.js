// Debug-only: relays this JS realm's console.log/warn/error calls to a
// small local logging server (see scripts/debug-log-server.mjs) in addition
// to the real console. Each realm - the main thread (main.js) and the
// worker (worker-engine.js) - has its OWN separate console object, so a
// single override only ever sees calls made in that same realm; this must
// be installed once per realm (see main.js's/worker-engine.js's own
// DEBUG_MODE/debugTelemetry-gated calls to it) to get a unified feed of
// both. Silently does nothing if the log server isn't running (fire-and-
// forget fetch, failure ignored) - so leaving this installed never breaks
// normal use when nobody's running the terminal tool.
const LOG_SERVER_URL = 'http://localhost:8790/log';

export function installDebugLogRelay(source) {
  for (const level of ['log', 'warn', 'error']) {
    const orig = console[level].bind(console);
    console[level] = (...args) => {
      orig(...args);
      const text = args.map((a) => (typeof a === 'string' ? a : safeStringify(a))).join(' ');
      fetch(LOG_SERVER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, level, text, t: Date.now() }),
      }).catch(() => {}); // log server not running (or not started yet) - fine, just skip this line
    };
  }
}

function safeStringify(v) {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
