#!/usr/bin/env node
// Tiny local logging relay for the AMODS web debug page (docs/debug.html).
//
// Run this (`node scripts/debug-log-server.mjs`), then open debug.html and
// use it as normal - every console.log/warn/error from BOTH the main thread
// (main.js) and the real-time worker (worker-engine.js) streams here live,
// tagged by source, instead of needing to open DevTools and copy-paste
// output by hand. See docs/js/debug-log.js for the browser-side half of
// this. Stop with Ctrl+C.
import http from 'node:http';

const PORT = 8790;

const RESET = '\x1b[0m';
const COLORS = { main: '\x1b[36m', worker: '\x1b[35m' }; // cyan / magenta
const LEVEL_COLORS = { error: '\x1b[31m', warn: '\x1b[33m' }; // red / yellow

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  if (req.method !== 'POST' || req.url !== '/log') {
    res.writeHead(404);
    res.end();
    return;
  }

  let body = '';
  req.on('data', (chunk) => { body += chunk; });
  req.on('end', () => {
    try {
      const { source, level, text, t } = JSON.parse(body);
      const time = new Date(t).toLocaleTimeString();
      const sourceColor = COLORS[source] || '';
      const levelColor = LEVEL_COLORS[level] || '';
      console.log(`${sourceColor}[${time}] [${source}]${RESET} ${levelColor}${text}${RESET}`);
    } catch (e) {
      console.error('debug-log-server: failed to parse log body:', e.message);
    }
    res.writeHead(204);
    res.end();
  });
});

server.listen(PORT, () => {
  console.log(`debug-log-server listening on http://localhost:${PORT}`);
  console.log('Open docs/debug.html (with DEBUG_MODE on) - logs from main.js and worker-engine.js will stream here live.');
});
