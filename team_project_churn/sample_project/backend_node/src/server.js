'use strict';
// server — 부트스트랩(무설치: 내장 http). 운영 권고=Fastify로 교체 가능. 추론 코드 없음.
const http = require('http');
const cfg = require('./config');
const routes = require('./interfaces/http/routes');

function matchRoute(method, pathname) {
  if (routes[`${method} ${pathname}`]) return { handler: routes[`${method} ${pathname}`], params: {} };
  for (const key of Object.keys(routes)) {
    const [m, pat] = key.split(' ');
    if (m !== method || !pat.includes(':')) continue;
    const pe = pat.split('/'), ae = pathname.split('/');
    if (pe.length !== ae.length) continue;
    const params = {}; let ok = true;
    for (let i = 0; i < pe.length; i++) {
      if (pe[i].startsWith(':')) params[pe[i].slice(1)] = decodeURIComponent(ae[i]);
      else if (pe[i] !== ae[i]) { ok = false; break; }
    }
    if (ok) return { handler: routes[key], params };
  }
  return null;
}

function send(res, code, obj) { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(obj)); }

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const r = matchRoute(req.method, url.pathname);
  if (!r) return send(res, 404, { error: 'not found', route: `${req.method} ${url.pathname}` });
  if (url.pathname !== '/health' && (req.headers['x-api-key'] || '') !== cfg.API_KEY) return send(res, 401, { error: 'invalid api key' });
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', async () => {
    try {
      const b = body ? JSON.parse(body) : {};
      const out = await r.handler(b, r.params);
      send(res, out && out._status ? out._status : 200, out);
    } catch (e) { send(res, 500, { error: String((e && e.message) || e) }); }
  });
});

if (require.main === module) server.listen(cfg.PORT, () => console.log(`[backend] http://localhost:${cfg.PORT}`));
module.exports = { server, matchRoute };
