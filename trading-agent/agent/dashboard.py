"""Real-time dashboard (stdlib only): PnL, win rate, positions, signals. Reads the SQLite DB."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .storage import Store

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Agent</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#0f1115;color:#e6e6e6}
h1{font-size:18px;margin:0 0 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:#181b22;border-radius:8px;padding:12px}.k{font-size:12px;color:#9aa}.v{font-size:20px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid #262a33;text-align:left}
.pos{color:#4ade80}.neg{color:#f87171}section{margin-top:14px}</style></head><body>
<h1>Trading Agent <span id="mode" class="k"></span></h1>
<div class="grid" id="stats"></div>
<section class="card"><canvas id="eq" height="90"></canvas></section>
<section class="card"><div class="k">Position</div><pre id="pos"></pre><div class="k">Strategy</div><pre id="strat"></pre><div class="k">Last signal</div><pre id="sig"></pre></section>
<section class="card"><div class="k">Recent trades</div><table id="trades"></table></section>
<section class="card"><div class="k">Log</div><table id="logs"></table></section>
<script>
let chart;
const f=(x,d=2)=>x==null?'-':Number(x).toFixed(d);
async function load(){const d=await (await fetch('/api/state')).json();
 document.getElementById('mode').textContent=d.mode||'';
 const s=[['Equity',f(d.equity)],['Total PnL',f(d.pnl_total)],['Win rate',d.win_rate==null?'-':(d.win_rate*100).toFixed(1)+'%'],
  ['Closed trades',d.closed],['Max DD',(d.max_dd*100).toFixed(2)+'%'],['Status',d.killed?'KILLED':'running']];
 document.getElementById('stats').innerHTML=s.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
 document.getElementById('pos').textContent=JSON.stringify(d.position,null,1)||'flat';
 document.getElementById('strat').textContent=JSON.stringify(d.strategy,null,1);
 document.getElementById('sig').textContent=JSON.stringify(d.last_signal,null,1);
 document.getElementById('trades').innerHTML='<tr><th>time</th><th>side</th><th>qty</th><th>price</th><th>pnl</th><th>reason</th></tr>'+
  d.trades.map(t=>`<tr><td>${t.ts}</td><td>${t.side}</td><td>${f(t.qty,6)}</td><td>${f(t.price)}</td><td class="${(t.pnl||0)>=0?'pos':'neg'}">${f(t.pnl)}</td><td>${t.reason}</td></tr>`).join('');
 document.getElementById('logs').innerHTML=d.logs.map(l=>`<tr><td>${l.ts}</td><td>${l.level}</td><td>${l.message}</td></tr>`).join('');
 const labels=d.equity_curve.map(e=>e.ts),vals=d.equity_curve.map(e=>e.equity);
 if(!chart){chart=new Chart(document.getElementById('eq'),{type:'line',data:{labels,datasets:[{label:'Equity',data:vals,borderColor:'#60a5fa',pointRadius:0,borderWidth:1.5}]},options:{animation:false,scales:{x:{display:false}}}});}
 else{chart.data.labels=labels;chart.data.datasets[0].data=vals;chart.update();}}
load();setInterval(load,5000);
</script></body></html>"""


def state(store: Store) -> dict:
    eq = store.rows("SELECT ts, equity FROM equity ORDER BY ts")
    step = max(1, len(eq) // 1000)
    sells = store.rows("SELECT pnl FROM trades WHERE side='sell'")
    pnl = [r["pnl"] or 0 for r in sells]
    peak, max_dd = 0.0, 0.0
    for e in eq:
        peak = max(peak, e["equity"])
        max_dd = max(max_dd, 1 - e["equity"] / peak if peak else 0)
    active = store.active_strategy()
    logs = store.rows("SELECT ts, level, message FROM logs ORDER BY id DESC LIMIT 30")
    return {
        "equity": eq[-1]["equity"] if eq else None,
        "pnl_total": (eq[-1]["equity"] - eq[0]["equity"]) if eq else 0,
        "win_rate": sum(p > 0 for p in pnl) / len(pnl) if pnl else None,
        "closed": len(pnl),
        "max_dd": max_dd,
        "position": store.get_state("position"),
        "strategy": {"spec": active[0], "validation": active[1]} if active else None,
        "last_signal": store.get_state("last_signal"),
        "killed": any("agent stopped" in l["message"] for l in logs[:5]),
        "trades": store.rows("SELECT ts, side, qty, price, pnl, reason FROM trades ORDER BY id DESC LIMIT 25"),
        "logs": logs,
        "equity_curve": eq[::step] + (eq[-1:] if eq and (len(eq) - 1) % step else []),
    }


def serve(db_path, port: int = 8050):
    store = Store(db_path)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/state"):
                body, ctype = json.dumps(state(store), default=str).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"dashboard on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
