"""lkhu memory dashboard — a single HTML page served by the daemon.

With no build step, it polls ``/api/stats``/``/api/memories`` with vanilla JS and renders.
It shows stored memories, strength/age distributions, consolidation (long-term transition)
status, and lifecycle settings at a glance.
"""
# ruff: noqa: E501 — this is an HTML/CSS/JS template string, so the line-length limit is not applied.

from __future__ import annotations

__all__ = ["render_dashboard"]

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>lkhu Memory Dashboard</title>
<style>
  :root { --bg:#0f1115; --card:#1a1d24; --line:#2a2f3a; --fg:#e6e8ec; --mut:#8b93a3;
          --green:#3fb950; --yellow:#d29922; --red:#f85149; --blue:#58a6ff; --purple:#bc8cff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Apple SD Gothic Neo",sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:16px 24px; border-bottom:1px solid var(--line); }
  header h1 { font-size:18px; margin:0; }
  header .sub { color:var(--mut); font-size:13px; }
  header .right { margin-left:auto; display:flex; gap:12px; align-items:center; font-size:13px; color:var(--mut); }
  main { padding:24px; max-width:1200px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .card .n { font-size:26px; font-weight:700; }
  .card .l { color:var(--mut); font-size:12px; margin-top:2px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:20px; }
  .panel h2 { font-size:14px; margin:0 0 12px; color:var(--fg); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media(max-width:760px){ .grid2{ grid-template-columns:1fr; } }
  .bar { display:flex; align-items:center; gap:8px; margin:5px 0; font-size:12px; }
  .bar .lbl { width:130px; color:var(--mut); flex-shrink:0; }
  .bar .track { flex:1; background:#11141a; border-radius:5px; height:14px; overflow:hidden; }
  .bar .fill { height:100%; background:var(--blue); }
  .bar .v { width:34px; text-align:right; color:var(--mut); }
  .life { font-size:12.5px; line-height:1.8; color:var(--mut); }
  .life code { background:#11141a; padding:1px 6px; border-radius:4px; color:var(--fg); }
  .flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px; font-size:12px; }
  .flow .step { background:#11141a; border:1px solid var(--line); padding:6px 10px; border-radius:8px; }
  .flow .arr { color:var(--mut); }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--mut); font-weight:600; cursor:pointer; user-select:none; position:sticky; top:0; background:var(--card); }
  td.text { white-space:normal; max-width:460px; }
  tr.archived { opacity:.45; }
  .badge { padding:2px 8px; border-radius:20px; font-size:11px; }
  .badge.turn{ background:#1f2937; color:#9db2cf; }
  .badge.explicit{ background:#14301c; color:var(--green); }
  .badge.summary{ background:#2a2140; color:var(--purple); }
  .badge.merged{ background:#33270f; color:var(--yellow); }
  .sbar { display:inline-block; width:70px; height:9px; background:#11141a; border-radius:5px; vertical-align:middle; overflow:hidden; }
  .sbar > i { display:block; height:100%; }
  .muted { color:var(--mut); }
  button { background:#21262d; color:var(--fg); border:1px solid var(--line); border-radius:7px; padding:6px 10px; cursor:pointer; font-size:12px; }
  label.tg { display:flex; gap:6px; align-items:center; cursor:pointer; }
</style>
</head>
<body>
<header>
  <h1>🌸 lkhu Memory Dashboard</h1>
  <span class="sub" id="datadir"></span>
  <div class="right">
    <label class="tg"><input type="checkbox" id="showArch"/> Show archived</label>
    <span id="updated"></span>
    <button onclick="load()">Refresh</button>
  </div>
</header>
<main>
  <div class="cards" id="cards"></div>

  <div class="panel">
    <h2>Lifecycle — when a memory becomes long-term, and when it is forgotten</h2>
    <div class="flow">
      <span class="step">Working memory (RAM)</span><span class="arr">→</span>
      <span class="step">Short-term accumulated scent</span><span class="arr">→</span>
      <span class="step">Long-term storage (shown here)</span><span class="arr">→</span>
      <span class="step">Consolidated summary</span><span class="arr">→</span>
      <span class="step muted">Archived/forgotten when weak</span>
    </div>
    <div class="life" id="life"></div>
  </div>

  <div class="grid2">
    <div class="panel"><h2>Strength distribution (survival likelihood)</h2><div id="strength"></div></div>
    <div class="panel"><h2>Age distribution · kind</h2><div id="agekind"></div></div>
  </div>

  <div class="panel">
    <h2>Stored memories <span class="muted" id="count"></span></h2>
    <div style="overflow:auto; max-height:60vh;">
      <table>
        <thead><tr>
          <th data-k="strength">Strength</th><th data-k="kind">Kind</th><th data-k="audit_text">Content</th>
          <th data-k="created_at">Created</th><th data-k="last_accessed_at">Last recall</th>
          <th data-k="access_count">Recalls</th><th data-k="session_id">Session</th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
</main>
<script>
let MEM=[], SORT={k:"strength",dir:-1};
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const scolor=s=> s<0.2?"var(--red)": s<0.5?"var(--yellow)": s<1.0?"var(--blue)":"var(--green)";
const when=iso=>{ if(!iso) return "-"; const d=new Date(iso), now=new Date();
  const mins=Math.floor((now-d)/60000);
  if(mins<60) return mins+"m ago"; if(mins<1440) return Math.floor(mins/60)+"h ago";
  return Math.floor(mins/1440)+"d ago"; };

function bars(obj, color){
  const max=Math.max(1,...Object.values(obj));
  return Object.entries(obj).map(([k,v])=>
    `<div class="bar"><span class="lbl">${esc(k)}</span>
     <span class="track"><span class="fill" style="width:${v/max*100}%;background:${color||'var(--blue)'}"></span></span>
     <span class="v">${v}</span></div>`).join("");
}

async function load(){
  const arch = document.getElementById("showArch").checked;
  const [stats, mem] = await Promise.all([
    fetch("/api/stats").then(r=>r.json()),
    fetch("/api/memories?archived="+(arch?1:0)).then(r=>r.json())
  ]);
  MEM = mem.memories;
  document.getElementById("datadir").textContent = stats.data_dir || "";
  document.getElementById("updated").textContent = "Updated "+new Date().toLocaleTimeString();

  document.getElementById("cards").innerHTML = [
    ["Total memories (active)", stats.total_active],
    ["Archived", stats.total_archived],
    ["Consolidated summaries", stats.summaries],
    ["Merged source count", stats.consolidated_sources],
  ].map(([l,n])=>`<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");

  const L=stats.lifecycle;
  document.getElementById("life").innerHTML =
    `• <b>Long-term storage</b>: every memory enters SQLite (here) immediately. The "transition" is split by kind —
       <span class="badge turn">turn</span>(auto-collected)·<span class="badge explicit">explicit</span>(explicitly stored)
       → grouped into <span class="badge summary">summary</span>(session merge) by the nightly <b>consolidation</b>.<br>
     • <b>Forgetting</b>: strength ×<code>${L.daily_decay}</code> daily, ×<code>${L.recall_boost}</code> boost on recall (max <code>${L.max_strength}</code>).<br>
     • <b>Consolidation condition</b>: <code>${L.min_session_size}</code>+ items in one session → merged. Schedule <code>${L.consolidation_cron}</code>.<br>
     • <b>Cleansing (forgetting finalized)</b>: strength < <code>${L.weak_strength}</code> and <code>${L.weak_min_age_days}</code>+ days → archived. Similarity > <code>${L.duplicate_threshold}</code> → merged. Schedule <code>${L.cleansing_cron}</code>.`;

  document.getElementById("strength").innerHTML = bars(stats.strength_buckets);
  document.getElementById("agekind").innerHTML =
    `<div class="muted" style="font-size:11px;margin-bottom:4px">Age</div>${bars(stats.age_buckets,'var(--purple)')}
     <div class="muted" style="font-size:11px;margin:10px 0 4px">Kind</div>${bars(stats.kinds,'var(--green)')}`;
  render();
}

function render(){
  MEM.sort((a,b)=>{ const k=SORT.k; let x=a[k],y=b[k];
    if(typeof x==="string"){x=x||"";y=y||"";return x<y?SORT.dir:x>y?-SORT.dir:0;}
    return ((x||0)-(y||0))*-SORT.dir; });
  document.getElementById("count").textContent = "· "+MEM.length+" items";
  document.getElementById("rows").innerHTML = MEM.map(m=>{
    const w=Math.min(100,(m.strength/1.5)*100);
    return `<tr class="${m.archived?'archived':''}">
      <td><span class="sbar"><i style="width:${w}%;background:${scolor(m.strength)}"></i></span> ${m.strength.toFixed(2)}</td>
      <td><span class="badge ${esc(m.kind)}">${esc(m.kind)}</span></td>
      <td class="text">${esc(m.audit_text).slice(0,300)}</td>
      <td>${when(m.created_at)}</td><td>${when(m.last_accessed_at)}</td>
      <td>${m.access_count}</td><td class="muted">${esc(m.session_id)||'-'}</td></tr>`;
  }).join("");
}

document.querySelectorAll("th").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; SORT.dir = (SORT.k===k)? -SORT.dir : -1; SORT.k=k; render();
});
document.getElementById("showArch").onchange=load;
load(); setInterval(load, 5000);
</script>
</body>
</html>"""


def render_dashboard() -> str:
    """Return the dashboard HTML string."""
    return _HTML
