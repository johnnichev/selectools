"""Visual Agent Builder — self-contained HTML interface for selectools serve --builder."""

BUILDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>selectools — Agent Builder</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0f172a;
  --surface: #1e293b;
  --border: #334155;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --cyan: #22d3ee;
  --blue: #3b82f6;
  --green: #22c55e;
  --red: #ef4444;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: ui-monospace, 'Cascadia Code', Menlo, monospace;
  font-size: 13px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
header {
  height: 48px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 12px;
  flex-shrink: 0;
  z-index: 10;
}
.logo { color: var(--cyan); font-weight: 700; font-size: 15px; letter-spacing: -0.02em; }
.badge {
  background: rgba(34,211,238,0.12);
  color: var(--cyan);
  padding: 2px 10px;
  border-radius: 9999px;
  font-size: 11px;
  border: 1px solid rgba(34,211,238,0.25);
}
.header-actions { margin-left: auto; display: flex; gap: 6px; align-items: center; }
.version { color: var(--muted); font-size: 11px; }

/* Buttons */
.btn {
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text);
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  transition: all 0.15s;
  white-space: nowrap;
}
.btn:hover { background: rgba(255,255,255,0.05); border-color: var(--cyan); color: var(--cyan); }
.btn-primary { background: var(--blue); border-color: var(--blue); color: #fff; }
.btn-primary:hover { background: #2563eb; border-color: #2563eb; color: #fff; }

/* Layout */
.main {
  flex: 1;
  display: flex;
  overflow: hidden;
  min-height: 0;
}

/* Palette */
.palette {
  width: 176px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 10px 8px;
  gap: 3px;
  flex-shrink: 0;
  overflow-y: auto;
}
.section-title {
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 6px 8px 4px;
}
.palette-item {
  padding: 7px 10px;
  border-radius: 5px;
  border: 1px solid var(--border);
  cursor: grab;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  user-select: none;
  transition: all 0.12s;
}
.palette-item:hover { border-color: var(--cyan); background: rgba(34,211,238,0.06); }
.palette-item:active { cursor: grabbing; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

.tip-box {
  margin-top: 4px;
  padding: 8px 10px;
  border-radius: 5px;
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 11px;
  line-height: 1.65;
}
kbd {
  background: rgba(255,255,255,0.08);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 1px 5px;
  font-size: 10px;
  font-family: inherit;
}

/* Canvas */
.canvas-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: var(--bg);
  background-image:
    radial-gradient(circle, rgba(51,65,85,0.6) 1px, transparent 1px);
  background-size: 22px 22px;
  cursor: default;
}
#canvas { width: 100%; height: 100%; }
.canvas-hint {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  color: rgba(148,163,184,0.5);
  font-size: 11px;
  pointer-events: none;
  white-space: nowrap;
}

/* Properties panel */
.props {
  width: 216px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-header {
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  flex-shrink: 0;
}
.props-scroll { flex: 1; overflow-y: auto; }
.props-empty {
  padding: 24px 14px;
  color: var(--muted);
  font-size: 12px;
  text-align: center;
  line-height: 1.7;
}
.props-body { padding: 10px 14px; display: flex; flex-direction: column; gap: 10px; }

.field { display: flex; flex-direction: column; gap: 3px; }
.field label {
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.field input, .field select, .field textarea {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 4px;
  padding: 5px 8px;
  font-family: inherit;
  font-size: 12px;
  width: 100%;
  transition: border-color 0.12s;
  resize: vertical;
}
.field input:focus, .field select:focus, .field textarea:focus {
  outline: none;
  border-color: var(--cyan);
}
.field textarea { min-height: 72px; }
select option { background: var(--surface); }

.delete-btn {
  margin-top: 4px;
  width: 100%;
  padding: 6px;
  border-radius: 4px;
  border: 1px solid rgba(239,68,68,0.3);
  background: transparent;
  color: var(--red);
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  transition: all 0.12s;
}
.delete-btn:hover { background: rgba(239,68,68,0.1); border-color: var(--red); }

/* Code panel */
.code-panel {
  height: 196px;
  background: var(--surface);
  border-top: 1px solid var(--border);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
}
.code-tabs {
  display: flex;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding: 0 10px;
  flex-shrink: 0;
}
.code-tab {
  padding: 8px 14px;
  cursor: pointer;
  font-size: 12px;
  color: var(--muted);
  border-bottom: 2px solid transparent;
  transition: all 0.12s;
  user-select: none;
}
.code-tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.code-tab:hover:not(.active) { color: var(--text); }
.code-actions { margin-left: auto; display: flex; gap: 6px; align-items: center; }
.code-output {
  flex: 1;
  overflow: auto;
  padding: 10px 14px;
  font-size: 11.5px;
  color: #7dd3fc;
  line-height: 1.6;
  white-space: pre;
  font-family: ui-monospace, monospace;
  min-height: 0;
}

/* SVG node classes */
.node { cursor: pointer; }
.node-start .body { fill: rgba(34,197,94,0.12); stroke: #22c55e; stroke-width: 1.5; rx: 8; }
.node-end   .body { fill: rgba(239,68,68,0.12);  stroke: #ef4444; stroke-width: 1.5; rx: 8; }
.node-agent .body { fill: rgba(34,211,238,0.09); stroke: #22d3ee; stroke-width: 1.5; rx: 8; }
.node-agent.sel .body { stroke: var(--blue); stroke-width: 2.5; }
.node-start.sel .body,
.node-end.sel   .body { stroke-width: 2.5; filter: drop-shadow(0 0 5px currentColor); }
.node-label    { fill: var(--text);  font-family: ui-monospace,monospace; font-size: 12px; font-weight: 600; }
.node-sublabel { fill: var(--muted); font-family: ui-monospace,monospace; font-size: 10px; }

.port circle { fill: var(--bg); stroke: #475569; stroke-width: 1.5; cursor: crosshair; }
.port:hover circle { fill: var(--cyan); stroke: var(--cyan); }
.port.active circle { fill: var(--blue); stroke: var(--blue); }

.edge-g { cursor: pointer; }
.edge-g path { fill: none; stroke: #475569; stroke-width: 2; }
.edge-g.sel path { stroke: var(--cyan); }
.edge-g:hover path { stroke: #64748b; }
.edge-lbl { fill: #64748b; font-family: ui-monospace,monospace; font-size: 10px; }
.edge-lbl-bg { fill: var(--bg); }
.preview { pointer-events: none; fill: none; stroke: var(--blue); stroke-width: 2; stroke-dasharray: 6,3; opacity: 0.65; }
</style>
</head>
<body>

<header>
  <span class="logo">selectools</span>
  <span class="badge">builder</span>
  <span class="version">v0.20.0</span>
  <div class="header-actions">
    <button class="btn" onclick="onClear()">Clear</button>
    <button class="btn" onclick="loadExample()">Example</button>
    <button class="btn" onclick="doExport('yaml')">Export YAML</button>
    <button class="btn" onclick="doExport('python')">Export Python</button>
  </div>
</header>

<div class="main">
  <!-- Palette -->
  <div class="palette">
    <div class="section-title">Add Nodes</div>

    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'agent')">
      <div class="dot" style="background:#22d3ee"></div> Agent
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'start')">
      <div class="dot" style="background:#22c55e"></div> START
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'end')">
      <div class="dot" style="background:#ef4444"></div> END
    </div>

    <div class="section-title" style="margin-top:10px">Tips</div>
    <div class="tip-box">
      Drag nodes onto canvas.<br>
      Click <span style="color:var(--cyan)">○</span> output port<br>
      then click input port<br>
      to connect.<br><br>
      Click node to edit.<br>
      Click edge to label.<br><br>
      <kbd>Del</kbd> delete selected<br>
      <kbd>Esc</kbd> cancel action
    </div>
  </div>

  <!-- Canvas -->
  <div class="canvas-wrap" id="canvasWrap"
       ondragover="event.preventDefault()"
       ondrop="canvasDrop(event)">
    <svg id="canvas"
         onmousemove="svgMouseMove(event)"
         onmouseup="svgMouseUp(event)"
         onclick="svgClick(event)">
      <defs>
        <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3"
                orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
        </marker>
        <marker id="arr-sel" markerWidth="9" markerHeight="9" refX="7" refY="3"
                orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L0,6 L9,3 z" fill="#22d3ee"/>
        </marker>
      </defs>
      <g id="layer-edges"></g>
      <g id="layer-nodes"></g>
      <path id="preview" class="preview" d="" style="display:none"/>
    </svg>
    <div class="canvas-hint">Drag nodes from the palette · Click ○ ports to connect</div>
  </div>

  <!-- Properties -->
  <div class="props">
    <div class="panel-header">Properties</div>
    <div class="props-scroll">
      <div id="propsEmpty" class="props-empty">Select a node or edge<br>to edit its properties.</div>
      <div id="propsBody" class="props-body" style="display:none"></div>
    </div>
  </div>
</div>

<!-- Code panel -->
<div class="code-panel">
  <div class="code-tabs">
    <span class="code-tab active" id="tab-python" onclick="switchTab('python')">Python</span>
    <span class="code-tab" id="tab-yaml" onclick="switchTab('yaml')">YAML</span>
    <div class="code-actions">
      <button class="btn" id="copyBtn" onclick="copyCode()">Copy</button>
      <button class="btn" onclick="downloadCode()">Download</button>
    </div>
  </div>
  <div class="code-output" id="codeOut"></div>
</div>

<script>
// ─── Constants ────────────────────────────────────────────────────────────
const NW = 164, NH = 70, PR = 6;

// ─── State ────────────────────────────────────────────────────────────────
let nodes = [], edges = [], sel = null;
let dragging = null, connecting = null;
let mouse = { x: 0, y: 0 };
let seq = 1, activeTab = 'python';
let palType = null;

// ─── Factories ────────────────────────────────────────────────────────────
function mkNode(type, x, y) {
  const id = type === 'start' ? '__start__'
           : type === 'end'   ? 'end_' + (seq++)
           : 'agent_' + (seq++);
  return {
    id, type, x, y,
    name: type === 'start' ? 'START'
        : type === 'end'   ? 'END'
        : 'Agent ' + seq,
    provider: 'openai', model: 'gpt-4o-mini',
    system_prompt: '', tools: '',
  };
}
function mkEdge(from, to) {
  return { id: 'e' + (seq++), from, to, label: '' };
}

// ─── Geometry ─────────────────────────────────────────────────────────────
function outPort(n) { return { x: n.x + NW, y: n.y + NH / 2 }; }
function inPort(n)  { return { x: n.x,      y: n.y + NH / 2 }; }
function bez(x1, y1, x2, y2) {
  const cx = (x1 + x2) / 2;
  return `M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}`;
}

// ─── SVG helpers ──────────────────────────────────────────────────────────
function S(tag, attrs, parent) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
function Stext(txt, x, y, a, parent) {
  const el = S('text', { x, y, ...a }, parent);
  el.textContent = txt;
  return el;
}

// ─── Render ───────────────────────────────────────────────────────────────
function render() { renderEdges(); renderNodes(); updateCode(); }

function renderNodes() {
  const layer = document.getElementById('layer-nodes');
  layer.innerHTML = '';
  for (const n of nodes) {
    const isSel = sel?.type === 'node' && sel.id === n.id;
    const g = S('g', {
      class: `node node-${n.type}${isSel ? ' sel' : ''}`,
      transform: `translate(${n.x},${n.y})`
    }, layer);

    S('rect', { class: 'body', width: NW, height: NH, rx: 8 }, g);

    const label = n.type === 'start' ? 'START'
                : n.type === 'end'   ? 'END'
                : n.name;
    Stext(label, NW / 2, n.type === 'agent' ? 25 : NH / 2, {
      class: 'node-label',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle'
    }, g);

    if (n.type === 'agent') {
      Stext(n.provider + ' · ' + n.model, NW / 2, 48, {
        class: 'node-sublabel',
        'text-anchor': 'middle',
        'dominant-baseline': 'middle'
      }, g);
    }

    // Input port (not for START)
    if (n.type !== 'start') {
      const pg = S('g', { class: 'port', transform: `translate(0,${NH/2})` }, g);
      S('circle', { cx: 0, cy: 0, r: PR }, pg);
      pg.dataset.nid = n.id; pg.dataset.side = 'in';
      pg.addEventListener('click', portClick);
    }
    // Output port (not for END)
    if (n.type !== 'end') {
      const pg = S('g', { class: 'port', transform: `translate(${NW},${NH/2})` }, g);
      S('circle', { cx: 0, cy: 0, r: PR }, pg);
      pg.dataset.nid = n.id; pg.dataset.side = 'out';
      pg.addEventListener('click', portClick);
    }

    g.addEventListener('mousedown', e => nodeMouseDown(e, n.id));
    g.addEventListener('click', e => nodeClick(e, n.id));
  }
}

function renderEdges() {
  const layer = document.getElementById('layer-edges');
  layer.innerHTML = '';
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn) continue;
    const p1 = outPort(fn), p2 = inPort(tn);
    const isSel = sel?.type === 'edge' && sel.id === e.id;
    const g = S('g', { class: 'edge-g' + (isSel ? ' sel' : '') }, layer);
    S('path', { d: bez(p1.x, p1.y, p2.x, p2.y), 'marker-end': isSel ? 'url(#arr-sel)' : 'url(#arr)' }, g);
    if (e.label) {
      const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
      S('rect', { x: mx - e.label.length * 3.2 - 4, y: my - 9, width: e.label.length * 6.4 + 8, height: 14, rx: 3, class: 'edge-lbl-bg' }, g);
      Stext(e.label, mx, my, { class: 'edge-lbl', 'text-anchor': 'middle', 'dominant-baseline': 'middle' }, g);
    }
    g.addEventListener('click', ev => { ev.stopPropagation(); selEdge(e.id); });
  }
}

// ─── Interaction ──────────────────────────────────────────────────────────
function portClick(e) {
  e.stopPropagation();
  const nid = e.currentTarget.dataset.nid;
  const side = e.currentTarget.dataset.side;
  if (!connecting) {
    if (side === 'out') { connecting = { nid }; render(); }
    return;
  }
  if (side === 'in' && connecting.nid !== nid) {
    if (!edges.some(ex => ex.from === connecting.nid && ex.to === nid))
      edges.push(mkEdge(connecting.nid, nid));
  }
  connecting = null;
  document.getElementById('preview').style.display = 'none';
  render();
}

function nodeMouseDown(e, id) {
  e.stopPropagation();
  const n = nodes.find(n => n.id === id);
  if (!n) return;
  const r = document.getElementById('canvas').getBoundingClientRect();
  dragging = { id, ox: e.clientX - r.left - n.x, oy: e.clientY - r.top - n.y };
}

function nodeClick(e, id) {
  e.stopPropagation();
  if (connecting) return;
  selNode(id);
}

function svgMouseMove(e) {
  const r = document.getElementById('canvas').getBoundingClientRect();
  mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
  if (dragging) {
    const n = nodes.find(n => n.id === dragging.id);
    if (n) { n.x = mouse.x - dragging.ox; n.y = mouse.y - dragging.oy; render(); }
  }
  if (connecting) {
    const fn = nodes.find(n => n.id === connecting.nid);
    if (fn) {
      const p = outPort(fn);
      const prev = document.getElementById('preview');
      prev.setAttribute('d', bez(p.x, p.y, mouse.x, mouse.y));
      prev.style.display = '';
    }
  }
}

function svgMouseUp() { dragging = null; }

function svgClick(e) {
  if (connecting) {
    connecting = null;
    document.getElementById('preview').style.display = 'none';
    render();
    return;
  }
  deselect();
}

function selNode(id) { sel = { type: 'node', id }; render(); showNodeProps(id); }
function selEdge(id) { sel = { type: 'edge', id }; render(); showEdgeProps(id); }

function deselect() {
  sel = null; render();
  document.getElementById('propsEmpty').style.display = '';
  document.getElementById('propsBody').style.display = 'none';
}

// ─── Properties ───────────────────────────────────────────────────────────
function showNodeProps(id) {
  const n = nodes.find(n => n.id === id); if (!n) return;
  document.getElementById('propsEmpty').style.display = 'none';
  const body = document.getElementById('propsBody');
  body.style.display = 'flex'; body.innerHTML = '';

  if (n.type !== 'agent') {
    const p = document.createElement('p');
    p.style.cssText = 'color:var(--muted);font-size:12px;line-height:1.6;padding:4px 0';
    p.textContent = n.type === 'start' ? 'Entry point of the graph.' : 'Terminal node. Graph exits here.';
    body.appendChild(p);
  } else {
    addField(body, 'Name', 'text', n.name, v => { n.name = v; render(); });
    addField(body, 'Provider', 'select', n.provider, v => { n.provider = v; render(); },
      ['openai', 'anthropic', 'gemini', 'ollama']);
    addField(body, 'Model', 'text', n.model, v => { n.model = v; render(); });
    addField(body, 'System Prompt', 'textarea', n.system_prompt, v => { n.system_prompt = v; updateCode(); });
    addField(body, 'Tools (comma-sep)', 'text', n.tools, v => { n.tools = v; updateCode(); });
  }
  if (n.type !== 'start') {
    const b = document.createElement('button');
    b.className = 'delete-btn'; b.textContent = 'Delete Node';
    b.onclick = delSelected; body.appendChild(b);
  }
}

function showEdgeProps(id) {
  const e = edges.find(e => e.id === id); if (!e) return;
  document.getElementById('propsEmpty').style.display = 'none';
  const body = document.getElementById('propsBody');
  body.style.display = 'flex'; body.innerHTML = '';
  addField(body, 'Condition Label', 'text', e.label, v => { e.label = v; render(); });
  const hint = document.createElement('p');
  hint.style.cssText = 'color:var(--muted);font-size:11px;line-height:1.5;padding:2px 0';
  hint.textContent = 'Leave blank for unconditional. Add a label (e.g. "approved") for conditional routing.';
  body.appendChild(hint);
  const b = document.createElement('button');
  b.className = 'delete-btn'; b.textContent = 'Delete Edge';
  b.onclick = delSelected; body.appendChild(b);
}

function addField(parent, label, type, value, onChange, opts) {
  const div = document.createElement('div'); div.className = 'field';
  const lbl = document.createElement('label'); lbl.textContent = label;
  div.appendChild(lbl);
  let inp;
  if (type === 'textarea') {
    inp = document.createElement('textarea'); inp.value = value;
  } else if (type === 'select') {
    inp = document.createElement('select');
    for (const o of opts) {
      const opt = document.createElement('option');
      opt.value = o; opt.textContent = o;
      if (o === value) opt.selected = true;
      inp.appendChild(opt);
    }
  } else {
    inp = document.createElement('input'); inp.type = type; inp.value = value;
  }
  inp.addEventListener('input', () => onChange(inp.value));
  inp.addEventListener('change', () => onChange(inp.value));
  div.appendChild(inp); parent.appendChild(div);
}

// ─── Keyboard ────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (connecting) { connecting = null; document.getElementById('preview').style.display='none'; render(); }
    else deselect();
  }
  if ((e.key === 'Delete' || e.key === 'Backspace') && sel) {
    if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
    delSelected();
  }
});

function delSelected() {
  if (!sel) return;
  if (sel.type === 'node') {
    const n = nodes.find(n => n.id === sel.id);
    if (n?.type === 'start') return;
    nodes = nodes.filter(n => n.id !== sel.id);
    edges = edges.filter(e => e.from !== sel.id && e.to !== sel.id);
  } else {
    edges = edges.filter(e => e.id !== sel.id);
  }
  sel = null; deselect();
}

// ─── Drag from palette ────────────────────────────────────────────────────
function palDragStart(e, type) {
  palType = type; e.dataTransfer.effectAllowed = 'copy';
}

function canvasDrop(e) {
  e.preventDefault();
  if (!palType) return;
  if (palType === 'start' && nodes.some(n => n.type === 'start')) { palType = null; return; }
  const wrap = document.getElementById('canvasWrap');
  const r = wrap.getBoundingClientRect();
  const n = mkNode(palType, Math.max(8, e.clientX - r.left - NW/2), Math.max(8, e.clientY - r.top - NH/2));
  nodes.push(n); palType = null;
  render(); selNode(n.id);
}

// ─── Code generation ──────────────────────────────────────────────────────
function updateCode() {
  document.getElementById('codeOut').textContent = activeTab === 'python' ? genPython() : genYaml();
}

function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-python').className = 'code-tab' + (tab === 'python' ? ' active' : '');
  document.getElementById('tab-yaml').className   = 'code-tab' + (tab === 'yaml'   ? ' active' : '');
  updateCode();
}

function varName(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'agent';
}

function genPython() {
  const agents = nodes.filter(n => n.type === 'agent');
  if (!agents.length) return '# Drag agent nodes onto the canvas to generate code';

  const L = [];
  L.push('from selectools import Agent, AgentConfig');
  L.push('from selectools.orchestration import AgentGraph');
  L.push('');
  L.push('# Initialise your provider, e.g.:');
  L.push('# from selectools import OpenAIProvider');
  L.push('# provider = OpenAIProvider(api_key="...")');
  L.push('');

  for (const n of agents) {
    const v = varName(n.name);
    const tools = n.tools ? n.tools.split(',').map(s => s.trim()).filter(Boolean) : [];
    L.push(`${v} = Agent(`);
    L.push(`    provider=provider,`);
    if (tools.length) L.push(`    tools=[${tools.join(', ')}],`);
    L.push(`    config=AgentConfig(`);
    L.push(`        name="${n.name}",`);
    if (n.system_prompt)
      L.push(`        system_prompt=${JSON.stringify(n.system_prompt)},`);
    L.push(`    )`);
    L.push(`)`);
    L.push('');
  }

  L.push('graph = AgentGraph()');
  for (const n of agents) L.push(`graph.add_node("${varName(n.name)}", ${varName(n.name)})`);

  // entry point from START edges
  const startEdge = edges.find(e => { const fn = nodes.find(n => n.id === e.from); return fn?.type === 'start'; });
  if (startEdge) {
    const tn = nodes.find(n => n.id === startEdge.to);
    if (tn?.type === 'agent') L.push(`graph.set_entry_point("${varName(tn.name)}")`);
  }
  L.push('');

  // group conditional edges by source
  const condMap = {};
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn || fn.type === 'start') continue;
    const fid = varName(fn.name);
    const tid = tn.type === 'end' ? 'AgentGraph.END' : `"${varName(tn.name)}"`;
    if (e.label) {
      (condMap[fid] = condMap[fid] || []).push({ lbl: e.label, tid });
    } else {
      L.push(`graph.add_edge("${fid}", ${tid})`);
    }
  }

  for (const [fid, conds] of Object.entries(condMap)) {
    const mapping = conds.map(c => `        "${c.lbl}": ${c.tid}`).join(',\n');
    L.push(`graph.add_conditional_edge(`);
    L.push(`    "${fid}",`);
    L.push(`    lambda state: state.data.get("route", "${conds[0].lbl}"),`);
    L.push(`    {`);
    L.push(mapping);
    L.push(`    }`);
    L.push(`)`);
    L.push('');
  }

  L.push('result = graph.run("Your task here")');
  L.push('print(result.content)');
  return L.join('\n');
}

function genYaml() {
  const agents = nodes.filter(n => n.type === 'agent');
  if (!agents.length) return '# Drag agent nodes onto the canvas to generate YAML';

  const L = [];
  L.push('# selectools agent graph — generated by builder');
  L.push('name: my-agent-graph');
  L.push('type: graph');
  L.push('');
  L.push('nodes:');
  for (const n of agents) {
    const id = varName(n.name);
    L.push(`  ${id}:`);
    L.push(`    provider: ${n.provider}`);
    L.push(`    model: ${n.model}`);
    if (n.system_prompt) {
      L.push(`    system_prompt: |`);
      for (const line of n.system_prompt.split('\\n')) L.push(`      ${line}`);
    }
    if (n.tools) {
      const ts = n.tools.split(',').map(s => s.trim()).filter(Boolean);
      if (ts.length) { L.push(`    tools:`); ts.forEach(t => L.push(`      - ${t}`)); }
    }
  }
  L.push('');
  L.push('edges:');
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn) continue;
    const fid = fn.type === 'start' ? 'START' : varName(fn.name);
    const tid = tn.type === 'end'   ? 'END'   : varName(tn.name);
    L.push(`  - from: ${fid}`);
    L.push(`    to: ${tid}`);
    if (e.label) L.push(`    condition: "${e.label}"`);
  }
  return L.join('\n');
}

// ─── Actions ──────────────────────────────────────────────────────────────
function onClear() {
  if (!confirm('Clear the graph?')) return;
  nodes = []; edges = []; sel = null; connecting = null; seq = 1;
  deselect();
}

function copyCode() {
  const code = document.getElementById('codeOut').textContent;
  navigator.clipboard.writeText(code).then(() => {
    const b = document.getElementById('copyBtn');
    b.textContent = 'Copied!';
    setTimeout(() => b.textContent = 'Copy', 1500);
  }).catch(() => {});
}

function downloadCode() {
  const code = document.getElementById('codeOut').textContent;
  const ext = activeTab === 'python' ? 'py' : 'yaml';
  const a = document.createElement('a');
  a.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(code);
  a.download = 'agent_graph.' + ext;
  a.click();
}

function doExport(tab) { switchTab(tab); downloadCode(); }

// ─── Load example ─────────────────────────────────────────────────────────
function loadExample() {
  nodes = []; edges = []; sel = null; seq = 1;

  const start = mkNode('start', 60, 185);
  nodes.push(start);

  const researcher = mkNode('agent', 280, 110);
  researcher.name = 'Researcher';
  researcher.system_prompt = 'You are a research assistant. Produce a structured summary.';
  nodes.push(researcher);

  const writer = mkNode('agent', 500, 110);
  writer.name = 'Writer';
  writer.system_prompt = 'You are a technical writer. Turn the research into a blog post.';
  nodes.push(writer);

  const reviewer = mkNode('agent', 500, 270);
  reviewer.name = 'Reviewer';
  reviewer.system_prompt = 'Review the draft. Output APPROVED or REVISION_NEEDED.';
  nodes.push(reviewer);

  const end = mkNode('end', 720, 185);
  nodes.push(end);

  edges.push(mkEdge(start.id, researcher.id));
  edges.push(mkEdge(researcher.id, writer.id));
  edges.push(mkEdge(writer.id, reviewer.id));
  edges.push({ id: 'ex1', from: reviewer.id, to: end.id,      label: 'approved' });
  edges.push({ id: 'ex2', from: reviewer.id, to: writer.id,   label: 'needs_revision' });

  deselect();
}

// ─── Init ─────────────────────────────────────────────────────────────────
loadExample();
</script>
</body>
</html>"""
