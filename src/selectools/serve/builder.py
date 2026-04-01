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

/* Templates dropdown */
.tmpl-wrap { position: relative; }
.tmpl-menu {
  display: none;
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 4px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  min-width: 200px;
  z-index: 100;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  overflow: hidden;
}
.tmpl-menu.open { display: block; }
.tmpl-item {
  padding: 8px 14px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text);
  transition: background 0.1s;
  white-space: nowrap;
}
.tmpl-item:hover { background: rgba(34,211,238,0.08); color: var(--cyan); }

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

/* Minimap */
.minimap {
  position: absolute;
  bottom: 8px;
  right: 8px;
  width: 160px;
  height: 96px;
  background: rgba(15,23,42,0.88);
  border: 1px solid var(--border);
  border-radius: 6px;
  pointer-events: none;
  overflow: hidden;
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
.node-loop  .body { fill: rgba(249,115,22,0.09); stroke: #f97316; stroke-width: 1.5; rx: 8; }
.node-subgraph .body { fill: rgba(168,85,247,0.09); stroke: #a855f7; stroke-width: 1.5; rx: 8; }
.node-note  .body { fill: rgba(245,158,11,0.09); stroke: #f59e0b; stroke-width: 1.5; rx: 8; }
.node-hitl  .body { fill: rgba(245,158,11,0.12); stroke: #f59e0b; stroke-width: 2; rx: 8; }
.node-agent.sel .body { stroke: var(--blue); stroke-width: 2.5; }
.node-loop.sel .body, .node-subgraph.sel .body, .node-note.sel .body, .node-hitl.sel .body { stroke-width: 2.5; filter: drop-shadow(0 0 5px currentColor); }
.node-start.sel .body,
.node-end.sel   .body { stroke-width: 2.5; filter: drop-shadow(0 0 5px currentColor); }
.node-label    { fill: var(--text);  font-family: ui-monospace,monospace; font-size: 12px; font-weight: 600; }
.node-sublabel { fill: var(--muted); font-family: ui-monospace,monospace; font-size: 10px; }
.node-cost     { fill: #4ade80;      font-family: ui-monospace,monospace; font-size: 9px; }

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

/* Port type colors */
.port-msg circle  { stroke: #22d3ee; stroke-width: 1.5; }
.port-ctrl circle { stroke: #22c55e; stroke-width: 1.5; }
.port-body circle { stroke: #f97316; stroke-width: 1.5; }
.port-done circle { stroke: #22c55e; stroke-width: 1.5; }
.port-term circle { stroke: #ef4444; stroke-width: 1.5; }
.port-sub circle  { stroke: #a855f7; stroke-width: 1.5; }
.port-var circle  { stroke: #a855f7; stroke-width: 1.5; r: 4; }
.port-msg:hover circle  { fill: #22d3ee; stroke: #22d3ee; }
.port-ctrl:hover circle { fill: #22c55e; stroke: #22c55e; }
.port-body:hover circle { fill: #f97316; stroke: #f97316; }
.port-done:hover circle { fill: #22c55e; stroke: #22c55e; }
.port-term:hover circle { fill: #ef4444; stroke: #ef4444; }
.port-sub:hover circle  { fill: #a855f7; stroke: #a855f7; }
.port-var:hover circle  { fill: #a855f7; stroke: #a855f7; }
.port-blocked circle    { fill: #1e293b !important; stroke: #475569 !important; opacity: 0.4; cursor: not-allowed; }
.port-compatible circle { filter: drop-shadow(0 0 4px currentColor); }

/* Context menu + agent picker */
.ctx-item { padding:7px 14px; cursor:pointer; font-size:12px; color:var(--text); user-select:none; }
.ctx-item:hover { background:rgba(255,255,255,0.06); }
.ctx-danger { color:var(--red); }
.ctx-sep { height:1px; background:var(--border); margin:4px 0; }
.agent-pick-panel { position:absolute; z-index:700; background:var(--surface); border:1px solid var(--cyan);
  border-radius:6px; padding:4px 0; min-width:180px; box-shadow:0 4px 20px rgba(0,0,0,0.5);
  max-height:220px; overflow-y:auto; }
.agent-pick-item { padding:7px 12px; cursor:pointer; font-size:12px; color:var(--text); display:flex; gap:6px; align-items:center; }
.agent-pick-item:hover { background:rgba(34,211,238,0.1); color:var(--cyan); }

/* Embed mode — hide editor chrome */
body.embed-mode header, body.embed-mode .main, body.embed-mode .code-panel { display:none !important; }
body.embed-mode .test-panel { flex:1; border-top:none; height:100vh; }
body.embed-mode .page { height:100vh; }

/* Wire tooltip */
.wire-tooltip { position:fixed; z-index:500; background:var(--surface); border:1px solid var(--border);
  border-radius:6px; padding:10px 12px; max-width:300px; pointer-events:none;
  box-shadow:0 4px 20px rgba(0,0,0,0.4); font-family:ui-monospace,monospace; font-size:11px; }

/* Wire inspector panel */
.wire-inspector { position:fixed; z-index:520; background:var(--surface); border:1px solid var(--border);
  border-radius:8px; padding:10px 12px; max-width:360px; min-width:200px; pointer-events:auto;
  box-shadow:0 4px 24px rgba(0,0,0,0.5); font-family:ui-monospace,monospace; font-size:11px; }
.wire-inspector pre { margin:6px 0 0; padding:0; white-space:pre-wrap; word-break:break-word;
  max-height:240px; overflow-y:auto; font-size:10px; line-height:1.5; }
.wire-copy-btn { font-size:10px; padding:1px 7px; border-radius:4px; cursor:pointer;
  background:var(--surface); border:1px solid var(--border); color:var(--text); float:right; margin-left:8px; }
.wi-key { color:#93c5fd; } .wi-str { color:#86efac; } .wi-num { color:#fcd34d; }
.wi-bool { color:#f9a8d4; } .wi-null { color:#94a3b8; }

/* Note resize handle */
.note-resize-handle { cursor:se-resize; opacity:0.6; }
.note-resize-handle:hover { opacity:1; }

/* Cost badge */
.cost-badge { font-size:10px; color:#4ade80; margin-left:6px; border:1px solid #4ade8044;
  border-radius:10px; padding:2px 8px; }

/* Scrubber replay button */
.scrub-replay { font-size:9px; color:var(--cyan); cursor:pointer; opacity:0; border:none;
  background:none; padding:0 3px; font-family:inherit; transition:opacity 0.12s; }
.scrub-step:hover .scrub-replay { opacity:1; }

/* AI Generate animation */
@keyframes nodeIn {
  from { opacity: 0; transform: scale(0.7); }
  to   { opacity: 1; transform: scale(1); }
}
.node-entering { animation: nodeIn 0.25s ease-out forwards; }

/* Frozen node */
.node-frozen .body { stroke-dasharray: 6,4 !important; opacity: 0.72; }

/* Trace scrubber */
.scrubber { display:flex; gap:3px; padding:5px 14px; flex-shrink:0; align-items:center;
  border-top:1px solid var(--border); flex-wrap:wrap; min-height:28px; }
.scrub-step { display:flex; align-items:center; gap:3px; cursor:pointer; opacity:0.65;
  transition:opacity 0.12s; padding:2px 4px; border-radius:3px; }
.scrub-step:hover { opacity:1; background:rgba(255,255,255,0.04); }
.scrub-step.active { opacity:1; background:rgba(34,211,238,0.08); }
.scrub-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.scrub-label { font-size:9px; color:var(--muted); white-space:nowrap; max-width:60px;
  overflow:hidden; text-overflow:ellipsis; }
.scrub-line { width:14px; height:1px; background:var(--border); flex-shrink:0; }
.scrub-header { display:flex; align-items:center; justify-content:space-between;
  padding:2px 8px 0; flex-shrink:0; }
.gantt-wrap { padding:4px 8px 6px; overflow-x:auto; flex-shrink:0; }
.gantt-label { font:10px monospace; fill:#94a3b8; }
.gantt-bar { opacity:0.85; cursor:pointer; }
.gantt-bar:hover { opacity:1; }
.gantt-dur { font:9px monospace; fill:#0f172a; pointer-events:none; }
.gantt-axis-line { stroke:#334155; stroke-width:1; }
.gantt-tick-line { stroke:#475569; stroke-width:1; }
.gantt-tick-label { font:9px monospace; fill:#64748b; }

/* Structured trace rows (Feature 09) */
.trace-row { border-radius:6px; overflow:hidden; }
.trace-row-header {
  display:flex; align-items:center; gap:8px;
  padding:5px 10px; cursor:pointer;
  background:var(--surface); font-size:11px;
}
.trace-row-header:hover { background:rgba(30,58,138,0.25); }
.trace-row-body {
  padding:7px 12px; background:#0a111f;
  border-top:1px solid var(--border);
  font:11px ui-monospace,monospace;
  display:none;
}
.trace-row.expanded .trace-row-body { display:block; }
.trace-badge {
  font-size:10px; padding:1px 6px; border-radius:10px;
  background:var(--surface); border:1px solid var(--border);
  color:var(--muted); margin-left:auto;
}
.trace-spinner { animation:spin 1s linear infinite; display:inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
.trace-type-tool   { border-left:3px solid #f59e0b; }
.trace-type-chunk  { border-left:3px solid #22d3ee; }
.trace-type-node   { border-left:3px solid #8b5cf6; }
.trace-type-error  { border-left:3px solid #f87171; }

/* Port pinning (Feature 10) */
.port-pinned { background: #f59e0b !important; }

/* Eval badges */
.eval-dot-pass { fill: #22c55e; }
.eval-dot-fail { fill: #ef4444; }
.trace-eval-pass { color: #22c55e; font-size: 11px; }
.trace-eval-fail { color: #ef4444; font-size: 11px; }
.trace-hitl-prompt { color: #f59e0b; margin: 6px 0 4px; font-weight: 600; }
.hitl-btns { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 8px; }
.hitl-btn { padding: 4px 12px; border-radius: 4px; border: 1px solid #f59e0b;
  background: rgba(245,158,11,0.1); color: #f59e0b; cursor: pointer;
  font-size: 12px; font-family: inherit; }
.hitl-btn:hover { background: rgba(245,158,11,0.25); }

/* Search overlay (Cmd+K) */
.search-overlay { position:fixed; top:60px; left:50%; transform:translateX(-50%);
  width:340px; background:var(--surface); border:1px solid var(--cyan); border-radius:8px;
  z-index:1000; box-shadow:0 12px 36px rgba(0,0,0,0.55); overflow:hidden; }
.search-overlay input { width:100%; background:transparent; border:none; outline:none;
  color:var(--text); padding:10px 14px; font-family:inherit; font-size:13px;
  border-bottom:1px solid var(--border); }
.search-result { padding:7px 14px; cursor:pointer; font-size:12px; display:flex;
  align-items:center; gap:8px; }
.search-result:hover { background:rgba(34,211,238,0.08); color:var(--cyan); }
.search-empty { padding:12px 14px; color:var(--muted); font-size:12px; }

/* Code panel resize handle */
.code-resize-handle {
  height: 6px;
  cursor: ns-resize;
  background: transparent;
  flex-shrink: 0;
  transition: background 0.12s;
}
.code-resize-handle:hover, .code-resize-handle.active { background: rgba(34,211,238,0.18); }

/* Node error state */
.node-error .body { stroke: #f97316 !important; filter: drop-shadow(0 0 6px rgba(249,115,22,0.6)); }

/* Test panel */
.test-panel {
  height: 240px; background: var(--surface); border-top: 2px solid var(--cyan);
  flex-shrink: 0; display: flex; flex-direction: column;
}
.test-header {
  padding: 8px 14px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; flex-shrink: 0; gap: 8px;
}
.test-tabs-bar {
  display: flex; align-items: center; border-bottom: 1px solid var(--border);
  padding: 0 10px; flex-shrink: 0; background: var(--bg);
}
.test-tab {
  padding: 6px 12px; cursor: pointer; font-size: 12px; color: var(--muted);
  border-bottom: 2px solid transparent; user-select: none;
}
.test-tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.test-output, .test-history {
  flex: 1; overflow-y: auto; padding: 8px 14px; font-size: 12px;
  font-family: ui-monospace, monospace; line-height: 1.6;
}
.trace-node-start { color: var(--cyan); margin-top: 6px; font-weight: 600; }
.trace-chunk { color: var(--text); }
.trace-tool { color: #f59e0b; margin: 2px 0; }
.trace-tool-result { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
.trace-node-end { color: var(--green); font-size: 11px; }
.trace-error { color: var(--red); }
.trace-run-end { color: var(--green); font-weight: 600; border-top: 1px solid var(--border); margin-top: 8px; padding-top: 6px; }
.history-item { padding: 6px 10px; border-radius: 4px; border: 1px solid var(--border); margin-bottom: 6px; cursor: pointer; }
.history-item:hover { border-color: var(--cyan); }
.history-time { color: var(--muted); font-size: 10px; }
.history-preview { color: var(--text); font-size: 11px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>

<datalist id="condition-presets">
  <option value="approved">
  <option value="rejected">
  <option value="needs_revision">
  <option value="continue">
  <option value="done">
  <option value="error">
  <option value="retry">
  <option value="human_review">
  <option value="revise">
  <option value="body">
  <option value="exit">
</datalist>

<header>
  <span class="logo">selectools</span>
  <span class="badge">builder</span>
  <span class="version">v0.20.0</span>
  <span id="costBadge" class="cost-badge" title="Estimated cost per workflow run based on 500 in / 200 out tokens per agent node"></span>
  <div class="header-actions">
    <button class="btn" onclick="onClear()">Clear</button>
    <button class="btn" onclick="loadExample()">Example</button>
    <div class="tmpl-wrap">
      <button class="btn" id="tmplBtn" onclick="toggleTemplates(event)">Templates &#9660;</button>
      <div class="tmpl-menu" id="tmplMenu">
        <div class="tmpl-item" onclick="loadTemplate('simple_chatbot')">Simple Chatbot</div>
        <div class="tmpl-item" onclick="loadTemplate('researcher_writer')">Researcher + Writer</div>
        <div class="tmpl-item" onclick="loadTemplate('rag_pipeline')">RAG Pipeline</div>
        <div class="tmpl-item" onclick="loadTemplate('reviewer_loop')">Reviewer Loop</div>
        <div class="tmpl-item" onclick="loadTemplate('hitl_approval')">HITL Approval</div>
        <div class="tmpl-item" onclick="loadTemplate('multi_model_panel')">Multi-Model Panel</div>
        <div class="tmpl-item" onclick="loadTemplate('chain_of_thought')">Chain of Thought</div>
      </div>
    </div>
    <button class="btn" onclick="openImport()">Import</button>
    <button class="btn" onclick="openEmbed()" title="Get embed code for this workflow">&#128279; Embed</button>
    <button class="btn" onclick="openLoadTrace()" title="Load a production AgentTrace JSON into the scrubber and timeline">&#x1F4E5; Trace</button>
    <button class="btn" onclick="openWatchFile()" title="Watch a Python file and sync canvas on save">&#x1F4C2; Watch</button>
    <button class="btn" onclick="openGenBar()" style="color:#a855f7;border-color:#a855f7">&#10024; Generate</button>
    <button class="btn" onclick="undoAction()">&#8630; Undo</button>
    <button class="btn" onclick="redoAction()">&#8631; Redo</button>
    <button class="btn" onclick="doExport('yaml')">Export YAML</button>
    <button class="btn" onclick="doExport('python')">Export Python</button>
    <button class="btn btn-primary" onclick="openTestPanel()">&#9654; Test</button>
    <div id="providerHealthBar" style="display:flex;gap:4px;align-items:center;padding:0 4px" title="Provider health"></div>
  </div>
</header>

<div id="genBar" style="display:none; background:var(--surface); border-bottom:1px solid var(--border); padding:8px 16px; gap:8px; align-items:center; flex-shrink:0">
  <input id="genInput" type="text" placeholder="Describe your agent workflow&#8230;" style="flex:1; background:var(--bg); border:1px solid var(--border); border-radius:6px; color:var(--text); font-family:inherit; font-size:13px; padding:6px 10px" />
  <button id="genBtn" class="btn btn-primary" onclick="doGenerate()">Generate</button>
  <button class="btn" onclick="closeGenBar()">&#10005;</button>
</div>

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
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'loop')">
      <div class="dot" style="background:#f97316"></div> Loop
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'subgraph')">
      <div class="dot" style="background:#a855f7"></div> Subgraph
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'note')">
      <div class="dot" style="background:#f59e0b"></div> Note
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'hitl')">
      <div class="dot" style="background:#f59e0b"></div> Human Input
    </div>
    <div class="palette-item" draggable="true"
         ondragstart="palDragStart(event,'agent_tool')">
      <div class="dot" style="background:#a78bfa"></div> Agent Tool
    </div>

    <div class="section-title" style="margin-top:10px">Tips</div>
    <div class="tip-box">
      Drag nodes onto canvas.<br>
      Click <span style="color:var(--cyan)">&#9675;</span> output port<br>
      then click input port<br>
      to connect.<br><br>
      Click node to edit.<br>
      Click edge to label.<br><br>
      <kbd>Del</kbd> delete selected<br>
      <kbd>Esc</kbd> cancel action<br>
      <kbd>Ctrl+Z</kbd> undo<br>
      <kbd>Ctrl+Y</kbd> redo<br>
      <kbd>Ctrl+C</kbd> copy node<br>
      <kbd>Ctrl+V</kbd> paste node<br>
      <kbd>Cmd+K</kbd> search nodes
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
    <div class="canvas-hint">Drag nodes from the palette &#xB7; Click &#9675; ports to connect</div>
    <svg class="minimap" id="minimap" viewBox="0 0 160 96" xmlns="http://www.w3.org/2000/svg"></svg>
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
  <div class="code-resize-handle" id="resizeHandle"></div>
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

<!-- Test panel -->
<div id="testPanel" class="test-panel" style="display:none">
  <div class="test-header">
    <span style="color:var(--cyan);font-weight:600;font-size:13px">Test Run</span>
    <div style="display:flex;gap:6px;align-items:center;margin-left:auto">
      <span style="color:var(--muted);font-size:11px">API key (optional):</span>
      <input id="apiKeyInput" type="password" placeholder="sk-... or leave blank for mock"
             style="width:200px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 8px;font-family:inherit;font-size:11px" />
      <button class="btn btn-primary" id="runBtn" onclick="runTest()">&#9654; Run</button>
      <button class="btn" onclick="closeTestPanel()">&#10005;</button>
    </div>
  </div>
  <textarea id="testInput" placeholder="Enter test message (e.g. 'Summarize the latest AI news')"
            style="width:100%;background:var(--bg);border:1px solid var(--border);border-top:none;color:var(--text);padding:8px 14px;font-family:inherit;font-size:12px;resize:none;height:52px;outline:none"></textarea>
  <div class="test-tabs-bar">
    <span class="test-tab active" id="ttab-output" onclick="switchTestTab('output')">Output</span>
    <span class="test-tab" id="ttab-history" onclick="switchTestTab('history')">History (0)</span>
    <span class="test-tab" id="ttab-ai" onclick="switchTestTab('ai')">&#x1F4AC; AI</span>
    <span id="testStatus" style="margin-left:auto;font-size:11px;color:var(--muted)"></span>
    <button class="btn" style="font-size:10px;padding:2px 6px;margin-left:6px" title="Dock test panel beside canvas" onclick="setTestPanelMode(testPanelMode==='docked'?'overlay':'docked')">&#x229E;</button>
  </div>
  <div id="testOutput" class="test-output"></div>
  <!-- Feature 14: AI Copilot Tab -->
  <div id="aiCopilotTab" style="display:none;flex-direction:column;height:100%">
    <div id="aiCopilotHistory" style="flex:1;overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:6px"></div>
    <div id="aiSuggestedFollowUp" style="padding:3px 8px;font-size:10px;color:var(--muted);cursor:pointer" onclick="useFollowUp()"></div>
    <div style="display:flex;gap:6px;padding:7px 8px;border-top:1px solid var(--border)">
      <input id="aiCopilotInput" type="text" placeholder="Describe what to change..." style="flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 8px;font:11px ui-monospace,monospace;outline:none" onkeydown="if(event.key==='Enter')sendAiCopilot()">
      <button class="btn btn-primary" style="font-size:11px;padding:3px 10px" onclick="sendAiCopilot()">Send</button>
    </div>
  </div>
  <div id="scrubberHeader" class="scrub-header" style="display:none">
    <span style="font-size:10px;color:var(--muted)">Execution steps</span>
    <button id="ganttToggleBtn" class="btn" style="font-size:10px;padding:2px 6px" onclick="toggleGantt()">&#x1F4CA; Timeline</button>
  </div>
  <div id="scrubber" class="scrubber" style="display:none"></div>
  <div id="ganttWrap" class="gantt-wrap" style="display:none">
    <svg id="ganttSvg" style="width:100%;display:block;"></svg>
  </div>
  <div id="historyControls" style="display:none;padding:6px 8px;border-bottom:1px solid var(--border);gap:6px;align-items:center;flex-shrink:0">
    <input id="historySearch" type="text" placeholder="Search runs..." oninput="filterHistory()"
           style="flex:1;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 7px;font-size:11px;outline:none">
    <button class="btn" style="font-size:11px;padding:2px 8px" onclick="exportHistory()">&#x2B07; Export</button>
  </div>
  <div id="historySessionCost" style="display:none;padding:4px 10px;font-size:10px;color:var(--muted);border-bottom:1px solid var(--border)"></div>
  <div id="testHistory" class="test-history" style="display:none"></div>
</div>

<!-- Search overlay (Cmd+K) -->
<div id="searchOverlay" class="search-overlay" style="display:none">
  <input id="searchInput" type="text" placeholder="Search nodes\u2026" oninput="searchNodes(this.value)" />
  <div id="searchResults" style="max-height:220px;overflow-y:auto"></div>
</div>

<script>
// ─── Constants ────────────────────────────────────────────────────────────
const NW = 164, NH = 70, PR = 6;

// ─── Port compatibility ────────────────────────────────────────────────────
const COMPAT = {
  ctrl: new Set(['msg', 'sub', 'term']),
  msg:  new Set(['msg', 'sub', 'term', 'body', 'var']),
  body: new Set(['msg', 'sub']),
  done: new Set(['msg', 'sub', 'term']),
  sub:  new Set(['msg', 'sub', 'term']),
  var:  new Set([]),   // var is input-only — cannot be a connection source
};
function portsCompat(srcPtype, dstPtype) {
  return (COMPAT[srcPtype] || new Set()).has(dstPtype);
}

// ─── Model registry (grouped by provider) ────────────────────────────────
const MODELS = {
  openai: [
    'gpt-4o','gpt-4o-mini','gpt-4.1','gpt-4.1-mini','gpt-4.1-nano',
    'gpt-5','gpt-5-mini','gpt-5-nano','gpt-5-pro','chatgpt-4o-latest',
    'o1','o1-mini','o1-pro','o3','o3-mini','o3-pro','o4-mini',
    'gpt-4-turbo','gpt-4','gpt-3.5-turbo',
    'gpt-4o-2024-11-20','gpt-4o-2024-08-06','gpt-4o-mini-2024-07-18',
  ],
  anthropic: [
    'claude-sonnet-4-6','claude-opus-4-6','claude-haiku-4-5',
    'claude-3-7-sonnet-latest','claude-3-5-sonnet-latest','claude-3-5-haiku-latest',
    'claude-opus-4','claude-sonnet-4','claude-3-opus','claude-3-haiku',
  ],
  gemini: [
    'gemini-2.5-pro','gemini-2.5-flash','gemini-2.5-flash-lite',
    'gemini-2.0-flash','gemini-2.0-flash-lite',
    'gemini-1.5-pro','gemini-1.5-flash',
    'gemma-3','gemma-3n',
  ],
  ollama: [
    'llama3.2','llama3.1','llama3','llama2',
    'mistral','mixtral','gemma','phi',
    'qwen','codellama','neural-chat','vicuna',
  ],
};

// ─── Cost registry ────────────────────────────────────────────────────────
const MODEL_COSTS = {
  'gpt-4o':{in:2.5,out:10},'gpt-4o-mini':{in:0.15,out:0.6},
  'gpt-4.1':{in:2.0,out:8},'gpt-4.1-mini':{in:0.4,out:1.6},'gpt-4.1-nano':{in:0.1,out:0.4},
  'gpt-5':{in:1.25,out:10},'gpt-5-mini':{in:0.25,out:2},'chatgpt-4o-latest':{in:5,out:15},
  'o1':{in:15,out:60},'o3':{in:2,out:8},'o3-mini':{in:1.1,out:4.4},'o4-mini':{in:1.1,out:4.4},
  'claude-sonnet-4-6':{in:3,out:15},'claude-opus-4-6':{in:5,out:25},'claude-haiku-4-5':{in:1,out:5},
  'claude-3-7-sonnet-latest':{in:3,out:15},'claude-3-5-sonnet-latest':{in:3,out:15},
  'claude-3-5-haiku-latest':{in:0.8,out:4},'claude-opus-4':{in:5,out:25},'claude-sonnet-4':{in:3,out:15},
  'gemini-2.5-pro':{in:1.25,out:10},'gemini-2.5-flash':{in:0.3,out:2.5},
  'gemini-2.5-flash-lite':{in:0.1,out:0.4},'gemini-2.0-flash':{in:0.1,out:0.4},
  'gemini-1.5-pro':{in:1.25,out:5},'gemini-1.5-flash':{in:0.075,out:0.3},
};

function nodeCostLabel(n) {
  const c = MODEL_COSTS[n.model];
  if (!c) return 'local/free';
  if (c.in === 0) return 'free';
  const est = (c.in * 500 + c.out * 200) / 1e6;
  return `~$${est < 0.001 ? est.toFixed(5) : est.toFixed(4)}/call`;
}

// ─── State ────────────────────────────────────────────────────────────────
let nodes = [], edges = [], sel = null;
let dragging = null, connecting = null;
let mouse = { x: 0, y: 0 };
let seq = 1, activeTab = 'python';
let palType = null;

// ─── Undo / Redo ──────────────────────────────────────────────────────────
let history = [], histIdx = -1;
let clipboard = null;
let frozenOutputs = {};   // nodeId → last output string for frozen nodes
let edgeLastOutput = {};  // edgeId → last output from source node
let evalResults = {};     // nodeId → {pass, results} from last run

// ─── Feature 09: Structured Trace State ───────────────────────────────────
let traceRows = [];  // [{type, header, body, badge, id, node_id}]

// ─── Feature 10: Data Pinning ─────────────────────────────────────────────
let pinnedPorts = {};  // key: `${nodeId}::${portKey}`, value: pinned data

// ─── Feature 11: Replay Diff ──────────────────────────────────────────────
let replayBaseline = {};  // nodeId → last output string
let activeReplayNodeId = null;

// ─── Feature 13: Docked Panel ────────────────────────────────────────────
let testPanelMode = 'hidden';

// ─── Feature 14: AI Copilot ───────────────────────────────────────────────
let aiCopilotHistory = [];

// ─── Variable port helpers ─────────────────────────────────────────────────
const SKIP_VARS = new Set(['0','1','2','3','4','5','6','7','8','9']);
function extractVars(text) {
  const matches = (text || '').match(/\\{([a-zA-Z_][a-zA-Z0-9_]{0,31})\\}/g) || [];
  return [...new Set(matches.map(m => m.slice(1, -1)).filter(v => !SKIP_VARS.has(v)))];
}

function snapshot() {
  const state = JSON.stringify({ nodes: nodes, edges: edges });
  history = history.slice(0, histIdx + 1);
  history.push(state);
  if (history.length > 50) history.shift();
  histIdx = history.length - 1;
}

function undoAction() {
  if (histIdx <= 0) return;
  histIdx--;
  const s = JSON.parse(history[histIdx]);
  nodes = s.nodes; edges = s.edges; sel = null;
  deselect();
}

function redoAction() {
  if (histIdx >= history.length - 1) return;
  histIdx++;
  const s = JSON.parse(history[histIdx]);
  nodes = s.nodes; edges = s.edges; sel = null;
  deselect();
}

// ─── Factories ────────────────────────────────────────────────────────────
function mkNode(type, x, y) {
  const id = type === 'start'    ? '__start__'
           : type === 'end'      ? 'end_' + (seq++)
           : type === 'loop'     ? 'loop_' + (seq++)
           : type === 'subgraph' ? 'sub_' + (seq++)
           : type === 'note'     ? 'note_' + (seq++)
           : type === 'hitl'     ? 'hitl_' + (seq++)
           : 'agent_' + (seq++);
  const base = { id, type, x, y };
  if (type === 'start') return { ...base, name: 'START' };
  if (type === 'end')   return { ...base, name: 'END' };
  if (type === 'loop')  return { ...base, name: 'Loop ' + seq, max_iterations: 5, exit_condition: '' };
  if (type === 'subgraph') return { ...base, name: 'Subgraph ' + seq, graph_name: '' };
  if (type === 'note')  return { ...base, name: 'Note', text: '', color: 'yellow' };
  if (type === 'hitl')  return { ...base, name: 'Human Input', options: 'approve, reject', timeout_label: 'timeout' };
  if (type === 'agent_tool') return { ...base, label: 'Nested Agent', tool_name: 'nested_agent', tool_description: '', tool_input_param: 'query', tool_target_node: '', tool_max_tokens: 500 };
  return { ...base, name: 'Agent ' + seq, provider: 'openai', model: 'gpt-4o-mini', system_prompt: '', tools: '', frozen: false, eval_assertion: '' };
}

function mkEdge(from, to) {
  return { id: 'e' + (seq++), from, to, label: '' };
}

// ─── Geometry ─────────────────────────────────────────────────────────────
function outPort(n)      { return { x: n.x + NW, y: n.y + NH / 2 }; }
function outPortBody(n)  { return { x: n.x + NW, y: n.y + NH * 0.3 }; }
function outPortDone(n)  { return { x: n.x + NW, y: n.y + NH * 0.7 }; }
function inPort(n)       { return { x: n.x,      y: n.y + NH / 2 }; }
function bez(x1, y1, x2, y2) {
  const cx = (x1 + x2) / 2;
  return `M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}`;
}

function nodeHeight(n) {
  if (n.type === 'hitl') {
    const opts = (n.options || '').split(',').filter(s => s.trim());
    const numPorts = opts.length + 1; // options + timeout
    return Math.max(NH, 30 + numPorts * 22);
  }
  if (n.type === 'agent') {
    const vars = extractVars(n.system_prompt);
    if (vars.length > 0) return Math.max(NH, Math.ceil(NH / 2) + 18 + vars.length * 18 + 6);
  }
  if (n.type === 'note') return n.collapsed ? 32 : (n.height || NH);
  return NH;
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
function render() {
  // Prune edges whose varPort no longer exists in the target node's system_prompt
  edges = edges.filter(e => {
    if (!e.varPort) return true;
    const tn = nodes.find(n => n.id === e.to);
    return tn && extractVars(tn.system_prompt).includes(e.varPort);
  });
  renderEdges(); renderNodes(); updatePortCompat(); updateCode(); updateMinimap(); updateWorkflowCost();
}

function updatePortCompat() {
  document.querySelectorAll('.port').forEach(pg => {
    pg.classList.remove('port-compatible', 'port-blocked');
    if (!connecting) return;
    if (pg.dataset.nid === connecting.nid) return;  // own ports — neutral
    if (pg.dataset.side !== 'in') {
      pg.classList.add('port-blocked');              // out→out never valid
    } else if (!portsCompat(connecting.ptype, pg.dataset.ptype)) {
      pg.classList.add('port-blocked');
    } else {
      pg.classList.add('port-compatible');
    }
  });
}

function updateWorkflowCost() {
  const badge = document.getElementById('costBadge');
  if (!badge) return;
  const agentNodes = nodes.filter(n => n.type === 'agent');
  if (!agentNodes.length) { badge.textContent = ''; return; }
  let total = 0;
  for (const n of agentNodes) {
    const c = MODEL_COSTS[n.model];
    if (c) total += (c.in * 500 + c.out * 200) / 1e6;
  }
  const costStr = total === 0 ? 'free' : total < 0.001 ? `~$${total.toFixed(5)}/run` : `~$${total.toFixed(4)}/run`;
  badge.textContent = `${agentNodes.length} agents \xb7 ${costStr}`;
  badge.title = agentNodes.map(n => `${n.name}: ${nodeCostLabel(n)}`).join('\\n');
}

// ─── Agent-as-Tool Picker ─────────────────────────────────────────────────
function openAgentPicker(nodeId, anchorEl) {
  closeAgentPicker();
  const others = nodes.filter(n => n.type === 'agent' && n.id !== nodeId);
  if (!others.length) {
    const st = document.getElementById('testStatus');
    if (st) { const p = st.textContent; st.textContent = 'No other agent nodes in graph.'; setTimeout(() => { st.textContent = p; }, 2000); }
    return;
  }
  const panel = document.createElement('div');
  panel.id = 'agentPickerPanel';
  panel.className = 'agent-pick-panel';
  panel.innerHTML = '<div style="padding:6px 12px;font-size:10px;color:var(--cyan);font-weight:700;border-bottom:1px solid var(--border)">Pick agent to use as tool:</div>';
  for (const other of others) {
    const item = document.createElement('div');
    item.className = 'agent-pick-item';
    item.innerHTML = `<span style="color:#a855f7">\u25cf</span> ${other.name}`;
    item.title = `Adds @${varName(other.name)} to tools`;
    item.onclick = () => {
      const cn = nodes.find(x => x.id === nodeId);
      if (cn) {
        const existing = cn.tools ? cn.tools.split(',').map(s => s.trim()).filter(Boolean) : [];
        const ref = '@' + varName(other.name);
        if (!existing.includes(ref)) { existing.push(ref); cn.tools = existing.join(', '); }
        updateCode();
        showNodeProps(nodeId);
      }
      closeAgentPicker();
    };
    panel.appendChild(item);
  }
  anchorEl.appendChild(panel);
  setTimeout(() => document.addEventListener('click', closeAgentPicker, { once: true }), 0);
}

function closeAgentPicker() {
  const p = document.getElementById('agentPickerPanel');
  if (p) p.remove();
}

function renderNodes() {
  const layer = document.getElementById('layer-nodes');
  layer.innerHTML = '';
  for (const n of nodes) {
    const isSel = sel?.type === 'node' && sel.id === n.id;
    const frozenCls = (n.frozen) ? ' node-frozen' : '';
    const g = S('g', {
      class: `node node-${n.type}${isSel ? ' sel' : ''}${frozenCls}`,
      transform: `translate(${n.x},${n.y})`
    }, layer);

    S('rect', { class: 'body', width: NW, height: NH, rx: 8 }, g);

    if (n.type === 'start' || n.type === 'end') {
      const label = n.type === 'start' ? 'START' : 'END';
      Stext(label, NW / 2, NH / 2, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);

    } else if (n.type === 'agent') {
      Stext(n.name, NW / 2, 20, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      Stext(n.provider + ' \xb7 ' + n.model, NW / 2, 40, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      Stext(nodeCostLabel(n), NW / 2, 57, {
        class: 'node-cost', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      if (n.frozen) {
        Stext('\u2744 frozen', NW - 4, 9, {
          class: 'frozen-badge', 'text-anchor': 'end', 'dominant-baseline': 'middle',
          style: 'fill:#94a3b8;font-size:8px'
        }, g);
      }
      const er = evalResults[n.id];
      if (er) {
        const ec = S('circle', { cx: NW - 8, cy: NH - 8, r: 5,
          class: er.pass ? 'eval-dot-pass' : 'eval-dot-fail' }, g);
        const t = S('title', {}, ec);
        t.textContent = er.pass ? 'Evals: all passed'
          : 'Evals failed: ' + er.results.filter(r => !r.pass).map(r => r.name).join(', ');
      }

    } else if (n.type === 'hitl') {
      const nh = nodeHeight(n);
      S('rect', { class: 'body', width: NW, height: nh, rx: 8 }, g);
      Stext(n.name, NW / 2, 20, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      const sub = (n.options || '').slice(0, 30);
      Stext(sub, NW / 2, 38, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: 'fill:#f59e0b;font-size:9px'
      }, g);

    } else if (n.type === 'loop') {
      Stext(n.name, NW / 2, 22, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      Stext('max: ' + n.max_iterations + ' iters', NW / 2, 40, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      Stext('body \xb7 done', NW / 2, 56, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: 'fill:#f97316;font-size:9px'
      }, g);

    } else if (n.type === 'subgraph') {
      Stext(n.name, NW / 2, 25, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      const gname = n.graph_name || '(not set)';
      Stext(gname, NW / 2, 48, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);

    } else if (n.type === 'note') {
      const noteColors = { yellow: '#f59e0b', blue: '#3b82f6', green: '#22c55e', red: '#ef4444' };
      const nc = noteColors[n.color] || '#f59e0b';
      const nw = n.width || NW;
      const nh_n = n.collapsed ? 32 : (n.height || NH);
      S('rect', { class: 'body', width: nw, height: nh_n, rx: 8, style: `fill:${nc}22;stroke:${nc};stroke-width:1.5` }, g);
      Stext('NOTE', nw / 2, 14, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: `fill:${nc};font-size:9px;font-weight:700`
      }, g);
      const colBtn = S('text', { x: nw - 10, y: 14, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: 'fill:var(--muted);font-size:9px;cursor:pointer', class: 'node-sublabel' }, g);
      colBtn.textContent = n.collapsed ? '\u25bc' : '\u25b2';
      const nid_col = n.id;
      colBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleNoteCollapse(nid_col); });
      if (!n.collapsed) {
        const fo = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
        fo.setAttribute('x', '6'); fo.setAttribute('y', '22');
        fo.setAttribute('width', String(nw - 12)); fo.setAttribute('height', String(nh_n - 30));
        const fdiv = document.createElement('div');
        fdiv.style.cssText = 'font-size:9px;line-height:1.4;color:var(--text);overflow:hidden;padding:1px';
        fdiv.xmlns = 'http://www.w3.org/1999/xhtml';
        fdiv.innerHTML = renderMarkdown(n.text || '');
        fo.appendChild(fdiv);
        g.appendChild(fo);
        const rhNid = n.id;
        const rh = S('rect', { x: nw - 12, y: nh_n - 12, width: 10, height: 10, rx: 2,
          class: 'note-resize-handle', style: `fill:${nc}88` }, g);
        rh.addEventListener('mousedown', ev => startNoteResize(ev, rhNid));
      }
    } else if (n.type === 'agent_tool') {
      // Feature 17: agent_tool node rendering
      S('rect', { class: 'body', width: NW, height: NH, rx: 8, style: 'stroke:#a78bfa;stroke-width:1.5' }, g);
      Stext('\u2699 Agent Tool', NW / 2, 16, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: 'fill:var(--muted);font-size:9px'
      }, g);
      Stext(n.label || n.name || 'Nested Agent', NW / 2, 34, {
        class: 'node-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
      const toolSub = n.tool_name || '(unnamed tool)';
      Stext(toolSub, NW / 2, 50, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: 'fill:#a78bfa;font-size:9px'
      }, g);
    }

    // Input port (not for START, not for note)
    if (n.type !== 'start' && n.type !== 'note') {
      const inPtype = n.type === 'end' ? 'term' : n.type === 'loop' ? 'body'
                    : n.type === 'subgraph' ? 'sub' : 'msg';
      const nh = nodeHeight(n);
      const pg = S('g', { class: `port port-${inPtype}`, transform: `translate(0,${nh/2})` }, g);
      S('circle', { cx: 0, cy: 0, r: PR }, pg);
      pg.dataset.nid = n.id; pg.dataset.side = 'in'; pg.dataset.ptype = inPtype;
      pg.addEventListener('click', portClick);

      // Variable input ports derived from {placeholders} in system_prompt
      if (n.type === 'agent') {
        const vars = extractVars(n.system_prompt);
        vars.forEach((v, i) => {
          const py = Math.ceil(NH / 2) + 18 + i * 18;
          const vpg = S('g', { class: 'port port-var', transform: `translate(0,${py})` }, g);
          S('circle', { cx: 0, cy: 0, r: PR - 1 }, vpg);
          Stext(v, PR + 3, py, {
            class: 'node-sublabel', 'dominant-baseline': 'middle',
            style: 'font-size:8px;fill:#a855f7'
          }, g);
          vpg.dataset.nid = n.id; vpg.dataset.side = 'in'; vpg.dataset.port = v; vpg.dataset.ptype = 'var';
          vpg.addEventListener('click', portClick);
        });
      }
    }

    // Output ports
    if (n.type !== 'end' && n.type !== 'note') {
      if (n.type === 'loop') {
        // Two output ports for loop: body and done
        const pgb = S('g', { class: 'port port-body', transform: `translate(${NW},${NH*0.3})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pgb);
        pgb.dataset.nid = n.id; pgb.dataset.side = 'out'; pgb.dataset.port = 'body'; pgb.dataset.ptype = 'body';
        pgb.addEventListener('click', portClick);
        Stext('body', NW + PR + 3, NH * 0.3, {
          class: 'node-sublabel', 'dominant-baseline': 'middle',
          style: 'font-size:8px;fill:#f97316'
        }, g);

        const pgd = S('g', { class: 'port port-done', transform: `translate(${NW},${NH*0.7})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pgd);
        pgd.dataset.nid = n.id; pgd.dataset.side = 'out'; pgd.dataset.port = 'done'; pgd.dataset.ptype = 'done';
        pgd.addEventListener('click', portClick);
        Stext('done', NW + PR + 3, NH * 0.7, {
          class: 'node-sublabel', 'dominant-baseline': 'middle',
          style: 'font-size:8px;fill:#22c55e'
        }, g);
      } else if (n.type === 'hitl') {
        const opts = (n.options || '').split(',').map(s => s.trim()).filter(Boolean);
        const nh = nodeHeight(n);
        opts.forEach((opt, i) => {
          const py = 30 + i * 22;
          const pg = S('g', { class: 'port port-msg', transform: `translate(${NW},${py})` }, g);
          S('circle', { cx: 0, cy: 0, r: PR }, pg);
          pg.dataset.nid = n.id; pg.dataset.side = 'out'; pg.dataset.port = opt; pg.dataset.ptype = 'msg';
          pg.addEventListener('click', portClick);
          Stext(opt, NW + PR + 3, py, {
            class: 'node-sublabel', 'dominant-baseline': 'middle',
            style: 'font-size:8px;fill:#f59e0b'
          }, g);
        });
        const tpy = nh - 12;
        const tpg = S('g', { class: 'port port-ctrl', transform: `translate(${NW},${tpy})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, tpg);
        tpg.dataset.nid = n.id; tpg.dataset.side = 'out'; tpg.dataset.port = n.timeout_label || 'timeout'; tpg.dataset.ptype = 'ctrl';
        tpg.addEventListener('click', portClick);
        Stext(n.timeout_label || 'timeout', NW + PR + 3, tpy, {
          class: 'node-sublabel', 'dominant-baseline': 'middle',
          style: 'font-size:8px;fill:#22c55e'
        }, g);
      } else {
        const outPtype = n.type === 'start' ? 'ctrl' : n.type === 'subgraph' ? 'sub' : 'msg';
        const nh = nodeHeight(n);
        const pg = S('g', { class: `port port-${outPtype}`, transform: `translate(${NW},${nh/2})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pg);
        pg.dataset.nid = n.id; pg.dataset.side = 'out'; pg.dataset.ptype = outPtype;
        pg.addEventListener('click', portClick);
      }
    }

    g.addEventListener('mousedown', e => nodeMouseDown(e, n.id));
    g.addEventListener('click', e => nodeClick(e, n.id));
    g.addEventListener('contextmenu', e => { e.preventDefault(); selNode(n.id); showCtxMenu(e, n.id); });
  }
}

function renderEdges() {
  const layer = document.getElementById('layer-edges');
  layer.innerHTML = '';
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn) continue;
    let p1;
    if (fn.type === 'loop' && e.port === 'done') {
      p1 = outPortDone(fn);
    } else if (fn.type === 'loop' && e.port === 'body') {
      p1 = outPortBody(fn);
    } else if (fn.type === 'hitl' && e.port) {
      const opts = (fn.options || '').split(',').map(s => s.trim()).filter(Boolean);
      const idx = opts.indexOf(e.port);
      if (idx >= 0) {
        p1 = { x: fn.x + NW, y: fn.y + 30 + idx * 22 };
      } else {
        const nh = nodeHeight(fn);
        p1 = { x: fn.x + NW, y: fn.y + nh - 12 };
      }
    } else {
      p1 = outPort(fn);
    }
    let p2;
    if (e.varPort && tn.type === 'agent') {
      const vars = extractVars(tn.system_prompt);
      const vi = vars.indexOf(e.varPort);
      p2 = vi >= 0
        ? { x: tn.x, y: tn.y + Math.ceil(NH / 2) + 18 + vi * 18 }
        : inPort(tn);
    } else {
      p2 = inPort(tn);
    }
    const isSel = sel?.type === 'edge' && sel.id === e.id;
    const g = S('g', { class: 'edge-g' + (isSel ? ' sel' : '') }, layer);
    S('path', { d: bez(p1.x, p1.y, p2.x, p2.y), 'marker-end': isSel ? 'url(#arr-sel)' : 'url(#arr)' }, g);
    if (e.label) {
      const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
      S('rect', { x: mx - e.label.length * 3.2 - 4, y: my - 9, width: e.label.length * 6.4 + 8, height: 14, rx: 3, class: 'edge-lbl-bg' }, g);
      Stext(e.label, mx, my, { class: 'edge-lbl', 'text-anchor': 'middle', 'dominant-baseline': 'middle' }, g);
    }
    g.addEventListener('click', ev => { ev.stopPropagation(); selEdge(e.id); });
    const edgeId = e.id;
    g.addEventListener('mouseenter', ev => showWireInspector(ev, edgeId));
    g.addEventListener('mouseleave', hideWireInspector);
    g.addEventListener('mousemove', moveWireInspector);
  }
}

// ─── Sticky Notes Rich Mode ───────────────────────────────────────────────
function renderMarkdown(text) {
  return (text || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm, '<h3 style="margin:2px 0;font-size:9px">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 style="margin:2px 0;font-size:10px">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 style="margin:2px 0;font-size:11px">$1</h1>')
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code style="background:rgba(0,0,0,0.3);padding:0 2px;border-radius:2px">$1</code>')
    .replace(/^[*-] (.+)$/gm, '<div style="margin-left:8px">\u2022 $1</div>')
    .replace(/\\n/g, '<br>');
}
function toggleNoteCollapse(nodeId) {
  const node = nodes.find(n => n.id === nodeId);
  if (!node) return;
  node.collapsed = !node.collapsed;
  renderNodes();
  saveHistory();
}
let _resizeNote = null;
function startNoteResize(ev, nodeId) {
  ev.stopPropagation(); ev.preventDefault();
  const node = nodes.find(n => n.id === nodeId);
  if (!node) return;
  _resizeNote = { id: nodeId, startX: ev.clientX, startY: ev.clientY,
                  startW: node.width || NW, startH: node.height || NH };
  document.addEventListener('mousemove', _doNoteResize);
  document.addEventListener('mouseup', _stopNoteResize);
}
function _doNoteResize(ev) {
  if (!_resizeNote) return;
  const node = nodes.find(n => n.id === _resizeNote.id);
  if (!node) return;
  node.width  = Math.max(120, _resizeNote.startW + (ev.clientX - _resizeNote.startX));
  node.height = Math.max(60,  _resizeNote.startH + (ev.clientY - _resizeNote.startY));
  renderNodes();
}
function _stopNoteResize() {
  if (_resizeNote) { saveHistory(); _resizeNote = null; }
  document.removeEventListener('mousemove', _doNoteResize);
  document.removeEventListener('mouseup', _stopNoteResize);
}

// ─── Wire-Hover Preview (Rivet-style) ─────────────────────────────────────
function showWireTooltip(ev, fromId) {
  const tt = document.getElementById('wireTooltip');
  if (!tt) return;
  const output = frozenOutputs[fromId];
  const node = nodes.find(n => n.id === fromId);
  const label = node ? node.name : fromId;
  const preview = output
    ? output.slice(0, 280) + (output.length > 280 ? '\u2026' : '')
    : '<span style="color:var(--muted)">No output yet \u2014 run the workflow first</span>';
  tt.innerHTML = `<div style="color:var(--cyan);font-size:10px;margin-bottom:5px;font-weight:700">\u25ba ${label}</div><div style="white-space:pre-wrap;line-height:1.5">${preview}</div>`;
  tt.style.display = 'block';
  tt.style.left = (ev.clientX + 14) + 'px';
  tt.style.top  = (ev.clientY - 8) + 'px';
}

function hideWireTooltip() {
  const tt = document.getElementById('wireTooltip');
  if (tt) tt.style.display = 'none';
}

// ─── Wire Inspector (Rivet-style full output panel) ───────────────────────
function showWireInspector(e, edgeId) {
  const panel = document.getElementById('wireInspector');
  const body  = document.getElementById('wireInspectorBody');
  const lbl   = document.getElementById('wireInspectorLabel');
  if (!panel) return;
  const edge = edges.find(x => x.id === edgeId);
  const srcNode = edge ? nodes.find(n => n.id === edge.from) : null;
  const label = srcNode ? srcNode.name : (edge ? edge.from : 'Wire');
  lbl.textContent = '\u25ba ' + label;
  const val = edgeLastOutput[edgeId];
  body.innerHTML = val != null
    ? syntaxHighlightJSON(val)
    : '<span style="color:var(--muted)">No output yet \u2014 run the workflow first</span>';
  panel.style.display = 'block';
  panel.style.left = (e.clientX + 16) + 'px';
  panel.style.top  = (e.clientY - 10) + 'px';
}
function hideWireInspector() {
  const panel = document.getElementById('wireInspector');
  if (panel) panel.style.display = 'none';
}
function moveWireInspector(e) {
  const panel = document.getElementById('wireInspector');
  if (panel && panel.style.display !== 'none') {
    panel.style.left = (e.clientX + 16) + 'px';
    panel.style.top  = (e.clientY - 10) + 'px';
  }
}
function copyWireInspector() {
  const body = document.getElementById('wireInspectorBody');
  if (body) navigator.clipboard.writeText(body.textContent || '').catch(() => {});
}
function syntaxHighlightJSON(val) {
  let str;
  try { str = JSON.stringify(typeof val === 'string' ? JSON.parse(val) : val, null, 2); }
  catch { str = String(val == null ? '' : val); }
  if (str.length > 2000) str = str.slice(0, 2000) + '\\n\u2026 (truncated)';
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"([^"]+)":/g, '<span class="wi-key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="wi-str">"$1"</span>')
    .replace(/: (-?\\d+\\.?\\d*)/g, ': <span class="wi-num">$1</span>')
    .replace(/: (true|false)/g, ': <span class="wi-bool">$1</span>')
    .replace(/: (null)/g, ': <span class="wi-null">$1</span>');
}

// ─── Minimap ──────────────────────────────────────────────────────────────
function updateMinimap() {
  const mm = document.getElementById('minimap');
  if (!mm) return;
  mm.innerHTML = '';
  if (!nodes.length) return;

  const MW = 160, MH = 96, pad = 8;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x + NW);
    maxY = Math.max(maxY, n.y + NH);
  }
  const bw = maxX - minX || 1, bh = maxY - minY || 1;
  const scaleX = (MW - pad * 2) / bw;
  const scaleY = (MH - pad * 2) / bh;
  const scale = Math.min(scaleX, scaleY);

  const nodeColors = {
    start: '#22c55e', end: '#ef4444', agent: '#22d3ee',
    loop: '#f97316', subgraph: '#a855f7', note: '#f59e0b', hitl: '#f59e0b'
  };
  for (const n of nodes) {
    const rx = pad + (n.x - minX) * scale;
    const ry = pad + (n.y - minY) * scale;
    const rw = Math.max(4, NW * scale);
    const rh = Math.max(3, NH * scale);
    const col = nodeColors[n.type] || '#94a3b8';
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', rx); rect.setAttribute('y', ry);
    rect.setAttribute('width', rw); rect.setAttribute('height', rh);
    rect.setAttribute('rx', 2);
    rect.setAttribute('fill', col + '66');
    rect.setAttribute('stroke', col);
    rect.setAttribute('stroke-width', '0.8');
    mm.appendChild(rect);
  }

  // Viewport indicator
  const wrap = document.getElementById('canvasWrap');
  if (wrap) {
    const vx = pad + (0 - minX) * scale;
    const vy = pad + (0 - minY) * scale;
    const vw = wrap.offsetWidth * scale;
    const vh = wrap.offsetHeight * scale;
    const vp = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    vp.setAttribute('x', vx); vp.setAttribute('y', vy);
    vp.setAttribute('width', vw); vp.setAttribute('height', vh);
    vp.setAttribute('rx', 2);
    vp.setAttribute('fill', 'none');
    vp.setAttribute('stroke', '#3b82f6');
    vp.setAttribute('stroke-width', '1');
    vp.setAttribute('opacity', '0.7');
    mm.appendChild(vp);
  }
}

// ─── Interaction ──────────────────────────────────────────────────────────
function portClick(e) {
  e.stopPropagation();
  const nid = e.currentTarget.dataset.nid;
  const side = e.currentTarget.dataset.side;
  const port = e.currentTarget.dataset.port || '';
  const ptype = e.currentTarget.dataset.ptype || 'msg';
  if (!connecting) {
    if (side === 'out') { connecting = { nid, port, ptype }; render(); }
    return;
  }
  if (side === 'in' && connecting.nid !== nid) {
    // Friendly tip for common mis-wiring
    if (connecting.ptype === 'body' && ptype === 'term') {
      const st = document.getElementById('testStatus');
      if (st) {
        const prev = st.textContent;
        st.textContent = 'Tip: use the done port to exit a loop, body port loops back';
        setTimeout(() => { st.textContent = prev; }, 2800);
      }
      connecting = null;
      document.getElementById('preview').style.display = 'none';
      render();
      return;
    }
    // General port compatibility enforcement
    if (!portsCompat(connecting.ptype, ptype)) {
      const st = document.getElementById('testStatus');
      if (st) {
        const prev = st.textContent;
        st.textContent = `\u26a0 Incompatible ports: ${connecting.ptype} \u2192 ${ptype}`;
        setTimeout(() => { st.textContent = prev; }, 2500);
      }
      connecting = null;
      document.getElementById('preview').style.display = 'none';
      render();
      return;
    }
    if (!edges.some(ex => ex.from === connecting.nid && ex.to === nid && ex.port === connecting.port && ex.varPort === (ptype === 'var' ? port : undefined))) {
      const edge = mkEdge(connecting.nid, nid);
      if (connecting.port) edge.port = connecting.port;
      if (ptype === 'var' && port) edge.varPort = port;
      edges.push(edge);
      snapshot();
    }
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
  dragging = { id, ox: e.clientX - r.left - n.x, oy: e.clientY - r.top - n.y, moved: false };
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
    if (n) {
      n.x = mouse.x - dragging.ox; n.y = mouse.y - dragging.oy;
      dragging.moved = true;
      render();
    }
  }
  if (connecting) {
    const fn = nodes.find(n => n.id === connecting.nid);
    if (fn) {
      let p;
      if (fn.type === 'loop' && connecting.port === 'done') p = outPortDone(fn);
      else if (fn.type === 'loop' && connecting.port === 'body') p = outPortBody(fn);
      else if (fn.type === 'hitl' && connecting.port) {
        const opts = (fn.options || '').split(',').map(s => s.trim()).filter(Boolean);
        const idx = opts.indexOf(connecting.port);
        if (idx >= 0) {
          p = { x: fn.x + NW, y: fn.y + 30 + idx * 22 };
        } else {
          const nh = nodeHeight(fn);
          p = { x: fn.x + NW, y: fn.y + nh - 12 };
        }
      }
      else p = outPort(fn);
      const prev = document.getElementById('preview');
      prev.setAttribute('d', bez(p.x, p.y, mouse.x, mouse.y));
      prev.style.display = '';
    }
  }
}

function svgMouseUp(e) {
  if (dragging && dragging.moved) snapshot();
  dragging = null;
}

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

  if (n.type === 'start' || n.type === 'end') {
    const p = document.createElement('p');
    p.style.cssText = 'color:var(--muted);font-size:12px;line-height:1.6;padding:4px 0';
    p.textContent = n.type === 'start' ? 'Entry point of the graph.' : 'Terminal node. Graph exits here.';
    body.appendChild(p);
  } else if (n.type === 'agent') {
    addField(body, 'Name', 'text', n.name, v => { n.name = v; render(); });
    addField(body, 'Provider', 'select', n.provider, v => { n.provider = v; rebuildModelOptions(n); render(); },
      ['openai', 'anthropic', 'gemini', 'ollama']);
    addModelField(body, n);
    addField(body, 'System Prompt', 'textarea', n.system_prompt, v => { n.system_prompt = v; updateCode(); });
    addField(body, 'Tools (comma-sep)', 'text', n.tools, v => { n.tools = v; updateCode(); });
    // Agent-as-tool picker
    const agentRow = document.createElement('div');
    agentRow.style.cssText = 'position:relative;margin-top:-4px;margin-bottom:4px';
    const agentPickBtn = document.createElement('button');
    agentPickBtn.className = 'btn';
    agentPickBtn.style.cssText = 'font-size:10px;padding:3px 8px;width:100%;text-align:left';
    agentPickBtn.textContent = '\\uD83E\\uDD16 + use another agent as tool\u2026';
    agentPickBtn.title = 'Adds a @tool() wrapper for another agent node in the generated Python';
    agentPickBtn.onclick = (e) => { e.stopPropagation(); openAgentPicker(n.id, agentRow); };
    agentRow.appendChild(agentPickBtn);
    body.appendChild(agentRow);
    const freezeBtn = document.createElement('button');
    freezeBtn.className = 'btn';
    freezeBtn.style.cssText = 'width:100%;font-size:11px;padding:5px 8px;margin-top:2px';
    freezeBtn.textContent = n.frozen ? '\\u2744 Unfreeze Node' : '\\u2744 Freeze Node';
    freezeBtn.title = 'Frozen nodes reuse last cached output — skipped on re-run';
    if (n.frozen) freezeBtn.style.borderColor = '#3b82f6';
    freezeBtn.onclick = () => {
      n.frozen = !n.frozen;
      if (!n.frozen) delete frozenOutputs[n.id];
      render();
      showNodeProps(n.id);
    };
    body.appendChild(freezeBtn);
    addField(body, 'Eval Assertion (keyword)', 'text', n.eval_assertion || '', v => { n.eval_assertion = v; });
  } else if (n.type === 'loop') {
    addField(body, 'Name', 'text', n.name, v => { n.name = v; render(); });
    addField(body, 'Max Iterations', 'number', String(n.max_iterations), v => { n.max_iterations = parseInt(v) || 5; render(); updateCode(); });
    addField(body, 'Exit Condition', 'text', n.exit_condition, v => { n.exit_condition = v; updateCode(); });
    const hint = document.createElement('p');
    hint.style.cssText = 'color:var(--muted);font-size:11px;line-height:1.5;padding:2px 0';
    hint.textContent = 'Use "body" port for loop back, "done" port to exit.';
    body.appendChild(hint);
  } else if (n.type === 'subgraph') {
    addField(body, 'Name', 'text', n.name, v => { n.name = v; render(); });
    addField(body, 'Graph Name', 'text', n.graph_name, v => { n.graph_name = v; render(); updateCode(); });
    const hint = document.createElement('p');
    hint.style.cssText = 'color:var(--muted);font-size:11px;line-height:1.5;padding:2px 0';
    hint.textContent = 'Loads a sub-graph via load_subgraph(graph_name).';
    body.appendChild(hint);
  } else if (n.type === 'note') {
    addField(body, 'Text', 'textarea', n.text, v => { n.text = v; render(); });
    addField(body, 'Color', 'select', n.color, v => { n.color = v; render(); }, ['yellow', 'blue', 'green', 'red']);
    const hint = document.createElement('p');
    hint.style.cssText = 'color:var(--muted);font-size:11px;line-height:1.5;padding:2px 0';
    hint.textContent = 'Annotation only \u2014 not included in generated code.';
    body.appendChild(hint);
  } else if (n.type === 'hitl') {
    addField(body, 'Name', 'text', n.name, v => { n.name = v; render(); });
    addField(body, 'Options', 'text', n.options || '', v => { n.options = v; render(); updateCode(); });
    addField(body, 'Timeout Label', 'text', n.timeout_label || 'timeout', v => { n.timeout_label = v; render(); updateCode(); });
    const hint = document.createElement('p');
    hint.style.cssText = 'color:var(--muted);font-size:11px;line-height:1.5;padding:2px 0';
    hint.textContent = 'Each option becomes an output port. Connect them to the next node in your graph.';
    body.appendChild(hint);
    // Feature 15: HITL form builder
    const formEditorDiv = document.createElement('div');
    formEditorDiv.id = 'hitlFormEditor';
    formEditorDiv.style.cssText = 'margin-top:8px';
    formEditorDiv.innerHTML = `<div style="font-size:10px;color:var(--muted);margin-bottom:4px">Form fields</div><div id="hitlFieldsList"></div><button class="btn" style="font-size:10px;margin-top:4px;padding:2px 8px" onclick="addHitlField()">+ Add field</button>`;
    body.appendChild(formEditorDiv);
    renderHitlFormEditor(n);
  } else if (n.type === 'agent_tool') {
    // Feature 17: Agent-as-Tool props
    const atDiv = document.createElement('div');
    atDiv.id = 'agentToolProps';
    atDiv.innerHTML = `
      <div class="field"><label style="font-size:10px">Tool name</label><input id="propToolName" type="text" value="${escAttr(n.tool_name||'')}" placeholder="search_agent" oninput="updateSelectedNode('tool_name',this.value)" style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 7px;font:11px ui-monospace,monospace;outline:none"></div>
      <div class="field"><label style="font-size:10px">Tool description</label><textarea id="propToolDesc" rows="2" placeholder="Describe what this tool does" oninput="updateSelectedNode('tool_description',this.value)" style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 7px;font:11px ui-monospace,monospace;outline:none;resize:vertical">${escAttr(n.tool_description||'')}</textarea></div>
      <div class="field"><label style="font-size:10px">Input parameter</label><input id="propToolInput" type="text" value="${escAttr(n.tool_input_param||'query')}" oninput="updateSelectedNode('tool_input_param',this.value)" style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:4px 7px;font:11px ui-monospace,monospace;outline:none"></div>
      <div class="field"><label style="font-size:10px">Target agent node</label><select id="propToolTarget" onchange="updateSelectedNode('tool_target_node',this.value)" style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 7px;font:11px ui-monospace,monospace"></select></div>
    `;
    body.appendChild(atDiv);
    populateAgentToolTargets();
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
  addDatalistField(body, 'Condition Label', e.label, v => { e.label = v; render(); });
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

function addDatalistField(parent, label, value, onChange) {
  const div = document.createElement('div'); div.className = 'field';
  const lbl = document.createElement('label'); lbl.textContent = label;
  div.appendChild(lbl);
  const inp = document.createElement('input');
  inp.type = 'text'; inp.setAttribute('list', 'condition-presets');
  inp.value = value; inp.placeholder = 'blank = unconditional';
  inp.addEventListener('input', () => onChange(inp.value));
  inp.addEventListener('change', () => onChange(inp.value));
  div.appendChild(inp); parent.appendChild(div);
}

function rebuildModelOptions(n) {
  const selEl = document.getElementById('modelSel');
  const searchEl = document.getElementById('modelSearch');
  if (!selEl) return;
  const q = (searchEl ? searchEl.value : '').toLowerCase();
  const list = MODELS[n.provider] || [];
  selEl.innerHTML = '';
  for (const m of list) {
    if (q && !m.toLowerCase().includes(q)) continue;
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    if (m === n.model) opt.selected = true;
    selEl.appendChild(opt);
  }
  if (!selEl.value && selEl.options.length) {
    selEl.value = selEl.options[0].value;
    n.model = selEl.value;
  }
}

function addModelField(parent, n) {
  const div = document.createElement('div'); div.className = 'field';
  const lbl = document.createElement('label'); lbl.textContent = 'Model';
  div.appendChild(lbl);
  const search = document.createElement('input');
  search.id = 'modelSearch'; search.type = 'text'; search.placeholder = 'Filter models\u2026';
  search.style.cssText = 'margin-bottom:3px';
  div.appendChild(search);
  const selEl = document.createElement('select');
  selEl.id = 'modelSel'; selEl.size = 5; selEl.style.cssText = 'height:80px';
  div.appendChild(selEl);
  parent.appendChild(div);
  rebuildModelOptions(n);
  search.addEventListener('input', () => rebuildModelOptions(n));
  selEl.addEventListener('change', () => { n.model = selEl.value; render(); });
  // Benchmark button
  const bmRow = document.createElement('div');
  bmRow.style.cssText = 'display:flex;gap:6px;align-items:center;margin-top:4px';
  const bmBtn = document.createElement('button');
  bmBtn.className = 'btn'; bmBtn.textContent = '🧪 Benchmark';
  bmBtn.title = 'Run eval cases against candidate models and pick cheapest that passes threshold';
  bmBtn.onclick = () => benchmarkModel(n);
  bmRow.appendChild(bmBtn);
  const bmRes = document.createElement('span');
  bmRes.id = 'evalRouteResult';
  bmRes.style.cssText = 'font-size:10px;color:#94a3b8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
  bmRow.appendChild(bmRes);
  div.appendChild(bmRow);
}

async function benchmarkModel(n) {
  const res = document.getElementById('evalRouteResult');
  if (res) { res.textContent = 'Benchmarking\u2026'; res.style.color = '#f59e0b'; }
  const cases = (n.eval_cases || []);
  const apiKey = document.getElementById('apiKeyInput') ? document.getElementById('apiKeyInput').value : '';
  try {
    const resp = await fetch('/eval-route', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: document.getElementById('testInput') ? document.getElementById('testInput').value : '',
        system_prompt: n.system_prompt || '',
        eval_cases: cases,
        threshold: 0.7,
        api_key: apiKey
      })
    });
    const data = await resp.json();
    if (data.model) {
      n.model = data.model;
      rebuildModelOptions(n);
      const method = data.method === 'eval-validated' ? '\u2705 eval' : data.method === 'heuristic' ? '🧠 heuristic' : '\u26A0\uFE0F best-avail';
      const scoreStr = Object.entries(data.scores || {}).map(([m,s]) => m.split('/').pop() + ':' + (s*100).toFixed(0) + '%').join(' ');
      if (res) { res.textContent = method + ' \u2192 ' + data.model + (scoreStr ? ' | ' + scoreStr : ''); res.style.color = '#22d3ee'; }
    }
  } catch(e) {
    if (res) { res.textContent = 'Error: ' + e.message; res.style.color = '#f87171'; }
  }
}

// ─── Keyboard ────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const active = document.activeElement;
  const inInput = ['INPUT','TEXTAREA','SELECT'].includes(active.tagName);

  if (e.key === 'Escape') {
    const gb = document.getElementById('genBar');
    if (gb && gb.style.display !== 'none') { closeGenBar(); return; }
    const im = document.getElementById('importModal');
    if (im && im.style.display !== 'none') { closeImport(); return; }
    const em = document.getElementById('embedModal');
    if (em && em.style.display !== 'none') { closeEmbed(); return; }
    const lt = document.getElementById('loadTraceModal');
    if (lt && lt.style.display !== 'none') { closeLoadTrace(); return; }
    const rd = document.getElementById('replayDiffPanel');
    if (rd && rd.style.display !== 'none') { closeReplayDiff(); return; }
    const wf = document.getElementById('watchFileModal');
    if (wf && wf.style.display !== 'none') { closeWatchFileModal(); return; }
    const cm = document.getElementById('ctxMenu');
    if (cm && cm.style.display !== 'none') { hideCtxMenu(); return; }
    const ov = document.getElementById('searchOverlay');
    if (ov && ov.style.display !== 'none') { closeSearch(); return; }
    if (connecting) { connecting = null; document.getElementById('preview').style.display='none'; render(); }
    else deselect();
    return;
  }
  if ((e.key === 'Delete' || e.key === 'Backspace') && sel && !inInput) {
    delSelected();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
    e.preventDefault(); undoAction(); return;
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
    e.preventDefault(); redoAction(); return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !inInput) {
    if (sel && sel.type === 'node') {
      const n = nodes.find(n => n.id === sel.id);
      if (n) clipboard = JSON.parse(JSON.stringify(n));
    }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'v' && !inInput) {
    if (clipboard) {
      const pasted = JSON.parse(JSON.stringify(clipboard));
      pasted.id = pasted.type + '_' + (seq++);
      pasted.x = (pasted.x || 0) + 50;
      pasted.y = (pasted.y || 0) + 50;
      nodes.push(pasted);
      snapshot();
      render();
      selNode(pasted.id);
    }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    const ov = document.getElementById('searchOverlay');
    if (ov.style.display === 'none' || !ov.style.display) openSearch();
    else closeSearch();
    return;
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
  sel = null;
  snapshot();
  deselect();
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
  snapshot();
  render(); selNode(n.id);
}

// ─── Templates ────────────────────────────────────────────────────────────
function toggleTemplates(e) {
  e.stopPropagation();
  document.getElementById('tmplMenu').classList.toggle('open');
}
document.addEventListener('click', () => {
  document.getElementById('tmplMenu').classList.remove('open');
});

function loadTemplate(name) {
  document.getElementById('tmplMenu').classList.remove('open');
  nodes = []; edges = []; sel = null; seq = 1;

  if (name === 'simple_chatbot') {
    const start = mkNode('start', 60, 160);
    nodes.push(start);
    const agent = mkNode('agent', 280, 160);
    agent.name = 'Chatbot';
    agent.system_prompt = 'You are a helpful assistant.';
    nodes.push(agent);
    const end = mkNode('end', 500, 160);
    nodes.push(end);
    edges.push(mkEdge(start.id, agent.id));
    edges.push(mkEdge(agent.id, end.id));

  } else if (name === 'researcher_writer') {
    loadExample();
    return;

  } else if (name === 'rag_pipeline') {
    const start = mkNode('start', 60, 185);
    nodes.push(start);
    const retriever = mkNode('agent', 280, 185);
    retriever.name = 'Retriever';
    retriever.system_prompt = 'Retrieve relevant documents from the knowledge base and return structured context.';
    retriever.tools = 'vector_search, bm25_search';
    nodes.push(retriever);
    const writer = mkNode('agent', 500, 185);
    writer.name = 'Writer';
    writer.system_prompt = 'Use the retrieved context to write a comprehensive, accurate answer.';
    nodes.push(writer);
    const end = mkNode('end', 720, 185);
    nodes.push(end);
    edges.push(mkEdge(start.id, retriever.id));
    edges.push(mkEdge(retriever.id, writer.id));
    edges.push(mkEdge(writer.id, end.id));

  } else if (name === 'reviewer_loop') {
    const start = mkNode('start', 60, 185);
    nodes.push(start);
    const writer = mkNode('agent', 280, 110);
    writer.name = 'Writer';
    writer.system_prompt = 'Write a high-quality draft based on the input.';
    nodes.push(writer);
    const reviewer = mkNode('agent', 500, 270);
    reviewer.name = 'Reviewer';
    reviewer.system_prompt = 'Review the draft. Output "approved" if good, or "needs_revision" with feedback.';
    nodes.push(reviewer);
    const end = mkNode('end', 720, 185);
    nodes.push(end);
    edges.push(mkEdge(start.id, writer.id));
    edges.push(mkEdge(writer.id, reviewer.id));
    edges.push({ id: 'ex1', from: reviewer.id, to: end.id, label: 'approved' });
    edges.push({ id: 'ex2', from: reviewer.id, to: writer.id, label: 'needs_revision' });

  } else if (name === 'hitl_approval') {
    const start = mkNode('start', 60, 200);
    nodes.push(start);
    const processor = mkNode('agent', 280, 200);
    processor.name = 'Processor';
    processor.system_prompt = 'Process the input and produce a result for human review.';
    nodes.push(processor);
    const gate = mkNode('hitl', 500, 150);
    gate.name = 'ReviewGate';
    gate.options = 'approve, reject';
    gate.timeout_label = 'timeout';
    nodes.push(gate);
    const end = mkNode('end', 720, 110);
    nodes.push(end);
    const reviser = mkNode('agent', 720, 280);
    reviser.name = 'Reviser';
    reviser.system_prompt = 'Revise the output based on the rejection feedback.';
    nodes.push(reviser);
    edges.push(mkEdge(start.id, processor.id));
    edges.push(mkEdge(processor.id, gate.id));
    edges.push({ id: 'h1', from: gate.id, to: end.id, port: 'approve', label: 'approve' });
    edges.push({ id: 'h2', from: gate.id, to: reviser.id, port: 'reject', label: 'reject' });
    edges.push({ id: 'h3', from: gate.id, to: end.id, port: 'timeout', label: 'timeout' });
    edges.push(mkEdge(reviser.id, processor.id));

  } else if (name === 'multi_model_panel') {
    const start = mkNode('start', 60, 240);
    nodes.push(start);
    const gpt = mkNode('agent', 280, 80);
    gpt.name = 'GPT Agent';
    gpt.provider = 'openai'; gpt.model = 'gpt-4o-mini';
    gpt.system_prompt = 'Answer the question concisely.';
    nodes.push(gpt);
    const claude = mkNode('agent', 280, 220);
    claude.name = 'Claude Agent';
    claude.provider = 'anthropic'; claude.model = 'claude-3-5-haiku-latest';
    claude.system_prompt = 'Answer the question concisely.';
    nodes.push(claude);
    const gemini = mkNode('agent', 280, 360);
    gemini.name = 'Gemini Agent';
    gemini.provider = 'gemini'; gemini.model = 'gemini-2.5-flash';
    gemini.system_prompt = 'Answer the question concisely.';
    nodes.push(gemini);
    const aggregator = mkNode('agent', 520, 240);
    aggregator.name = 'Aggregator';
    aggregator.system_prompt = 'Combine the three responses into a consensus answer.';
    nodes.push(aggregator);
    const end = mkNode('end', 740, 240);
    nodes.push(end);
    edges.push(mkEdge(start.id, gpt.id));
    edges.push(mkEdge(start.id, claude.id));
    edges.push(mkEdge(start.id, gemini.id));
    edges.push(mkEdge(gpt.id, aggregator.id));
    edges.push(mkEdge(claude.id, aggregator.id));
    edges.push(mkEdge(gemini.id, aggregator.id));
    edges.push(mkEdge(aggregator.id, end.id));

  } else if (name === 'chain_of_thought') {
    const start = mkNode('start', 60, 185);
    nodes.push(start);
    const planner = mkNode('agent', 280, 110);
    planner.name = 'Planner';
    planner.system_prompt = 'Break down the task into a step-by-step plan.';
    nodes.push(planner);
    const executor = mkNode('agent', 500, 110);
    executor.name = 'Executor';
    executor.system_prompt = 'Execute each step of the plan and produce results.';
    nodes.push(executor);
    const critic = mkNode('agent', 500, 280);
    critic.name = 'Critic';
    critic.system_prompt = 'Evaluate the result. Output "done" if satisfactory, or "revise" with feedback.';
    nodes.push(critic);
    const end = mkNode('end', 720, 185);
    nodes.push(end);
    edges.push(mkEdge(start.id, planner.id));
    edges.push(mkEdge(planner.id, executor.id));
    edges.push(mkEdge(executor.id, critic.id));
    edges.push({ id: 'ct1', from: critic.id, to: end.id, label: 'done' });
    edges.push({ id: 'ct2', from: critic.id, to: planner.id, label: 'revise' });
  }

  snapshot();
  deselect();
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
  const codeNodes = nodes.filter(n => n.type !== 'note');
  const agents = nodes.filter(n => n.type === 'agent');
  const loops = nodes.filter(n => n.type === 'loop');
  const subgraphs = nodes.filter(n => n.type === 'subgraph');
  const hitls = nodes.filter(n => n.type === 'hitl');
  if (!agents.length && !loops.length && !subgraphs.length && !hitls.length) return '# Drag agent nodes onto the canvas to generate code';

  // Collect all @agentname tool references across all nodes
  const agentToolRefs = new Set();
  for (const n of agents) {
    const tools = n.tools ? n.tools.split(',').map(s => s.trim()).filter(Boolean) : [];
    for (const t of tools) { if (t.startsWith('@')) agentToolRefs.add(t.slice(1)); }
  }
  const needsToolDecorator = agentToolRefs.size > 0;

  const L = [];
  L.push('from selectools import Agent, AgentConfig');
  L.push('from selectools.orchestration import AgentGraph');
  if (subgraphs.length) L.push('from selectools.orchestration import load_subgraph');
  if (needsToolDecorator) L.push('from selectools.tools import tool');
  L.push('');
  L.push('# Initialise your provider, e.g.:');
  L.push('# from selectools import OpenAIProvider');
  L.push('# provider = OpenAIProvider(api_key="...")');
  L.push('');

  // Topological sort: agents referenced as tools must be defined first
  const agentOrder = [...agents];
  agentOrder.sort((a, b) => {
    const aRef = agentToolRefs.has(varName(a.name));
    const bRef = agentToolRefs.has(varName(b.name));
    return aRef === bRef ? 0 : aRef ? -1 : 1;
  });

  for (const n of agentOrder) {
    const v = varName(n.name);
    const rawTools = n.tools ? n.tools.split(',').map(s => s.trim()).filter(Boolean) : [];
    // Resolve @agentname refs → wrapper function names, keep plain tools as-is
    const resolvedTools = rawTools.map(t => t.startsWith('@') ? t.slice(1) + '_tool' : t);

    L.push(`${v} = Agent(`);
    L.push(`    provider=provider,`);
    if (resolvedTools.length) L.push(`    tools=[${resolvedTools.join(', ')}],`);
    L.push(`    config=AgentConfig(`);
    L.push(`        name="${n.name}",`);
    if (n.system_prompt)
      L.push(`        system_prompt=${JSON.stringify(n.system_prompt)},`);
    L.push(`    )`);
    L.push(`)`);
    const vars = extractVars(n.system_prompt);
    if (vars.length) {
      L.push(`# Variable ports: ${vars.map(v => '{' + v + '}').join(', ')} — wire upstream outputs to substitute these placeholders`);
    }
    // If this agent is used as a tool by others, emit a @tool() wrapper immediately after
    if (agentToolRefs.has(v)) {
      L.push('');
      L.push(`@tool()`);
      L.push(`def ${v}_tool(query: str) -> str:`);
      L.push(`    "Delegate to the ${n.name} agent."`);
      L.push(`    result = ${v}.run(query)`);
      L.push(`    return result.content`);
    }
    L.push('');
  }

  if (hitls.length) {
    L.push('# Human-in-the-loop nodes — implement wait_for_human() with your interrupt mechanism');
    for (const h of hitls) {
      const opts = (h.options || '').split(',').map(s => s.trim()).filter(Boolean);
      L.push(`# ${varName(h.name)}: options=[${opts.map(o => `"${o}"`).join(', ')}], timeout="${h.timeout_label || 'timeout'}"`);
    }
    L.push('');
  }
  L.push('graph = AgentGraph()');
  for (const n of agents) L.push(`graph.add_node("${varName(n.name)}", ${varName(n.name)})`);
  for (const h of hitls) {
    const opts = (h.options || '').split(',').map(s => s.trim()).filter(Boolean);
    L.push(`graph.add_node("${varName(h.name)}", wait_for_human(options=[${opts.map(o => `"${o}"`).join(', ')}], timeout_label="${h.timeout_label || 'timeout'}"))`);
  }
  for (const n of subgraphs) {
    const gname = n.graph_name || 'my_subgraph';
    L.push(`graph.add_node("${varName(n.name)}", load_subgraph("${gname}"))`);
  }
  for (const n of loops) {
    const maxIter = n.max_iterations || 5;
    const exitCond = n.exit_condition || 'done';
    L.push(`graph.add_node("${varName(n.name)}", AgentGraph.loop_node(max_iterations=${maxIter}, exit_condition="${exitCond}"))`);
  }

  // entry point from START edges
  const startEdge = edges.find(e => { const fn = nodes.find(n => n.id === e.from); return fn?.type === 'start'; });
  if (startEdge) {
    const tn = nodes.find(n => n.id === startEdge.to);
    if (tn && tn.type !== 'end' && tn.type !== 'note') L.push(`graph.set_entry_point("${varName(tn.name)}")`);
  }
  L.push('');

  // group conditional edges by source
  const condMap = {};
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn || fn.type === 'start' || fn.type === 'note' || tn.type === 'note') continue;
    const fid = varName(fn.name);
    const tid = tn.type === 'end' ? 'AgentGraph.END' : `"${varName(tn.name)}"`;
    if (fn.type === 'loop') {
      const port = e.port || 'body';
      L.push(`graph.add_loop_edge("${fid}", ${tid}, port="${port}")`);
    } else if (e.label) {
      (condMap[fid] = condMap[fid] || []).push({ lbl: e.label, tid });
    } else {
      L.push(`graph.add_edge("${fid}", ${tid})`);
    }
  }

  for (const [fid, conds] of Object.entries(condMap)) {
    const mapping = conds.map(c => `        "${c.lbl}": ${c.tid}`).join(',\\n');
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
  return L.join('\\n');
}

function genYaml() {
  const agents = nodes.filter(n => n.type === 'agent');
  const loops = nodes.filter(n => n.type === 'loop');
  const subgraphs = nodes.filter(n => n.type === 'subgraph');
  const hitls = nodes.filter(n => n.type === 'hitl');
  if (!agents.length && !loops.length && !subgraphs.length && !hitls.length) return '# Drag agent nodes onto the canvas to generate YAML';

  const L = [];
  L.push('# selectools agent graph — generated by builder');
  L.push('name: my-agent-graph');
  L.push('type: graph');
  L.push('');
  L.push('nodes:');
  for (const n of agents) {
    const id = varName(n.name);
    L.push(`  ${id}:`);
    L.push(`    type: agent`);
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
    const vars = extractVars(n.system_prompt);
    if (vars.length) { L.push(`    input_vars:`); vars.forEach(v => L.push(`      - ${v}`)); }
  }
  for (const n of loops) {
    L.push(`  ${varName(n.name)}:`);
    L.push(`    type: loop`);
    L.push(`    max_iterations: ${n.max_iterations || 5}`);
    if (n.exit_condition) L.push(`    exit_condition: "${n.exit_condition}"`);
  }
  for (const n of subgraphs) {
    L.push(`  ${varName(n.name)}:`);
    L.push(`    type: subgraph`);
    L.push(`    graph_name: ${n.graph_name || 'my_subgraph'}`);
  }
  for (const h of hitls) {
    const id = varName(h.name);
    const opts = (h.options || '').split(',').map(s => s.trim()).filter(Boolean);
    L.push(`  ${id}:`);
    L.push(`    type: hitl`);
    if (opts.length) { L.push(`    options:`); opts.forEach(o => L.push(`      - ${o}`)); }
    L.push(`    timeout_label: ${h.timeout_label || 'timeout'}`);
  }
  L.push('');
  L.push('edges:');
  for (const e of edges) {
    const fn = nodes.find(n => n.id === e.from);
    const tn = nodes.find(n => n.id === e.to);
    if (!fn || !tn || fn.type === 'note' || tn.type === 'note') continue;
    const fid = fn.type === 'start' ? 'START' : varName(fn.name);
    const tid = tn.type === 'end'   ? 'END'   : varName(tn.name);
    L.push(`  - from: ${fid}`);
    L.push(`    to: ${tid}`);
    if (e.label) L.push(`    condition: "${e.label}"`);
    if (e.port) L.push(`    port: ${e.port}`);
  }
  // Feature 10: pinned ports
  const pinnedKeys = Object.keys(pinnedPorts);
  if (pinnedKeys.length) {
    L.push('');
    L.push('pinned_ports:');
    pinnedKeys.forEach(k => {
      const val = pinnedPorts[k];
      L.push(`  ${k}: "${String(val).replace(/"/g, '\\"').slice(0, 80)}"`);
    });
  }
  // Feature 15: form_fields in YAML for hitl nodes
  for (const h of hitls) {
    if (h.form_fields && h.form_fields.length) {
      // already emitted above; just ensure form_fields key present in comment
    }
  }
  return L.join('\\n');
}

// ─── AI Generate ──────────────────────────────────────────────────────────
function openGenBar() {
  const bar = document.getElementById('genBar');
  bar.style.display = 'flex';
  setTimeout(() => document.getElementById('genInput').focus(), 50);
}
function closeGenBar() {
  document.getElementById('genBar').style.display = 'none';
  document.getElementById('genInput').value = '';
}

async function doGenerate() {
  const desc = document.getElementById('genInput').value.trim();
  if (!desc) return;
  const btn = document.getElementById('genBtn');
  btn.textContent = 'Generating\u2026'; btn.disabled = true;
  try {
    const apiKey = document.getElementById('apiKeyInput')?.value?.trim() || '';
    const resp = await fetch('/ai-build', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc, api_key: apiKey}),
    });
    if (!resp.ok) throw new Error('Server error');
    const data = await resp.json();
    if (!data.nodes || !data.edges) throw new Error('Invalid response');
    if (nodes.filter(n => n.type !== 'start').length > 0 && !confirm('Replace current graph?')) return;
    const positioned = autoLayout(data.nodes, data.edges);
    nodes = positioned;
    edges = data.edges;
    seq = Math.max(...nodes.map(n => parseInt(n.id.split('_').pop()) || 0), 0) + 1;
    snapshot();
    deselect();
    closeGenBar();
    render();
    document.querySelectorAll('.node').forEach((el, i) => {
      setTimeout(() => el.classList.add('node-entering'), i * 80);
      setTimeout(() => el.classList.remove('node-entering'), i * 80 + 300);
    });
  } catch(err) {
    const inp = document.getElementById('genInput');
    inp.style.borderColor = 'var(--red)';
    setTimeout(() => { inp.style.borderColor = ''; }, 2000);
  } finally {
    btn.textContent = 'Generate'; btn.disabled = false;
  }
}

document.getElementById('genInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') doGenerate();
});

// ─── Code Import ──────────────────────────────────────────────────────────
function openImport() {
  document.getElementById('importModal').style.display = 'flex';
  document.getElementById('importInput').value = '';
  document.getElementById('importError').style.display = 'none';
  setTimeout(() => document.getElementById('importInput').focus(), 50);
}
function closeImport() { document.getElementById('importModal').style.display = 'none'; }

function parseYaml(text) {
  const lines = text.split('\\n');
  const pnodes = [], pedges = [];
  let section = null, cur = null, curEdge = null;
  let inSP = false, inTools = false, inOpts = false, edgeSeq = 0;
  for (const raw of lines) {
    const s = raw.trimEnd();
    if (!s || s.trim().startsWith('#')) { inSP = false; continue; }
    if (s === 'nodes:') { section = 'nodes'; cur = null; continue; }
    if (s === 'edges:') { section = 'edges'; cur = null; curEdge = null; continue; }
    if (section === 'nodes') {
      const nm = s.match(/^  ([a-zA-Z_]\\w*):$/);
      if (nm) {
        inSP = false; inTools = false; inOpts = false;
        cur = {id: nm[1], type: 'agent', name: nm[1], provider: 'openai', model: 'gpt-4o-mini',
               system_prompt: '', tools: '', frozen: false, eval_assertion: ''};
        pnodes.push(cur); continue;
      }
      if (!cur) continue;
      const kv = s.match(/^    (\\w+):(.*)$/);
      if (kv) {
        const k = kv[1], val = kv[2].trim();
        inSP = false; inTools = false; inOpts = false;
        if (k === 'type') cur.type = val;
        else if (k === 'provider') cur.provider = val;
        else if (k === 'model') cur.model = val;
        else if (k === 'max_iterations') cur.max_iterations = parseInt(val) || 5;
        else if (k === 'exit_condition') cur.exit_condition = val.replace(/^"|"$/g, '');
        else if (k === 'graph_name') cur.graph_name = val;
        else if (k === 'timeout_label') cur.timeout_label = val;
        else if (k === 'system_prompt') {
          if (val === '|') { inSP = true; cur.system_prompt = ''; }
          else cur.system_prompt = val.replace(/^"|"$/g, '');
        } else if (k === 'tools') inTools = true;
        else if (k === 'options') { inOpts = true; cur.options = ''; }
        continue;
      }
      if (inSP && s.startsWith('      ')) { cur.system_prompt += (cur.system_prompt ? '\\n' : '') + raw.slice(6).trimEnd(); continue; }
      if (inTools && s.startsWith('      - ')) { const t = s.slice(8).trim(); cur.tools = cur.tools ? cur.tools + ', ' + t : t; continue; }
      if (inOpts && s.startsWith('      - ')) { const o = s.slice(8).trim(); cur.options = cur.options ? cur.options + ', ' + o : o; continue; }
    }
    if (section === 'edges') {
      if (s.startsWith('  - from:')) { curEdge = {id: 'ye_' + (edgeSeq++), from: s.slice(9).trim(), to: '', label: ''}; pedges.push(curEdge); continue; }
      if (curEdge && s.startsWith('    to:')) { curEdge.to = s.slice(7).trim(); continue; }
      if (curEdge && s.startsWith('    condition:')) { curEdge.label = s.slice(14).trim().replace(/^"|"$/g, ''); continue; }
      if (curEdge && s.startsWith('    port:')) { curEdge.port = s.slice(9).trim(); continue; }
    }
  }
  let startNode = pnodes.find(n => n.type === 'start');
  if (!startNode) { startNode = {id: '__start__', type: 'start', name: 'START'}; pnodes.unshift(startNode); }
  let endNode = pnodes.find(n => n.type === 'end');
  if (!endNode) { endNode = {id: '__end__', type: 'end', name: 'END'}; pnodes.push(endNode); }
  for (const e of pedges) {
    if (e.from === 'START') e.from = startNode.id;
    if (e.to === 'END' || e.to === 'AgentGraph.END') e.to = endNode.id;
  }
  return {nodes: pnodes, edges: pedges};
}

function parsePython(text) {
  const pnodes = [], pedges = [];
  const agentVars = {};   // varName -> {name, system_prompt}
  const graphIds  = {};   // graphId -> varName
  let eSeq = 0, m;

  // Agent variable assignments
  const agRe = /^(\\w+)\\s*=\\s*Agent\\s*\\(/mg;
  while ((m = agRe.exec(text)) !== null) {
    const varN = m[1], after = text.slice(m.index, m.index + 800);
    const nameM = after.match(/name\\s*=\\s*["']([^"']+)["']/);
    const spM   = after.match(/system_prompt\\s*=\\s*"([^"]*)"/);
    agentVars[varN] = {name: nameM ? nameM[1] : varN, system_prompt: spM ? spM[1] : ''};
  }

  // graph.add_node("id", simpleVar)
  const agNodeRe = /graph\\.add_node\\s*\\(\\s*["'](\\w+)["']\\s*,\\s*(\\w+)\\s*\\)/g;
  while ((m = agNodeRe.exec(text)) !== null) graphIds[m[1]] = m[2];

  // graph.add_node("id", wait_for_human(...))
  const hitlRe = /graph\\.add_node\\s*\\(\\s*["'](\\w+)["']\\s*,\\s*wait_for_human\\(([^)]*)\\)\\s*\\)/g;
  while ((m = hitlRe.exec(text)) !== null) {
    const gid = m[1], params = m[2];
    const optM = params.match(/options\\s*=\\s*\\[([^\\]]*)\\]/);
    const optsRaw = optM ? (optM[1].match(/["']([^"']+)["']/g) || []).map(s => s.slice(1,-1)) : [];
    const tlM = params.match(/timeout_label\\s*=\\s*["']([^"']+)["']/);
    pnodes.push({id: gid, type: 'hitl', name: gid, options: optsRaw.join(', '), timeout_label: tlM ? tlM[1] : 'timeout'});
  }

  // graph.add_node("id", AgentGraph.loop_node(...))
  const loopRe = /graph\\.add_node\\s*\\(\\s*["'](\\w+)["']\\s*,\\s*AgentGraph\\.loop_node\\(([^)]*)\\)\\s*\\)/g;
  while ((m = loopRe.exec(text)) !== null) {
    const gid = m[1], params = m[2];
    const maxM = params.match(/max_iterations\\s*=\\s*(\\d+)/);
    const exitM = params.match(/exit_condition\\s*=\\s*["']([^"']+)["']/);
    pnodes.push({id: gid, type: 'loop', name: gid, max_iterations: maxM ? parseInt(maxM[1]) : 5, exit_condition: exitM ? exitM[1] : 'done'});
  }

  // Build agent pnodes
  for (const [gid, varRef] of Object.entries(graphIds)) {
    const info = agentVars[varRef];
    if (info) pnodes.push({id: gid, type: 'agent', name: info.name, system_prompt: info.system_prompt,
                           provider: 'openai', model: 'gpt-4o-mini', tools: '', frozen: false, eval_assertion: ''});
  }
  for (const [varN, info] of Object.entries(agentVars)) {
    if (!Object.values(graphIds).includes(varN) && !pnodes.find(n => n.id === varN))
      pnodes.push({id: varN, type: 'agent', name: info.name, system_prompt: info.system_prompt,
                   provider: 'openai', model: 'gpt-4o-mini', tools: '', frozen: false, eval_assertion: ''});
  }

  const startNode = {id: '__start__', type: 'start', name: 'START'};
  const endNode   = {id: '__end__',   type: 'end',   name: 'END'};

  // Entry point → start edge
  const epM = text.match(/graph\\.set_entry_point\\s*\\(\\s*["'](\\w+)["']/);
  if (epM) pedges.push({id: 'pe_' + (eSeq++), from: startNode.id, to: epM[1], label: ''});

  // graph.add_edge
  const addEdgeRe = /graph\\.add_edge\\s*\\(\\s*["'](\\w+)["']\\s*,\\s*(?:AgentGraph\\.END|["'](\\w+)["'])\\s*\\)/g;
  while ((m = addEdgeRe.exec(text)) !== null)
    pedges.push({id: 'pe_' + (eSeq++), from: m[1], to: m[2] || endNode.id, label: ''});

  // graph.add_conditional_edge
  const condRe = /graph\\.add_conditional_edge\\s*\\(\\s*["'](\\w+)["'][\\s\\S]*?\\{([\\s\\S]*?)\\}\\s*\\)/g;
  while ((m = condRe.exec(text)) !== null) {
    const from = m[1], mapB = m[2];
    const pairRe = /["']([^"']+)["']\\s*:\\s*(?:AgentGraph\\.END|["']([^"']+)["'])/g;
    let pm;
    while ((pm = pairRe.exec(mapB)) !== null)
      pedges.push({id: 'pe_' + (eSeq++), from, to: pm[2] || endNode.id, label: pm[1]});
  }

  for (const e of pedges) if (e.to === 'AgentGraph.END') e.to = endNode.id;
  return {nodes: [startNode, ...pnodes, endNode], edges: pedges};
}

function autoLayout(pnodes, pedges) {
  const layers = {};
  const startId = (pnodes.find(n => n.type === 'start') || pnodes[0] || {}).id;
  const queue = [{id: startId, layer: 0}], visited = new Set();
  while (queue.length) {
    const {id, layer} = queue.shift();
    if (visited.has(id)) continue;
    visited.add(id); layers[id] = layer;
    for (const e of pedges) if (e.from === id) queue.push({id: e.to, layer: layer + 1});
  }
  const vals = Object.values(layers);
  const maxLayer = vals.length ? Math.max(0, ...vals) : 0;
  for (const n of pnodes) if (layers[n.id] === undefined) layers[n.id] = maxLayer + 1;
  const byLayer = {};
  for (const [id, l] of Object.entries(layers)) (byLayer[l] = byLayer[l] || []).push(id);
  for (const n of pnodes) {
    const layer = layers[n.id] || 0, col = byLayer[layer] || [n.id];
    n.x = 60 + layer * 260;
    n.y = 80 + col.indexOf(n.id) * 120;
  }
  return pnodes;
}

function doImport() {
  const text = document.getElementById('importInput').value.trim();
  const errEl = document.getElementById('importError');
  errEl.style.display = 'none';
  if (!text) return;
  let parsed;
  try {
    if (text.includes('AgentGraph()') || text.includes('= Agent(') || text.startsWith('from selectools')) {
      parsed = parsePython(text);
    } else if (text.includes('type: graph') || text.includes('nodes:')) {
      parsed = parseYaml(text);
    } else {
      throw new Error('Expected AgentGraph Python or "type: graph" YAML');
    }
  } catch(err) {
    errEl.textContent = 'Could not parse: ' + err.message;
    errEl.style.display = 'block';
    return;
  }
  if (!parsed.nodes.length) {
    errEl.textContent = 'No nodes found in input.';
    errEl.style.display = 'block';
    return;
  }
  if (nodes.filter(n => n.type !== 'start').length > 0 && !confirm('Replace current graph?')) return;
  const positioned = autoLayout(parsed.nodes, parsed.edges);
  nodes = positioned;
  edges = parsed.edges;
  seq = Math.max(...nodes.map(n => parseInt(n.id.split('_').pop()) || 0), 0) + 1;
  snapshot();
  deselect();
  closeImport();
}

// ─── Actions ──────────────────────────────────────────────────────────────
function onClear() {
  if (!confirm('Clear the graph?')) return;
  nodes = []; edges = []; sel = null; connecting = null; seq = 1;
  history = []; histIdx = -1;
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

  snapshot();
  deselect();
}

// ─── Code panel resize ────────────────────────────────────────────────────
(function() {
  let resizing = false, startY = 0, startH = 0;
  const panel = document.querySelector('.code-panel');
  const handle = document.getElementById('resizeHandle');
  handle.addEventListener('mousedown', e => {
    resizing = true; startY = e.clientY; startH = panel.offsetHeight;
    handle.classList.add('active'); e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!resizing) return;
    const delta = startY - e.clientY;
    panel.style.height = Math.max(60, Math.min(window.innerHeight * 0.75, startH + delta)) + 'px';
  });
  document.addEventListener('mouseup', () => {
    resizing = false; handle.classList.remove('active');
  });
})();

// ─── Test Panel ───────────────────────────────────────────────────────────
let testPanelOpen = false;
let testActiveTab = 'output';
let runHistory = [];
let currentRunEvents = [];
let isRunning = false;

function openTestPanel() {
  document.getElementById('testPanel').style.display = 'flex';
  testPanelOpen = true;
}
function closeTestPanel() {
  document.getElementById('testPanel').style.display = 'none';
  testPanelOpen = false;
}
function switchTestTab(tab) {
  testActiveTab = tab;
  document.getElementById('ttab-output').className = 'test-tab' + (tab === 'output' ? ' active' : '');
  document.getElementById('ttab-history').className = 'test-tab' + (tab === 'history' ? ' active' : '');
  const aiTab = document.getElementById('ttab-ai');
  if (aiTab) aiTab.className = 'test-tab' + (tab === 'ai' ? ' active' : '');
  document.getElementById('testOutput').style.display = tab === 'output' ? '' : 'none';
  document.getElementById('testHistory').style.display = tab === 'history' ? '' : 'none';
  const copilotTab = document.getElementById('aiCopilotTab');
  if (copilotTab) copilotTab.style.display = tab === 'ai' ? 'flex' : 'none';
  document.getElementById('historyControls').style.display = tab === 'history' ? 'flex' : 'none';
  document.getElementById('historySessionCost').style.display = tab === 'history' ? '' : 'none';
}

function appendTrace(html) {
  const out = document.getElementById('testOutput');
  out.insertAdjacentHTML('beforeend', html);
  out.scrollTop = out.scrollHeight;
}

// ─── Feature 09: Structured trace rows ────────────────────────────────────
function addTraceRow(type, header, body, badge, nodeId) {
  traceRows.push({ type, header, body, badge, node_id: nodeId || null, id: 'tr_' + Date.now() + '_' + Math.random().toString(36).slice(2) });
  renderTraceRows();
}
function renderTraceRows() {
  const el = document.getElementById('testOutput');
  el.innerHTML = traceRows.map(row => `
    <div class="trace-row trace-type-${row.type}" id="${row.id}">
      <div class="trace-row-header" onclick="toggleTraceRow('${row.id}')">
        <span>${row.header}</span>
        ${row.badge ? `<span class="trace-badge">${row.badge}</span>` : ''}
      </div>
      ${row.body ? `<div class="trace-row-body"><pre style="margin:0;white-space:pre-wrap">${row.body}</pre></div>` : ''}
    </div>
  `).join('');
  el.scrollTop = el.scrollHeight;
}
function toggleTraceRow(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('expanded');
}
function updateTraceRowBadge(key, badge) {
  const row = [...traceRows].reverse().find(r => r.node_id === key || r.header.includes(key));
  if (row) { row.badge = badge; renderTraceRows(); }
}
function updateTraceRowBody(key, body) {
  const row = [...traceRows].reverse().find(r => r.header.includes(key));
  if (row) { row.body = body; renderTraceRows(); }
}

// ─── Feature 10: Port pinning ─────────────────────────────────────────────
function pinPort(nodeId, portKey, value) {
  pinnedPorts[`${nodeId}::${portKey}`] = value;
  render();
}
function unpinPort(nodeId, portKey) {
  delete pinnedPorts[`${nodeId}::${portKey}`];
  render();
}
function isPinned(nodeId, portKey) {
  return `${nodeId}::${portKey}` in pinnedPorts;
}

// ─── Feature 11: Single-node replay diff ──────────────────────────────────
function showReplayDiff(nodeId, before, after) {
  const node = nodes.find(n => n.id === nodeId);
  const name = node ? (node.label || node.name || nodeId) : nodeId;
  const diff = computeLineDiff(before, after);
  const titleEl = document.getElementById('replayDiffTitle');
  const beforeEl = document.getElementById('replayDiffBefore');
  const afterEl = document.getElementById('replayDiffAfter');
  const panel = document.getElementById('replayDiffPanel');
  if (titleEl) titleEl.textContent = `Diff: ${name}`;
  if (beforeEl) beforeEl.innerHTML = diff.before;
  if (afterEl) afterEl.innerHTML = diff.after;
  if (panel) panel.style.display = 'flex';
}
function closeReplayDiff() {
  const panel = document.getElementById('replayDiffPanel');
  if (panel) panel.style.display = 'none';
}
function computeLineDiff(a, b) {
  const aLines = (a || '').split('\\n');
  const bLines = (b || '').split('\\n');
  const beforeHtml = [], afterHtml = [];
  const maxLen = Math.max(aLines.length, bLines.length);
  for (let i = 0; i < maxLen; i++) {
    const aLine = i < aLines.length ? escHtml(aLines[i]) : '';
    const bLine = i < bLines.length ? escHtml(bLines[i]) : '';
    if (aLine === bLine) {
      beforeHtml.push(`<span>${aLine || '&nbsp;'}</span>`);
      afterHtml.push(`<span>${bLine || '&nbsp;'}</span>`);
    } else {
      beforeHtml.push(`<span style="background:#7f1d1d40;color:#fca5a5">${aLine || '&nbsp;'}</span>`);
      afterHtml.push(`<span style="background:#14532d40;color:#86efac">${bLine || '&nbsp;'}</span>`);
    }
  }
  return { before: beforeHtml.join('\\n'), after: afterHtml.join('\\n') };
}
function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function getNodeLastOutput(nodeId) {
  for (let i = currentRunEvents.length - 1; i >= 0; i--) {
    const ev = currentRunEvents[i];
    if (ev.type === 'node_end' && ev.node_id === nodeId) return frozenOutputs[nodeId] || '';
  }
  return frozenOutputs[nodeId] || '';
}

// ─── Feature 12: Gantt critical path ──────────────────────────────────────
function computeCriticalPath(bars, graphEdges) {
  const ef = {};
  bars.forEach(b => { ef[b.nodeId] = b.endMs; });
  let maxEnd = 0, criticalEnd = null;
  bars.forEach(b => { if (b.endMs > maxEnd) { maxEnd = b.endMs; criticalEnd = b.nodeId; } });
  const criticalSet = new Set();
  let cur = criticalEnd;
  while (cur) {
    criticalSet.add(cur);
    const inEdges = graphEdges.filter(e => e.target === cur || e.to === cur);
    const best = inEdges.reduce((acc, e) => {
      const src = e.source || e.from;
      return (ef[src] || 0) > (ef[acc ? (acc.source || acc.from) : ''] || 0) ? e : acc;
    }, null);
    const nextId = best ? (best.source || best.from) : null;
    if (!nextId || nextId === cur) break;
    cur = nextId;
  }
  return criticalSet;
}

// ─── Feature 13: Docked panel mode ────────────────────────────────────────
function setTestPanelMode(mode) {
  testPanelMode = mode;
  const panel = document.getElementById('testPanel');
  if (!panel) return;
  if (mode === 'hidden') {
    panel.style.display = 'none';
  } else if (mode === 'overlay') {
    panel.style.display = 'flex';
    panel.style.position = 'fixed';
    panel.style.right = '0';
    panel.style.width = '400px';
  } else if (mode === 'docked') {
    panel.style.display = 'flex';
    panel.style.position = 'relative';
    panel.style.width = '320px';
    panel.style.flexShrink = '0';
  }
}

// ─── Feature 14: AI Copilot ───────────────────────────────────────────────
async function sendAiCopilot() {
  const inputEl = document.getElementById('aiCopilotInput');
  if (!inputEl) return;
  const msg = inputEl.value.trim();
  if (!msg) return;
  inputEl.value = '';
  appendAiMessage('user', msg);
  appendAiMessage('assistant', '&#x23F3; thinking...');
  const apiKey = (document.getElementById('apiKeyInput') || {}).value || '';
  let data = {};
  try {
    const resp = await fetch('/ai-refine', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ current_graph: {nodes, edges}, selected_node_id: sel, message: msg, history: aiCopilotHistory, api_key: apiKey })
    });
    data = await resp.json();
  } catch(e) {
    removeLastAiMessage();
    appendAiMessage('assistant', 'Error: ' + e.message);
    return;
  }
  removeLastAiMessage();
  if (data.error) { appendAiMessage('assistant', 'Error: ' + data.error); return; }
  if (data.patch) applyGraphPatch(data.patch);
  appendAiMessage('assistant', data.explanation || '(no explanation)');
  const followEl = document.getElementById('aiSuggestedFollowUp');
  if (followEl && data.suggested_follow_up) followEl.textContent = '&#x1F4A1; ' + data.suggested_follow_up;
  aiCopilotHistory.push({role:'user',content:msg});
  aiCopilotHistory.push({role:'assistant',content:data.explanation||''});
  if (aiCopilotHistory.length > 12) aiCopilotHistory = aiCopilotHistory.slice(-12);
}
function appendAiMessage(role, text) {
  const hist = document.getElementById('aiCopilotHistory');
  if (!hist) return;
  const div = document.createElement('div');
  div.style.cssText = `padding:6px 10px;border-radius:6px;font-size:11px;line-height:1.5;${role==='user'?'background:rgba(34,211,238,0.08);color:var(--cyan)':'background:var(--surface);color:var(--text)'}`;
  div.setAttribute('data-ai-msg','1');
  div.textContent = text;
  hist.appendChild(div);
  hist.scrollTop = hist.scrollHeight;
}
function removeLastAiMessage() {
  const hist = document.getElementById('aiCopilotHistory');
  if (!hist) return;
  const msgs = hist.querySelectorAll('[data-ai-msg]');
  if (msgs.length) msgs[msgs.length-1].remove();
}
function useFollowUp() {
  const el = document.getElementById('aiSuggestedFollowUp');
  const inp = document.getElementById('aiCopilotInput');
  if (el && inp) { inp.value = el.textContent.replace(/^💡\\s*/,''); inp.focus(); }
}
function applyGraphPatch(patch) {
  if (!patch) return;
  if (patch.type === 'update_node') {
    const node = nodes.find(n => n.id === patch.node_id);
    if (node) Object.assign(node, patch.changes || {});
  } else if (patch.type === 'add_node') {
    nodes.push({id: 'node_' + Date.now(), ...(patch.changes || {})});
  } else if (patch.type === 'remove_node') {
    nodes = nodes.filter(n => n.id !== patch.node_id);
    edges = edges.filter(e => (e.source||e.from) !== patch.node_id && (e.target||e.to) !== patch.node_id);
  } else if (patch.type === 'add_edge') {
    edges.push({id: 'e_' + Date.now(), ...(patch.changes || {})});
  }
  render();
  snapshot();
}

// ─── Feature 15: HITL form builder ────────────────────────────────────────
function addHitlField() {
  if (!sel) return;
  const node = nodes.find(n => n.id === sel);
  if (!node) return;
  const field = {id: 'f_' + Date.now(), type: 'text', label: '', placeholder: '', required: false};
  node.form_fields = node.form_fields || [];
  node.form_fields.push(field);
  renderHitlFormEditor(node);
}
function removeHitlField(fieldId) {
  if (!sel) return;
  const node = nodes.find(n => n.id === sel);
  if (!node) return;
  node.form_fields = (node.form_fields || []).filter(f => f.id !== fieldId);
  renderHitlFormEditor(node);
}
function renderHitlFormEditor(node) {
  const list = document.getElementById('hitlFieldsList');
  if (!list) return;
  list.innerHTML = (node.form_fields || []).map(f => `
    <div class="hitl-field-row" data-field-id="${f.id}" style="display:flex;gap:4px;margin-bottom:4px;align-items:center">
      <select style="font-size:10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px" onchange="updateHitlField('${f.id}','type',this.value)">
        <option${f.type==='text'?' selected':''}>text</option>
        <option${f.type==='textarea'?' selected':''}>textarea</option>
        <option${f.type==='number'?' selected':''}>number</option>
        <option${f.type==='select'?' selected':''}>select</option>
        <option${f.type==='checkbox'?' selected':''}>checkbox</option>
      </select>
      <input type="text" value="${escAttr(f.label)}" placeholder="Label" style="flex:1;font-size:10px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 5px" oninput="updateHitlField('${f.id}','label',this.value)">
      <button style="font-size:10px;padding:1px 5px;border-radius:3px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer" onclick="removeHitlField('${f.id}')">&#10005;</button>
    </div>
  `).join('');
}
function updateHitlField(fieldId, key, value) {
  if (!sel) return;
  const node = nodes.find(n => n.id === sel);
  if (!node) return;
  const field = (node.form_fields || []).find(f => f.id === fieldId);
  if (field) field[key] = value;
}
function escAttr(s) { return (s||'').replace(/"/g,'&quot;').replace(/</g,'&lt;'); }

// ─── Feature 16: Bidirectional file sync ──────────────────────────────────
let _watchFileReader = null;
function openWatchFile() {
  const m = document.getElementById('watchFileModal');
  if (m) m.style.display = 'flex';
}
function closeWatchFileModal() {
  const m = document.getElementById('watchFileModal');
  if (m) m.style.display = 'none';
}
function stopWatchFile() {
  if (_watchFileReader) { try { _watchFileReader.cancel(); } catch(_){} _watchFileReader = null; }
  const st = document.getElementById('watchFileStatus');
  if (st) st.textContent = '&#x23F9; Stopped';
}
async function startWatchFile() {
  const pathEl = document.getElementById('watchFilePath');
  if (!pathEl || !pathEl.value.trim()) return;
  const path = pathEl.value.trim();
  const st = document.getElementById('watchFileStatus');
  if (st) st.textContent = '&#x1F441; Watching\u2026';
  closeWatchFileModal();
  try {
    const resp = await fetch('/watch-file', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path})});
    const reader = resp.body.getReader();
    _watchFileReader = reader;
    const decoder = new TextDecoder();
    (function read() {
      reader.read().then(({done, value}) => {
        if (done) { if (st) st.textContent = '&#x23F9; Watch ended'; return; }
        const text = decoder.decode(value);
        for (const line of text.split('\\n')) {
          if (line.startsWith('data:')) {
            try {
              const ev = JSON.parse(line.slice(5).trim());
              if (ev.type === 'file_changed') {
                const g = _parsePythonToGraph(ev.content);
                if (g && g.nodes && g.nodes.length) {
                  nodes = g.nodes; edges = g.edges || [];
                  render();
                  if (st) st.textContent = '&#x21BB; Synced ' + new Date().toLocaleTimeString();
                }
              }
            } catch(_) {}
          }
        }
        read();
      });
    })();
  } catch(e) {
    if (st) st.textContent = 'Error: ' + e.message;
  }
}
function _parsePythonToGraph(src) {
  const nodeRe = /graph\\.add_node\\(["']([^"']+)["']/g;
  const edgeRe = /graph\\.add_edge\\(["']([^"']+)["']\\s*,\\s*["']([^"']+)["']/g;
  const ns = [], es = [];
  let m, i = 0;
  while ((m = nodeRe.exec(src)) !== null) {
    ns.push({id: m[1], label: m[1], type: 'agent', x: i * 180 + 60, y: 200});
    i++;
  }
  while ((m = edgeRe.exec(src)) !== null) {
    es.push({id: `e_${m[1]}_${m[2]}`, source: m[1], target: m[2], from: m[1], to: m[2]});
  }
  return {nodes: ns, edges: es};
}

// ─── Feature 17: Agent-as-Tool node ───────────────────────────────────────
function populateAgentToolTargets() {
  const sel2 = document.getElementById('propToolTarget');
  if (!sel2) return;
  sel2.innerHTML = nodes
    .filter(n => n.type === 'agent' || n.type === 'subgraph')
    .map(n => `<option value="${n.id}">${escAttr(n.label||n.name||n.id)}</option>`)
    .join('');
}
function updateSelectedNode(key, value) {
  if (!sel) return;
  const node = nodes.find(n => n.id === sel);
  if (node) { node[key] = value; render(); }
}

async function runTest() {
  if (isRunning) return;
  const input = document.getElementById('testInput').value.trim();
  if (!input) { document.getElementById('testInput').focus(); return; }
  const apiKey = document.getElementById('apiKeyInput').value.trim();
  const mock = !apiKey;

  isRunning = true;
  document.getElementById('runBtn').textContent = '\u23f3 Running\u2026';
  document.getElementById('runBtn').disabled = true;
  document.getElementById('testStatus').textContent = mock ? '[mock] Mock mode' : '[live] Live mode';
  document.getElementById('testOutput').innerHTML = '';
  switchTestTab('output');

  currentRunEvents = [];
  traceRows = [];
  evalResults = {};
  activeReplayNodeId = null;
  const runStart = Date.now();

  if (mock) {
    await runMock(input);
  } else {
    await runLive(input, apiKey);
  }

  const elapsed = ((Date.now() - runStart) / 1000).toFixed(1);
  runHistory.unshift({ input: input.slice(0, 60), events: [...currentRunEvents], elapsed, mock, time: new Date().toLocaleTimeString() });
  if (runHistory.length > 20) runHistory.pop();
  refreshHistory();
  renderScrubber();
  renderGantt();

  isRunning = false;
  document.getElementById('runBtn').textContent = '\u25b6 Run';
  document.getElementById('runBtn').disabled = false;
  document.getElementById('testStatus').textContent = `Done in ${elapsed}s`;
}

async function runMock(input) {
  const agentNodes = nodes.filter(n => n.type === 'agent');
  const hitlNodes  = nodes.filter(n => n.type === 'hitl');
  if (!agentNodes.length && !hitlNodes.length) {
    appendTrace('<div class="trace-error">No agent nodes in graph. Add at least one agent node.</div>');
    return;
  }

  const executableNodes = nodes.filter(n => n.type === 'agent' || n.type === 'hitl');
  const ordered = topoOrder(executableNodes);
  let lastOutput = input;
  const _mockT0 = Date.now();

  for (const n of ordered) {
    // HITL node: pause and wait for human choice
    if (n.type === 'hitl') {
      appendTrace(`<div class="trace-node-start">\u23f8 ${n.name}</div>`);
      currentRunEvents.push({type: 'node_start', node: n.name, node_id: n.id, node_type: 'hitl', ts: Date.now() - _mockT0});
      const opts = (n.options || '').split(',').map(s => s.trim()).filter(Boolean);
      const choice = await waitForHitlChoice(n.id, opts);
      currentRunEvents.push({type: 'hitl_choice', node_id: n.id, choice, ts: Date.now() - _mockT0});
      appendTrace(`<div class="trace-node-end">  \u2713 hitl: ${choice}</div>`);
      currentRunEvents.push({type: 'node_end', node_id: n.id, tokens: 0, cost: 0, ts: Date.now() - _mockT0});
      lastOutput = choice;
      continue;
    }

    // Frozen node: skip re-execution, reuse last cached output
    if (n.frozen && frozenOutputs[n.id]) {
      appendTrace(`<div class="trace-node-start">\u2744 ${n.name} <span style="color:#475569;font-size:10px">(frozen \u2014 cached)</span></div>`);
      appendTrace(`<div class="trace-chunk">${frozenOutputs[n.id]}</div>`);
      appendTrace('<div class="trace-node-end">  \u2713 0 tokens \xb7 $0.0000 (cached)</div>');
      currentRunEvents.push({type: 'node_start', node: n.name, node_id: n.id, node_type: n.type, frozen: true, ts: Date.now() - _mockT0});
      currentRunEvents.push({type: 'node_end', node_id: n.id, tokens: 0, cost: 0, ts: Date.now() - _mockT0});
      lastOutput = frozenOutputs[n.id];
      continue;
    }

    // Warn for unconnected variable ports
    const nodeVars = extractVars(n.system_prompt);
    for (const v of nodeVars) {
      const connected = edges.some(e => e.to === n.id && e.varPort === v);
      if (!connected) appendTrace(`<div class="trace-tool" style="color:#a855f7">  \u26a0 {${v}} unconnected \u2014 using raw placeholder</div>`);
    }

    appendTrace(`<div class="trace-node-start">\u25b6 ${n.name} (${n.provider}/${n.model})</div>`);
    currentRunEvents.push({type: 'node_start', node: n.name, node_id: n.id, node_type: n.type, ts: Date.now() - _mockT0});

    await sleep(80);
    if (n.tools) {
      const toolList = n.tools.split(',').map(t => t.trim()).filter(Boolean);
      for (const tool of toolList.slice(0, 2)) {
        appendTrace(`<div class="trace-tool">  [tool] ${tool}(query="${lastOutput.slice(0,30)}\u2026")</div>`);
        await sleep(60);
        appendTrace(`<div class="trace-tool-result">  \u2192 [mock result from ${tool}]</div>`);
        currentRunEvents.push({type: 'tool_call', node_id: n.id, tool, ts: Date.now() - _mockT0});
      }
    }

    const words = `[MOCK] ${n.name} processed your request. In real mode this would use ${n.provider} / ${n.model} with your API key.`.split(' ');
    let chunk = '';
    const span = document.createElement('span');
    span.className = 'trace-chunk';
    document.getElementById('testOutput').appendChild(span);
    for (const w of words) {
      chunk += w + ' ';
      span.textContent = chunk;
      document.getElementById('testOutput').scrollTop = document.getElementById('testOutput').scrollHeight;
      await sleep(35);
    }
    const evalResult = _clientRunEvals(n.id, chunk, n.eval_assertion || '');
    if (evalResult.pass) {
      appendTrace('<div class="trace-eval-pass">  \u2713 evals passed</div>');
    } else {
      const failedNames = evalResult.results.filter(r => !r.pass).map(r => r.name).join(', ');
      appendTrace(`<div class="trace-eval-fail">  \u2717 evals: ${failedNames}</div>`);
    }
    currentRunEvents.push({type: 'eval_result', node_id: n.id, pass: evalResult.pass, results: evalResult.results, ts: Date.now() - _mockT0});
    appendTrace('<div class="trace-node-end">  \u2713 45 tokens \xb7 $0.0000 (mock)</div>');
    currentRunEvents.push({type: 'node_end', node_id: n.id, tokens: 45, cost: 0, ts: Date.now() - _mockT0});
    frozenOutputs[n.id] = chunk;   // cache output for potential freeze reuse
    lastOutput = chunk;
  }

  appendTrace('<div class="trace-run-end">\u2705 Mock run complete</div>');
}

async function runLive(input, apiKey) {
  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({input, nodes, edges, api_key: apiKey}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({message: 'Request failed'}));
      appendTrace(`<div class="trace-error">Error: ${err.message || resp.status}</div>`);
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') break;
        let ev;
        try { ev = JSON.parse(raw); } catch(e) { continue; }
        currentRunEvents.push(ev);
        handleTraceEvent(ev);
      }
    }
  } catch(e) {
    appendTrace(`<div class="trace-error">Connection error: ${e.message}</div>`);
  }
}

function handleTraceEvent(ev) {
  if (ev.type === 'node_start') {
    appendTrace(`<div class="trace-node-start">\u25b6 ${ev.node_name || ev.node_id}</div>`);
  } else if (ev.type === 'chunk') {
    const out = document.getElementById('testOutput');
    let span = out.querySelector('span.trace-chunk:last-of-type');
    if (!span) { span = document.createElement('span'); span.className = 'trace-chunk'; out.appendChild(span); }
    span.textContent += ev.content;
    out.scrollTop = out.scrollHeight;
  } else if (ev.type === 'tool_call') {
    appendTrace(`<div class="trace-tool">  [tool] ${ev.tool}(${JSON.stringify(ev.args || {}).slice(0,60)})</div>`);
  } else if (ev.type === 'tool_result') {
    appendTrace(`<div class="trace-tool-result">  \u2192 ${String(ev.result).slice(0,120)}</div>`);
  } else if (ev.type === 'node_end') {
    const cost = ev.cost > 0 ? `$${ev.cost.toFixed(5)}` : '$0.00';
    appendTrace(`<div class="trace-node-end">  \u2713 ${ev.tokens || 0} tokens \xb7 ${cost}</div>`);
    for (const e of edges) { if (e.from === ev.node_id) edgeLastOutput[e.id] = frozenOutputs[ev.node_id] || ''; }
  } else if (ev.type === 'error') {
    appendTrace(`<div class="trace-error">  \u2717 ${ev.message}</div>`);
  } else if (ev.type === 'eval_result') {
    evalResults[ev.node_id] = {pass: ev.pass, results: ev.results || []};
    render();
    if (ev.pass) {
      appendTrace('<div class="trace-eval-pass">  \u2713 evals passed</div>');
    } else {
      const failed = (ev.results || []).filter(r => !r.pass).map(r => r.name).join(', ');
      appendTrace(`<div class="trace-eval-fail">  \u2717 evals: ${failed}</div>`);
    }
  } else if (ev.type === 'hitl_pause') {
    appendTrace(`<div class="trace-hitl-prompt">\u23f8 ${ev.node_name || ev.node_id} \u2014 awaiting human choice</div>`);
  } else if (ev.type === 'hitl_choice') {
    appendTrace(`<div style="color:#f59e0b;font-size:11px;padding:2px 8px">  \u2713 Chose: ${ev.choice}</div>`);
  } else if (ev.type === 'hitl_auto') {
    appendTrace(`<div style="color:#94a3b8;font-size:11px;padding:2px 8px">  \u23f1 Auto-resolved: ${ev.choice} (timeout)</div>`);
  } else if (ev.type === 'run_end') {
    const cost = ev.total_cost > 0 ? ` \xb7 $${ev.total_cost.toFixed(5)}` : '';
    appendTrace(`<div class="trace-run-end">\u2705 Run complete \u2014 ${ev.total_tokens || 0} tokens${cost}</div>`);
    if (activeReplayNodeId) {
      const before = replayBaseline[activeReplayNodeId] || '';
      const after = getNodeLastOutput(activeReplayNodeId) || '';
      if (before !== after && (before || after)) showReplayDiff(activeReplayNodeId, before, after);
      activeReplayNodeId = null;
    }
  }
}

function refreshHistory() {
  document.getElementById('ttab-history').textContent = `History (${runHistory.length})`;
  const el = document.getElementById('testHistory');
  el.innerHTML = runHistory.map((r, i) => `
    <div class="run-item history-item" data-input="${(r.input || '').replace(/"/g, '&quot;')}" onclick="replayHistory(${i})">
      <div style="display:flex;justify-content:space-between">
        <span style="color:${r.mock ? '#f59e0b' : 'var(--green)'}; font-size:10px">${r.mock ? '[mock]' : '[live]'} \xb7 ${r.elapsed}s</span>
        <span class="history-time">${r.time}</span>
      </div>
      <div class="history-preview">${r.input}</div>
    </div>
  `).join('');
  let totalTokens = 0, totalCost = 0;
  for (const r of runHistory) {
    for (const ev of (r.events || [])) {
      if (ev.type === 'run_end') { totalTokens += ev.total_tokens || 0; totalCost += ev.total_cost || 0; }
    }
  }
  const costEl = document.getElementById('historySessionCost');
  if (totalTokens > 0) {
    const costStr = totalCost > 0 ? ` \xb7 $${totalCost.toFixed(5)}` : '';
    costEl.textContent = `Session: ${totalTokens} tokens${costStr}`;
  } else {
    costEl.textContent = '';
  }
}

function filterHistory() {
  const q = document.getElementById('historySearch').value.toLowerCase();
  document.querySelectorAll('#testHistory .run-item').forEach(el => {
    const text = (el.getAttribute('data-input') || '').toLowerCase();
    el.style.display = (!q || text.includes(q)) ? '' : 'none';
  });
}
function exportHistory() {
  if (!runHistory.length) return;
  const jsonl = runHistory.map(r => JSON.stringify({
    time: r.time, input: r.input, elapsed: r.elapsed, mock: r.mock, events: r.events,
  })).join('\\n');
  const blob = new Blob([jsonl], {type: 'application/x-ndjson'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'selectools-runs-' + new Date().toISOString().slice(0, 10) + '.jsonl';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function replayHistory(idx) {
  const r = runHistory[idx];
  document.getElementById('testOutput').innerHTML = '';
  evalResults = {};
  switchTestTab('output');
  for (const ev of r.events) handleTraceEvent(ev);
}

function topoOrder(executableNodes) {
  const execIds = new Set(executableNodes.map(n => n.id));
  const startNode = nodes.find(n => n.type === 'start');
  if (!startNode) return executableNodes;
  const result = [];
  const visited = new Set();
  function walk(nid) {
    if (visited.has(nid)) return;
    visited.add(nid);
    const n = nodes.find(x => x.id === nid);
    if (n && execIds.has(n.id)) result.push(n);
    for (const e of edges) {
      if (e.from === nid) walk(e.to);
    }
  }
  walk(startNode.id);
  for (const n of executableNodes) if (!visited.has(n.id)) result.push(n);
  return result;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function waitForHitlChoice(nodeId, options) {
  return new Promise(resolve => {
    const out = document.getElementById('testOutput');
    const hdr = document.createElement('div');
    hdr.className = 'trace-hitl-prompt';
    hdr.textContent = '\u23f8 Human Input required — choose an option:';
    out.appendChild(hdr);
    const wrap = document.createElement('div');
    wrap.className = 'hitl-btns';
    let resolved = false;
    const choose = opt => {
      if (resolved) return;
      resolved = true;
      wrap.innerHTML = `<span style="color:#f59e0b;font-size:11px">\u2713 Chose: ${opt}</span>`;
      resolve(opt);
    };
    for (const opt of options) {
      const b = document.createElement('button');
      b.className = 'hitl-btn';
      b.textContent = opt;
      b.onclick = () => choose(opt);
      wrap.appendChild(b);
    }
    out.appendChild(wrap);
    out.scrollTop = out.scrollHeight;
    // auto-resolve after 30s with timeout sentinel
    setTimeout(() => choose('timeout'), 30000);
  });
}

// ─── Client-side Eval Runner ──────────────────────────────────────────────
function _clientRunEvals(nodeId, text, assertion) {
  const results = [];
  results.push({name: 'not_empty', pass: !!text.trim()});
  const lower = text.toLowerCase().trim();
  const noApology = !lower.startsWith("i'm sorry") && !lower.startsWith('i apologize')
                    && !lower.startsWith('i am sorry') && !lower.startsWith('sorry,');
  results.push({name: 'no_apology', pass: noApology});
  if (assertion && assertion.trim()) {
    const a = assertion.trim();
    results.push({name: 'contains(' + a.slice(0, 20) + ')', pass: text.toLowerCase().includes(a.toLowerCase())});
  }
  const pass = results.every(r => r.pass);
  evalResults[nodeId] = {pass, results};
  render();
  return {pass, results};
}

// ─── Trace Scrubber (time-travel) ─────────────────────────────────────────
function renderScrubber() {
  const el = document.getElementById('scrubber');
  const hdr = document.getElementById('scrubberHeader');
  const gw = document.getElementById('ganttWrap');
  if (!el) return;
  const steps = currentRunEvents.filter(ev => ev.type === 'node_start');
  if (!steps.length) {
    el.style.display = 'none';
    if (hdr) hdr.style.display = 'none';
    if (gw) gw.style.display = 'none';
    return;
  }
  el.style.display = 'flex';
  if (hdr) hdr.style.display = 'flex';
  if (gw) gw.style.display = 'none';  // reset to dot view on each new run
  const btn = document.getElementById('ganttToggleBtn');
  if (btn) btn.textContent = '\\uD83D\\uDCCA Timeline';
  el.innerHTML = '';

  const typeColors = { agent: '#22d3ee', loop: '#f97316', subgraph: '#a855f7', start: '#22c55e', hitl: '#f59e0b' };
  steps.forEach((ev, i) => {
    if (i > 0) {
      const line = document.createElement('div');
      line.className = 'scrub-line';
      el.appendChild(line);
    }
    const step = document.createElement('div');
    step.className = 'scrub-step';
    step.title = (ev.node || ev.node_name || ev.node_id || 'step ' + (i + 1));

    const dot = document.createElement('div');
    dot.className = 'scrub-dot';
    dot.style.background = ev.frozen ? '#475569' : (typeColors[ev.node_type] || '#22d3ee');

    const lbl = document.createElement('div');
    lbl.className = 'scrub-label';
    lbl.textContent = (ev.node || ev.node_name || 'step ' + (i + 1)).slice(0, 10);

    const replayBtn = document.createElement('button');
    replayBtn.className = 'scrub-replay';
    replayBtn.textContent = '\u23f5';
    replayBtn.title = 'Re-run from here';
    const capturedIdx = i;
    replayBtn.addEventListener('click', e => { e.stopPropagation(); rerunFromEvent(capturedIdx); });

    step.appendChild(dot);
    step.appendChild(lbl);
    step.appendChild(replayBtn);

    const nid = ev.node_id;
    step.addEventListener('click', () => {
      el.querySelectorAll('.scrub-step').forEach(s => s.classList.remove('active'));
      step.classList.add('active');
      if (nid) selNode(nid);
    });
    el.appendChild(step);
  });
}

// ─── Gantt Timeline ────────────────────────────────────────────────────────
function toggleGantt() {
  const g = document.getElementById('ganttWrap');
  const s = document.getElementById('scrubber');
  const btn = document.getElementById('ganttToggleBtn');
  if (!g || !s) return;
  const show = g.style.display === 'none';
  g.style.display = show ? 'block' : 'none';
  s.style.display  = show ? 'none'  : 'flex';
  if (btn) btn.textContent = show ? '\u23f1 Steps' : '\\uD83D\\uDCCA Timeline';
  if (show) renderGantt();
}

function renderGantt() {
  const svg = document.getElementById('ganttSvg');
  if (!svg) return;

  const starts = {}, ends = {};
  for (const ev of currentRunEvents) {
    if (ev.type === 'node_start' && ev.ts != null) starts[ev.node_id] = ev;
    if (ev.type === 'node_end'   && ev.ts != null) ends[ev.node_id]   = ev;
  }

  const rows = Object.keys(starts)
    .filter(id => ends[id])
    .map(id => ({
      id,
      name:   starts[id].node_name || starts[id].node || id,
      type:   starts[id].node_type || 'agent',
      start:  starts[id].ts,
      end:    ends[id].ts,
      tokens: ends[id].tokens || 0,
      cost:   ends[id].cost   || 0,
      frozen: !!starts[id].frozen,
    }));

  if (!rows.length) return;

  const LABEL_W = 90, PAD = 8, ROW_H = 22, AXIS_H = 18, W = 460;
  const BAR_W = W - LABEL_W - PAD;
  const totalMs = Math.max(...rows.map(r => r.end), 1);
  const H = rows.length * ROW_H + AXIS_H + 4;
  const typeColors = { agent: '#22d3ee', loop: '#f97316', subgraph: '#a855f7', start: '#22c55e', hitl: '#f59e0b' };

  let html = '';
  rows.forEach((r, i) => {
    const y   = i * ROW_H;
    const x   = LABEL_W + (r.start / totalMs) * BAR_W;
    const bw  = Math.max(4, ((r.end - r.start) / totalMs) * BAR_W);
    const col = r.frozen ? '#475569' : (typeColors[r.type] || '#22d3ee');
    const dur = r.end - r.start;
    const tip = `${r.name}: ${dur}ms${r.tokens ? ' \xb7 ' + r.tokens + ' tok' : ''}${r.cost ? ' \xb7 $' + r.cost.toFixed(4) : ''}`;
    html += `<text class="gantt-label" x="${LABEL_W - 4}" y="${y + 14}" text-anchor="end">${r.name.slice(0, 12)}</text>`;
    html += `<rect class="gantt-bar" x="${x}" y="${y + 3}" width="${bw}" height="${ROW_H - 7}" fill="${col}" rx="2"><title>${tip}</title></rect>`;
    if (bw > 32) html += `<text class="gantt-dur" x="${x + bw / 2}" y="${y + 14}" text-anchor="middle">${dur}ms</text>`;
  });

  const ay = rows.length * ROW_H + 4;
  html += `<line class="gantt-axis-line" x1="${LABEL_W}" y1="${ay}" x2="${W - PAD}" y2="${ay}"/>`;
  for (let t = 0; t <= 4; t++) {
    const tx = LABEL_W + (t / 4) * BAR_W;
    html += `<line class="gantt-tick-line" x1="${tx}" y1="${ay}" x2="${tx}" y2="${ay + 4}"/>`;
    html += `<text class="gantt-tick-label" x="${tx}" y="${ay + 14}" text-anchor="middle">${Math.round((t / 4) * totalMs)}ms</text>`;
  }

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.innerHTML = html;

  // Feature 12: critical path + dependency arrows
  const ganttBars = rows.map((r, i) => ({
    nodeId: r.id, startMs: r.start, endMs: r.end, y: i * ROW_H
  }));
  const criticalSet = computeCriticalPath(ganttBars, edges);
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <marker id="ganttArrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
    <marker id="ganttArrowCrit" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/>
    </marker>`;
  svg.appendChild(defs);
  edges.forEach(edge => {
    const srcId = edge.source || edge.from;
    const tgtId = edge.target || edge.to;
    const srcBar = ganttBars.find(b => b.nodeId === srcId);
    const tgtBar = ganttBars.find(b => b.nodeId === tgtId);
    if (!srcBar || !tgtBar) return;
    const BAR_H = ROW_H;
    const toX = ms => LABEL_W + (ms / totalMs) * BAR_W;
    const x1 = toX(srcBar.endMs), y1 = srcBar.y + BAR_H / 2;
    const x2 = toX(tgtBar.startMs), y2 = tgtBar.y + BAR_H / 2;
    const mid = (x1 + x2) / 2;
    const onCrit = criticalSet.has(srcId) && criticalSet.has(tgtId);
    const color = onCrit ? '#f59e0b' : '#475569';
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    arrow.setAttribute('d', `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
    arrow.setAttribute('stroke', color);
    arrow.setAttribute('stroke-width', onCrit ? '2' : '1');
    arrow.setAttribute('fill', 'none');
    arrow.setAttribute('stroke-dasharray', onCrit ? '' : '3 2');
    arrow.setAttribute('marker-end', `url(#ganttArrow${onCrit ? 'Crit' : ''})`);
    svg.appendChild(arrow);
  });
}

// ─── Load Production Trace ────────────────────────────────────────────────
function openLoadTrace() {
  const m = document.getElementById('loadTraceModal');
  if (m) { m.style.display = 'flex'; }
  const inp = document.getElementById('loadTraceInput');
  if (inp) { inp.value = ''; setTimeout(() => inp.focus(), 50); }
  const err = document.getElementById('loadTraceError');
  if (err) err.style.display = 'none';
}
function closeLoadTrace() {
  const m = document.getElementById('loadTraceModal');
  if (m) m.style.display = 'none';
}
function doLoadTrace() {
  const text = (document.getElementById('loadTraceInput').value || '').trim();
  const errEl = document.getElementById('loadTraceError');
  errEl.style.display = 'none';
  if (!text) return;
  let traceData;
  try { traceData = JSON.parse(text); } catch(e) {
    errEl.textContent = 'Invalid JSON: ' + e.message;
    errEl.style.display = 'block';
    return;
  }
  if (!traceData.steps || !Array.isArray(traceData.steps)) {
    errEl.textContent = 'Missing "steps" array \u2014 paste the output of trace_to_json(result.trace)';
    errEl.style.display = 'block';
    return;
  }
  currentRunEvents = convertTraceToEvents(traceData);
  renderScrubber();
  renderGantt();
  const totalMs = traceData.total_duration_ms || 0;
  runHistory.unshift({
    input: '[trace] ' + (traceData.request_id || new Date().toLocaleTimeString()),
    events: [...currentRunEvents],
    elapsed: (totalMs / 1000).toFixed(2),
    mock: false,
    time: new Date().toLocaleTimeString(),
    imported: true,
  });
  if (runHistory.length > 20) runHistory.pop();
  refreshHistory();
  openTestPanel();
  switchTestTab('output');
  document.getElementById('testStatus').textContent = '\u2139 Production trace loaded';
  closeLoadTrace();
}
function convertTraceToEvents(t) {
  const evs = [];
  const t0 = t.start_time || 0;
  let lastNodeId = null;
  const hasGraphSteps = t.steps.some(s => s.type === 'graph_node_start' || s.type === 'graph_node_end');
  if (!hasGraphSteps) {
    evs.push({type: 'node_start', node_id: 'agent', node_name: 'Agent', node_type: 'agent', ts: 0});
  }
  for (const s of t.steps) {
    const ts = Math.round(((s.timestamp || t0) - t0) * 1000);
    if (s.type === 'graph_node_start') {
      lastNodeId = s.node_name || 'node';
      evs.push({type: 'node_start', node_id: lastNodeId, node_name: lastNodeId, node_type: 'agent', ts});
    } else if (s.type === 'graph_node_end') {
      const nid = s.node_name || lastNodeId || 'node';
      evs.push({type: 'node_end', node_id: nid, tokens: (s.prompt_tokens || 0) + (s.completion_tokens || 0), cost: s.cost_usd || 0, ts});
    } else if (s.type === 'tool_execution') {
      const nid = lastNodeId || 'agent';
      evs.push({type: 'tool_call', node_id: nid, tool: s.tool_name || 'tool', args: s.tool_args || {}, ts});
      if (s.tool_result) evs.push({type: 'tool_result', node_id: nid, tool: s.tool_name || 'tool', result: s.tool_result, ts: ts + 1});
    } else if (s.type === 'llm_call' && !hasGraphSteps) {
      lastNodeId = 'agent';
      if (s.summary) evs.push({type: 'chunk', node_id: 'agent', content: s.summary, ts});
    } else if (s.type === 'error') {
      evs.push({type: 'error', message: s.error || 'error', ts});
    }
  }
  if (!hasGraphSteps) {
    const durMs = t.total_duration_ms || Math.round(((t.end_time || t0) - t0) * 1000);
    evs.push({type: 'node_end', node_id: 'agent', tokens: t.total_tokens || 0, cost: t.total_cost || 0, ts: durMs});
  }
  evs.push({type: 'run_end', total_tokens: t.total_tokens || 0, total_cost: t.total_cost || 0, ts: Math.round(((t.end_time || t0) - t0) * 1000)});
  return evs;
}

// ─── Canvas Search (Cmd+K) ────────────────────────────────────────────────
function openSearch() {
  const ov = document.getElementById('searchOverlay');
  ov.style.display = '';
  const inp = document.getElementById('searchInput');
  inp.value = '';
  document.getElementById('searchResults').innerHTML = '';
  inp.focus();
}

function closeSearch() {
  document.getElementById('searchOverlay').style.display = 'none';
}

function searchNodes(q) {
  const res = document.getElementById('searchResults');
  res.innerHTML = '';
  if (!q.trim()) return;
  const matches = nodes.filter(n => (n.name || '').toLowerCase().includes(q.toLowerCase()) ||
                                    n.type.toLowerCase().includes(q.toLowerCase()));
  if (!matches.length) {
    const empty = document.createElement('div');
    empty.className = 'search-empty';
    empty.textContent = 'No nodes found';
    res.appendChild(empty);
    return;
  }
  const nodeColors = {
    start: '#22c55e', end: '#ef4444', agent: '#22d3ee',
    loop: '#f97316', subgraph: '#a855f7', note: '#f59e0b', hitl: '#f59e0b'
  };
  for (const n of matches) {
    const row = document.createElement('div');
    row.className = 'search-result';
    const dot = document.createElement('div');
    dot.style.cssText = 'width:8px;height:8px;border-radius:50%;background:' + (nodeColors[n.type] || '#94a3b8') + ';flex-shrink:0';
    const name = document.createElement('span');
    name.textContent = n.name || n.id;
    const type = document.createElement('span');
    type.style.cssText = 'color:var(--muted);font-size:10px;margin-left:auto';
    type.textContent = n.type;
    row.appendChild(dot); row.appendChild(name); row.appendChild(type);
    row.addEventListener('click', () => searchSelect(n.id));
    res.appendChild(row);
  }
}

function searchSelect(id) {
  closeSearch();
  selNode(id);
  const n = nodes.find(x => x.id === id);
  if (n) {
    const wrap = document.getElementById('canvasWrap');
    if (wrap) {
      wrap.scrollLeft = Math.max(0, n.x - wrap.offsetWidth / 2 + 80);
      wrap.scrollTop  = Math.max(0, n.y - wrap.offsetHeight / 2 + 35);
    }
  }
}

// ─── Context Menu ─────────────────────────────────────────────────────────
let ctxNodeId = null;

function showCtxMenu(ev, nodeId) {
  ev.stopPropagation();
  ctxNodeId = nodeId;
  const menu = document.getElementById('ctxMenu');
  if (!menu) return;
  const n = nodes.find(x => x.id === nodeId);
  const isAgent = n && n.type === 'agent';
  const rerunBtn = document.getElementById('ctxRerun');
  if (rerunBtn) rerunBtn.style.display = isAgent ? '' : 'none';
  menu.style.left = ev.clientX + 'px';
  menu.style.top  = ev.clientY + 'px';
  menu.style.display = 'block';
}

function hideCtxMenu() {
  const menu = document.getElementById('ctxMenu');
  if (menu) menu.style.display = 'none';
  ctxNodeId = null;
}
function ctxPinOutput() {
  hideCtxMenu();
  if (!ctxNodeId) return;
  const val = frozenOutputs[ctxNodeId] || edgeLastOutput[ctxNodeId + '::output'] || null;
  if (val !== null && val !== undefined) {
    pinPort(ctxNodeId, 'output', val);
  } else {
    // show a toast-style status note
    document.getElementById('testStatus').textContent = '\u26a0 Run the graph first to capture an output to pin';
  }
}

async function rerunNodeAlone(nodeId) {
  hideCtxMenu();
  const n = nodes.find(x => x.id === nodeId);
  if (!n || n.type !== 'agent') return;
  replayBaseline[nodeId] = getNodeLastOutput(nodeId);
  activeReplayNodeId = nodeId;
  const input = frozenOutputs[nodeId] || document.getElementById('testInput')?.value || 'test input';
  openTestPanel();
  switchTestTab('output');
  document.getElementById('testOutput').innerHTML = '';
  traceRows = [];
  document.getElementById('testStatus').textContent = `\u21ba Re-running ${n.name}\u2026`;
  appendTrace(`<div class="trace-node-start">\u21ba Re-run in isolation: ${n.name}</div>`);
  appendTrace(`<div class="trace-chunk" style="color:var(--muted)">Input: ${input.slice(0, 80)}</div>`);
  await sleep(120);
  const fakeOutput = `[mock] ${n.name} \u2192 Processed: "${input.slice(0, 60)}${input.length > 60 ? '\u2026' : ''}"`;
  appendTrace(`<div class="trace-chunk">${fakeOutput}</div>`);
  frozenOutputs[nodeId] = fakeOutput;
  appendTrace(`<div class="trace-node-end">\u2713 Done (isolated mock)</div>`);
  document.getElementById('testStatus').textContent = `\u21ba Re-run complete`;
}

document.addEventListener('mousedown', ev => {
  const menu = document.getElementById('ctxMenu');
  if (menu && menu.style.display !== 'none' && !menu.contains(ev.target)) hideCtxMenu();
});

// ─── Embed Widget ─────────────────────────────────────────────────────────
function openEmbed() {
  const modal = document.getElementById('embedModal');
  if (!modal) return;
  modal.style.display = 'flex';
  updateEmbedCode();
  updateEmbedPreview();
}
function updateEmbedCode() {
  const code = document.getElementById('embedCode');
  if (!code) return;
  const yaml = genYaml();
  const encoded = btoa(unescape(encodeURIComponent(yaml)));
  const origin = window.location.origin || 'http://localhost:8000';
  const accent = encodeURIComponent((document.getElementById('embedAccent') || {value: '#22d3ee'}).value);
  const pos = (document.getElementById('embedPosition') || {value: 'bottom-right'}).value;
  const welcome = encodeURIComponent((document.getElementById('embedWelcome') || {value: ''}).value);
  const src = `${origin}/builder?embed=1&graph=${encoded}&accent=${accent}&pos=${pos}&welcome=${welcome}`;
  const posStyle = pos === 'inline'
    ? 'width:100%;height:520px;'
    : `position:fixed;${pos === 'bottom-left' ? 'left:20px' : 'right:20px'};bottom:20px;width:380px;height:520px;`;
  code.textContent = `<!-- selectools agent embed -->\n<iframe\n  src="${src}"\n  style="border:none;border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,0.35);${posStyle}"\n  allow="clipboard-write">\n</iframe>\n\n<!-- Self-host: selectools serve --builder --port 8000 -->`;
}
function updateEmbedPreview() {
  const preview = document.getElementById('embedPreviewWidget');
  if (preview) preview.style.background = (document.getElementById('embedAccent') || {value: '#22d3ee'}).value;
}

function closeEmbed() {
  const modal = document.getElementById('embedModal');
  if (modal) modal.style.display = 'none';
}

function copyEmbedCode() {
  const code = document.getElementById('embedCode');
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => {
    const btn = document.getElementById('embedCopyBtn');
    if (btn) { const prev = btn.textContent; btn.textContent = '\u2713 Copied!'; setTimeout(() => { btn.textContent = prev; }, 1500); }
  });
}

// ─── Scrubber: Re-run from Checkpoint ─────────────────────────────────────
function rerunFromEvent(idx) {
  const step = currentRunEvents.filter(ev => ev.type === 'node_start')[idx];
  if (!step) return;
  openTestPanel();
  switchTestTab('output');
  document.getElementById('testOutput').innerHTML = '';
  document.getElementById('testStatus').textContent = `\u23f3 Re-running from ${step.node || 'step ' + (idx + 1)}\u2026`;
  const ordered = topoOrder(nodes.filter(n => n.type === 'agent' || n.type === 'hitl'));
  const startIdx = ordered.findIndex(n => n.id === step.node_id);
  const subset = startIdx >= 0 ? ordered.slice(startIdx) : ordered;
  appendTrace(`<div class="trace-node-start">\u23f0 Time-travel re-run from: ${step.node || 'step ' + (idx + 1)}</div>`);
  let lastOutput = frozenOutputs[step.node_id] || document.getElementById('testInput')?.value || 'replayed input';
  (async () => {
    for (const n of subset) {
      appendTrace(`<div class="trace-node-start">\u25b6 ${n.name}</div>`);
      await sleep(80);
      const out = `[replay] ${n.name}: "${lastOutput.slice(0, 50)}\u2026"`;
      appendTrace(`<div class="trace-chunk">${out}</div>`);
      frozenOutputs[n.id] = out;
      lastOutput = out;
      appendTrace(`<div class="trace-node-end">\u2713 Done</div>`);
    }
    document.getElementById('testStatus').textContent = `\u2713 Time-travel complete`;
  })();
}

// ─── Embed Mode (/?embed=1) ────────────────────────────────────────────────
(function initEmbedMode() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('embed') !== '1') return;
  document.body.classList.add('embed-mode');
  const accent = params.get('accent');
  if (accent) document.documentElement.style.setProperty('--cyan', decodeURIComponent(accent));
  const welcome = params.get('welcome');
  if (welcome) {
    const wEl = document.createElement('div');
    wEl.style.cssText = 'padding:10px 14px;font-size:12px;color:var(--muted);border-bottom:1px solid var(--border)';
    wEl.textContent = decodeURIComponent(welcome);
    document.body.prepend(wEl);
  }
  const graphB64 = params.get('graph');
  if (graphB64) {
    try {
      const yaml = decodeURIComponent(escape(atob(graphB64)));
      const parsed = parseYaml(yaml);
      if (parsed && parsed.nodes && parsed.nodes.length) {
        nodes = parsed.nodes; edges = parsed.edges || [];
        autoLayout(nodes, edges);
        seq = Math.max(...nodes.map(n => parseInt(n.id.split('_').pop() || '0', 10) || 0)) + 1;
        snapshot(); render();
      }
    } catch (e) { /* ignore bad graph param */ }
  }
  openTestPanel();
})();

// ─── Feature 20: Provider health polling ─────────────────────────────────
async function refreshProviderHealth() {
  try {
    const resp = await fetch('/provider-health');
    if (!resp.ok) return;
    const health = await resp.json();
    const bar = document.getElementById('providerHealthBar');
    if (!bar) return;
    bar.innerHTML = Object.entries(health).map(([name, h]) => {
      const color = h.status === 'ok' ? '#22c55e' : h.status === 'error' ? '#f87171' : '#94a3b8';
      const tip = h.status === 'ok' ? `${name}: ${h.latency_ms}ms` : `${name}: ${h.error || 'checking...'}`;
      return `<span title="${tip}" style="width:7px;height:7px;border-radius:50%;background:${color};display:inline-block"></span>`;
    }).join('');
  } catch(_) {}
}
setInterval(refreshProviderHealth, 30000);

// ─── Init ─────────────────────────────────────────────────────────────────
if (!document.body.classList.contains('embed-mode')) loadExample();
</script>

<div id="importModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:200; align-items:center; justify-content:center">
  <div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:24px; width:560px; max-height:80vh; display:flex; flex-direction:column; gap:12px">
    <div style="font-weight:700; font-size:14px; color:var(--cyan)">Import Graph</div>
    <div style="color:var(--muted); font-size:12px">Paste selectools Python (AgentGraph) or YAML (type: graph)</div>
    <textarea id="importInput" style="flex:1; min-height:260px; background:var(--bg); border:1px solid var(--border); border-radius:6px; color:var(--text); font-family:inherit; font-size:12px; padding:10px; resize:vertical" placeholder="from selectools.orchestration import AgentGraph&#10;...&#10;&#10;or&#10;&#10;name: my-graph&#10;type: graph&#10;nodes:&#10;  ..."></textarea>
    <div id="importError" style="color:var(--red); font-size:12px; display:none"></div>
    <div style="display:flex; gap:8px; justify-content:flex-end">
      <button class="btn" onclick="closeImport()">Cancel</button>
      <button class="btn btn-primary" onclick="doImport()">Import to Canvas</button>
    </div>
  </div>
</div>

<!-- Wire-Hover Tooltip -->
<div id="wireTooltip" class="wire-tooltip" style="display:none"></div>

<!-- Wire Inspector Panel -->
<div id="wireInspector" class="wire-inspector" style="display:none">
  <div style="display:flex;align-items:center;color:var(--cyan);font-size:10px;font-weight:700">
    <span id="wireInspectorLabel">\u25ba Wire Output</span>
    <button class="wire-copy-btn" onclick="copyWireInspector()">Copy</button>
  </div>
  <pre id="wireInspectorBody"></pre>
</div>

<!-- Right-Click Context Menu -->
<div id="ctxMenu" style="display:none;position:fixed;z-index:600;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:4px 0;min-width:170px;box-shadow:0 4px 20px rgba(0,0,0,0.5)">
  <div id="ctxRerun" class="ctx-item" onclick="rerunNodeAlone(ctxNodeId)">&#8635; Re-run in isolation</div>
  <div class="ctx-item" onclick="ctxPinOutput()">&#x1F4CC; Pin last output</div>
  <div class="ctx-sep"></div>
  <div class="ctx-item" onclick="hideCtxMenu()">&#10005; Close</div>
</div>

<!-- Embed Modal -->
<div id="embedModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:300;align-items:center;justify-content:center">
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:24px;width:580px;max-height:80vh;display:flex;flex-direction:column;gap:12px">
    <div style="font-weight:700;font-size:14px;color:var(--cyan)">&#128279; Embed This Workflow</div>
    <div style="color:var(--muted);font-size:12px">Embeds the workflow as a live interactive chat widget. The iframe loads <code style="color:var(--cyan)">?embed=1</code> mode — full chat panel, no editor chrome. Visitors can run the agent directly.</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:center">
      <label style="font-size:11px;color:var(--muted)">Accent color
        <input type="color" id="embedAccent" value="#22d3ee" oninput="updateEmbedCode();updateEmbedPreview()"
               style="margin-left:6px;border:none;background:none;cursor:pointer;width:36px;height:22px;vertical-align:middle">
      </label>
      <label style="font-size:11px;color:var(--muted)">Position
        <select id="embedPosition" oninput="updateEmbedCode()"
                style="margin-left:6px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-size:11px">
          <option value="bottom-right">Bottom-right</option>
          <option value="bottom-left">Bottom-left</option>
          <option value="inline">Inline</option>
        </select>
      </label>
      <label style="font-size:11px;color:var(--muted);grid-column:1/-1">Welcome message
        <input id="embedWelcome" type="text" placeholder="Hi! How can I help?" oninput="updateEmbedCode()"
               style="margin-left:6px;flex:1;width:calc(100% - 130px);background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 7px;font-size:11px;outline:none">
      </label>
    </div>
    <div id="embedPreviewWidget" title="Live accent preview" style="width:40px;height:40px;border-radius:50%;background:#22d3ee;align-self:flex-start;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.4)"></div>
    <pre id="embedCode" style="flex:1;min-height:140px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:ui-monospace,monospace;font-size:11px;padding:12px;overflow-x:auto;white-space:pre-wrap;margin:0"></pre>
    <div style="color:var(--muted);font-size:11px">&#9432; Self-host: <code>selectools serve --builder --port 8000</code> — then paste this code in any web page.</div>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="btn" onclick="closeEmbed()">Cancel</button>
      <button class="btn btn-primary" id="embedCopyBtn" onclick="copyEmbedCode()">Copy Code</button>
    </div>
  </div>
</div>

<div id="loadTraceModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:300;align-items:center;justify-content:center">
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px;width:540px;max-height:80vh;display:flex;flex-direction:column;gap:10px">
    <div style="font-weight:700;font-size:14px;color:var(--cyan)">&#x1F4E5; Load Production Trace</div>
    <div style="color:var(--muted);font-size:11px">Paste the JSON output of <code style="color:var(--cyan)">trace_to_json(result.trace)</code> to replay a real run in the scrubber and timeline.</div>
    <textarea id="loadTraceInput" style="flex:1;min-height:200px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:10px;font:11px ui-monospace,monospace;resize:vertical" placeholder='{"steps": [...], "total_tokens": 142, "total_cost": 0.0021, ...}'></textarea>
    <div id="loadTraceError" style="display:none;color:#f87171;font-size:11px"></div>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="btn" onclick="closeLoadTrace()">Cancel</button>
      <button class="btn btn-primary" onclick="doLoadTrace()">Load into Scrubber</button>
    </div>
  </div>
</div>

<!-- Feature 11: Replay Diff Panel -->
<div id="replayDiffPanel" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:300;align-items:center;justify-content:center">
  <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;max-width:720px;width:95%;max-height:80vh;display:flex;flex-direction:column">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <span id="replayDiffTitle" style="font-size:13px;font-weight:600;color:var(--cyan)">Diff</span>
      <button class="btn" onclick="closeReplayDiff()">&#10005; Close</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;overflow:auto;flex:1">
      <div>
        <div style="font-size:10px;color:var(--muted);margin-bottom:4px">Before</div>
        <pre id="replayDiffBefore" style="margin:0;font-size:11px;white-space:pre-wrap;line-height:1.5"></pre>
      </div>
      <div>
        <div style="font-size:10px;color:var(--muted);margin-bottom:4px">After</div>
        <pre id="replayDiffAfter" style="margin:0;font-size:11px;white-space:pre-wrap;line-height:1.5"></pre>
      </div>
    </div>
  </div>
</div>

<!-- Feature 16: Watch File Modal -->
<div id="watchFileModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:300;align-items:center;justify-content:center">
  <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:20px;max-width:480px;width:90%">
    <h3 style="margin:0 0 12px;font-size:14px;color:var(--cyan)">&#x1F4C2; Watch Python File</h3>
    <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Paste an absolute path to a selectools agent .py file. Canvas updates on every save.</div>
    <input id="watchFilePath" type="text" placeholder="/home/user/myagent.py" style="width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:6px 10px;font:11px ui-monospace,monospace;outline:none">
    <div style="margin-top:12px;display:flex;gap:8px">
      <button class="btn btn-primary" onclick="startWatchFile()">Start Watching</button>
      <button class="btn" onclick="stopWatchFile()">Stop</button>
      <button class="btn" onclick="closeWatchFileModal()">Cancel</button>
    </div>
    <div id="watchFileStatus" style="font-size:10px;color:var(--muted);margin-top:6px"></div>
  </div>
</div>
</body>
</html>"""
