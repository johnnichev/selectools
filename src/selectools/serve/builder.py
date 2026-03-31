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
.node-agent.sel .body { stroke: var(--blue); stroke-width: 2.5; }
.node-loop.sel .body, .node-subgraph.sel .body, .node-note.sel .body { stroke-width: 2.5; filter: drop-shadow(0 0 5px currentColor); }
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
    <button class="btn" onclick="undoAction()">&#8630; Undo</button>
    <button class="btn" onclick="redoAction()">&#8631; Redo</button>
    <button class="btn" onclick="doExport('yaml')">Export YAML</button>
    <button class="btn" onclick="doExport('python')">Export Python</button>
    <button class="btn btn-primary" onclick="openTestPanel()">&#9654; Test</button>
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
      <kbd>Ctrl+V</kbd> paste node
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
    <span id="testStatus" style="margin-left:auto;font-size:11px;color:var(--muted)"></span>
  </div>
  <div id="testOutput" class="test-output"></div>
  <div id="testHistory" class="test-history" style="display:none"></div>
</div>

<script>
// ─── Constants ────────────────────────────────────────────────────────────
const NW = 164, NH = 70, PR = 6;

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
           : 'agent_' + (seq++);
  const base = { id, type, x, y };
  if (type === 'start') return { ...base, name: 'START' };
  if (type === 'end')   return { ...base, name: 'END' };
  if (type === 'loop')  return { ...base, name: 'Loop ' + seq, max_iterations: 5, exit_condition: '' };
  if (type === 'subgraph') return { ...base, name: 'Subgraph ' + seq, graph_name: '' };
  if (type === 'note')  return { ...base, name: 'Note', text: '', color: 'yellow' };
  return { ...base, name: 'Agent ' + seq, provider: 'openai', model: 'gpt-4o-mini', system_prompt: '', tools: '' };
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
function render() { renderEdges(); renderNodes(); updateCode(); updateMinimap(); }

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
      S('rect', { class: 'body', width: NW, height: NH, rx: 8, style: `fill:${nc}22;stroke:${nc};stroke-width:1.5` }, g);
      Stext('NOTE', NW / 2, 18, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
        style: `fill:${nc};font-size:9px;font-weight:700`
      }, g);
      const txt = (n.text || '').slice(0, 40);
      Stext(txt, NW / 2, NH / 2 + 6, {
        class: 'node-sublabel', 'text-anchor': 'middle', 'dominant-baseline': 'middle'
      }, g);
    }

    // Input port (not for START, not for note)
    if (n.type !== 'start' && n.type !== 'note') {
      const pg = S('g', { class: 'port', transform: `translate(0,${NH/2})` }, g);
      S('circle', { cx: 0, cy: 0, r: PR }, pg);
      pg.dataset.nid = n.id; pg.dataset.side = 'in';
      pg.addEventListener('click', portClick);
    }

    // Output ports
    if (n.type !== 'end' && n.type !== 'note') {
      if (n.type === 'loop') {
        // Two output ports for loop: body and done
        const pgb = S('g', { class: 'port', transform: `translate(${NW},${NH*0.3})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pgb);
        pgb.dataset.nid = n.id; pgb.dataset.side = 'out'; pgb.dataset.port = 'body';
        pgb.addEventListener('click', portClick);
        Stext('body', NW + PR + 3, NH * 0.3, {
          class: 'node-sublabel', 'dominant-baseline': 'middle',
          style: 'font-size:8px;fill:#f97316'
        }, g);

        const pgd = S('g', { class: 'port', transform: `translate(${NW},${NH*0.7})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pgd);
        pgd.dataset.nid = n.id; pgd.dataset.side = 'out'; pgd.dataset.port = 'done';
        pgd.addEventListener('click', portClick);
        Stext('done', NW + PR + 3, NH * 0.7, {
          class: 'node-sublabel', 'dominant-baseline': 'middle',
          style: 'font-size:8px;fill:#f97316'
        }, g);
      } else {
        const pg = S('g', { class: 'port', transform: `translate(${NW},${NH/2})` }, g);
        S('circle', { cx: 0, cy: 0, r: PR }, pg);
        pg.dataset.nid = n.id; pg.dataset.side = 'out';
        pg.addEventListener('click', portClick);
      }
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
    let p1;
    if (fn.type === 'loop' && e.port === 'done') {
      p1 = outPortDone(fn);
    } else if (fn.type === 'loop' && e.port === 'body') {
      p1 = outPortBody(fn);
    } else {
      p1 = outPort(fn);
    }
    const p2 = inPort(tn);
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
    loop: '#f97316', subgraph: '#a855f7', note: '#f59e0b'
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
  if (!connecting) {
    if (side === 'out') { connecting = { nid, port }; render(); }
    return;
  }
  if (side === 'in' && connecting.nid !== nid) {
    if (!edges.some(ex => ex.from === connecting.nid && ex.to === nid && ex.port === connecting.port)) {
      const edge = mkEdge(connecting.nid, nid);
      if (connecting.port) edge.port = connecting.port;
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
    hint.textContent = 'Annotation only — not included in generated code.';
    body.appendChild(hint);
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
}

// ─── Keyboard ────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const active = document.activeElement;
  const inInput = ['INPUT','TEXTAREA','SELECT'].includes(active.tagName);

  if (e.key === 'Escape') {
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
    const start = mkNode('start', 60, 185);
    nodes.push(start);
    const processor = mkNode('agent', 280, 185);
    processor.name = 'Processor';
    processor.system_prompt = 'Process the input and produce a result for human review.';
    nodes.push(processor);
    const reviewer = mkNode('agent', 500, 185);
    reviewer.name = 'HumanReviewer';
    reviewer.system_prompt = 'A human reviews the output. Route to "approved" or "rejected".';
    nodes.push(reviewer);
    const end = mkNode('end', 720, 185);
    nodes.push(end);
    edges.push(mkEdge(start.id, processor.id));
    edges.push(mkEdge(processor.id, reviewer.id));
    edges.push({ id: 'h1', from: reviewer.id, to: end.id, label: 'approved' });
    edges.push({ id: 'h2', from: reviewer.id, to: processor.id, label: 'rejected' });

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
  if (!agents.length && !loops.length && !subgraphs.length) return '# Drag agent nodes onto the canvas to generate code';

  const L = [];
  L.push('from selectools import Agent, AgentConfig');
  L.push('from selectools.orchestration import AgentGraph');
  if (subgraphs.length) L.push('from selectools.orchestration import load_subgraph');
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
  if (!agents.length && !loops.length && !subgraphs.length) return '# Drag agent nodes onto the canvas to generate YAML';

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
  return L.join('\\n');
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
  document.getElementById('testOutput').style.display = tab === 'output' ? '' : 'none';
  document.getElementById('testHistory').style.display = tab === 'history' ? '' : 'none';
}

function appendTrace(html) {
  const out = document.getElementById('testOutput');
  out.insertAdjacentHTML('beforeend', html);
  out.scrollTop = out.scrollHeight;
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

  isRunning = false;
  document.getElementById('runBtn').textContent = '\u25b6 Run';
  document.getElementById('runBtn').disabled = false;
  document.getElementById('testStatus').textContent = `Done in ${elapsed}s`;
}

async function runMock(input) {
  const agentNodes = nodes.filter(n => n.type === 'agent');
  if (!agentNodes.length) {
    appendTrace('<div class="trace-error">No agent nodes in graph. Add at least one agent node.</div>');
    return;
  }

  const ordered = topoOrder(agentNodes);
  let lastOutput = input;

  for (const n of ordered) {
    appendTrace(`<div class="trace-node-start">\u25b6 ${n.name} (${n.provider}/${n.model})</div>`);
    currentRunEvents.push({type: 'node_start', node: n.name});

    await sleep(80);
    if (n.tools) {
      const toolList = n.tools.split(',').map(t => t.trim()).filter(Boolean);
      for (const tool of toolList.slice(0, 2)) {
        appendTrace(`<div class="trace-tool">  [tool] ${tool}(query="${lastOutput.slice(0,30)}\u2026")</div>`);
        await sleep(60);
        appendTrace(`<div class="trace-tool-result">  \u2192 [mock result from ${tool}]</div>`);
        currentRunEvents.push({type: 'tool_call', tool});
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
    appendTrace('<div class="trace-node-end">  \u2713 45 tokens \xb7 $0.0000 (mock)</div>');
    currentRunEvents.push({type: 'node_end', tokens: 45, cost: 0});
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
  } else if (ev.type === 'error') {
    appendTrace(`<div class="trace-error">  \u2717 ${ev.message}</div>`);
  } else if (ev.type === 'run_end') {
    const cost = ev.total_cost > 0 ? ` \xb7 $${ev.total_cost.toFixed(5)}` : '';
    appendTrace(`<div class="trace-run-end">\u2705 Run complete \u2014 ${ev.total_tokens || 0} tokens${cost}</div>`);
  }
}

function refreshHistory() {
  document.getElementById('ttab-history').textContent = `History (${runHistory.length})`;
  const el = document.getElementById('testHistory');
  el.innerHTML = runHistory.map((r, i) => `
    <div class="history-item" onclick="replayHistory(${i})">
      <div style="display:flex;justify-content:space-between">
        <span style="color:${r.mock ? '#f59e0b' : 'var(--green)'}; font-size:10px">${r.mock ? '[mock]' : '[live]'} \xb7 ${r.elapsed}s</span>
        <span class="history-time">${r.time}</span>
      </div>
      <div class="history-preview">${r.input}</div>
    </div>
  `).join('');
}

function replayHistory(idx) {
  const r = runHistory[idx];
  document.getElementById('testOutput').innerHTML = '';
  switchTestTab('output');
  for (const ev of r.events) handleTraceEvent(ev);
}

function topoOrder(agentNodes) {
  const startNode = nodes.find(n => n.type === 'start');
  if (!startNode) return agentNodes;
  const result = [];
  const visited = new Set();
  function walk(nid) {
    if (visited.has(nid)) return;
    visited.add(nid);
    const n = nodes.find(x => x.id === nid);
    if (n && n.type === 'agent') result.push(n);
    for (const e of edges) {
      if (e.from === nid) walk(e.to);
    }
  }
  walk(startNode.id);
  for (const n of agentNodes) if (!visited.has(n.id)) result.push(n);
  return result;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Init ─────────────────────────────────────────────────────────────────
loadExample();
</script>
</body>
</html>"""
