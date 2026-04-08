#!/usr/bin/env python3
"""Build the self-contained examples gallery HTML page.

Usage:
    python scripts/build_examples_gallery.py > site/examples/index.html

Scans examples/*.py, extracts metadata from docstrings, embeds syntax-highlighted
source code, and generates a single HTML file with filtering, search, and copy.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
REPO_URL = "https://github.com/johnnichev/selectools"
BUILDER_URL = "builder/"

# Map categories to documentation pages
CAT_DOCS = {
    "agent": "modules/AGENT/",
    "audit": "modules/AUDIT/",
    "caching": "modules/SEMANTIC_CACHE/",
    "config": "modules/TEMPLATES/",
    "deployment": "modules/builder/",
    "evals": "modules/EVALS/",
    "guardrails": "modules/GUARDRAILS/",
    "memory": "modules/MEMORY/",
    "multi-agent": "modules/ORCHESTRATION/",
    "observability": "modules/TRACE_STORE/",
    "patterns": "modules/PATTERNS/",
    "pipeline": "modules/PIPELINE/",
    "rag": "modules/RAG/",
    "sessions": "modules/SESSIONS/",
    "streaming": "modules/STREAMING/",
    "structured": "modules/AGENT/",
    "tools": "modules/TOOLS/",
}

CAT_ICONS = {
    "agent": "&#9889;",
    "audit": "&#128203;",
    "caching": "&#128230;",
    "config": "&#9881;",
    "deployment": "&#128640;",
    "evals": "&#128202;",
    "guardrails": "&#128737;",
    "memory": "&#128024;",
    "multi-agent": "&#129302;",
    "observability": "&#128269;",
    "patterns": "&#129504;",
    "pipeline": "&#128295;",
    "rag": "&#128270;",
    "sessions": "&#128190;",
    "streaming": "&#9889;",
    "structured": "&#128196;",
    "tools": "&#128295;",
}


def extract_metadata(path: str) -> dict:
    """Extract title, description, categories, and source from an example file."""
    with open(path, encoding="utf-8") as f:
        source = f.read()

    m = re.search(r'"""(.*?)"""', source, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", source, re.DOTALL)
    doc = m.group(1).strip() if m else ""

    lines = doc.split("\n")
    title = lines[0].strip().rstrip(".") if lines else ""
    title = re.sub(r"^Example\s+\d+:\s*", "", title)
    fname = os.path.basename(path)
    if title.startswith("from ") or title.startswith("import ") or not title:
        title = fname.replace(".py", "").replace("_", " ").title()
        title = re.sub(r"^\d+\s+", "", title)

    desc_lines = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln or ln.startswith("Prerequisites") or ln.startswith("Run:"):
            continue
        if ln.startswith("python ") or ln.startswith("pip ") or ln.startswith("selectools "):
            continue
        desc_lines.append(ln)
        if len(desc_lines) >= 2:
            break
    desc = " ".join(desc_lines)
    if len(desc) > 220:
        desc = desc[:217] + "..."

    num_match = re.match(r"(\d+)", fname)
    num = int(num_match.group(1)) if num_match else 0

    cats: set[str] = set()
    if "AgentGraph" in source or "agent_graph" in fname:
        cats.add("multi-agent")
    if "SupervisorAgent" in source:
        cats.add("multi-agent")
    if any(
        p in source
        for p in ["PlanAndExecuteAgent", "ReflectiveAgent", "DebateAgent", "TeamLeadAgent"]
    ):
        cats.add("patterns")
    if "rag" in fname or "RAG" in source or "VectorStore" in source:
        cats.add("rag")
    if "stream" in fname or "astream" in source:
        cats.add("streaming")
    if "Pipeline" in source and "pipeline" in fname:
        cats.add("pipeline")
    if "@tool" in source and num <= 13:
        cats.add("tools")
    if "guardrail" in fname.lower() or "Guardrail" in source:
        cats.add("guardrails")
    if "eval" in fname.lower() or "EvalSuite" in source:
        cats.add("evals")
    if "memory" in fname or "ConversationMemory" in source:
        cats.add("memory")
    if "session" in fname or "SessionStore" in source:
        cats.add("sessions")
    if "observer" in fname or "Observer" in source:
        cats.add("observability")
    if "audit" in fname or "AuditLogger" in source:
        cats.add("audit")
    if "structured" in fname or "response_format" in source:
        cats.add("structured")
    if "cache" in fname.lower() or "InMemoryCache" in source:
        cats.add("caching")
    if "serve" in fname or "builder" in fname:
        cats.add("deployment")
    if "yaml" in fname or "template" in fname:
        cats.add("config")
    if not cats:
        cats.add("agent")

    needs_key = "LocalProvider" not in source and "stubs" not in source
    has_graph = "AgentGraph" in source and fname != "76_visual_builder.py"

    return {
        "num": num,
        "file": fname,
        "title": title,
        "desc": desc,
        "categories": sorted(cats),
        "needs_key": needs_key,
        "has_graph": has_graph,
        "lines": source.count("\n") + 1,
        "source": source,
    }


def highlight_python(src: str) -> str:
    """Basic Python syntax highlighting via regex."""
    s = html.escape(src)
    kw = (
        r"\b(from|import|def|class|return|if|elif|else|for|while|with|as|try|except|"
        r"finally|raise|yield|async|await|and|or|not|in|is|True|False|None|lambda|"
        r"pass|break|continue|global|nonlocal|assert|del)\b"
    )
    s = re.sub(kw, r'<span class="kw">\1</span>', s)
    s = re.sub(r"(@\w+(?:\([^)]*\))?)", r'<span class="dec">\1</span>', s)
    s = re.sub(r"(#[^\n]*)", r'<span class="cmt">\1</span>', s)
    s = re.sub(r"\b(\d+\.?\d*)\b", r'<span class="num">\1</span>', s)
    return s


def build_gallery(examples: list[dict]) -> str:
    """Generate the full HTML gallery page."""
    all_cats = sorted(set(c for e in examples for c in e["categories"]))
    total = len(examples)
    no_key = sum(1 for e in examples if not e["needs_key"])

    cat_btns = [f'<button class="cb on" data-cat="all">All ({total})</button>']
    for c in all_cats:
        n = sum(1 for e in examples if c in e["categories"])
        icon = CAT_ICONS.get(c, "")
        label = c.replace("-", " ").title()
        cat_btns.append(f'<button class="cb" data-cat="{c}">{icon} {label} ({n})</button>')

    # Build a JSON object of raw sources for lazy rendering
    sources_dict = {ex["file"]: ex["source"] for ex in examples}
    sources_json = json.dumps(sources_dict)

    cards = []
    for ex in examples:
        cats_str = " ".join(ex["categories"])
        # Category tags link to their doc pages
        cat_parts = []
        for c in ex["categories"]:
            label = c.replace("-", " ").title()
            if c in CAT_DOCS:
                cat_parts.append(f'<a href="../{CAT_DOCS[c]}" class="ec1">{label}</a>')
            else:
                cat_parts.append(f'<span class="ec1">{label}</span>')
        cats_html = "".join(cat_parts)
        key_badge = (
            '<span class="ek">API Key</span>'
            if ex["needs_key"]
            else '<span class="enk">No Key</span>'
        )
        graph_btn = ""
        if ex["has_graph"]:
            graph_btn = f'<a href="../{BUILDER_URL}" class="eab ebu">Open in Builder</a>'
        # Doc link — use first category with a doc page
        doc_btn = ""
        for c in ex["categories"]:
            if c in CAT_DOCS:
                doc_btn = f'<a href="../{CAT_DOCS[c]}" class="eab">Docs</a>'
                break

        cards.append(
            f'<div class="ec" data-cats="{cats_str}" '
            f'data-title="{html.escape(ex["title"].lower())}" data-file="{ex["file"]}">'
            f'<div class="eh" onclick="toggle(this)">'
            f'<div class="en">{ex["num"]:02d}</div>'
            f'<div class="ei"><div class="et">{html.escape(ex["title"])}</div>'
            f'<div class="ed">{html.escape(ex["desc"])}</div></div>'
            f'<div class="em">{key_badge}<span class="eln">{ex["lines"]}L</span></div>'
            f'<div class="ev">&#9660;</div></div>'
            f'<div class="eb" style="display:none">'
            f'<div class="eg">{cats_html}</div>'
            f'<div class="ea">'
            f'<button class="eab" onclick="cpSrc(this)">Copy</button>'
            f'<a href="{REPO_URL}/blob/main/examples/{ex["file"]}" class="eab" '
            f'target="_blank">GitHub</a>{doc_btn}{graph_btn}</div>'
            f'<pre class="ep"></pre>'
            f"</div></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>selectools \u2014 {total} Example Scripts</title>
  <meta name="description" content="{total} runnable Python examples for selectools: agents, RAG, multi-agent graphs, evals, streaming, guardrails, and more." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0f172a;--sf:#1e293b;--bd:#334155;--tx:#e2e8f0;--dm:#94a3b8;--ft:#64748b;--cy:#22d3ee;--bl:#3b82f6;--gn:#22c55e;--font:'Plus Jakarta Sans',system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,monospace;--gr:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E");--exec-color:#22d3ee;--exec-glow:rgba(34,211,238,0.55);--exec-glow-soft:rgba(34,211,238,0.18);--exec-pulse-dur:1.6s;--exec-step-dur:0.55s;--exec-ease-step:cubic-bezier(0.4,0,0.2,1);--exec-ease-soft:cubic-bezier(0.16,1,0.3,1);--exec-blink-dur:1.05s}}
html{{scroll-behavior:smooth;-webkit-font-smoothing:antialiased}}
body{{background:var(--bg);color:var(--tx);font-family:var(--font);font-size:14px}}
nav{{position:sticky;top:0;z-index:50;background:rgba(15,23,42,0.85);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--bd);height:52px}}
nav .w{{max-width:960px;margin:0 auto;padding:0 20px;display:flex;align-items:center;justify-content:space-between;height:100%}}
.nl{{font-weight:800;font-size:15px;color:#fff;text-decoration:none}}.nl span{{color:var(--dm);font-weight:500;margin-left:8px;font-size:13px}}
.nr{{display:flex;gap:20px;font-size:13px;color:var(--dm)}}.nr a{{color:inherit;text-decoration:none}}.nr a:hover{{color:#fff}}
.ex-term{{max-width:960px;margin:32px auto 24px;background:#0b1220;border:1px solid var(--bd);border-radius:14px;box-shadow:0 20px 60px -28px rgba(0,0,0,0.55),0 0 0 1px rgba(34,211,238,0.05);overflow:hidden}}
.ex-term__bar{{display:flex;align-items:center;gap:8px;padding:12px 16px;border-bottom:1px solid var(--bd);background:rgba(15,23,42,0.7)}}
.ex-term__dot{{width:11px;height:11px;border-radius:999px}}
.ex-term__dot--r{{background:rgba(239,68,68,0.85)}}
.ex-term__dot--y{{background:rgba(250,204,21,0.85)}}
.ex-term__dot--g{{background:rgba(34,197,94,0.85)}}
.ex-term__name{{margin-left:8px;font-family:var(--mono);font-size:12px;color:var(--ft)}}
.ex-term__shell{{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--ft);letter-spacing:0.08em}}
.ex-term__body{{padding:22px 22px 24px;font-family:var(--mono)}}
.ex-prompt{{font-family:var(--mono);font-size:13px;line-height:1.75;white-space:pre;overflow-x:auto}}
.ex-prompt__user{{color:var(--cy)}}
.ex-prompt__at{{color:var(--ft)}}
.ex-prompt__host{{color:var(--cy)}}
.ex-prompt__colon{{color:var(--ft)}}
.ex-prompt__path{{color:#fbbf24}}
.ex-prompt__glyph{{color:var(--gn);margin:0 6px}}
.ex-prompt__cmd{{color:var(--tx)}}
.ex-prompt__flags{{color:#fbbf24}}
.ex-prompt__grep{{color:var(--ft)}}
.ex-subtitle{{margin-top:14px;font-family:var(--font);font-size:14px;color:var(--dm);max-width:600px;line-height:1.6}}
@media(max-width:640px){{.ex-prompt__user,.ex-prompt__at,.ex-prompt__host,.ex-prompt__colon,.ex-prompt__path{{display:none}}.ex-prompt__glyph{{margin-left:0}}}}
@media(prefers-reduced-motion:reduce){{.ex-prompt .exec-caret{{animation:none;opacity:1}}}}
.ct{{max-width:960px;margin:0 auto;padding:0 20px 16px;display:flex;flex-direction:column;gap:10px;position:sticky;top:52px;z-index:40;background:var(--bg);padding-top:10px}}
.si{{flex:1;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;color:var(--tx);font-family:var(--font);font-size:14px;outline:none}}
.si:focus{{border-color:var(--cy);box-shadow:0 0 0 2px rgba(34,211,238,0.12)}}.si::placeholder{{color:var(--ft)}}
.cr{{display:flex;flex-wrap:wrap;gap:6px}}
.cb{{font-family:var(--font);font-size:12px;font-weight:500;padding:6px 14px;border-radius:100px;border:1px solid rgba(51,65,85,0.6);background:rgba(30,41,59,0.7);color:var(--dm);cursor:pointer;transition:all .15s;-webkit-backdrop-filter:blur(4px);backdrop-filter:blur(4px)}}
.cb:hover{{background:rgba(51,65,85,0.5);border-color:var(--dm);color:var(--tx)}}.cb.on{{background:rgba(34,211,238,0.12);border-color:rgba(34,211,238,0.35);color:var(--cy);box-shadow:0 0 12px rgba(34,211,238,0.08)}}
.rc{{font-family:var(--mono);font-size:11px;color:var(--ft);padding:2px 0}}
.el{{max-width:960px;margin:0 auto;padding:0 20px 60px;display:flex;flex-direction:column;gap:2px}}
.ec{{border:1px solid var(--bd);border-radius:8px;overflow:hidden;background:var(--sf);background-image:var(--gr);transition:border-color .15s}}
.ec:hover{{border-color:rgba(34,211,238,0.2)}}.ec.op{{border-color:rgba(34,211,238,0.3)}}
.eh{{display:flex;align-items:center;gap:14px;padding:14px 18px;cursor:pointer;user-select:none}}
.en{{font-family:var(--mono);font-size:12px;font-weight:500;color:var(--cy);min-width:24px;flex-shrink:0}}
.ei{{flex:1;min-width:0}}.et{{font-weight:600;font-size:13px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.ed{{font-size:12px;color:var(--dm);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.em{{display:flex;gap:8px;align-items:center;flex-shrink:0}}
.ek{{font-family:var(--mono);font-size:10px;color:var(--ft);background:rgba(100,116,139,0.15);padding:2px 8px;border-radius:100px}}
.enk{{font-family:var(--mono);font-size:10px;color:var(--gn);background:rgba(34,197,94,0.1);padding:2px 8px;border-radius:100px}}
.eln{{font-family:var(--mono);font-size:10px;color:var(--ft)}}
.ev{{font-size:10px;color:var(--ft);transition:transform .2s}}.ec.op .ev{{transform:rotate(180deg)}}
.eb{{padding:0 18px 18px}}.eg{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.ec1{{font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:4px;background:rgba(59,130,246,0.1);color:#93c5fd;text-decoration:none;transition:background .12s}}
a.ec1:hover{{background:rgba(59,130,246,0.2);color:#bfdbfe}}
.ea{{display:flex;gap:8px;margin-bottom:12px}}
.eab{{font-family:var(--font);font-size:12px;font-weight:500;padding:5px 12px;border-radius:6px;border:1px solid var(--bd);background:transparent;color:var(--tx);cursor:pointer;text-decoration:none;transition:all .12s;display:inline-block}}
.eab:hover{{border-color:var(--dm);color:#fff}}
.ebu{{color:var(--cy);border-color:rgba(34,211,238,0.3)}}.ebu:hover{{border-color:var(--cy);background:rgba(34,211,238,0.08)}}
.ep{{font-family:var(--mono);font-size:12px;line-height:1.65;background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:16px;overflow-x:auto;max-height:500px;overflow-y:auto;white-space:pre;margin:0}}
.ep .kw{{color:#c084fc}}.ep .cmt{{color:var(--ft)}}.ep .num{{color:#fb923c}}.ep .dec{{color:#fbbf24}}
@media(max-width:640px){{.em,.ed{{display:none}}.nr{{gap:12px}}}}
.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
.exec-dot{{display:inline-block;width:8px;height:8px;border-radius:999px;background:var(--exec-color);box-shadow:0 0 0 0 var(--exec-glow);animation:exec-pulse var(--exec-pulse-dur) var(--exec-ease-soft) infinite;vertical-align:middle}}
.exec-dot--lg{{width:10px;height:10px}}
.exec-dot--sm{{width:6px;height:6px}}
@keyframes exec-pulse{{0%{{box-shadow:0 0 0 0 var(--exec-glow)}}60%{{box-shadow:0 0 0 8px rgba(34,211,238,0)}}100%{{box-shadow:0 0 0 0 rgba(34,211,238,0)}}}}
.exec-caret{{display:inline-block;width:0.55em;height:1.1em;vertical-align:text-bottom;background:var(--exec-color);box-shadow:0 0 6px var(--exec-glow);animation:exec-blink var(--exec-blink-dur) steps(2,jump-none) infinite;margin-left:2px}}
.exec-caret--thin{{width:2px;box-shadow:0 0 4px var(--exec-glow-soft)}}
@keyframes exec-blink{{0%,49%{{opacity:1}}50%,100%{{opacity:0}}}}
.exec-scan{{position:relative;overflow:hidden}}
.exec-scan.in-view::after{{content:"";position:absolute;top:0;left:-25%;width:25%;height:100%;background:linear-gradient(90deg,rgba(34,211,238,0) 0%,rgba(34,211,238,0.18) 40%,rgba(34,211,238,0.55) 50%,rgba(34,211,238,0.18) 60%,rgba(34,211,238,0) 100%);pointer-events:none;animation:exec-scan-sweep 1.4s var(--exec-ease-step) 0.2s 1 forwards}}
@keyframes exec-scan-sweep{{0%{{transform:translateX(0)}}100%{{transform:translateX(520%)}}}}
@keyframes exec-stamp{{0%{{transform:scale(0.92);box-shadow:0 0 0 0 var(--exec-glow)}}40%{{transform:scale(1.02);box-shadow:0 0 0 6px var(--exec-glow-soft)}}100%{{transform:scale(1);box-shadow:0 0 0 1px rgba(34,211,238,0.18)}}}}
@media(prefers-reduced-motion:reduce){{.exec-dot{{animation:none;box-shadow:0 0 6px var(--exec-glow)}}.exec-caret{{animation:none;opacity:1}}.exec-scan.in-view::after{{animation:none;display:none}}}}
  </style>
</head>
<body>
<nav><div class="w">
  <a href="../" class="nl"><span class="exec-dot"></span>&nbsp;selectools <span>examples</span></a>
  <div class="nr"><a href="../builder/">Builder</a><a href="../QUICKSTART/">Docs</a><a href="{REPO_URL}" target="_blank">GitHub</a></div>
</div></nav>
<header class="ex-term">
  <div class="ex-term__bar">
    <span class="ex-term__dot ex-term__dot--r" aria-hidden="true"></span>
    <span class="ex-term__dot ex-term__dot--y" aria-hidden="true"></span>
    <span class="ex-term__dot ex-term__dot--g" aria-hidden="true"></span>
    <span class="ex-term__name">~/selectools/examples</span>
    <span class="ex-term__shell">zsh</span>
  </div>
  <div class="ex-term__body">
    <div class="ex-prompt" aria-hidden="true"><span class="ex-prompt__user">selectools</span><span class="ex-prompt__at">@</span><span class="ex-prompt__host">examples.dev</span><span class="ex-prompt__colon">:</span><span class="ex-prompt__path">~/selectools/examples</span><span class="ex-prompt__glyph">$</span><span class="ex-prompt__cmd" id="ex-cmd"></span><span class="ex-prompt__flags" id="ex-flags"></span><span class="ex-prompt__grep" id="ex-grep"></span><span class="exec-caret"></span></div>
    <h1 class="sr-only">Selectools examples — {total} runnable Python scripts</h1>
    <p class="ex-subtitle">{total} runnable scripts covering agents, RAG, multi-agent graphs, evals, streaming, and guardrails. {no_key} run without an API key.</p>
  </div>
</header>
<div class="ct">
  <input class="si" type="text" placeholder="Search examples\u2026" oninput="flt();syncPrompt()" id="si" />
  <div class="cr">{chr(10).join(cat_btns)}</div>
  <div class="rc" id="rc">{total} examples</div>
</div>
<div class="el" id="el">
{chr(10).join(cards)}
</div>
<script>
const SRC={sources_json};
let ac='all';
function flt(){{const q=document.getElementById('si').value.toLowerCase();let c=0;document.querySelectorAll('.ec').forEach(d=>{{const t=d.dataset.title,f=d.dataset.file,cats=d.dataset.cats;const cm=ac==='all'||cats.includes(ac);const sm=!q||t.includes(q)||f.includes(q)||cats.includes(q);const s=cm&&sm;d.style.display=s?'':'none';if(s)c++}});document.getElementById('rc').textContent='# '+c+' files match'}}
document.querySelectorAll('.cb').forEach(b=>{{b.addEventListener('click',()=>{{document.querySelectorAll('.cb').forEach(x=>x.classList.remove('on'));b.classList.add('on');ac=b.dataset.cat;flt()}});}});
function hl(s){{s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');s=s.replace(/\\b(from|import|def|class|return|if|elif|else|for|while|with|as|try|except|finally|raise|yield|async|await|and|or|not|in|is|True|False|None|lambda|pass|break|continue)\\b/g,'<span class="kw">$1</span>');s=s.replace(/(#[^\\n]*)/g,'<span class="cmt">$1</span>');s=s.replace(/(@\\w+(?:\\([^)]*\\))?)/g,'<span class="dec">$1</span>');return s}}
function toggle(h){{const c=h.closest('.ec'),b=c.querySelector('.eb'),p=c.querySelector('.ep');c.classList.toggle('op');const open=c.classList.contains('op');b.style.display=open?'':'none';if(open&&!p.dataset.loaded){{p.innerHTML=hl(SRC[c.dataset.file]||'');p.dataset.loaded='1'}}}}
function cpSrc(b){{const f=b.closest('.ec').dataset.file;navigator.clipboard.writeText(SRC[f]||'');b.textContent='Copied!';setTimeout(()=>b.textContent='Copy',1500)}}
function syncPrompt(){{const q=document.getElementById('si').value;document.getElementById('ex-grep').textContent=q?' | grep -i '+q:'';document.getElementById('ex-flags').textContent=ac==='all'?'':' --tags '+ac}}
function typeLine(target,text,perChar,done){{let i=0;const tick=()=>{{if(i<=text.length){{target.textContent=text.slice(0,i);i++;setTimeout(tick,perChar)}}else if(done){{done()}}}};tick()}}
(function bootPrompt(){{const cmd=document.getElementById('ex-cmd');if(!cmd)return;const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;if(reduced){{cmd.textContent='ls examples/';syncPrompt();return}}typeLine(cmd,'ls examples/',35,syncPrompt)}})();
</script>
</body>
</html>"""


def main() -> None:
    examples = []
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(EXAMPLES_DIR, fname)
        examples.append(extract_metadata(path))

    sys.stdout.write(build_gallery(examples))


if __name__ == "__main__":
    main()
