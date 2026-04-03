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

    num = int(re.match(r"(\d+)", fname).group(1))

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

    cat_btns = [f'<button class="cat-btn active" data-cat="all">All ({total})</button>']
    for c in all_cats:
        n = sum(1 for e in examples if c in e["categories"])
        label = c.replace("-", " ").title()
        cat_btns.append(f'<button class="cat-btn" data-cat="{c}">{label} ({n})</button>')

    # Build a JSON object of raw sources for lazy rendering
    sources_dict = {ex["file"]: ex["source"] for ex in examples}
    sources_json = json.dumps(sources_dict)

    cards = []
    for ex in examples:
        cats_str = " ".join(ex["categories"])
        cats_html = "".join(
            f'<span class="ec1">{c.replace("-"," ").title()}</span>' for c in ex["categories"]
        )
        key_badge = (
            '<span class="ek">API Key</span>'
            if ex["needs_key"]
            else '<span class="enk">No Key</span>'
        )
        graph_btn = ""
        if ex["has_graph"]:
            graph_btn = f'<a href="../{BUILDER_URL}" class="eab ebu">Open in Builder</a>'

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
            f'target="_blank">GitHub</a>{graph_btn}</div>'
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
:root{{--bg:#0f172a;--sf:#1e293b;--bd:#334155;--tx:#e2e8f0;--dm:#94a3b8;--ft:#64748b;--cy:#22d3ee;--bl:#3b82f6;--gn:#22c55e;--font:'Plus Jakarta Sans',system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,monospace;--gr:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E")}}
html{{scroll-behavior:smooth;-webkit-font-smoothing:antialiased}}
body{{background:var(--bg);color:var(--tx);font-family:var(--font);font-size:14px}}
nav{{position:sticky;top:0;z-index:50;background:rgba(15,23,42,0.85);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--bd);height:52px}}
nav .w{{max-width:960px;margin:0 auto;padding:0 20px;display:flex;align-items:center;justify-content:space-between;height:100%}}
.nl{{font-weight:800;font-size:15px;color:#fff;text-decoration:none}}.nl span{{color:var(--dm);font-weight:500;margin-left:8px;font-size:13px}}
.nr{{display:flex;gap:20px;font-size:13px;color:var(--dm)}}.nr a{{color:inherit;text-decoration:none}}.nr a:hover{{color:#fff}}
.ph{{max-width:960px;margin:0 auto;padding:48px 20px 24px}}
.ph h1{{font-size:28px;letter-spacing:-0.03em;margin-bottom:8px;font-weight:800}}.ph p{{color:var(--dm);font-size:15px;max-width:600px;line-height:1.6}}
.ct{{max-width:960px;margin:0 auto;padding:0 20px 16px;display:flex;flex-direction:column;gap:10px;position:sticky;top:52px;z-index:40;background:var(--bg);padding-top:10px}}
.si{{flex:1;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;color:var(--tx);font-family:var(--font);font-size:14px;outline:none}}
.si:focus{{border-color:var(--cy);box-shadow:0 0 0 2px rgba(34,211,238,0.12)}}.si::placeholder{{color:var(--ft)}}
.cr{{display:flex;flex-wrap:wrap;gap:6px}}
.cb{{font-family:var(--font);font-size:12px;font-weight:500;padding:5px 12px;border-radius:100px;border:1px solid var(--bd);background:transparent;color:var(--dm);cursor:pointer;transition:all .15s}}
.cb:hover{{border-color:var(--dm);color:var(--tx)}}.cb.on{{background:rgba(34,211,238,0.1);border-color:rgba(34,211,238,0.3);color:var(--cy)}}
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
.ec1{{font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:4px;background:rgba(59,130,246,0.1);color:#93c5fd}}
.ea{{display:flex;gap:8px;margin-bottom:12px}}
.eab{{font-family:var(--font);font-size:12px;font-weight:500;padding:5px 12px;border-radius:6px;border:1px solid var(--bd);background:transparent;color:var(--tx);cursor:pointer;text-decoration:none;transition:all .12s;display:inline-block}}
.eab:hover{{border-color:var(--dm);color:#fff}}
.ebu{{color:var(--cy);border-color:rgba(34,211,238,0.3)}}.ebu:hover{{border-color:var(--cy);background:rgba(34,211,238,0.08)}}
.ep{{font-family:var(--mono);font-size:12px;line-height:1.65;background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:16px;overflow-x:auto;max-height:500px;overflow-y:auto;white-space:pre;margin:0}}
.ep .kw{{color:#c084fc}}.ep .cmt{{color:var(--ft)}}.ep .num{{color:#fb923c}}.ep .dec{{color:#fbbf24}}
@media(max-width:640px){{.em,.ed{{display:none}}.nr{{gap:12px}}}}
  </style>
</head>
<body>
<nav><div class="w">
  <a href="../" class="nl">selectools <span>examples</span></a>
  <div class="nr"><a href="../builder/">Builder</a><a href="../QUICKSTART/">Docs</a><a href="{REPO_URL}" target="_blank">GitHub</a></div>
</div></nav>
<div class="ph"><h1>{total} Example Scripts</h1><p>Runnable Python examples covering agents, RAG, multi-agent graphs, evals, streaming, guardrails, and more. {no_key} run without an API key.</p></div>
<div class="ct">
  <input class="si" type="text" placeholder="Search examples\u2026" oninput="flt()" id="si" />
  <div class="cr">{chr(10).join(cat_btns)}</div>
  <div class="rc" id="rc">{total} examples</div>
</div>
<div class="el" id="el">
{chr(10).join(cards)}
</div>
<script>
const SRC={sources_json};
let ac='all';
function flt(){{const q=document.getElementById('si').value.toLowerCase();let c=0;document.querySelectorAll('.ec').forEach(d=>{{const t=d.dataset.title,f=d.dataset.file,cats=d.dataset.cats;const cm=ac==='all'||cats.includes(ac);const sm=!q||t.includes(q)||f.includes(q)||cats.includes(q);const s=cm&&sm;d.style.display=s?'':'none';if(s)c++}});document.getElementById('rc').textContent=c+' example'+(c!==1?'s':'')}}
document.querySelectorAll('.cb').forEach(b=>{{b.addEventListener('click',()=>{{document.querySelectorAll('.cb').forEach(x=>x.classList.remove('on'));b.classList.add('on');ac=b.dataset.cat;flt()}});}});
function hl(s){{s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');s=s.replace(/\\b(from|import|def|class|return|if|elif|else|for|while|with|as|try|except|finally|raise|yield|async|await|and|or|not|in|is|True|False|None|lambda|pass|break|continue)\\b/g,'<span class="kw">$1</span>');s=s.replace(/(#[^\\n]*)/g,'<span class="cmt">$1</span>');s=s.replace(/(@\\w+(?:\\([^)]*\\))?)/g,'<span class="dec">$1</span>');return s}}
function toggle(h){{const c=h.closest('.ec'),b=c.querySelector('.eb'),p=c.querySelector('.ep');c.classList.toggle('op');const open=c.classList.contains('op');b.style.display=open?'':'none';if(open&&!p.dataset.loaded){{p.innerHTML=hl(SRC[c.dataset.file]||'');p.dataset.loaded='1'}}}}
function cpSrc(b){{const f=b.closest('.ec').dataset.file;navigator.clipboard.writeText(SRC[f]||'');b.textContent='Copied!';setTimeout(()=>b.textContent='Copy',1500)}}
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
