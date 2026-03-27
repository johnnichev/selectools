"""Self-contained chat playground HTML. Zero JS dependencies."""

PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Selectools Playground</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
header { padding: 16px 24px; border-bottom: 1px solid #1e293b; display: flex; align-items: center; gap: 12px; }
header h1 { font-size: 18px; font-weight: 600; }
header .badge { background: #1e293b; color: #06b6d4; font-size: 11px; padding: 2px 8px; border-radius: 99px; }
#chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
.msg.user { align-self: flex-end; background: #3b82f6; color: white; }
.msg.assistant { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
.msg.system { align-self: center; background: transparent; color: #64748b; font-size: 12px; font-style: italic; }
.meta { font-size: 11px; color: #64748b; margin-top: 4px; }
#input-area { padding: 16px 24px; border-top: 1px solid #1e293b; display: flex; gap: 8px; }
#prompt { flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; color: #e2e8f0; font-size: 14px; outline: none; resize: none; font-family: inherit; }
#prompt:focus { border-color: #3b82f6; }
#send { background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 12px 20px; font-size: 14px; cursor: pointer; font-weight: 500; }
#send:hover { background: #2563eb; }
#send:disabled { opacity: 0.5; cursor: not-allowed; }
.typing { display: inline-block; width: 8px; height: 8px; background: #64748b; border-radius: 50%; animation: blink 1s infinite; margin-right: 4px; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>
<header>
  <h1>selectools</h1>
  <span class="badge">playground</span>
</header>
<div id="chat">
  <div class="msg system">Type a message to chat with the agent.</div>
</div>
<div id="input-area">
  <textarea id="prompt" rows="1" placeholder="Send a message..." autofocus></textarea>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('prompt');
const btn = document.getElementById('send');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

async function send() {
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  btn.disabled = true;

  addMsg('user', prompt);
  const assistantEl = addMsg('assistant', '<span class="typing"></span><span class="typing" style="animation-delay:0.2s"></span><span class="typing" style="animation-delay:0.4s"></span>');

  try {
    const res = await fetch('/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    let meta = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\\n')) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') break;
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'chunk') {
            text += parsed.content;
            assistantEl.innerHTML = text;
          } else if (parsed.type === 'result') {
            if (!text) text = parsed.content;
            meta = `${parsed.iterations} iterations`;
            assistantEl.innerHTML = text + '<div class="meta">' + meta + '</div>';
          }
        } catch {}
      }
    }
    if (!text) assistantEl.innerHTML = '<em style="color:#64748b">No response</em>';
  } catch (err) {
    assistantEl.innerHTML = '<em style="color:#ef4444">Error: ' + err.message + '</em>';
  }
  btn.disabled = false;
  input.focus();
}

function addMsg(role, content) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = content;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}
</script>
</body>
</html>"""
