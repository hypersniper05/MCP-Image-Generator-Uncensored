"""A small built-in web page: server status, image upload (to use as edit inputs) and recent images."""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Gen MCP</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#667085;--line:#e3e6eb;--accent:#3b6ef5;--ok:#1a9c5b;--warn:#c98a07;--bad:#d64545}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e7eaf0;--muted:#98a2b3;--line:#2b3240;--accent:#6d93ff}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:16px}
.muted{color:var(--muted)}code{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--bg);padding:2px 6px;border-radius:6px;word-break:break-all}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:13px;font-weight:600;border:1px solid var(--line)}
.ready{color:var(--ok)}.error{color:var(--bad)}.busy{color:var(--warn)}
#drop{border:2px dashed var(--line);border-radius:12px;padding:32px;text-align:center;cursor:pointer;transition:border-color .15s}
#drop.over{border-color:var(--accent)}
.row{display:flex;gap:12px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.row:first-child{border-top:0}.row img{width:56px;height:56px;object-fit:cover;border-radius:8px;background:var(--bg)}
.row .grow{flex:1;min-width:0}
button{font:inherit;padding:5px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.grid a{display:block;border-radius:10px;overflow:hidden;border:1px solid var(--line);background:
 repeating-conic-gradient(#8882 0 25%,transparent 0 50%) 50%/16px 16px}
.grid img{width:100%;height:130px;object-fit:cover;display:block}
.grid span{display:block;font-size:12px;padding:4px 6px;background:var(--card);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style></head><body><main>
<h1>Image Gen MCP</h1>
<div class="muted">MCP endpoint (Streamable HTTP, no auth): <code id="ep"></code></div>
<div class="card"><h2>Status</h2><div id="status" class="muted">loading...</div></div>
<div class="card" id="upload"><h2>Upload images</h2>
<p class="muted" style="margin-top:0">Upload photos here, then give their reference to the assistant as input images for
<code>edit_image</code>, <code>generate_panorama</code> or <code>remove_background</code>.</p>
<div id="drop">Drop images here or click to choose<input id="file" type="file" accept="image/*" multiple hidden></div>
<div id="uploaded" style="margin-top:12px"></div></div>
<div class="card"><h2>Recent images</h2><div id="recent" class="grid"></div></div>
</main><script>
const $=s=>document.querySelector(s);
$('#ep').textContent=location.origin+'__MCP_PATH__';
async function status(){try{const r=await fetch('api/status');const s=await r.json();
 const cls=s.state==='ready'?'ready':s.state==='error'?'error':'busy';
 let t=`<span class="pill ${cls}">${s.state}</span> &nbsp; ${s.device.toUpperCase()} &middot; ${s.model.variant} ${s.model.quant}`;
 if(s.placement)t+=` &middot; diffusion: ${s.placement.diffusion}, text encoder: ${s.placement.text_encoder}, VAE: ${s.placement.vae}`;
 if(s.state==='downloading'&&s.download.percent!=null)t+=`<br>downloading models: ${s.download.percent}% of ${s.download.total_gb} GB`;
 if(s.error)t+=`<br><span class="error">${s.error.replace(/</g,'&lt;')}</span>`;
 if(s.running_jobs&&s.running_jobs.length)t+=`<br>running: `+s.running_jobs.map(j=>`${j.kind} ${j.progress_percent}%`).join(', ');
 $('#status').innerHTML=t;}catch(e){$('#status').textContent='server not reachable';}}
async function recent(){try{const r=await fetch('api/images');const d=await r.json();
 $('#recent').innerHTML=d.outputs_and_uploads.map(e=>{const pano=!!e.viewer_url||/panorama-/.test(e.file);
  const href=pano?'view/'+e.file:e.url;return `<a href="${href}" target="_blank" title="${e.file}"><img loading="lazy" src="outputs/${e.file}"><span>${e.file}</span></a>`}).join('')
  ||'<span class="muted">nothing yet</span>';}catch(e){}}
function row(res,name){const d=document.createElement('div');d.className='row';
 d.innerHTML=res.error?`<div class="grow error">${name}: ${res.error}</div>`:
 `<img src="outputs/${res.file}"><div class="grow"><div>${name} &middot; ${res.width}x${res.height}</div><code>${res.file}</code></div><button>Copy</button>`;
 const b=d.querySelector('button');if(b)b.onclick=()=>{navigator.clipboard.writeText(res.file);b.textContent='Copied'};
 $('#uploaded').prepend(d);}
async function send(files){for(const f of files){try{const r=await fetch('upload?name='+encodeURIComponent(f.name),
 {method:'POST',headers:{'Content-Type':f.type||'application/octet-stream'},body:f});row(await r.json(),f.name);}
 catch(e){row({error:String(e)},f.name)}}recent();}
const drop=$('#drop');drop.onclick=()=>$('#file').click();$('#file').onchange=e=>send(e.target.files);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');send(e.dataTransfer.files)};
status();recent();setInterval(status,5000);
</script></body></html>"""
