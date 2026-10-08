"""The in-chat image viewer (MCP Apps, io.modelcontextprotocol/ui).

Chat apps that support MCP Apps show this page in the chat under an image
tool's result. When the job is not finished yet, the page polls the app-only job_status tool and shows the image as
soon as it is ready, so the user sees it without the model calling get_job. Plain JSON-RPC over postMessage (ext-apps
spec 2026-01-26), no external scripts; images arrive as data: URIs, which the default app CSP allows. Clients without
MCP Apps ignore it and use get_job as before.
"""

URI = "ui://imagegen/job-viewer.html"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Image job viewer</title>
<style>
  /* Fallbacks for the standard MCP Apps style variables; the host may override them (hostContext.styles). */
  :root {
    color-scheme: light dark;
    --color-background-secondary: light-dark(#f0f0f0, #2a2a2a);
    --color-text-primary: light-dark(#171717, #fafafa);
    --color-text-secondary: light-dark(#5c5c5c, #a3a3a3);
    --color-text-danger: light-dark(#b42318, #f97066);
    --color-border-primary: light-dark(#d4d4d4, #404040);
    --color-ring-primary: light-dark(#2563eb, #60a5fa);
    --font-sans: system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-text-sm-size: 13px;
    --border-radius-md: 8px;
  }
  html, body { margin: 0; padding: 0; background: transparent; }
  body { font-family: var(--font-sans); font-size: var(--font-text-sm-size); color: var(--color-text-primary); }
  #root { padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  #status { color: var(--color-text-secondary); }
  #status.error { color: var(--color-text-danger); }
  .bar { height: 6px; border-radius: 3px; background: var(--color-background-secondary); overflow: hidden; }
  .bar > div { height: 100%; width: 0; background: var(--color-ring-primary); transition: width .4s ease; }
  .hidden { display: none; }
  #images { display: grid; gap: 8px; }
  #images img { display: block; max-width: 100%; height: auto; max-height: var(--img-max-h, none); justify-self: center;
                border-radius: var(--border-radius-md); border: 1px solid var(--color-border-primary); }
  .caption { color: var(--color-text-secondary); word-break: break-all; }
</style>
</head>
<body>
<div id="root">
  <div id="status">Waiting for the tool result...</div>
  <div class="bar hidden" id="bar"><div id="fill"></div></div>
  <div id="images"></div>
</div>
<script>
(function () {
  "use strict";
  // MCP Apps view (io.modelcontextprotocol/ui), plain JS, no bundler, no CDN.
  // Protocol: JSON-RPC 2.0 over postMessage to window.parent (ext-apps spec 2026-01-26).
  var STATUS_TOOL = "job_status";   // app-only tool on the same server (visibility ["app"])
  var POLL_MS = 2000;
  var MAX_FAILS = 5;
  var IMAGE_TYPES = { "image/png": 1, "image/jpeg": 1, "image/webp": 1, "image/gif": 1 };

  var statusEl = document.getElementById("status");
  var barEl = document.getElementById("bar");
  var fillEl = document.getElementById("fill");
  var imagesEl = document.getElementById("images");

  var nextId = 1, pending = {}, stopped = false, timer = null, jobId = null, fails = 0, gen = 0;
  var sawRunning = false;   // this view watched the job run (so the model has only "not finished yet")

  // ---------------------------------------------------------------- JSON-RPC over postMessage
  function post(msg) { msg.jsonrpc = "2.0"; window.parent.postMessage(msg, "*"); }
  function request(method, params) {
    var id = nextId++;
    return new Promise(function (resolve, reject) {
      pending[id] = { resolve: resolve, reject: reject };
      post({ id: id, method: method, params: params || {} });
    });
  }
  function notify(method, params) {
    var msg = { method: method };
    if (params !== undefined) msg.params = params;
    post(msg);
  }

  window.addEventListener("message", function (ev) {
    if (ev.source !== window.parent) return;          // only the host (or its sandbox proxy)
    var m = ev.data;
    if (!m || typeof m !== "object" || m.jsonrpc !== "2.0") return;
    if (m.method === undefined) {                     // response to one of our requests
      var p = pending[m.id];
      if (!p) return;
      delete pending[m.id];
      if (m.error) p.reject(new Error(m.error.message || "request failed"));
      else p.resolve(m.result || {});
      return;
    }
    if (m.id !== undefined && m.id !== null) {        // request from the host
      if (m.method === "ui/resource-teardown") { stop(); post({ id: m.id, result: {} }); }
      else if (m.method === "ping") post({ id: m.id, result: {} });
      else post({ id: m.id, error: { code: -32601, message: "Method not found: " + m.method } });
      return;
    }
    switch (m.method) {                               // notifications from the host
      case "ui/notifications/tool-result": onToolResult(m.params || {}); break;
      case "ui/notifications/tool-cancelled":
        stop(); hideBar(); setStatus("Cancelled" + (m.params && m.params.reason ? ": " + m.params.reason : "."), true); break;
      case "ui/notifications/host-context-changed": applyHostContext(m.params || {}); break;
      // ui/notifications/tool-input(-partial): the arguments; not needed here.
    }
  });

  // ---------------------------------------------------------------- UI helpers
  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.className = isError ? "error" : "";
    statusEl.classList.toggle("hidden", !text);
  }
  function showBar(pct) {
    barEl.classList.remove("hidden");
    fillEl.style.width = Math.max(0, Math.min(100, Number(pct) || 0)) + "%";
  }
  function hideBar() { barEl.classList.add("hidden"); }
  function textOf(result) {
    return (result.content || []).filter(function (c) { return c && c.type === "text"; })
      .map(function (c) { return c.text; }).join("\n");
  }
  function stop() { stopped = true; gen++; if (timer) clearTimeout(timer); timer = null; }

  // Draws every ImageContent block of a CallToolResult as a data: URI (allowed by the default
  // MCP Apps CSP "img-src 'self' data:", so no csp.resourceDomains and no network access is needed).
  function showImages(result) {
    var blocks = (result.content || []).filter(function (c) {
      return c && c.type === "image" && typeof c.data === "string" && IMAGE_TYPES[c.mimeType];
    });
    if (!blocks.length) return false;
    var meta = (result.structuredContent && result.structuredContent.images) || [];
    imagesEl.textContent = "";
    blocks.forEach(function (c, i) {
      var img = document.createElement("img");
      img.alt = "Generated image " + (i + 1);
      img.src = "data:" + c.mimeType + ";base64," + c.data;
      imagesEl.appendChild(img);
      var info = meta[i];
      if (info) {
        var cap = document.createElement("div");
        cap.className = "caption";
        cap.textContent = [info.width && info.height ? info.width + "x" + info.height : "", info.file || ""]
          .filter(Boolean).join("  ");
        imagesEl.appendChild(cap);
      }
    });
    hideBar();
    setStatus("");
    return true;
  }

  // ---------------------------------------------------------------- tool result and polling
  function isFinal(status) { return status === "completed" || status === "failed" || status === "cancelled"; }

  function onToolResult(result) {
    stop(); stopped = false;
    if (showImages(result)) return;                    // finished inside the tool call
    if (result.isError) { hideBar(); setStatus(textOf(result) || "The tool failed.", true); return; }
    var sc = result.structuredContent || {};
    // "Not finished yet" (running), or finished but the image blocks were dropped by the host or not inlined:
    // fetch through the app-only status tool. The result may also be an old one, replayed when a conversation is
    // reopened, so check the job's real state before showing anything.
    if (sc.job_id && sc.status !== "failed" && sc.status !== "cancelled") {
      jobId = sc.job_id; fails = 0; sawRunning = false;
      hideBar();
      setStatus(sc.status === "completed" || Array.isArray(sc.images) ? "Loading the image..." : "Checking the job...");
      schedule(0);
      return;
    }
    hideBar();
    setStatus(textOf(result) || "No image in this result.");
  }

  function about(s) { return s < 90 ? "about " + Math.max(10, Math.round(s / 10) * 10) + " s" : "about " + Math.round(s / 60) + " min"; }
  function progress(sc) {
    showBar(sc.progress_percent);
    var where = sc.queue_position ? "waiting, " + sc.queue_position + " ahead in line" : "working, " + Math.round(sc.progress_percent || 0) + "% done";
    var eta = typeof sc.eta_seconds === "number" ? (sc.eta_seconds > 5 ? ", " + about(sc.eta_seconds) + " left" : ", almost done") : "";
    setStatus("Generating: " + where + eta);
  }

  // Optional: tell the model the job finished (it only saw "still working"). Hosts may refuse; ignore errors.
  function tellModel(r) {
    var imgs = (r.structuredContent && r.structuredContent.images) || [];
    if (!imgs.length) return;
    var lines = imgs.map(function (i) { return "Image job " + jobId + " finished: " + (i.file || "") + (i.url ? " " + i.url : ""); });
    request("ui/update-model-context", { content: [{ type: "text", text: lines.join("\n") }] })
      .catch(function () {});
  }

  function schedule(ms) {
    if (stopped) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(poll, ms);
  }

  function poll() {
    timer = null;
    if (stopped) return;
    var g = gen;   // answers that arrive after a newer tool-result or a teardown are ignored
    request("tools/call", { name: STATUS_TOOL, arguments: { job_id: jobId } }).then(function (r) {
      if (stopped || g !== gen) return;
      fails = 0;
      var sc = r.structuredContent || {};
      if (sc.status === "expired") { hideBar(); setStatus(textOf(r)); return; }   // old result: neutral note
      if (r.isError) { hideBar(); setStatus(textOf(r) || "The job failed.", true); return; }
      if (sc.status === "completed") {
        if (showImages(r)) { if (sawRunning && !sc.delivered) tellModel(r); }
        else { hideBar(); setStatus(textOf(r) || "Finished, but no image came back."); }
        return;
      }
      if (isFinal(sc.status)) { hideBar(); setStatus("Job " + sc.status + (sc.error ? ": " + sc.error : "."), true); return; }
      sawRunning = true;
      progress(sc);
      schedule(POLL_MS);
    }, function (err) {
      if (stopped || g !== gen) return;
      fails += 1;
      if (fails >= MAX_FAILS) {
        hideBar();
        setStatus("The viewer cannot reach the server (" + err.message + "). Ask the assistant to call get_job with job_id " + jobId + ".", sawRunning);
        return;
      }
      schedule(POLL_MS * (fails + 1));
    });
  }

  // ---------------------------------------------------------------- host context (theme, styles)
  function applyHostContext(ctx) {
    var root = document.documentElement;
    if (ctx.theme === "light" || ctx.theme === "dark") {
      root.setAttribute("data-theme", ctx.theme);
      root.style.colorScheme = ctx.theme;
    }
    var vars = ctx.styles && ctx.styles.variables;
    if (vars) Object.keys(vars).forEach(function (k) { if (vars[k]) root.style.setProperty(k, vars[k]); });
    var fonts = ctx.styles && ctx.styles.css && ctx.styles.css.fonts;
    if (fonts && !document.getElementById("host-fonts")) {
      var s = document.createElement("style");
      s.id = "host-fonts";
      s.textContent = fonts;
      document.head.appendChild(s);
    }
    var ins = ctx.safeAreaInsets;
    if (ins) document.body.style.padding = ins.top + "px " + ins.right + "px " + ins.bottom + "px " + ins.left + "px";
    var cd = ctx.containerDimensions, cap = cd && (cd.maxHeight || cd.height);
    if (typeof cap === "number" && cap > 0) {   // keep tall images inside the host's height limit
      var pad = ins ? (ins.top || 0) + (ins.bottom || 0) : 0;
      root.style.setProperty("--img-max-h", Math.max(120, cap - pad - 70) + "px");
    }
  }

  // ---------------------------------------------------------------- size reporting
  var lastW = 0, lastH = 0, sizeQueued = false;
  function reportSize() {
    if (sizeQueued) return;
    sizeQueued = true;
    requestAnimationFrame(function () {
      sizeQueued = false;
      var html = document.documentElement, old = html.style.height;
      html.style.height = "max-content";               // measure content, not the iframe viewport
      var h = Math.ceil(html.getBoundingClientRect().height);
      html.style.height = old;
      var w = Math.ceil(window.innerWidth);
      if (w !== lastW || h !== lastH) {
        lastW = w; lastH = h;
        notify("ui/notifications/size-changed", { width: w, height: h });
      }
    });
  }

  // ---------------------------------------------------------------- handshake
  request("ui/initialize", {
    appInfo: { name: "imagegen-job-viewer", version: "1.0.0" },
    appCapabilities: { availableDisplayModes: ["inline"] },
    protocolVersion: "2026-01-26"
  }).then(function (res) {
    applyHostContext(res.hostContext || {});
    notify("ui/notifications/initialized");
    reportSize();
    if (typeof ResizeObserver === "function") {
      var ro = new ResizeObserver(reportSize);
      ro.observe(document.documentElement);
      ro.observe(document.body);
    }
  }, function (err) {
    setStatus("Could not connect to the host: " + err.message, true);
  });
})();
</script>
</body>
</html>
"""
