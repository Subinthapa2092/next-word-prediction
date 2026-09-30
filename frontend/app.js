(() => {
  // Empty = call the same server that served this page (Space or localhost:8000).
  // On Vercel, frontend/index.html sets window.API_BASE to the Space URL before this file loads.
  const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
  const K = 6;

  const $ = (id) => document.getElementById(id);
  const editor = $("editor"), mirror = $("mirror"), list = $("list"), status = $("status");
  let baseStatus = status.textContent.trim() || "GRU";
  let timer = null, requestId = 0, wakeTries = 0;
  let current = [], lastMode = "next", suggestedFor = "";

  // Draw the typed text plus the faded top suggestion, only while it still matches the text
  const drawGhost = () => {
    const text = editor.value;
    mirror.replaceChildren(document.createTextNode(text));
    if (current.length && suggestedFor === text && text.trim()) {
      const top = current[0].word;
      let tail;
      if (lastMode === "complete") {
        const prefix = (text.match(/\S+$/) || [""])[0];
        tail = top.toLowerCase().startsWith(prefix.toLowerCase()) ? top.slice(prefix.length) : "";
      } else {
        tail = (/\s$/.test(text) ? "" : " ") + top;
      }
      if (tail) {
        const g = document.createElement("span");
        g.className = "ghost";
        g.textContent = tail;
        mirror.append(g);
      }
    }
    mirror.append("\u200b");
  };

  const grow = () => { editor.style.height = "auto"; editor.style.height = editor.scrollHeight + "px"; };

  const setMessage = (text, isError = false) => {
    const p = document.createElement("p");
    p.className = "msg" + (isError ? " error" : "");
    p.textContent = text;
    list.replaceChildren(p);
    current = [];
    drawGhost();
  };

  const render = (suggestions, mode) => {
    if (!suggestions.length) return setMessage("No suggestion for that. Keep typing.");
    const max = suggestions[0].probability || 1;
    list.replaceChildren(...suggestions.map((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "row" + (i === 0 ? " top" : "");
      b.setAttribute("role", "option");
      const word = document.createElement("span");
      word.className = "word";
      word.textContent = s.word;
      const track = document.createElement("span");
      track.className = "track";
      const bar = document.createElement("span");
      bar.className = "bar";
      bar.style.width = Math.max(4, (s.probability / max) * 100) + "%";
      track.append(bar);
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = (s.probability * 100).toFixed(s.probability < 0.1 ? 1 : 0) + "%";
      b.append(word, track, pct);
      b.addEventListener("click", () => accept(s.word, mode));
      return b;
    }));
  };

  // Replace the half-typed word (complete mode) or append after the space (next mode)
  const accept = (word, mode) => {
    let text = editor.value;
    if (mode === "complete") text = text.replace(/\S+$/, "");
    editor.value = text + word + " ";
    grow();
    editor.focus();
    editor.setSelectionRange(editor.value.length, editor.value.length);
    fetchSuggestions();
  };

  async function fetchSuggestions() {
    const text = editor.value;
    if (!text.trim()) return setMessage("Suggestions appear here as you type.");
    const id = ++requestId;
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, k: K }),
      });
      if (id !== requestId) return; // a newer keystroke replaced this request
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return setMessage(err.detail || "The server could not answer. Try again.", true);
      }
      const data = await res.json();
      wakeTries = 0;
      lastMode = data.mode;
      suggestedFor = text;
      render(data.suggestions, data.mode);
      current = data.suggestions;
      drawGhost();
      status.textContent = `${baseStatus} \u00b7 ${data.latency_ms} ms`;
    } catch {
      if (id !== requestId) return;
      setMessage("Waking up the model. This can take up to a minute.", true);
      if (++wakeTries < 20) setTimeout(() => { if (id === requestId) fetchSuggestions(); }, 4000);
    }
  }

  // Show real model info, and wake a sleeping Space as soon as the page opens
  fetch(`${API_BASE}/health`)
    .then((r) => r.json())
    .then((h) => {
      if (h.vocab_size) {
        baseStatus = `${baseStatus.split(" \u00b7 ")[0]} \u00b7 ${h.vocab_size.toLocaleString()} words`;
        status.textContent = baseStatus;
      }
    })
    .catch(() => { status.textContent = `${baseStatus} \u00b7 waking up`; });

  editor.addEventListener("input", () => {
    grow();
    drawGhost();
    clearTimeout(timer);
    timer = setTimeout(fetchSuggestions, 120);
  });
  editor.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey && current.length && suggestedFor === editor.value) {
      e.preventDefault();
      accept(current[0].word, lastMode);
    }
  });

  $("aboutBtn").addEventListener("click", () => $("about").showModal());
  $("aboutClose").addEventListener("click", () => $("about").close());
  grow();
  drawGhost();
})();
