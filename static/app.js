(() => {
  // Empty = call the same server that served this page (Space or localhost:8000).
  // On Vercel, frontend/index.html sets window.API_BASE to the Space URL before this file loads.
  const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
  const K = 9;

  const $ = (id) => document.getElementById(id);
  const editor = $("editor"), grid = $("grid"), status = $("status");
  let baseStatus = status.textContent.trim() || "GRU";
  let timer = null, requestId = 0, current = [], lastMode = "next", wakeTries = 0;

  const setMessage = (text, isError = false) => {
    const p = document.createElement("p");
    p.className = "msg" + (isError ? " error" : "");
    p.textContent = text;
    grid.replaceChildren(p);
    current = [];
  };

  const render = (suggestions, mode) => {
    if (!suggestions.length) return setMessage("No suggestion for that. Keep typing.");
    current = suggestions;
    grid.replaceChildren(...suggestions.map((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (i === 0 ? " top" : "");
      b.setAttribute("role", "option");
      const word = document.createElement("span");
      word.textContent = s.word;
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = (s.probability * 100).toFixed(s.probability < 0.1 ? 1 : 0) + "%";
      b.append(word, pct);
      b.addEventListener("click", () => accept(s.word, mode));
      return b;
    }));
  };

  const grow = () => { editor.style.height = "auto"; editor.style.height = editor.scrollHeight + "px"; };

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
      render(data.suggestions, data.mode);
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
    clearTimeout(timer);
    timer = setTimeout(fetchSuggestions, 120);
  });
  editor.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey && current.length) {
      e.preventDefault();
      accept(current[0].word, lastMode);
    }
  });

  // Menu drawer (phones and tablets) and About dialog
  const setMenu = (open) => {
    document.body.classList.toggle("menu-open", open);
    $("menu").setAttribute("aria-expanded", String(open));
  };
  $("menu").addEventListener("click", () => setMenu(!document.body.classList.contains("menu-open")));
  $("scrim").addEventListener("click", () => setMenu(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });
  $("aboutBtn").addEventListener("click", () => { setMenu(false); $("about").showModal(); });
  $("aboutClose").addEventListener("click", () => $("about").close());
})();