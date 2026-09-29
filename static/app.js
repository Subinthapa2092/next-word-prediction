(() => {
  const editor = document.getElementById("editor");
  const strip = document.getElementById("strip");
  const status = document.getElementById("status");
  const baseStatus = status.textContent;
  let timer = null;
  let requestId = 0;
  let current = [];

  const setMessage = (text, isError = false) => {
    strip.replaceChildren();
    const p = document.createElement("p");
    p.className = "empty" + (isError ? " error" : "");
    p.textContent = text;
    strip.appendChild(p);
    current = [];
  };

  const render = (suggestions, mode) => {
    if (!suggestions.length) return setMessage("No suggestion for that. Keep typing.");
    strip.replaceChildren();
    current = suggestions;
    suggestions.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (i === 0 ? " top" : "");
      b.setAttribute("role", "option");
      const word = document.createElement("span");
      word.textContent = s.word;
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = (s.probability * 100).toFixed(s.probability < 0.1 ? 1 : 0) + "%";
      const bar = document.createElement("span");
      bar.className = "bar";
      bar.style.setProperty("--w", Math.max(3, s.probability * 100) + "%");
      b.append(word, pct, bar);
      b.addEventListener("click", () => accept(s.word, mode));
      strip.appendChild(b);
    });
  };

  // Replace the half-typed word (complete mode) or append after the space (next mode)
  const accept = (word, mode) => {
    let text = editor.value;
    if (mode === "complete") text = text.replace(/\S+$/, "");
    editor.value = text + word + " ";
    editor.focus();
    editor.setSelectionRange(editor.value.length, editor.value.length);
    fetchSuggestions();
  };

  let lastMode = "next";
  async function fetchSuggestions() {
    const text = editor.value;
    if (!text.trim()) return setMessage("Suggestions appear here as you type.");
    const id = ++requestId;
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, k: 5 }),
      });
      if (id !== requestId) return; // a newer keystroke already replaced this request
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return setMessage(err.detail || "The server could not answer. Try again.", true);
      }
      const data = await res.json();
      lastMode = data.mode;
      render(data.suggestions, data.mode);
      status.textContent = `${baseStatus} · ${data.latency_ms} ms`;
    } catch {
      if (id === requestId) setMessage("Cannot reach the server. Check that it is running.", true);
    }
  }

  editor.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(fetchSuggestions, 120);
  });

  editor.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey && current.length) {
      e.preventDefault();
      accept(current[0].word, lastMode);
    }
  });
})();
