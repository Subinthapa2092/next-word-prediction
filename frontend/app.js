// (() => {
//   // Empty = call the same server that served this page (Space or localhost:8000).
//   // On Vercel, frontend/index.html sets window.API_BASE to the Space URL before this file loads.
//   const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
//   const K = 6;

//   const $ = (id) => document.getElementById(id);
//   const editor = $("editor"), mirror = $("mirror"), list = $("list"), status = $("status");
//   let baseStatus = status.textContent.trim() || "GRU";
//   let timer = null, requestId = 0, wakeTries = 0;
//   let current = [], lastMode = "next", suggestedFor = "";

//   // Draw the typed text plus the faded top suggestion, only while it still matches the text
//   const drawGhost = () => {
//     const text = editor.value;
//     mirror.replaceChildren(document.createTextNode(text));
//     if (current.length && suggestedFor === text && text.trim()) {
//       const top = current[0].word;
//       let tail;
//       if (lastMode === "complete") {
//         const prefix = (text.match(/\S+$/) || [""])[0];
//         tail = top.toLowerCase().startsWith(prefix.toLowerCase()) ? top.slice(prefix.length) : "";
//       } else {
//         tail = (/\s$/.test(text) ? "" : " ") + top;
//       }
//       if (tail) {
//         const g = document.createElement("span");
//         g.className = "ghost";
//         g.textContent = tail;
//         mirror.append(g);
//       }
//     }
//     mirror.append("\u200b");
//   };

//   const grow = () => { editor.style.height = "auto"; editor.style.height = editor.scrollHeight + "px"; };

//   const setMessage = (text, isError = false) => {
//     const p = document.createElement("p");
//     p.className = "msg" + (isError ? " error" : "");
//     p.textContent = text;
//     list.replaceChildren(p);
//     current = [];
//     drawGhost();
//   };

//   const render = (suggestions, mode) => {
//     if (!suggestions.length) return setMessage("No suggestion for that. Keep typing.");
//     const max = suggestions[0].probability || 1;
//     list.replaceChildren(...suggestions.map((s, i) => {
//       const b = document.createElement("button");
//       b.type = "button";
//       b.className = "row" + (i === 0 ? " top" : "");
//       b.setAttribute("role", "option");
//       const word = document.createElement("span");
//       word.className = "word";
//       word.textContent = s.word;
//       const track = document.createElement("span");
//       track.className = "track";
//       const bar = document.createElement("span");
//       bar.className = "bar";
//       bar.style.width = Math.max(4, (s.probability / max) * 100) + "%";
//       track.append(bar);
//       const pct = document.createElement("span");
//       pct.className = "pct";
//       pct.textContent = (s.probability * 100).toFixed(s.probability < 0.1 ? 1 : 0) + "%";
//       b.append(word, track, pct);
//       b.addEventListener("click", () => accept(s.word, mode));
//       return b;
//     }));
//   };

//   // Replace the half-typed word (complete mode) or append after the space (next mode)
//   const accept = (word, mode) => {
//     let text = editor.value;
//     if (mode === "complete") text = text.replace(/\S+$/, "");
//     editor.value = text + word + " ";
//     grow();
//     editor.focus();
//     editor.setSelectionRange(editor.value.length, editor.value.length);
//     fetchSuggestions();
//   };

//   async function fetchSuggestions() {
//     const text = editor.value;
//     if (!text.trim()) return setMessage("Suggestions appear here as you type.");
//     const id = ++requestId;
//     try {
//       const res = await fetch(`${API_BASE}/api/predict`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ text, k: K }),
//       });
//       if (id !== requestId) return; // a newer keystroke replaced this request
//       if (!res.ok) {
//         const err = await res.json().catch(() => ({}));
//         return setMessage(err.detail || "The server could not answer. Try again.", true);
//       }
//       const data = await res.json();
//       wakeTries = 0;
//       lastMode = data.mode;
//       suggestedFor = text;
//       render(data.suggestions, data.mode);
//       current = data.suggestions;
//       drawGhost();
//       status.textContent = `${baseStatus} \u00b7 ${data.latency_ms} ms`;
//     } catch {
//       if (id !== requestId) return;
//       setMessage("Waking up the model. This can take up to a minute.", true);
//       if (++wakeTries < 20) setTimeout(() => { if (id === requestId) fetchSuggestions(); }, 4000);
//     }
//   }

//   // Show real model info, and wake a sleeping Space as soon as the page opens
//   fetch(`${API_BASE}/health`)
//     .then((r) => r.json())
//     .then((h) => {
//       if (h.vocab_size) {
//         baseStatus = `${baseStatus.split(" \u00b7 ")[0]} \u00b7 ${h.vocab_size.toLocaleString()} words`;
//         status.textContent = baseStatus;
//       }
//     })
//     .catch(() => { status.textContent = `${baseStatus} \u00b7 waking up`; });

//   editor.addEventListener("input", () => {
//     grow();
//     drawGhost();
//     clearTimeout(timer);
//     timer = setTimeout(fetchSuggestions, 120);
//   });
//   editor.addEventListener("keydown", (e) => {
//     if (e.key === "Tab" && !e.shiftKey && current.length && suggestedFor === editor.value) {
//       e.preventDefault();
//       accept(current[0].word, lastMode);
//     }
//   });

//   $("aboutBtn").addEventListener("click", () => $("about").showModal());
//   $("aboutClose").addEventListener("click", () => $("about").close());
//   grow();
//   drawGhost();
// })();

(() => {
  const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
  const K = 6;

  const editor = document.getElementById("editor");
  const mirror = document.getElementById("mirror");
  const list = document.getElementById("list");
  const status = document.getElementById("status");

  let current = [];
  let lastMode = "next";
  let suggestedFor = "";
  let timer = null;
  let requestId = 0;

  function grow() {
    editor.style.height = "auto";
    editor.style.height = editor.scrollHeight + "px";
  }

  function drawGhost() {
    const text = editor.value;

    mirror.replaceChildren(document.createTextNode(text));

    if (current.length && suggestedFor === text && text.trim()) {
      const word = current[0].word;
      let tail = "";

      if (lastMode === "complete") {
        const match = text.match(/\S+$/);
        const prefix = match ? match[0] : "";

        if (word.toLowerCase().startsWith(prefix.toLowerCase())) {
          tail = word.slice(prefix.length);
        }
      } else {
        tail = (/\s$/.test(text) ? "" : " ") + word;
      }

      if (tail) {
        const ghost = document.createElement("span");
        ghost.className = "ghost";
        ghost.textContent = tail;
        mirror.appendChild(ghost);
      }
    }

    mirror.appendChild(document.createTextNode("\u200b"));
  }

  function showMessage(message, error = false) {
    const p = document.createElement("p");

    p.className = error ? "msg error" : "msg";
    p.textContent = message;

    list.replaceChildren(p);

    current = [];
    suggestedFor = "";

    drawGhost();
  }

  function renderSuggestions(suggestions, mode) {
    if (!Array.isArray(suggestions) || suggestions.length === 0) {
      showMessage("No suggestion for that. Keep typing.");
      return;
    }

    const max = suggestions[0].probability || 1;

    const buttons = suggestions.map((suggestion, index) => {
      const button = document.createElement("button");

      button.type = "button";
      button.className = index === 0 ? "row top" : "row";
      button.setAttribute("role", "option");

      const word = document.createElement("span");
      word.className = "word";
      word.textContent = suggestion.word;

      const track = document.createElement("span");
      track.className = "track";

      const bar = document.createElement("span");
      bar.className = "bar";
      bar.style.width =
        Math.max(4, (suggestion.probability / max) * 100) + "%";

      track.appendChild(bar);

      const percentage = document.createElement("span");
      percentage.className = "pct";
      percentage.textContent =
        (suggestion.probability * 100).toFixed(
          suggestion.probability < 0.1 ? 1 : 0
        ) + "%";

      button.appendChild(word);
      button.appendChild(track);
      button.appendChild(percentage);

      button.addEventListener("click", () => {
        acceptWord(suggestion.word, mode);
      });

      return button;
    });

    list.replaceChildren(...buttons);
  }

  function acceptWord(word, mode) {
    let text = editor.value;

    if (mode === "complete") {
      text = text.replace(/\S+$/, "");
    }

    editor.value = text + word + " ";

    grow();
    editor.focus();

    editor.setSelectionRange(
      editor.value.length,
      editor.value.length
    );

    fetchSuggestions();
  }

  async function fetchSuggestions() {
    const text = editor.value;

    if (!text.trim()) {
      showMessage("Suggestions appear here as you type.");
      return;
    }

    const id = ++requestId;

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: text,
          k: K
        })
      });

      if (id !== requestId) {
        return;
      }

      const data = await response.json();

      if (!response.ok) {
        console.error("API error:", data);

        showMessage(
          data.detail || "The server could not answer. Try again.",
          true
        );

        return;
      }

      console.log("Prediction response:", data);

      current = data.suggestions || [];
      lastMode = data.mode || "next";
      suggestedFor = text;

      renderSuggestions(current, lastMode);
      drawGhost();

      status.textContent =
        `GRU · 10,000 words · ${data.latency_ms} ms`;

    } catch (error) {
      if (id !== requestId) {
        return;
      }

      console.error("Prediction request failed:", error);

      showMessage(
        "Could not connect to the prediction server.",
        true
      );
    }
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`);

      if (!response.ok) {
        throw new Error("Health check failed");
      }

      const data = await response.json();

      console.log("Health response:", data);

      if (data.vocab_size) {
        status.textContent =
          `GRU · ${data.vocab_size.toLocaleString()} words`;
      }

    } catch (error) {
      console.error("Health check failed:", error);
      status.textContent = "GRU · offline";
    }
  }

  editor.addEventListener("input", () => {
    grow();
    drawGhost();

    clearTimeout(timer);

    timer = setTimeout(() => {
      fetchSuggestions();
    }, 120);
  });

  editor.addEventListener("keydown", (event) => {
    if (
      event.key === "Tab" &&
      !event.shiftKey &&
      current.length &&
      suggestedFor === editor.value
    ) {
      event.preventDefault();
      acceptWord(current[0].word, lastMode);
    }
  });

  document.getElementById("aboutBtn").addEventListener("click", () => {
    document.getElementById("about").showModal();
  });

  document.getElementById("aboutClose").addEventListener("click", () => {
    document.getElementById("about").close();
  });

  grow();
  drawGhost();
  checkHealth();
})();