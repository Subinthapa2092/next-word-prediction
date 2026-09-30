(() => {
  const API_BASE = (window.API_BASE || "").replace(/\/$/, "");
  const K = 6;

  const $ = (id) => document.getElementById(id);

  const editor = $("editor");
  const mirror = $("mirror");
  const list = $("list");
  const status = $("status");

  let baseStatus = status.textContent.trim() || "GRU";
  let timer = null;
  let requestId = 0;

  let current = [];
  let lastMode = "next";
  let suggestedFor = "";

  const drawGhost = () => {
    const text = editor.value;

    mirror.replaceChildren(document.createTextNode(text));

    if (current.length && suggestedFor === text && text.trim()) {
      const top = current[0].word;
      let tail = "";

      if (lastMode === "complete") {
        const prefix = (text.match(/\S+$/) || [""])[0];

        if (top.toLowerCase().startsWith(prefix.toLowerCase())) {
          tail = top.slice(prefix.length);
        }
      } else {
        tail = (/\s$/.test(text) ? "" : " ") + top;
      }

      if (tail) {
        const ghost = document.createElement("span");
        ghost.className = "ghost";
        ghost.textContent = tail;
        mirror.appendChild(ghost);
      }
    }

    mirror.appendChild(document.createTextNode("\u200b"));
  };

  const grow = () => {
    editor.style.height = "auto";
    editor.style.height = editor.scrollHeight + "px";
  };

  const setMessage = (text, isError = false) => {
    const p = document.createElement("p");

    p.className = "msg" + (isError ? " error" : "");
    p.textContent = text;

    list.replaceChildren(p);

    current = [];
    suggestedFor = "";

    drawGhost();
  };

  const render = (suggestions, mode) => {
    if (!Array.isArray(suggestions) || !suggestions.length) {
      setMessage("No suggestion for that. Keep typing.");
      return;
    }

    const max = suggestions[0].probability || 1;

    list.replaceChildren(
      ...suggestions.map((s, i) => {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "row" + (i === 0 ? " top" : "");
        button.setAttribute("role", "option");

        const word = document.createElement("span");
        word.className = "word";
        word.textContent = s.word;

        const track = document.createElement("span");
        track.className = "track";

        const bar = document.createElement("span");
        bar.className = "bar";
        bar.style.width =
          Math.max(4, (s.probability / max) * 100) + "%";

        track.appendChild(bar);

        const percentage = document.createElement("span");
        percentage.className = "pct";
        percentage.textContent =
          (s.probability * 100).toFixed(
            s.probability < 0.1 ? 1 : 0
          ) + "%";

        button.appendChild(word);
        button.appendChild(track);
        button.appendChild(percentage);

        button.addEventListener("click", () => {
          accept(s.word, mode);
        });

        return button;
      })
    );
  };

  const accept = (word, mode) => {
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
  };

  async function fetchSuggestions() {
    const text = editor.value;

    if (!text.trim()) {
      setMessage("Suggestions appear here as you type.");
      return;
    }

    const id = ++requestId;

    try {
      const response = await fetch(
        `${API_BASE}/api/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            text: text,
            k: K
          })
        }
      );

      if (id !== requestId) {
        return;
      }

      const data = await response.json();

      if (!response.ok) {
        console.error("API error:", data);

        setMessage(
          data.detail ||
            "The server could not answer. Try again.",
          true
        );

        return;
      }

      current = data.suggestions || [];
      lastMode = data.mode || "next";
      suggestedFor = text;

      render(current, lastMode);
      drawGhost();

      status.textContent =
        `${baseStatus} · ${data.latency_ms} ms`;

    } catch (error) {
      if (id !== requestId) {
        return;
      }

      console.error("Prediction request failed:", error);

      setMessage(
        "Could not connect to the prediction server.",
        true
      );
    }
  }

  fetch(`${API_BASE}/health`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Health check failed");
      }

      return response.json();
    })
    .then((health) => {
      if (health.vocab_size) {
        baseStatus =
          `${baseStatus.split(" · ")[0]} · ` +
          `${health.vocab_size.toLocaleString()} words`;

        status.textContent = baseStatus;
      }
    })
    .catch(() => {
      status.textContent = `${baseStatus} · offline`;
    });

  editor.addEventListener("input", () => {
    grow();
    drawGhost();

    clearTimeout(timer);

    timer = setTimeout(fetchSuggestions, 120);
  });

  editor.addEventListener("keydown", (event) => {
    if (
      event.key === "Tab" &&
      !event.shiftKey &&
      current.length &&
      suggestedFor === editor.value
    ) {
      event.preventDefault();
      accept(current[0].word, lastMode);
    }
  });

  $("aboutBtn").addEventListener("click", () => {
    $("about").showModal();
  });

  $("aboutClose").addEventListener("click", () => {
    $("about").close();
  });

  grow();
  drawGhost();
})();