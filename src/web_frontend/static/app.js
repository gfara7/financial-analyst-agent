const form = document.getElementById("ask-form");
const textarea = document.getElementById("query");
const messages = document.getElementById("messages");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");

function addMessage(kind, title, html) {
  const card = document.createElement("article");
  card.className = `msg ${kind}`;
  card.innerHTML = `<h2>${title}</h2><div>${html}</div>`;
  messages.appendChild(card);
  messages.scrollTop = messages.scrollHeight;
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Running..." : "Run Analysis";
  statusEl.textContent = isLoading ? "Analyzing query..." : "Idle";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = textarea.value.trim();
  if (!query) return;

  addMessage("msg-user", "You", `<p>${query.replaceAll("<", "&lt;")}</p>`);
  textarea.value = "";
  setLoading(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const payload = await response.json();
    if (!response.ok) {
      const err = payload.error || "Request failed.";
      addMessage("msg-error", "Error", `<p>${err.replaceAll("<", "&lt;")}</p>`);
      statusEl.textContent = "Failed";
      return;
    }

    const rendered = marked.parse(payload.answer || "");
    addMessage("msg-assistant", "Assistant", rendered);
    statusEl.textContent = `Done in ${payload.elapsed_s ?? "?"}s`;
  } catch (err) {
    addMessage("msg-error", "Error", `<p>${String(err).replaceAll("<", "&lt;")}</p>`);
    statusEl.textContent = "Failed";
  } finally {
    setLoading(false);
  }
});

