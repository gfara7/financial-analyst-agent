from __future__ import annotations

import time

from flask import Flask, jsonify, render_template, request

from src.agent.graph import get_graph
from src.agent.state import initial_state


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()

        if len(query) < 3:
            return jsonify({"error": "Please enter a longer question."}), 400
        if len(query) > 2000:
            return jsonify({"error": "Question is too long (max 2000 characters)."}), 400

        started = time.perf_counter()
        try:
            graph = get_graph()
            final_state = graph.invoke(initial_state(query))
            report = (
                final_state.get("final_report")
                or final_state.get("draft_report")
                or "No report generated."
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            return jsonify({"error": f"Agent execution failed: {exc}"}), 500

        elapsed = round(time.perf_counter() - started, 2)
        return jsonify({"answer": report, "elapsed_s": elapsed})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

