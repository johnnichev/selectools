"""
Example 76: Visual Agent Builder (v0.20.0)

selectools now ships a zero-install visual builder — drag-drop AgentGraph
topology in a browser, generate Python or YAML, copy or download.

Start the builder with one command (no config file needed):

    selectools serve --builder --port 8080

Then open http://localhost:8080/builder in your browser.

You can also run it alongside a live agent:

    selectools serve agent.yaml --builder

This exposes:
  GET /builder   — Visual builder UI
  GET /playground — Chat UI (with agent)
  POST /invoke    — JSON API
  POST /stream    — SSE streaming

The builder generates code compatible with AgentGraph.
Export as Python or YAML using the buttons at the top right.
"""

from selectools.serve.app import BuilderServer


def main() -> None:
    srv = BuilderServer(port=8080)
    print("Visual builder: http://localhost:8080/builder")
    print("Press Ctrl+C to stop.")
    srv.serve()


if __name__ == "__main__":
    main()
