"""DocRAG from the command line and from MCP clients, over the REST API.

Everything here talks HTTP to a running DocRAG (``DOCRAG_URL``, ``DOCRAG_API_KEY``), so
every server-side guard — the key, the read-only key, rate limits, question slots, the
indexing backlog — applies exactly as it does to the web UI.
"""
