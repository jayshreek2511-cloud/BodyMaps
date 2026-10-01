import os


def register_interactive_if_enabled(app, blueprint, url_prefix: str) -> bool:
    """Keep interactive routes absent unless explicitly enabled by the server."""
    if os.environ.get("NNINTERACTIVE_ENABLED", "false").lower() != "true":
        return False
    app.register_blueprint(blueprint, url_prefix=url_prefix)
    return True
