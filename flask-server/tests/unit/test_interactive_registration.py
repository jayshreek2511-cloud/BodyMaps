from flask import Flask

from api.interactive import interactive_blueprint
from services.interactive_registration import register_interactive_if_enabled


def test_interactive_routes_are_not_registered_by_default(monkeypatch):
    monkeypatch.delenv("NNINTERACTIVE_ENABLED", raising=False)
    app = Flask(__name__)

    assert register_interactive_if_enabled(app, interactive_blueprint, "/api") is False
    assert app.test_client().get("/api/interactive/config").status_code == 404


def test_interactive_routes_register_only_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("NNINTERACTIVE_ENABLED", "true")
    app = Flask(__name__)

    assert register_interactive_if_enabled(app, interactive_blueprint, "/api") is True
    response = app.test_client().get("/api/interactive/config")
    assert response.status_code == 200
    assert response.json["enabled"] is True
