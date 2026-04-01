from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )

    from .routes import bp

    app.register_blueprint(bp)
    return app
