from __future__ import annotations

from flask import Flask

from .config import AppConfig
from .logging_utils import configure_logging
from .web.routes import web_bp


def create_app(config: AppConfig | None = None) -> Flask:
    app_config = config or AppConfig.from_env()
    configure_logging(app_config.log_file)

    app = Flask(
        __name__,
        template_folder=str(app_config.base_dir / "app" / "web" / "templates"),
        static_folder=str(app_config.base_dir / "app" / "web" / "static"),
    )
    app.config["APP_CONFIG"] = app_config

    # Disable static file caching in development to prevent Ctrl+Shift+R issues
    if app_config.environment == "development":
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    app.register_blueprint(web_bp)

    return app
