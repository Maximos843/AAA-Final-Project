from pathlib import Path

import aiohttp_jinja2
import jinja2
from aiohttp.web import Application

from src.lib import views
from src.lib.model.models import create_model, create_vectorizer


lib = Path("src/lib")


def create_app() -> Application:
    app = Application()
    # setup routes
    app.router.add_static("/static/", lib / "static")
    app.router.add_view("/", views.IndexView, name="index")
    # setup templates
    aiohttp_jinja2.setup(
        app=app,
        loader=jinja2.FileSystemLoader(lib / "templates"),
    )
    app["model"] = create_model()
    app['vectorizer'] = create_vectorizer()
    return app


async def async_create_app() -> Application:
    return create_app()
