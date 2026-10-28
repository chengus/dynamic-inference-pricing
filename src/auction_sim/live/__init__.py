"""Local, synthetic live order-book dashboard."""

from .server import LiveDashboard, make_server, serve_dashboard

__all__ = ["LiveDashboard", "make_server", "serve_dashboard"]
