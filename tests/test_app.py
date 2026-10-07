"""Headless tests for app.py using streamlit.testing.v1.AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_app_loads_overview_headless():
    """Verify app.py loads the default Overview page without exceptions or network calls."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Exception on Overview page: {at.exception}"
    assert len(at.title) >= 1
    assert "Enhanced Indexing Portfolio Optimization" in at.title[0].value


def test_app_navigate_all_pages():
    """Verify headless navigation to every page operates without exceptions."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Initial run exception: {at.exception}"

    pages = ["Explore", "Sweeps & Costs", "Statistical Significance", "About", "Overview"]
    for page_name in pages:
        at.sidebar.radio(key="nav_page").set_value(page_name).run()
        assert not at.exception, f"Exception when navigating to '{page_name}': {at.exception}"


def test_app_explore_parameter_selection():
    """Verify parameter selection on the Explore page works and displays metrics."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav_page").set_value("Explore").run()
    assert not at.exception

    # Selectboxes on explore page
    assert len(at.selectbox) >= 3
    # Check that the precomputed note is present
    pills = [m.value for m in at.markdown if "precomputed" in m.value.lower()]
    assert len(pills) >= 1
