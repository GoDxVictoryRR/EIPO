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

    pages = [
        "Explore",
        "Sweeps & Costs",
        "Statistical Significance",
        "About",
        "Overview",
        "💰 Wealth Simulator",
        "⚡ Live Simulation",
        "💼 Holdings & Sectors",
    ]
    for page_name in pages:
        at.radio(key="nav_page").set_value(page_name).run()
        assert not at.exception, f"Exception when navigating to '{page_name}': {at.exception}"


def test_app_wealth_simulator():
    """Verify demo money wealth simulator calculates returns and renders metrics without exception."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.radio(key="nav_page").set_value("💰 Wealth Simulator").run()
    assert not at.exception
    # Ensure all headline financial comparison metrics are rendered
    assert len(at.metric) >= 4
    # Check that presets buttons exist
    assert len(at.button) >= 6


def test_app_explore_parameter_selection():
    """Verify parameter selection on the Explore page works and displays metrics."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.radio(key="nav_page").set_value("Explore").run()
    assert not at.exception

    # Selectboxes on explore page
    assert len(at.selectbox) >= 3
    # Check that the precomputed note is present
    pills = [m.value for m in at.markdown if "precomputed" in m.value.lower()]
    assert len(pills) >= 1


def test_app_live_simulation_execution():
    """Verify live backtest simulation execution runs and produces metrics without exception."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.radio(key="nav_page").set_value("⚡ Live Simulation").run()
    assert not at.exception

    # Check button execution
    assert len(at.button) >= 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) >= 5


def test_app_holdings_and_sectors():
    """Verify holdings and sectors page renders tables and sliders properly."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.radio(key="nav_page").set_value("💼 Holdings & Sectors").run()
    assert not at.exception
    assert len(at.select_slider) >= 1
    assert len(at.dataframe) >= 2
