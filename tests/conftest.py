"""Keep gameplay regressions independent of locally trained policy files."""
import pytest


@pytest.fixture(autouse=True)
def isolate_optional_tactical_models(monkeypatch):
    # Tests that exercise model loading explicitly override this setting.
    monkeypatch.setenv('SHADOW_CONTEXT_MODELS', 'off')
