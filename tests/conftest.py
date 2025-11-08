import pytest
from fastapi.testclient import TestClient
from pathlib import Path

try:
    from app.main import app
except Exception:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
