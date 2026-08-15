"""
API-layer contract tests.

These cover behaviour that spans routes rather than any single model: the security
middleware, the shared OHLCV cache, request guards on upload and batch endpoints,
and the JSON-safety of responses. Everything here runs offline — network access is
stubbed — so the suite stays deterministic.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.security import API_KEY_ENV, API_KEY_HEADER, limiter
from src.data import ohlcv


@pytest.fixture(autouse=True)
def _isolate_api_state(monkeypatch):
    """Each test starts with an empty cache, a clean limiter and auth disabled."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    limiter.reset()
    ohlcv.cache_clear()
    yield
    limiter.reset()
    ohlcv.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


def _frame(rows: int = 5, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="D")
    close = [start + i for i in range(rows)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1_000_000 + i for i in range(rows)],
        },
        index=idx,
    )


# ── Security: authentication ────────────────────────────────────────────────

def test_auth_disabled_by_default(client):
    """Local development must keep working with no configuration."""
    assert client.get("/api/data/cache").status_code == 200


def test_auth_required_when_key_configured(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "s3cret")

    assert client.get("/api/data/cache").status_code == 401
    assert client.get("/api/data/cache", headers={API_KEY_HEADER: "wrong"}).status_code == 401
    assert client.get("/api/data/cache", headers={API_KEY_HEADER: "s3cret"}).status_code == 200


@pytest.mark.parametrize("path", ["/health", "/"])
def test_public_paths_never_require_a_key(client, monkeypatch, path):
    """Liveness and service metadata must stay reachable for probes."""
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    assert client.get(path).status_code == 200


# ── Security: rate limiting ─────────────────────────────────────────────────

def test_expensive_endpoints_are_rate_limited(client, monkeypatch):
    """Training is capped far below the default budget — it starts real work."""
    monkeypatch.setattr(
        "src.api.routes.training.threading.Thread",
        lambda *a, **k: type("_NoopThread", (), {"start": lambda self: None})(),
    )
    body = {"symbol": "AAPL", "model_type": "xgboost"}
    codes = [client.post("/api/training/train", json=body).status_code for _ in range(12)]

    assert codes.count(200) == 10, codes
    assert codes.count(429) == 2, codes


def test_rate_limited_response_carries_retry_after(client, monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.training.threading.Thread",
        lambda *a, **k: type("_NoopThread", (), {"start": lambda self: None})(),
    )
    body = {"symbol": "AAPL", "model_type": "xgboost"}
    last = None
    for _ in range(12):
        last = client.post("/api/training/train", json=body)

    assert last.status_code == 429
    assert int(last.headers["Retry-After"]) >= 1


# ── Batch quotes ────────────────────────────────────────────────────────────

def _stub_batch_download(monkeypatch, frame):
    monkeypatch.setattr("src.api.routes.data.yf.download", lambda *a, **k: frame)


def test_batch_quotes_returns_price_and_change(client, monkeypatch):
    frame = pd.concat({"AAPL": _frame(), "MSFT": _frame(start=200.0)}, axis=1)
    _stub_batch_download(monkeypatch, frame)

    payload = client.get("/api/data/quotes?symbols=AAPL,MSFT").json()

    assert payload["returned"] == 2
    assert payload["quotes"]["AAPL"]["price"] == pytest.approx(104.0)
    # Close rises by 1.0 per bar from 103.0 -> 104.0.
    assert payload["quotes"]["AAPL"]["change_pct"] == pytest.approx(100 * 1.0 / 103.0, rel=1e-3)


def test_batch_quotes_omits_unknown_symbols_without_failing(client, monkeypatch):
    """One bad ticker must not blank the caller's whole grid."""
    frame = pd.concat({"AAPL": _frame()}, axis=1)
    _stub_batch_download(monkeypatch, frame)

    payload = client.get("/api/data/quotes?symbols=AAPL,NOSUCHTICKER").json()

    assert payload["returned"] == 1
    assert payload["missing"] == ["NOSUCHTICKER"]


def test_batch_quotes_deduplicates_and_uppercases(client, monkeypatch):
    frame = pd.concat({"AAPL": _frame()}, axis=1)
    _stub_batch_download(monkeypatch, frame)

    payload = client.get("/api/data/quotes?symbols=aapl,AAPL, aapl ").json()

    assert payload["requested"] == 1


def test_batch_quotes_rejects_empty_and_oversized_requests(client):
    assert client.get("/api/data/quotes?symbols=").status_code == 400
    too_many = ",".join(f"SYM{i}" for i in range(101))
    assert client.get(f"/api/data/quotes?symbols={too_many}").status_code == 400


# ── Upload guards ───────────────────────────────────────────────────────────

VALID_CSV = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2026-01-01,1,2,0.5,1.5,100\n"
    b"2026-01-02,1,2,0.5,1.6,120\n"
)


def test_upload_sanitises_path_traversal_filenames(client):
    resp = client.post(
        "/api/data/upload",
        files={"file": ("../../etc/evil.csv", io.BytesIO(VALID_CSV), "text/csv")},
    )

    assert resp.status_code == 200
    # No directory components survive into the stored key or the echoed name.
    assert resp.json()["filename"] == "evil.csv"


def test_upload_rejects_non_csv_and_empty_files(client):
    assert client.post(
        "/api/data/upload",
        files={"file": ("notes.txt", io.BytesIO(VALID_CSV), "text/plain")},
    ).status_code == 400

    assert client.post(
        "/api/data/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    ).status_code == 400


def test_upload_rejects_files_over_the_size_cap(client, monkeypatch):
    # Shrink the cap rather than building a 50 MB payload in the test.
    monkeypatch.setattr("src.api.routes.data.MAX_UPLOAD_BYTES", 512)
    oversized = b"Date,Open,High,Low,Close,Volume\n" + b"2026-01-01,1,2,0.5,1.5,100\n" * 200

    resp = client.post(
        "/api/data/upload",
        files={"file": ("big.csv", io.BytesIO(oversized), "text/csv")},
    )

    assert resp.status_code == 413


# ── Shared OHLCV cache ──────────────────────────────────────────────────────

def test_fetch_ohlcv_serves_second_call_from_cache(monkeypatch):
    calls = []

    def fake_download(symbol, **kwargs):
        calls.append(symbol)
        return _frame()

    monkeypatch.setattr("src.data.ohlcv.yf.download", fake_download)

    first = ohlcv.fetch_ohlcv("AAPL", "1d")
    second = ohlcv.fetch_ohlcv("AAPL", "1d")

    assert len(calls) == 1, "second call should have been served from cache"
    assert first.equals(second)


def test_fetch_ohlcv_drops_duplicate_columns(monkeypatch):
    """
    Guards the yfinance thread-safety workaround: concurrent downloads can return a
    frame with two symbols' columns merged, which otherwise makes row["Close"] a Series.
    """
    merged = pd.concat([_frame(), _frame(start=200.0)], axis=1)
    assert merged.columns.duplicated().any()
    monkeypatch.setattr("src.data.ohlcv.yf.download", lambda *a, **k: merged)

    result = ohlcv.fetch_ohlcv("AAPL", "1d")

    assert not result.columns.duplicated().any()
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_cache_endpoint_reports_bounds(client):
    payload = client.get("/api/data/cache").json()

    assert payload["maxsize"] == ohlcv.MAX_CACHED_FRAMES
    assert "entries_open_cache" in payload and "entries_closed_cache" in payload


# ── Response JSON-safety ────────────────────────────────────────────────────

def test_model_listing_serialises_nan_metrics(client, monkeypatch):
    """
    Training metrics legitimately contain NaN; NaN is not valid JSON and previously
    turned this endpoint into a 500.
    """
    monkeypatch.setattr(
        "src.api.routes.training.list_model_metadata",
        lambda: [{"version_id": "v1", "metrics": {"rmse": float("nan"), "mae": 1.5}}],
    )

    resp = client.get("/api/training/models")

    assert resp.status_code == 200
    model = resp.json()["models"][0]
    assert model["metrics"]["rmse"] is None
    assert model["metrics"]["mae"] == 1.5
