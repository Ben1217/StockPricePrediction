import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.features.feature_engineering import create_sequences
from src.models.model_bundle import list_model_metadata, load_model_bundle, save_model_bundle
from src.models.random_forest_model import RandomForestModel


def test_create_sequences_include_current_row():
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    X_seq, y_seq = create_sequences(X, y, sequence_length=2)

    assert X_seq.shape == (2, 2, 1)
    assert y_seq.tolist() == [20.0, 30.0]
    assert X_seq[0, :, 0].tolist() == [1.0, 2.0]
    assert X_seq[1, :, 0].tolist() == [2.0, 3.0]


def test_model_bundle_round_trip(tmp_path):
    X = np.array(
        [
            [0.1, 1.0],
            [0.2, 1.1],
            [0.3, 1.2],
            [0.4, 1.3],
            [0.5, 1.4],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0, 1], dtype=np.float32)

    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    model = RandomForestModel(params={"n_estimators": 8, "random_state": 42, "n_jobs": 1})
    model.fit(X_scaled, y)

    models_dir = tmp_path / "bundles"
    meta = save_model_bundle(
        model=model,
        model_type="random_forest",
        symbol="AAPL",
        horizon=1,
        feature_columns=["feature_a", "feature_b"],
        scaler=scaler,
        metadata={
            "horizons": [1],
            "feature_config": {"include_technical": True, "include_lags": True},
            "target_type": "direction",
            "objective": "next_day_direction",
            "metrics": {"test": {"accuracy": 0.6, "f1": 0.6667, "roc_auc": 0.75}},
        },
        models_dir=models_dir,
    )

    bundle_dir = models_dir / "AAPL" / "random_forest"
    assert bundle_dir.exists()
    assert (bundle_dir / "model.joblib").exists()
    assert (bundle_dir / "scaler.joblib").exists()
    assert (bundle_dir / "feature_columns.json").exists()
    assert (bundle_dir / "metadata.json").exists()

    loaded = load_model_bundle(
        model_type="random_forest",
        symbol="AAPL",
        horizon=5,
        bundles_dir=models_dir,
    )

    assert loaded is not None
    assert loaded.symbol == "AAPL"
    assert loaded.horizon == 1
    assert loaded.supported_horizons == [1]
    assert loaded.target_type == "direction"
    assert loaded.feature_columns == ["feature_a", "feature_b"]
    assert loaded.scaler is not None
    assert loaded.metadata["bundle_dir"] == str(bundle_dir)
    assert meta["training_horizon"] == 1

    listed = list_model_metadata(
        model_type="random_forest",
        symbol="AAPL",
        horizon=5,
        bundles_dir=models_dir,
        metadata_dir=tmp_path / "unused_metadata",
    )
    assert len(listed) == 1
    assert listed[0]["bundle_dir"] == str(bundle_dir)

    preds = loaded.model.predict(loaded.scaler.transform(X[:2]).astype(np.float32))
    assert len(preds) == 2


def test_model_bundle_round_trip_preserves_index_symbol(tmp_path):
    X = np.array(
        [
            [0.1, 1.0],
            [0.2, 1.1],
            [0.3, 1.2],
            [0.4, 1.3],
            [0.5, 1.4],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0, 1], dtype=np.float32)

    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    model = RandomForestModel(params={"n_estimators": 8, "random_state": 42, "n_jobs": 1})
    model.fit(X_scaled, y)

    models_dir = tmp_path / "bundles"
    save_model_bundle(
        model=model,
        model_type="random_forest",
        symbol="^GSPC",
        horizon=1,
        feature_columns=["feature_a", "feature_b"],
        scaler=scaler,
        metadata={"metrics": {"test": {"accuracy": 0.6}}, "horizons": [1], "target_type": "direction"},
        models_dir=models_dir,
    )

    bundle_dir = models_dir / "^GSPC" / "random_forest"
    loaded = load_model_bundle(
        model_type="random_forest",
        symbol="^GSPC",
        horizon=1,
        bundles_dir=models_dir,
    )
    listed = list_model_metadata(
        model_type="random_forest",
        symbol="^GSPC",
        horizon=1,
        bundles_dir=models_dir,
        metadata_dir=tmp_path / "unused_metadata",
    )

    assert bundle_dir.exists()
    assert loaded is not None
    assert loaded.symbol == "^GSPC"
    assert loaded.supported_horizons == [1]
    assert listed[0]["symbol"] == "^GSPC"
