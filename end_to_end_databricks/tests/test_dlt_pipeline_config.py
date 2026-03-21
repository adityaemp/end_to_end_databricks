import json
from pathlib import Path


def test_dlt_pipeline_json_exists_and_valid():
    path = Path(__file__).resolve().parents[1] / "resources" / "dlt-sensor-streaming-pipeline.json"
    assert path.exists(), f"DLT pipeline config not found at {path}"

    config = json.loads(path.read_text())
    assert config.get("name") is not None
    assert config.get("target_schema") == "end_to_end_databricks"
    assert config.get("continuous") is True


def test_sensor_features_module_import():
    # Sanity check module imports and function signature without requiring Spark context
    from end_to_end_databricks.feature_engineering.features.sensor_features import compute_features_fn

    assert callable(compute_features_fn)
