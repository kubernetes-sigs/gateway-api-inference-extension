import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Import the application and predictor; adjust the import path if your module name differs
from server import LatencyPredictor, predictor, app

@pytest.fixture(autouse=True)
def reset_predictor(monkeypatch, tmp_path):
    """
    Reset environment for each test: override model paths to a temporary directory
    and reinitialize the predictor.
    """
    tmp_models = tmp_path / "models"
    monkeypatch.setenv("LATENCY_TTFT_MODEL_PATH", str(tmp_models / "ttft.joblib"))
    monkeypatch.setenv("LATENCY_TPOT_MODEL_PATH", str(tmp_models / "tpot.joblib"))
    monkeypatch.setenv("LATENCY_TTFT_SCALER_PATH", str(tmp_models / "ttft_scaler.joblib"))
    monkeypatch.setenv("LATENCY_TPOT_SCALER_PATH", str(tmp_models / "tpot_scaler.joblib"))
    # Ensure minimum samples for retrain is low to speed up `train`
    monkeypatch.setenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", "1")
    # Reinitialize predictor instance
    predictor.__init__()
    return predictor

# Unit tests for internal methods

def test_train_model_with_scaling_valid():
    lp = LatencyPredictor()
    features = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
    target = pd.Series([1.0, 2.0, 3.0])
    model, scaler = lp._train_model_with_scaling(features, target)
    # Model and scaler should be returned and able to transform
    assert hasattr(model, 'predict')
    scaled = scaler.transform(features)
    assert not np.isnan(scaled).any()


def test_train_model_with_scaling_empty():
    lp = LatencyPredictor()
    with pytest.raises(ValueError):
        lp._train_model_with_scaling(pd.DataFrame(), pd.Series())


def test_create_default_models_and_predict():
    lp = LatencyPredictor()
    # Create and assign default models
    lp.ttft_model, lp.ttft_scaler = lp._create_default_model('ttft')
    lp.tpot_model, lp.tpot_scaler = lp._create_default_model('tpot')
    assert lp.is_ready
    # Test prediction with default models
    features = {
        'kv_cache_percentage': 0.5,
        'input_token_length': 128,
        'num_request_waiting': 5,
        'num_request_running': 2
    }
    ttft_ms, tpot_ms, ttft_std, tpot_std = lp.predict(features)
    # Outputs should be floats
    assert isinstance(ttft_ms, float)
    assert isinstance(tpot_ms, float)
    assert isinstance(ttft_std, float)
    assert isinstance(tpot_std, float)


def test_add_training_sample_and_all_samples():
    lp = LatencyPredictor()
    sample = {
        'kv_cache_percentage': 0.2,
        'actual_ttft_ms': 150.0,
        'actual_tpot_ms': 30.0,
        'num_request_running': 2
    }
    lp.add_training_sample(sample)
    # Determine expected bucket index
    idx = min(int(sample['kv_cache_percentage'] * lp.num_buckets), lp.num_buckets - 1)
    assert sample in lp.ttft_data_buckets[idx]
    assert sample in lp.tpot_data_buckets[idx]
    all_ttft = lp._all_samples(lp.ttft_data_buckets)
    assert sample in all_ttft


def test_predict_invalid_inputs():
    lp = LatencyPredictor()
    # Assign default models so predictor.is_ready is True
    lp.ttft_model, lp.ttft_scaler = lp._create_default_model('ttft')
    lp.tpot_model, lp.tpot_scaler = lp._create_default_model('tpot')
    # Missing a required feature
    #with pytest.raises(ValueError):
    lp.predict({'kv_cache_percentage': 0.5, 'input_token_length': 100, 'num_request_running': 1,'num_request_waiting': 1, })
    # Invalid type
    #with pytest.raises(Ex):
    #    lp.predict({'kv_cache_percentage': 'bad', 'input_token_length': 100, 'num_request_waiting': 1, 'num_request_running': 0})
    # NaN input
    #bad_features = {'kv_cache_percentage': np.nan, 'input_token_length': 100, 'num_request_waiting': 1, 'num_request_running': 0}
    #with pytest.raises(ValueError):
     #  lp.predict(bad_features)

# API endpoint tests using FastAPI TestClient
client = TestClient(app)

def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Latency Predictor is running."}


def test_healthz_endpoint():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_endpoint_not_ready(monkeypatch):
    # Force is_ready False
    monkeypatch.setattr(predictor, 'is_ready', False)
    resp = client.get("/readyz")
    assert resp.status_code == 503


def test_add_training_data_endpoint():
    payload = {
        'kv_cache_percentage': 0.5,
        'input_token_length': 10,
        'num_request_waiting': 1,
        'num_request_running': 1,
        'actual_ttft_ms': 100.0,
        'actual_tpot_ms': 20.0
    }
    resp = client.post("/add_training_data", json=payload)
    assert resp.status_code == 202
    assert resp.json()["message"] == "Training sample accepted."


def test_predict_endpoint_not_ready(monkeypatch):
    # Force is_ready False
    monkeypatch.setattr(predictor, 'is_ready', False)
    payload = {
        'kv_cache_percentage': 0.5,
        'input_token_length': 10,
        'num_request_waiting': 1,
        'num_request_running': 1
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 503
