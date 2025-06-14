# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch, Mock, mock_open, call
import yaml
import os
import datetime
from sidecar import LoraReconciler, LoraAdapter, CONFIG_MAP_FILE, BASE_FIELD, ADAPTER_STATUS_METRICS

# Update TEST_CONFIG_DATA to include the new configuration parameters
TEST_CONFIG_DATA = {
    BASE_FIELD: {
        "host": "localhost",
        "name": "sql-loras-llama",
        "port": 8000,
        "ensureExist": {
            "models": [
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "sql-lora-v1",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "sql-lora-v3",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "already_exists",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
            ]
        },
        "ensureNotExist": {
            "models": [
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "sql-lora-v2",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "sql-lora-v3",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
                {
                    "base-model": "meta-llama/Llama-3.1-8B-Instruct",
                    "id": "to_remove",
                    "source": "yard1/llama-2-7b-sql-lora-test",
                },
            ]
        },
    }
}

EXIST_ADAPTERS = [
    LoraAdapter(a["id"], a["source"], a["base-model"])
    for a in TEST_CONFIG_DATA[BASE_FIELD]["ensureExist"]["models"]
]

NOT_EXIST_ADAPTERS = [
    LoraAdapter(a["id"], a["source"], a["base-model"])
    for a in TEST_CONFIG_DATA[BASE_FIELD]["ensureNotExist"]["models"]
]
RESPONSES = {
    "v1/models": {
        "object": "list",
        "data": [
            {
                "id": "already_exists",
                "object": "model",
                "created": 1729693000,
                "owned_by": "vllm",
                "root": "meta-llama/Llama-3.1-8B-Instruct",
                "parent": None,
                "max_model_len": 4096,
            },
            {
                "id": "to_remove",
                "object": "model",
                "created": 1729693000,
                "owned_by": "vllm",
                "root": "yard1/llama-2-7b-sql-lora-test",
                "parent": "base1",
                "max_model_len": None,
            },
        ],
    },
}


def getMockResponse(status_return_value: object = None) -> object:
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    return mock_response


class LoraReconcilerTest(unittest.TestCase):
    @patch(
        "builtins.open", new_callable=mock_open, read_data=yaml.dump(TEST_CONFIG_DATA)
    )
    @patch("sidecar.requests.get")
    def setUp(self, mock_get, mock_file):
        with patch.object(LoraReconciler, "is_server_healthy", return_value=True):
            mock_response = getMockResponse()
            mock_response.json.return_value = RESPONSES["v1/models"]
            mock_get.return_value = mock_response
            
            # Create reconciler with command line argument values instead of config file values
            self.reconciler = LoraReconciler(
                config_file=CONFIG_MAP_FILE,
                health_check_timeout=180,
                health_check_interval=10,
                reconcile_trigger_seconds=30,
                config_validation=False
            )
            self.maxDiff = None

    @patch("sidecar.requests.get")
    @patch("sidecar.requests.post")
    def test_load_adapter(self, mock_post: Mock, mock_get: Mock):
        mock_response = getMockResponse()
        mock_response.json.return_value = RESPONSES["v1/models"]
        mock_get.return_value = mock_response
        mock_file = mock_open(read_data=yaml.dump(TEST_CONFIG_DATA))
        with patch("builtins.open", mock_file):
            with patch.object(LoraReconciler, "is_server_healthy", return_value=True):
                mock_post.return_value = getMockResponse()
                # loading a new adapter
                adapter = EXIST_ADAPTERS[0]
                url = "http://localhost:8000/v1/load_lora_adapter"
                payload = {
                    "lora_name": adapter.id,
                    "lora_path": adapter.source,
                    "base_model_name": adapter.base_model,
                }
                self.reconciler.load_adapter(adapter)
                # adapter 2 already exists `id:already_exists`
                already_exists = EXIST_ADAPTERS[2]
                self.reconciler.load_adapter(already_exists)
                mock_post.assert_called_once_with(url, json=payload)

    @patch("sidecar.requests.get")
    @patch("sidecar.requests.post")
    def test_unload_adapter(self, mock_post: Mock, mock_get: Mock):
        mock_response = getMockResponse()
        mock_response.json.return_value = RESPONSES["v1/models"]
        mock_get.return_value = mock_response
        mock_file = mock_open(read_data=yaml.dump(TEST_CONFIG_DATA))
        with patch("builtins.open", mock_file):
            with patch.object(LoraReconciler, "is_server_healthy", return_value=True):
                mock_post.return_value = getMockResponse()
                # unloading an existing adapter `id:to_remove`
                adapter = NOT_EXIST_ADAPTERS[2]
                self.reconciler.unload_adapter(adapter)
                payload = {"lora_name": adapter.id}
                adapter = NOT_EXIST_ADAPTERS[0]
                self.reconciler.unload_adapter(adapter)
                mock_post.assert_called_once_with(
                    "http://localhost:8000/v1/unload_lora_adapter",
                    json=payload,
                )

    @patch(
        "builtins.open", new_callable=mock_open, read_data=yaml.dump(TEST_CONFIG_DATA)
    )
    @patch("sidecar.requests.get")
    @patch("sidecar.requests.post")
    def test_reconcile(self, mock_post, mock_get, mock_file):
        with patch("builtins.open", mock_file):
            with patch.object(LoraReconciler, "is_server_healthy", return_value=True):
                with patch.object(
                    LoraReconciler, "load_adapter", return_value=""
                ) as mock_load:
                    with patch.object(
                        LoraReconciler, "unload_adapter", return_value=""
                    ) as mock_unload:
                        mock_get_response = getMockResponse()
                        mock_get_response.json.return_value = RESPONSES["v1/models"]
                        mock_get.return_value = mock_get_response
                        mock_post.return_value = getMockResponse()
                        
                        # Create reconciler with command line argument values
                        self.reconciler = LoraReconciler(
                            config_file=CONFIG_MAP_FILE,
                            health_check_timeout=180,
                            health_check_interval=10,
                            reconcile_trigger_seconds=30,
                            config_validation=False
                        )
                        self.reconciler.reconcile()
                        
                        # First check the call count
                        self.assertEqual(mock_load.call_count, 2, "Expected 2 load adapter calls")
                        self.assertEqual(mock_unload.call_count, 2, "Expected 2 unload adapter calls")
                        
                        # Check that the adapters with the correct IDs were loaded
                        loaded_ids = [call.args[0].id for call in mock_load.call_args_list]
                        self.assertIn("sql-lora-v1", loaded_ids, "sql-lora-v1 should have been loaded")
                        self.assertIn("already_exists", loaded_ids, "already_exists should have been loaded")
                        
                        # Check that the adapters with the correct IDs were unloaded
                        unloaded_ids = [call.args[0].id for call in mock_unload.call_args_list]
                        self.assertIn("sql-lora-v2", unloaded_ids, "sql-lora-v2 should have been unloaded")
                        self.assertIn("to_remove", unloaded_ids, "to_remove should have been unloaded")

    def test_health_check_settings(self):
        """Test that health check settings are properly initialized from command line args"""
        # Create reconciler with specific values
        reconciler = LoraReconciler(
            config_file=CONFIG_MAP_FILE,
            health_check_timeout=240,
            health_check_interval=15,
            reconcile_trigger_seconds=45,
            config_validation=False
        )

        # Check that values are properly set
        self.assertEqual(reconciler.health_check_timeout, datetime.timedelta(seconds=240))
        self.assertEqual(reconciler.health_check_interval, datetime.timedelta(seconds=15))
        self.assertEqual(reconciler.reconcile_trigger_seconds, 45)

    def test_update_adapter_status_metrics(self):
        """Test that update_adapter_status_metrics method works correctly"""
        # Clear any existing metrics
        ADAPTER_STATUS_METRICS.clear()

        # Create reconciler
        reconciler = LoraReconciler(
            config_file=CONFIG_MAP_FILE,
            health_check_timeout=180,
            health_check_interval=10,
            reconcile_trigger_seconds=30,
            config_validation=False
        )

        # Test setting loaded status
        reconciler.update_adapter_status_metrics("test-adapter-1", is_loaded=True)
        reconciler.update_adapter_status_metrics("test-adapter-2", is_loaded=False)

        # Get all metric samples
        metric_samples = list(ADAPTER_STATUS_METRICS.collect())[0].samples

        # Check that metrics were set correctly
        adapter_metrics = {}
        for sample in metric_samples:
            adapter_name = sample.labels['adapter_name']
            adapter_metrics[adapter_name] = sample.value

        self.assertEqual(adapter_metrics.get('test-adapter-1'), 1.0, "test-adapter-1 should be marked as loaded")
        self.assertEqual(adapter_metrics.get('test-adapter-2'), 0.0, "test-adapter-2 should be marked as not loaded")

    def test_metrics_endpoint(self):
        """Test that Prometheus metrics can be collected"""
        from prometheus_client import generate_latest

        # Clear metrics and set a test value
        ADAPTER_STATUS_METRICS.clear()
        ADAPTER_STATUS_METRICS.labels(adapter_name='test-adapter').set(1)

        # Test that generate_latest produces valid output
        metrics_bytes = generate_latest()
        metrics = metrics_bytes.decode('utf-8')
        self.assertIn('lora_syncer_adapter_status{adapter_name="test-adapter"} 1.0', metrics)


if __name__ == "__main__":
    unittest.main()
