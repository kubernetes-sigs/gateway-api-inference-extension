import unittest
from unittest.mock import patch, Mock, mock_open, call
import yaml
import os
import datetime
from sidecar import LoraReconciler, LoraAdapter, CONFIG_MAP_FILE, BASE_FIELD

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


if __name__ == "__main__":
    unittest.main()