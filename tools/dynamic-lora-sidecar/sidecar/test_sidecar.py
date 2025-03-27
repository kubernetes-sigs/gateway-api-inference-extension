import unittest
from unittest.mock import patch, Mock, mock_open, call
import yaml
import os
from sidecar import LoraReconciler, CONFIG_MAP_FILE, BASE_FIELD, LoraAdapter

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
    LoraAdapter(a["id"], a["base-model"], a["source"])
    for a in TEST_CONFIG_DATA[BASE_FIELD]["ensureExist"]["models"]
]

NOT_EXIST_ADAPTERS = [
    LoraAdapter(a["id"], a["base-model"], a["source"])
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
            self.reconciler = LoraReconciler(False)
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
                        self.reconciler = LoraReconciler()
                        self.reconciler.reconcile()
                        
                        # 1 adapter is in both exist and not exist list, only 2 are expected to be loaded
                        mock_load.assert_has_calls(
                            calls=[call(EXIST_ADAPTERS[0]), call(EXIST_ADAPTERS[2])]
                        )
                        assert mock_load.call_count == 2
                        
                        # 1 adapter is in both exist and not exist list, only 2 are expected to be unloaded
                        mock_unload.assert_has_calls(
                            calls=[call(NOT_EXIST_ADAPTERS[0]), call(NOT_EXIST_ADAPTERS[2])]
                        )
                        assert mock_unload.call_count == 2

if __name__ == "__main__":
    unittest.main()
