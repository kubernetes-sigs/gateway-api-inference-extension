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

import requests
import yaml
import time
import argparse
from jsonschema import validate
from watchfiles import awatch
from dataclasses import dataclass
import asyncio
import logging
import datetime
import os
import sys
from prometheus_client import Gauge, start_http_server
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

# Initialize Prometheus metrics
ADAPTER_STATUS_METRICS = Gauge(
    'lora_syncer_adapter_status',
    'Status of LoRA adapters (1=loaded, 0=not_loaded)',
    ['adapter_name']
)

CONFIG_MAP_FILE = os.environ.get(
    "DYNAMIC_LORA_ROLLOUT_CONFIG", "/config/configmap.yaml"
)
BASE_FIELD = "vLLMLoRAConfig"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d -  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.Formatter.converter = time.localtime


def current_time_human() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='vLLM LoRA Adapter Reconciler')
    parser.add_argument('--health-check-timeout', type=int, default=300,
                        help='Health check timeout in seconds (default: 300)')
    parser.add_argument('--health-check-interval', type=int, default=2,
                        help='Health check interval in seconds (default: 2)')
    parser.add_argument('--reconcile-trigger', type=int, default=5,
                        help='Reconciliation trigger interval in seconds (default: 5)')
    parser.add_argument('--config', type=str, default=CONFIG_MAP_FILE,
                        help=f'Path to config map file (default: {CONFIG_MAP_FILE})')
    parser.add_argument('--config-validation', action='store_true', default=True,
                        help='Enable config validation (default: True)')
    parser.add_argument('--metrics-port', type=int, default=8080,
                        help='Port to listen for Prometheus metrics (default: 8080)')
    return parser.parse_args()


class FileChangeHandler(FileSystemEventHandler):
    """Custom event handler that handles file modifications."""

    def __init__(self, reconciler, config_file):
        super().__init__()
        self.reconciler = reconciler
        self.config_file = config_file

    def on_modified(self, event):
        logging.info("modified!")
        logging.info(f"Config '{self.config_file}' modified!")
        self.reconciler.reconcile()
        logging.info(f"model server reconcile to Config '{self.config_file}' !")


@dataclass
class LoraAdapter:
    """Class representation of lora adapters in config"""

    def __init__(self, id, source="", base_model=""):
        self.id = id
        self.source = source
        self.base_model = base_model

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class LoraReconciler:
    """
    Reconciles adapters registered on vllm server with adapters listed in configmap in current state
    """

    def __init__(self, config_file, health_check_timeout, health_check_interval, 
                 reconcile_trigger_seconds, config_validation=True):
        self.config_file = config_file
        self.config_validation = config_validation
        self.health_check_timeout = datetime.timedelta(seconds=health_check_timeout)
        self.health_check_interval = datetime.timedelta(seconds=health_check_interval)
        self.reconcile_trigger_seconds = reconcile_trigger_seconds
        
        logging.info(f"Settings initialized: health check timeout={health_check_timeout}s, "
                     f"interval={health_check_interval}s, "
                     f"reconcile trigger={self.reconcile_trigger_seconds}s")

    def validate_config(self, c) -> bool:
        try:
            with open("validation.yaml", "r") as f:
                schema = yaml.safe_load(f)
                validate(instance=c, schema=schema)
                return True
        except Exception as e:
            logging.error(f"Cannot load config {self.config_file} validation error: {e}")
            return False

    @property
    def config(self):
        """Load configmap into memory"""
        try:
            with open(self.config_file, "r") as f:
                c = yaml.safe_load(f)
                if self.config_validation and not self.validate_config(c):
                    return {}
                if c is None:
                    c = {}
                c = c.get("vLLMLoRAConfig", {})
                return c
        except Exception as e:
            logging.error(f"cannot load config {self.config_file} {e}")
            return {}

    @property
    def host(self):
        """Model server host"""
        return self.config.get("host", "localhost")

    @property
    def port(self):
        """Model server port"""
        return self.config.get("port", 8000)

    @property
    def model_server(self):
        """Model server {host}:{port}"""
        return f"{self.host}:{self.port}"
    
    @property
    def default_base_model(self):
        """Default base model to use when not specified at adapter level"""
        return self.config.get("defaultBaseModel", "")

    @property
    def ensure_exist_adapters(self):
        """Lora adapters in config under key `ensureExist` in set"""
        adapters = self.config.get("ensureExist", {}).get("models", set())
        default_model = self.default_base_model
        
        return set(
            [
                LoraAdapter(
                    adapter["id"], 
                    adapter["source"], 
                    adapter.get("base-model", default_model)
                )
                for adapter in adapters
            ]
        )

    @property
    def ensure_not_exist_adapters(self):
        """Lora adapters in config under key `ensureNotExist` in set"""
        adapters = self.config.get("ensureNotExist", {}).get("models", set())
        default_model = self.default_base_model
        
        return set(
            [
                LoraAdapter(
                    adapter["id"], 
                    adapter["source"], 
                    adapter.get("base-model", default_model)
                )
                for adapter in adapters
            ]
        )

    @property
    def registered_adapters(self):
        """Lora Adapters registered on model server"""
        url = f"http://{self.model_server}/v1/models"
        if not self.is_server_healthy:
            logging.error(f"vllm server at {self.model_server} not healthy")
            return set()
        try:
            response = requests.get(url)
            adapters = [
                LoraAdapter(a.get("id", ""), a.get("")) for a in response.json()["data"]
            ]
            return set(adapters)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with vLLM server: {e}")
            return set()

    @property
    def is_server_healthy(self) -> bool:
        """probe server's health endpoint until timeout or success"""

        def check_health() -> bool:
            """Checks server health"""
            url = f"http://{self.model_server}/health"
            try:
                response = requests.get(url)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False

        start_time = datetime.datetime.now()
        while datetime.datetime.now() - start_time < self.health_check_timeout:
            if check_health():
                return True
            time.sleep(self.health_check_interval.seconds)
        return False

    def load_adapter(self, adapter: LoraAdapter) -> None | str:
        """Sends a request to load the specified model."""
        if adapter in self.registered_adapters:
            logging.info(
                f"{adapter.id} already present on model server {self.model_server}"
            )
            return
        url = f"http://{self.model_server}/v1/load_lora_adapter"
        payload = {
            "lora_name": adapter.id,
            "lora_path": adapter.source,
            "base_model_name": adapter.base_model,
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logging.info(f"loaded model {adapter.id}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"error loading model {adapter.id}: {e}")
            return f"error loading model {adapter.id}: {e}"

    def unload_adapter(self, adapter: LoraAdapter) -> None | str:
        """Sends a request to unload the specified model."""
        if adapter not in self.registered_adapters:
            logging.info(
                f"{adapter.id} already doesn't exist on model server {self.model_server}"
            )
            return
        url = f"http://{self.model_server}/v1/unload_lora_adapter"
        payload = {"lora_name": adapter.id}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logging.info(f"unloaded model {adapter.id}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"error unloading model {adapter.id}: {e}")
            return f"error unloading model {adapter.id}: {e}"

    def reconcile(self):
        """Reconciles model server with current version of configmap"""
        logging.info(
            f"reconciling model server {self.model_server} with config stored at {self.config_file}"
        )
        
        if not self.is_server_healthy:
            logging.error(f"vllm server at {self.model_server} not healthy")
            return
        invalid_adapters = ", ".join(
            str(a.id)
            for a in self.ensure_exist_adapters & self.ensure_not_exist_adapters
        )
        logging.warning(
            f"skipped adapters found in both `ensureExist` and `ensureNotExist` {invalid_adapters}"
        )
        adapters_to_load = self.ensure_exist_adapters - self.ensure_not_exist_adapters
        adapters_to_load_id = ", ".join(str(a.id) for a in adapters_to_load)
        logging.info(f"adapter to load {adapters_to_load_id}")
        for adapter in adapters_to_load:
            err = self.load_adapter(adapter)
            if err is None:
                self.update_adapter_status_metrics(adapter.id, is_loaded=True)
        adapters_to_unload = self.ensure_not_exist_adapters - self.ensure_exist_adapters
        adapters_to_unload_id = ", ".join(str(a.id) for a in adapters_to_unload)
        logging.info(f"adapters to unload {adapters_to_unload_id}")
        for adapter in adapters_to_unload:
            err = self.unload_adapter(adapter)
            if err is None:
                self.update_adapter_status_metrics(adapter.id, is_loaded=False)

    def update_adapter_status_metrics(self, adapter_id: str, is_loaded: bool):
        """Update adapter status metrics"""
        status = 1 if is_loaded else 0
        ADAPTER_STATUS_METRICS.labels(adapter_name=adapter_id).set(status)



async def main():
    args = parse_arguments()

    # Update CONFIG_MAP_FILE with argument value
    config_file = args.config

    reconciler_instance = LoraReconciler(
        config_file=config_file,
        health_check_timeout=args.health_check_timeout,
        health_check_interval=args.health_check_interval,
        reconcile_trigger_seconds=args.reconcile_trigger,
        config_validation=args.config_validation
    )

    # Start metrics server
    logging.info(f"Starting metrics server on port {args.metrics_port}")
    start_http_server(args.metrics_port)

    logging.info(f"Running initial reconcile for config map {config_file}")
    reconciler_instance.reconcile()

    event_handler = FileChangeHandler(reconciler_instance, config_file)
    observer = Observer()
    observer.schedule(
        event_handler, path=os.path.dirname(config_file), recursive=False
    )
    observer.start()

    try:
        logging.info(f"Starting to watch {config_file} for changes and performing periodic reconciliation...")
        while True:
            # Get current trigger interval from reconciler
            trigger_seconds = reconciler_instance.reconcile_trigger_seconds
            logging.info(f"Waiting {trigger_seconds}s before next reconciliation...")
            # Wait for configured trigger interval
            await asyncio.sleep(trigger_seconds)
            # Force trigger reconciliation
            logging.info("Periodic reconciliation triggered")
            reconciler_instance.reconcile()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    asyncio.run(main())
