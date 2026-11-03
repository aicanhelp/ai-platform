# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Container info collector for eagle-eye: collects container metadata (container name,
pod name, namespace) mapped to NPU devices.
  1. Connect to containerd socket (CRI + OCI clients on unix:///run/containerd/containerd.sock)
  2. CRI ListContainers() -> get container IDs + K8s labels (namespace, pod name, container name)
  3. OCI Get() with namespace k8s.io -> get container spec (device files + env vars)
  4. Extract NPU device IDs from ASCEND_DEVICE_ID env or from spec.Linux.Resources.Devices
  5. Map device_id -> container info and expose as card_id:device_id -> {container_name, pod_name, namespace}
"""

import logging
import os
import re
import json
import grpc
from typing import Dict, List, Optional
from dataclasses import dataclass
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.protobuf import json_format

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)

from common.base.base_info_collector import BaseInfoCollector
from hardware_monitor.collector.container.v1 import containerd_pb2
from hardware_monitor.collector.container.v1 import containerd_pb2_grpc
from common.npu_metrics.constants import MAX_INT32

# Import CRI runtime protos - import v1 first since most containerd versions support it
# Only import one version at module level to avoid duplicate proto file name conflict
try:
    from hardware_monitor.collector.runtime.v1 import runtime_pb2 as cri_v1_pb2
    from hardware_monitor.collector.runtime.v1 import runtime_pb2_grpc as cri_v1_grpc

    cri_v1alpha2_pb2 = None
    cri_v1alpha2_grpc = None
except (ImportError, Exception) as e:
    logger.warning("Failed to import CRI v1, will try v1alpha2: %s", str(e))
    cri_v1_pb2 = None
    cri_v1_grpc = None
    # v1alpha2 will be imported lazily in init() if needed
    cri_v1alpha2_pb2 = None
    cri_v1alpha2_grpc = None

# Constants
DEFAULT_CONTAINERD_SOCK = "unix:///run/containerd/containerd.sock"
NAMESPACE_K8S = "k8s.io"
GRPC_HEADER_NAMESPACE = "containerd-namespace"

# Kubernetes CRI labels
LABEL_K8S_POD_NAMESPACE = "io.kubernetes.pod.namespace"
LABEL_K8S_POD_NAME = "io.kubernetes.pod.name"
LABEL_CONTAINER_NAME = "io.kubernetes.container.name"

# Support both env var names used in the field (ASCEND_VISIBLE_DEVICES is used by some runtimes)
ASCEND_DEVICE_ID_ENV = "ASCEND_DEVICE_ID"
ASCEND_VISIBLE_DEVICES_ENV = "ASCEND_VISIBLE_DEVICES"
ASCEND_DEVICE_ENV_NAMES = (ASCEND_DEVICE_ID_ENV, ASCEND_VISIBLE_DEVICES_ENV)
CHAR_DEVICE = "c"
PROC_DEVICES_PATH = "/proc/devices"
NPU_MAJOR_PATTERN = re.compile(r"^(\d{1,3})\s+[v]?devdrv-cdev$")

# Limits
MAX_DEVICES_NUM = 1000
MAX_ENV_NUM = 10000
MAX_ENV_LENGTH = 1024
MAX_PARSE_WORKERS = 8

# Cache key for container devices
CONTAINERS_DEVICES_CACHE_KEY = "containers-devices"


@dataclass
class CommonContainer:
    """Container with ID and labels"""
    id: str
    labels: Dict[str, str]


@dataclass
class DevicesInfo:
    """Container device information"""
    container_id: str
    namespace: str
    pod_name: str
    container_name: str
    device_ids: List[int]  # List of NPU device IDs (logic IDs or minor numbers)


class ContainerdRuntimeOperator:
    """
    Runtime operator for containerd using CRI and OCI APIs.
    """

    def __init__(self,
                 cri_endpoint: str = DEFAULT_CONTAINERD_SOCK,
                 oci_endpoint: str = DEFAULT_CONTAINERD_SOCK,
                 namespace: str = NAMESPACE_K8S):
        self.cri_endpoint = cri_endpoint
        self.oci_endpoint = oci_endpoint
        self.namespace = namespace

        # gRPC connections and clients
        self.cri_channel = None
        self.oci_channel = None
        self.cri_client_v1alpha2 = None
        self.cri_client_v1 = None
        self.oci_client = None

        # Cache for NPU major IDs
        self._npu_major_ids = None
        self._npu_major_lock = Lock()

    def init(self) -> None:
        """Initialize CRI and OCI clients"""
        global cri_v1alpha2_pb2, cri_v1alpha2_grpc, cri_v1_pb2, cri_v1_grpc

        # Connect to CRI endpoint
        logger.info("Connecting to CRI endpoint: %s", self.cri_endpoint)
        self.cri_channel = grpc.insecure_channel(self.cri_endpoint)

        # Try v1 first if available (most containerd versions support it), fallback to v1alpha2
        if cri_v1_grpc:
            try:
                self.cri_client_v1 = cri_v1_grpc.RuntimeServiceStub(self.cri_channel)
                # Test connection by calling ListContainers
                request = self._gen_container_request_v1()
                self.cri_client_v1.ListContainers(request, timeout=5)
            except Exception as e:
                logger.warning("CRI v1 connection failed, trying v1alpha2: %s", str(e))
                if cri_v1alpha2_grpc is None:
                    try:
                        from hardware_monitor.collector.runtime.v1alpha2 import runtime_pb2 as cri_v1alpha2_pb2
                        from hardware_monitor.collector.runtime.v1alpha2 import runtime_pb2_grpc as cri_v1alpha2_grpc
                    except Exception as e2:
                        logger.error("Failed to import CRI v1alpha2: %s", str(e2))
                        raise
                self.cri_client_v1alpha2 = cri_v1alpha2_grpc.RuntimeServiceStub(self.cri_channel)
                self._test_cri_connection()
        elif cri_v1alpha2_grpc:
            self.cri_client_v1alpha2 = cri_v1alpha2_grpc.RuntimeServiceStub(self.cri_channel)
            self._test_cri_connection()
        else:
            raise RuntimeError("No CRI runtime proto modules available")

        # Connect to OCI endpoint
        logger.info("Connecting to OCI endpoint: %s", self.oci_endpoint)
        self.oci_channel = grpc.insecure_channel(self.oci_endpoint)
        self.oci_client = containerd_pb2_grpc.ContainersStub(self.oci_channel)

        logger.info("Successfully initialized containerd clients")

    def _test_cri_connection(self):
        """Test CRI connection"""
        request = self._gen_container_request_v1alpha2()
        self.cri_client_v1alpha2.ListContainers(request, timeout=5)

    def close(self) -> None:
        """Close connections"""
        if self.cri_channel:
            self.cri_channel.close()
        if self.oci_channel:
            self.oci_channel.close()

    def get_containers(self) -> List[CommonContainer]:
        """Get all running containers using CRI API"""
        containers = []

        try:
            if self.cri_client_v1alpha2:
                request = self._gen_container_request_v1alpha2()
                response = self.cri_client_v1alpha2.ListContainers(request, timeout=10)

                for container in response.containers:
                    containers.append(CommonContainer(
                        id=container.id,
                        labels=dict(container.labels)
                    ))
            elif self.cri_client_v1:
                request = self._gen_container_request_v1()
                response = self.cri_client_v1.ListContainers(request, timeout=10)

                for container in response.containers:
                    containers.append(CommonContainer(
                        id=container.id,
                        labels=dict(container.labels)
                    ))
        except Exception as e:
            logger.error("Failed to list containers via CRI: %s", str(e))
            raise

        logger.debug("Found %d running containers", len(containers))
        return containers

    def get_container_info_by_id(self, container_id: str) -> Dict:
        """Get container spec using OCI API"""
        request = containerd_pb2.GetContainerRequest(id=container_id)
        metadata = [(GRPC_HEADER_NAMESPACE, self.namespace)]

        try:
            response = self.oci_client.Get(request, metadata=metadata, timeout=10)
            # The spec is stored as protobuf Any type with type_url and value bytes
            spec_any = response.container.spec

            # The spec value is stored as bytes - containerd typically stores OCI spec as JSON bytes
            if hasattr(spec_any, 'value') and spec_any.value:
                try:
                    spec_json_str = spec_any.value.decode('utf-8')
                    spec_dict = json.loads(spec_json_str)
                    return spec_dict
                except (UnicodeDecodeError, json.JSONDecodeError) as decode_err:
                    logger.debug("Failed to parse spec value as JSON for %s: %s", container_id, str(decode_err))
                    raise ValueError(f"Spec value is not JSON: {decode_err}")

            # If no value field, try MessageToJson (will likely fail without descriptor)
            try:
                spec_json = json_format.MessageToJson(spec_any)
                spec_dict = json.loads(spec_json)
                return spec_dict
            except Exception as json_err:
                logger.debug("MessageToJson failed for %s (expected if descriptor missing): %s", container_id,
                             str(json_err))
                raise ValueError(f"Cannot parse spec: spec has no value field and MessageToJson failed: {json_err}")

        except Exception as e:
            logger.error("Failed to get container info for %s: %s", container_id, str(e))
            raise

    def _gen_container_request_v1alpha2(self):
        """Generate CRI ListContainers request for v1alpha2"""
        filter_obj = cri_v1alpha2_pb2.ContainerFilter()
        state_value = cri_v1alpha2_pb2.ContainerStateValue()
        state_value.state = cri_v1alpha2_pb2.CONTAINER_RUNNING
        filter_obj.state.CopyFrom(state_value)

        request = cri_v1alpha2_pb2.ListContainersRequest()
        request.filter.CopyFrom(filter_obj)
        return request

    def _gen_container_request_v1(self):
        """Generate CRI ListContainers request for v1"""
        filter_obj = cri_v1_pb2.ContainerFilter()
        state_value = cri_v1_pb2.ContainerStateValue()
        state_value.state = cri_v1_pb2.CONTAINER_RUNNING
        filter_obj.state.CopyFrom(state_value)

        request = cri_v1_pb2.ListContainersRequest()
        request.filter.CopyFrom(filter_obj)
        return request

    def get_npu_major_ids(self) -> List[str]:
        """Get NPU major device IDs from /proc/devices"""
        with self._npu_major_lock:
            if self._npu_major_ids is not None:
                return self._npu_major_ids

            major_ids = []
            try:
                if not os.path.exists(PROC_DEVICES_PATH):
                    logger.warning("%s not found", PROC_DEVICES_PATH)
                    return []

                with open(PROC_DEVICES_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if NPU_MAJOR_PATTERN.match(line):
                            fields = line.split()
                            major_ids.append(fields[0])

                self._npu_major_ids = major_ids
                logger.debug("Found NPU major IDs: %s", major_ids)
            except Exception as e:
                logger.error("Failed to read NPU major IDs: %s", str(e))

            return major_ids


class DevicesParser:
    """Parse NPU device information from container specs"""

    def __init__(self, runtime_operator: ContainerdRuntimeOperator):
        self.runtime_operator = runtime_operator

    def parse_devices(self, container: CommonContainer) -> Optional[DevicesInfo]:
        """
        Parse devices from container spec.
        1) Try env vars (ASCEND_DEVICE_ID / ASCEND_VISIBLE_DEVICES) first
        2) Fallback to spec.linux.resources.devices (major/minor) when present.
        """
        try:
            try:
                spec = self.runtime_operator.get_container_info_by_id(container.id)
            except (ValueError, Exception) as spec_err:
                logger.debug("Cannot parse spec for container %s (will skip): %s", container.id, str(spec_err))
                return None

            # Labels (required for DevicesInfo)
            namespace = container.labels.get(LABEL_K8S_POD_NAMESPACE, '')
            pod_name = container.labels.get(LABEL_K8S_POD_NAME, '')
            container_name = container.labels.get(LABEL_CONTAINER_NAME, '')
            if not namespace or not pod_name or not container_name:
                logger.debug("Container %s missing K8s labels", container.id)
                return None

            env_vars = []
            if spec.get('process') and spec['process'].get('env'):
                env_vars = spec['process']['env']
            if len(env_vars) <= MAX_ENV_NUM:
                for env in reversed(env_vars):
                    for env_name in ASCEND_DEVICE_ENV_NAMES:
                        if env_name in env:
                            device_ids = self._get_devices_with_ascend_runtime(env, container.id)
                            if device_ids:
                                return DevicesInfo(
                                    container_id=container.id,
                                    namespace=namespace,
                                    pod_name=pod_name,
                                    container_name=container_name,
                                    device_ids=device_ids
                                )
                            break

            # 2) Fallback: linux.resources.devices (major/minor) - only when spec has them
            if spec.get('linux') and spec['linux'].get('resources'):
                devices = spec['linux']['resources'].get('devices', [])
                if len(devices) <= MAX_DEVICES_NUM:
                    device_ids = self._get_devices_without_ascend_runtime(spec)
                    if device_ids:
                        return DevicesInfo(
                            container_id=container.id,
                            namespace=namespace,
                            pod_name=pod_name,
                            container_name=container_name,
                            device_ids=device_ids
                        )

            logger.debug(
                "Container %s (ns=%s pod=%s): no NPU devices from env or linux.resources",
                container.id, namespace, pod_name
            )
            return None

        except Exception as e:
            logger.debug("Failed to parse devices for container %s: %s", container.id, str(e))
            return None

    def _get_devices_with_ascend_runtime(self, ascend_dev_env: str, container_id: str) -> List[int]:
        """Extract device IDs from ASCEND_DEVICE_ID env var"""
        logger.debug("Getting device info from env var: %s", ascend_dev_env)

        parts = ascend_dev_env.split('=', 1)
        if len(parts) != 2:
            logger.error("Invalid ASCEND_DEVICE_ID format: %s", ascend_dev_env)
            return []

        device_str = parts[1]
        if len(device_str) > MAX_ENV_LENGTH:
            logger.warning("Device string too long: %d", len(device_str))
            return []

        return self._parse_diff_env_fmt(device_str, container_id)

    def _get_devices_without_ascend_runtime(self, spec: Dict) -> List[int]:
        """Extract device IDs from device files in spec"""
        if not spec.get('linux') or not spec['linux'].get('resources'):
            return []

        devices = spec['linux']['resources'].get('devices', [])
        major_ids = self.runtime_operator.get_npu_major_ids()

        device_ids = []
        for dev in devices:
            if dev.get('major') is None or dev.get('minor') is None:
                continue

            major = str(dev['major'])
            try:
                minor = int(dev['minor'])  # JSON may give int or string
            except (TypeError, ValueError):
                continue
            dev_type = dev.get('type', '')

            if dev_type == CHAR_DEVICE and major in major_ids:
                if minor > MAX_INT32:
                    logger.warning("Device minor ID too large: %d", minor)
                    continue
                device_ids.append(minor)

        return device_ids

    def _parse_diff_env_fmt(self, devices: str, container_id: str) -> List[int]:
        """Parse different env var formats for device IDs"""
        if 'Ascend' in devices:
            return self._get_device_ids_by_ascend_style(devices, container_id)
        elif '-' in devices and ',' in devices:
            return self._get_device_ids_by_comma_minus_style(devices, container_id)
        elif '-' in devices:
            return self._get_device_ids_by_minus_style(devices, container_id)
        else:
            return self._get_device_ids_by_comma_style(devices, container_id)

    def _get_device_ids_by_comma_style(self, devices: str, container_id: str) -> List[int]:
        """Parse format: "0,1,2" """
        device_ids = []
        for dev_id in devices.split(','):
            try:
                device_ids.append(int(dev_id.strip()))
            except ValueError:
                logger.warning("Container %s has invalid device ID: %s", container_id, dev_id)
        return device_ids

    def _get_device_ids_by_ascend_style(self, devices: str, container_id: str) -> List[int]:
        """Parse format: "Ascend-0,Ascend-1" """
        device_ids = []
        for sub_device in devices.split(','):
            parts = sub_device.split('-')
            if len(parts) != 2:
                logger.warning("Container %s has invalid Ascend format: %s", container_id, sub_device)
                continue
            try:
                device_ids.append(int(parts[1]))
            except ValueError:
                logger.warning("Container %s has invalid device ID: %s", container_id, parts[1])
        return device_ids

    def _get_device_ids_by_comma_minus_style(self, devices: str, container_id: str) -> List[int]:
        """Parse format: "0-1,2-3" """
        device_ids = []
        for range_str in devices.split(','):
            if '-' in range_str:
                parts = range_str.split('-')
                if len(parts) == 2:
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                        device_ids.extend(range(start, end + 1))
                    except ValueError:
                        logger.warning("Container %s has invalid range: %s", container_id, range_str)
        return device_ids

    def _get_device_ids_by_minus_style(self, devices: str, container_id: str) -> List[int]:
        """Parse format: "0-2" """
        parts = devices.split('-')
        if len(parts) != 2:
            return []
        try:
            start = int(parts[0])
            end = int(parts[1])
            return list(range(start, end + 1))
        except ValueError:
            logger.warning("Container %s has invalid range: %s", container_id, devices)
            return []


class ContainerInfoCollector(BaseInfoCollector):
    """
    Collects container information (container name, pod name, namespace)
    mapped to NPU devices (card_id, device_id) using containerd gRPC APIs.
    """

    def __init__(self, update_interval: int = 5):
        super().__init__(update_interval)
        self.runtime_operator = None
        self.parser = None
        self._initialized = False

    def _init_containerd(self):
        """Initialize containerd clients"""
        if self._initialized:
            return True

        try:
            self.runtime_operator = ContainerdRuntimeOperator()
            self.runtime_operator.init()
            self.parser = DevicesParser(self.runtime_operator)
            self._initialized = True
            logger.info("Container info collector initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize containerd clients: %s", str(e))
            return False

    def collect_info(self):
        """
        Collects container info and updates cache
        """
        # Initialize containerd if not already done
        if not self._init_containerd():
            # Return early if initialization failed
            logger.warning("Containerd initialization failed, skipping container info collection")
            return

        try:
            # Step 1: Get all containers via CRI
            containers = self.runtime_operator.get_containers()

            # Step 2: Parse devices for each container (OCI Get + extract NPU IDs)
            # Use parallel parsing when there are many containers to reduce latency
            # Build container_id -> DevicesInfo map
            containers_devices_infos = {}  # container_id -> DevicesInfo
            workers = min(MAX_PARSE_WORKERS, max(1, len(containers)))

            if len(containers) <= 4:
                for container in containers:
                    devices_info = self.parser.parse_devices(container)
                    if devices_info and devices_info.device_ids:
                        containers_devices_infos[devices_info.container_id] = devices_info
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_container = {
                        executor.submit(self.parser.parse_devices, c): c for c in containers
                    }
                    for future in as_completed(future_to_container, timeout=30):
                        try:
                            devices_info = future.result()
                            if devices_info and devices_info.device_ids:
                                containers_devices_infos[devices_info.container_id] = devices_info
                        except Exception as e:
                            logger.debug("Parse devices failed for one container: %s", str(e))

            # This is container_id -> DevicesInfo mapping; we store as list of one element
            self.cache.clear(CONTAINERS_DEVICES_CACHE_KEY)
            self.cache.set(CONTAINERS_DEVICES_CACHE_KEY, containers_devices_infos)
            logger.debug("Updated container devices cache with %d containers", len(containers_devices_infos))

        except Exception as e:
            logger.error("Error collecting container info: %s", str(e), exc_info=True)

    def stop(self):
        """Override stop to close containerd connections"""
        if self.runtime_operator:
            self.runtime_operator.close()
        super().stop()

