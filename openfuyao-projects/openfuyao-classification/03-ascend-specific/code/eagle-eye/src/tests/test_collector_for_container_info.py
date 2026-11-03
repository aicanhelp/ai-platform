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
Unit tests for ContainerInfoCollector and DevicesParser: collect_info with mocked
containerd runtime, parser device ID formats, and get_npu_major_ids.
"""

import unittest

from unittest.mock import patch, Mock, mock_open

from common.base.cache_manager import CacheManager
from hardware_monitor.collector.container_info.containerd.collector_for_container_info import (
    ContainerInfoCollector,
    ContainerdRuntimeOperator,
    DevicesParser,
    DevicesInfo,
    CommonContainer,
    CONTAINERS_DEVICES_CACHE_KEY,
)


def _clear_container_devices_cache():
    """Clear container devices cache key for test isolation."""
    CacheManager().clear(CONTAINERS_DEVICES_CACHE_KEY)


class TestContainerInfoCollector(unittest.TestCase):
    """
    Test ContainerInfoCollector.collect_info with mocked init, get_containers, and parser.
    """

    def setUp(self):
        _clear_container_devices_cache()

    def tearDown(self):
        _clear_container_devices_cache()

    @patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.ContainerdRuntimeOperator")
    @patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.DevicesParser")
    def test_collect_info_init_fails(self, mock_parser_cls, mock_runtime_cls):
        """When _init_containerd fails, collect_info returns without updating cache."""
        mock_runtime_cls.side_effect = RuntimeError("containerd not available")
        collector = ContainerInfoCollector()
        collector._initialized = False
        collector.collect_info()
        cached = collector.cache.get(CONTAINERS_DEVICES_CACHE_KEY)
        self.assertIsNone(cached)

    @patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.ContainerdRuntimeOperator")
    @patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.DevicesParser")
    def test_collect_info_empty_containers(self, mock_parser_cls, mock_runtime_cls):
        """When get_containers returns empty list, cache is updated with empty dict."""
        mock_runtime = Mock()
        mock_runtime.get_containers.return_value = []
        mock_runtime_cls.return_value = mock_runtime
        mock_parser_cls.return_value = Mock()

        collector = ContainerInfoCollector()
        collector._initialized = True
        collector.runtime_operator = mock_runtime
        collector.parser = mock_parser_cls.return_value
        collector.collect_info()

        cached = collector.cache.get(CONTAINERS_DEVICES_CACHE_KEY)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0], {})
        mock_runtime.get_containers.assert_called_once()

    @patch(
        "hardware_monitor.collector.container_info.containerd.collector_for_container_info.ContainerdRuntimeOperator"
    )
    @patch(
        "hardware_monitor.collector.container_info.containerd.collector_for_container_info.DevicesParser"
    )
    def test_collect_info_with_devices(self, mock_parser_cls, mock_runtime_cls):
        """When containers have device info, cache is updated with container_id -> DevicesInfo."""
        c1 = CommonContainer(id="cid1", labels={
            "io.kubernetes.pod.namespace": "ns1",
            "io.kubernetes.pod.name": "pod1",
            "io.kubernetes.container.name": "c1",
        })
        info1 = DevicesInfo(
            container_id="cid1",
            namespace="ns1",
            pod_name="pod1",
            container_name="c1",
            device_ids=[0, 1],
        )

        mock_runtime = Mock()
        mock_runtime.get_containers.return_value = [c1]
        mock_runtime_cls.return_value = mock_runtime
        mock_parser = Mock()
        mock_parser.parse_devices.side_effect = [info1]
        mock_parser_cls.return_value = mock_parser

        collector = ContainerInfoCollector()
        collector._initialized = True
        collector.runtime_operator = mock_runtime
        collector.parser = mock_parser
        collector.collect_info()

        cached = collector.cache.get(CONTAINERS_DEVICES_CACHE_KEY)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 1)
        data = cached[0]
        self.assertIn("cid1", data)
        self.assertEqual(data["cid1"].namespace, "ns1")
        self.assertEqual(data["cid1"].pod_name, "pod1")
        self.assertEqual(data["cid1"].container_name, "c1")
        self.assertEqual(data["cid1"].device_ids, [0, 1])


class TestDevicesParser(unittest.TestCase):
    """
    Test DevicesParser device ID parsing (ASCEND env and spec devices) with mocked runtime.
    """

    def test_parse_devices_no_linux_resources_returns_none(self):
        """Container spec without linux.resources returns None."""
        mock_runtime = Mock()
        mock_runtime.get_container_info_by_id.return_value = {}
        parser = DevicesParser(mock_runtime)
        container = CommonContainer(id="c1", labels={
            "io.kubernetes.pod.namespace": "ns",
            "io.kubernetes.pod.name": "pod",
            "io.kubernetes.container.name": "cnt",
        })
        result = parser.parse_devices(container)
        self.assertIsNone(result)

    def test_parse_devices_missing_labels_returns_none(self):
        """Container with missing K8s labels returns None."""
        mock_runtime = Mock()
        mock_runtime.get_container_info_by_id.return_value = {
            "linux": {"resources": {"devices": []}},
            "process": {"env": []},
        }
        parser = DevicesParser(mock_runtime)
        container = CommonContainer(id="c1", labels={})
        result = parser.parse_devices(container)
        self.assertIsNone(result)

    def test_parse_devices_ascend_env_comma_style(self):
        """ASCEND_DEVICE_ID with comma format (0,1,2) is parsed."""
        mock_runtime = Mock()
        mock_runtime.get_container_info_by_id.return_value = {
            "linux": {"resources": {"devices": []}},
            "process": {"env": ["PATH=/usr", "ASCEND_DEVICE_ID=0,1,2"]},
        }
        parser = DevicesParser(mock_runtime)
        container = CommonContainer(id="c1", labels={
            "io.kubernetes.pod.namespace": "ns",
            "io.kubernetes.pod.name": "pod",
            "io.kubernetes.container.name": "cnt",
        })
        result = parser.parse_devices(container)
        self.assertIsNotNone(result)
        self.assertEqual(result.device_ids, [0, 1, 2])
        self.assertEqual(result.namespace, "ns")
        self.assertEqual(result.pod_name, "pod")
        self.assertEqual(result.container_name, "cnt")

    def test_parse_devices_ascend_env_ascend_style(self):
        """ASCEND_DEVICE_ID with Ascend-0,Ascend-1 format is parsed."""
        mock_runtime = Mock()
        mock_runtime.get_container_info_by_id.return_value = {
            "linux": {"resources": {"devices": []}},
            "process": {"env": ["ASCEND_DEVICE_ID=Ascend-0,Ascend-1"]},
        }
        parser = DevicesParser(mock_runtime)
        container = CommonContainer(id="c1", labels={
            "io.kubernetes.pod.namespace": "default",
            "io.kubernetes.pod.name": "mypod",
            "io.kubernetes.container.name": "main",
        })
        result = parser.parse_devices(container)
        self.assertIsNotNone(result)
        self.assertEqual(result.device_ids, [0, 1])

    def test_parse_devices_ascend_env_minus_style(self):
        """ASCEND_DEVICE_ID with range format 0-2 is parsed."""
        mock_runtime = Mock()
        mock_runtime.get_container_info_by_id.return_value = {
            "linux": {"resources": {"devices": []}},
            "process": {"env": ["ASCEND_DEVICE_ID=0-2"]},
        }
        parser = DevicesParser(mock_runtime)
        container = CommonContainer(id="c1", labels={
            "io.kubernetes.pod.namespace": "ns",
            "io.kubernetes.pod.name": "pod",
            "io.kubernetes.container.name": "c",
        })
        result = parser.parse_devices(container)
        self.assertIsNotNone(result)
        self.assertEqual(result.device_ids, [0, 1, 2])

    def test_get_device_ids_by_comma_style(self):
        """_get_device_ids_by_comma_style parses '0, 1, 2'."""
        mock_runtime = Mock()
        parser = DevicesParser(mock_runtime)
        ids = parser._get_device_ids_by_comma_style("0, 1, 2", "c1")
        self.assertEqual(ids, [0, 1, 2])

    def test_get_device_ids_by_ascend_style(self):
        """_get_device_ids_by_ascend_style parses 'Ascend-0,Ascend-1'."""
        mock_runtime = Mock()
        parser = DevicesParser(mock_runtime)
        ids = parser._get_device_ids_by_ascend_style("Ascend-0,Ascend-1", "c1")
        self.assertEqual(ids, [0, 1])

    def test_get_device_ids_by_minus_style(self):
        """_get_device_ids_by_minus_style parses '0-2'."""
        mock_runtime = Mock()
        parser = DevicesParser(mock_runtime)
        ids = parser._get_device_ids_by_minus_style("0-2", "c1")
        self.assertEqual(ids, [0, 1, 2])

    def test_get_device_ids_by_comma_minus_style(self):
        """_get_device_ids_by_comma_minus_style parses '0-1,2-3'."""
        mock_runtime = Mock()
        parser = DevicesParser(mock_runtime)
        ids = parser._get_device_ids_by_comma_minus_style("0-1,2-3", "c1")
        self.assertEqual(ids, [0, 1, 2, 3])


class TestContainerdRuntimeOperatorNpuMajor(unittest.TestCase):
    """
    Test ContainerdRuntimeOperator.get_npu_major_ids (read /proc/devices).
    """

    @patch("builtins.open", new_callable=mock_open, read_data=" 100 devdrv-cdev\n200 vdevdrv-cdev\n")
    def test_get_npu_major_ids_parses_proc_devices(self, mock_file):
        """get_npu_major_ids reads /proc/devices and returns NPU major IDs."""
        operator = ContainerdRuntimeOperator()
        operator._npu_major_ids = None
        with patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.os.path.exists", return_value=True):
            ids = operator.get_npu_major_ids()
        self.assertEqual(ids, ["100", "200"])
        mock_file.assert_called_once()

    @patch("hardware_monitor.collector.container_info.containerd.collector_for_container_info.os.path.exists")
    def test_get_npu_major_ids_no_proc_devices(self, mock_exists):
        """When /proc/devices does not exist, get_npu_major_ids returns []."""
        mock_exists.return_value = False
        operator = ContainerdRuntimeOperator()
        operator._npu_major_ids = None
        ids = operator.get_npu_major_ids()
        self.assertEqual(ids, [])

    def test_get_npu_major_ids_cached(self):
        """get_npu_major_ids returns cached value on second call."""
        operator = ContainerdRuntimeOperator()
        operator._npu_major_ids = ["42"]
        ids = operator.get_npu_major_ids()
        self.assertEqual(ids, ["42"])


if __name__ == "__main__":
    unittest.main()
