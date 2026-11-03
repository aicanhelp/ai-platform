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
Unit tests for NPUDeviceInfoCollector: collect_info with get_card_list,
get_device_num_in_card, get_device_logic_id, get_phy_id success and error paths.
"""

import unittest
from unittest.mock import patch

from common.base.cache_manager import CacheManager
from common.npu_device_info.collector_for_npu_device_info import NPUDeviceInfoCollector

_DEVICE_INFO_KEYS = ("logic_id", "card_id", "device_id", "phy_id")


def _clear_device_info_cache():
    """Clear cache keys used by NPUDeviceInfoCollector for test isolation."""
    cache = CacheManager()
    for key in _DEVICE_INFO_KEYS:
        cache.clear(key)


class TestNPUDeviceInfoCollector(unittest.TestCase):
    """
    TestNPUDeviceInfoCollector tests collect_info: card list error/empty,
    device num error, logic_id/phy_id errors, and full success path.
    """

    def setUp(self):
        _clear_device_info_cache()

    def tearDown(self):
        _clear_device_info_cache()

    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_get_card_list_error(self, mock_get_card_list):
        """
        When get_card_list returns an error, collect_info returns without populating cache.
        """
        mock_get_card_list.return_value = (-2, [], [], RuntimeError("function not found"))
        collector = NPUDeviceInfoCollector()
        collector.collect_info()
        self.assertIsNone(CacheManager().get("logic_id"))
        self.assertIsNone(CacheManager().get("card_id"))

    @patch("common.npu_device_info.collector_for_npu_device_info.get_phy_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_logic_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_num_in_card")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_success(
        self, mock_get_card_list, mock_get_device_num, mock_get_logic_id, mock_get_phy_id
    ):
        """
        When all calls succeed, collect_info fills cache with logic_id, card_id, device_id, phy_id lists.
        """
        mock_get_card_list.return_value = (1, [0], [], None)
        mock_get_device_num.return_value = (2, None)
        mock_get_logic_id.side_effect = [(0, None), (1, None)]
        mock_get_phy_id.side_effect = [(10, None), (11, None)]

        collector = NPUDeviceInfoCollector()
        collector.collect_info()

        self.assertEqual(CacheManager().get("logic_id"), [0, 1])
        self.assertEqual(CacheManager().get("card_id"), [0, 0])
        self.assertEqual(CacheManager().get("device_id"), [0, 1])
        self.assertEqual(CacheManager().get("phy_id"), [10, 11])
        self.assertEqual(mock_get_device_num.call_count, 1)
        self.assertEqual(mock_get_logic_id.call_count, 2)
        self.assertEqual(mock_get_phy_id.call_count, 2)

    @patch("common.npu_device_info.collector_for_npu_device_info.get_phy_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_logic_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_num_in_card")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_device_num_error_skips_card(
        self, mock_get_card_list, mock_get_device_num, mock_get_logic_id, mock_get_phy_id
    ):
        """
        When get_device_num_in_card returns an error for a card, that card is skipped.
        """
        mock_get_card_list.return_value = (2, [0, 1], [], None)
        mock_get_device_num.side_effect = [(2, None), (-1, ValueError("invalid card"))]
        mock_get_logic_id.side_effect = [(0, None), (1, None)]
        mock_get_phy_id.side_effect = [(10, None), (11, None)]

        collector = NPUDeviceInfoCollector()
        collector.collect_info()

        self.assertEqual(CacheManager().get("logic_id"), [0, 1])
        self.assertEqual(CacheManager().get("card_id"), [0, 0])
        self.assertEqual(CacheManager().get("device_id"), [0, 1])
        self.assertEqual(CacheManager().get("phy_id"), [10, 11])
        mock_get_device_num.assert_any_call(0)
        mock_get_device_num.assert_any_call(1)
        self.assertEqual(mock_get_phy_id.call_count, 2)

    @patch("common.npu_device_info.collector_for_npu_device_info.get_phy_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_logic_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_num_in_card")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_logic_id_error_skips_device(
        self, mock_get_card_list, mock_get_device_num, mock_get_logic_id, mock_get_phy_id
    ):
        """
        When get_device_logic_id returns an error for a device, that device is skipped.
        """
        mock_get_card_list.return_value = (1, [0], [], None)
        mock_get_device_num.return_value = (2, None)
        mock_get_logic_id.side_effect = [(0, None), (-1, RuntimeError("ret: -8001"))]
        mock_get_phy_id.return_value = (10, None)

        collector = NPUDeviceInfoCollector()
        collector.collect_info()

        self.assertEqual(CacheManager().get("logic_id"), [0])
        self.assertEqual(CacheManager().get("card_id"), [0])
        self.assertEqual(CacheManager().get("device_id"), [0])
        self.assertEqual(CacheManager().get("phy_id"), [10])
        mock_get_phy_id.assert_called_once_with(0)

    @patch("common.npu_device_info.collector_for_npu_device_info.get_phy_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_logic_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_num_in_card")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_phy_id_error_skips_device(
        self, mock_get_card_list, mock_get_device_num, mock_get_logic_id, mock_get_phy_id
    ):
        """
        When get_phy_id returns an error for a device, that device is skipped.
        """
        mock_get_card_list.return_value = (1, [0], [], None)
        mock_get_device_num.return_value = (2, None)
        mock_get_logic_id.side_effect = [(0, None), (1, None)]
        mock_get_phy_id.side_effect = [(10, None), (-1, RuntimeError("invalid phy"))]

        collector = NPUDeviceInfoCollector()
        collector.collect_info()

        self.assertEqual(CacheManager().get("logic_id"), [0])
        self.assertEqual(CacheManager().get("card_id"), [0])
        self.assertEqual(CacheManager().get("device_id"), [0])
        self.assertEqual(CacheManager().get("phy_id"), [10])

    @patch("common.npu_device_info.collector_for_npu_device_info.get_phy_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_logic_id")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_device_num_in_card")
    @patch("common.npu_device_info.collector_for_npu_device_info.get_card_list")
    def test_collect_info_clears_cache_before_fill(
        self, mock_get_card_list, mock_get_device_num, mock_get_logic_id, mock_get_phy_id
    ):
        """
        collect_info clears device info keys before populating (no accumulation from previous run).
        """
        mock_get_card_list.return_value = (1, [0], [], None)
        mock_get_device_num.return_value = (1, None)
        mock_get_logic_id.return_value = (0, None)
        mock_get_phy_id.return_value = (0, None)

        collector = NPUDeviceInfoCollector()
        collector.collect_info()
        self.assertEqual(CacheManager().get("logic_id"), [0])
        collector.collect_info()
        self.assertEqual(CacheManager().get("logic_id"), [0])


if __name__ == "__main__":
    unittest.main()
