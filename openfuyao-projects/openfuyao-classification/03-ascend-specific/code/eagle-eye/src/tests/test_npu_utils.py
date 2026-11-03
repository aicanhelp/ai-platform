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
This module defines unit tests for NPU utility helpers that wrap low-level DCMI
bindings.
"""

import unittest
import ctypes

from unittest.mock import patch, Mock
from ctypes import c_int, c_ubyte, c_uint

from common.npu_metrics.constants import (HIAI_MAX_CARD_NUM, HIAI_MAX_DEVICE_NUM,
                                          RET_INVALID_VALUE, RET_NO_FUNCTION, PERCENT, INVALID_ID,
                                          MAX_INT32,
                                          A300I_A2_Board_Id, ASCEND_910A3, ASCEND_310, ASCEND_310B,
                                          ASCEND_310P, ASCEND_910, ASCEND_910B)
from common.npu_metrics.dcmi.dcmi_wrapper import (DcmiChipInfoV2Struct, DcmiChipInfoStruct,
                                                  DcmiBoardInfoStruct)
from common.npu_metrics.utils import (is_valid_logic_id_or_phy_id, is_valid_card_id, is_valid_device_id, get_phy_id,
                                      get_device_logic_id, get_device_num_in_card, get_card_list,
                                      is_valid_card_and_device_id, is_valid_chip_info, is_valid_board_info,
                                      uchar_array_to_str, is_910a3_chip, get_device_type_by_chip_name,
                                      get_device_type, is_training_card, get_chip_info,
                                      get_board_info, fetch_device_info_and_type)
from hardware_monitor.collector.metrics.npu.npu_utils import (is_valid_utilization_rate,
                                                              is_greater_than_or_equal_int32,
                                                              init_device_info, get_utilization_rate)


class TestNPUUtils(unittest.TestCase):
    """
    TestNPUUtils verifies NPU utility validators and mappings (IDs, utilization, chip/board types).
    """

    def test_is_valid_logic_id_or_phy_id(self):
        """
        Tests the is_valid_logic_id_or_phy_id function for valid and invalid logic/physical IDs.
        """
        self.assertTrue(is_valid_logic_id_or_phy_id(0))
        self.assertFalse(is_valid_logic_id_or_phy_id(HIAI_MAX_CARD_NUM * HIAI_MAX_DEVICE_NUM))

    def test_is_valid_card_id(self):
        """
        Tests the is_valid_card_id function for valid and invalid card IDs.
        """
        self.assertTrue(is_valid_card_id(0))
        self.assertTrue(is_valid_card_id(HIAI_MAX_CARD_NUM - 1))
        self.assertFalse(is_valid_card_id(-1))
        self.assertFalse(is_valid_card_id(HIAI_MAX_CARD_NUM))

    def test_is_valid_device_id(self):
        """
        Tests the is_valid_device_id function for valid and invalid device IDs.
        """
        self.assertTrue(is_valid_device_id(0))
        self.assertTrue(is_valid_device_id(HIAI_MAX_DEVICE_NUM - 1))
        self.assertFalse(is_valid_device_id(-1))
        self.assertFalse(is_valid_device_id(HIAI_MAX_DEVICE_NUM))

    def test_is_valid_utilization_rate(self):
        """
        Tests the is_valid_utilization_rate function for valid and invalid utilization rates.
        """
        self.assertTrue(is_valid_utilization_rate(0))
        self.assertTrue(is_valid_utilization_rate(PERCENT))
        self.assertFalse(is_valid_utilization_rate(PERCENT + 1))

    def test_is_valid_card_and_device_id(self):
        """
        Tests the is_valid_card_and_device_id function for
        valid and invalid combinations of card and device IDs.
        """
        self.assertTrue(is_valid_card_and_device_id(0, 0))
        self.assertTrue(is_valid_card_and_device_id(HIAI_MAX_CARD_NUM - 1, HIAI_MAX_DEVICE_NUM - 1))
        self.assertFalse(is_valid_card_and_device_id(-1, 0))
        self.assertFalse(is_valid_card_and_device_id(0, -1))
        self.assertFalse(is_valid_card_and_device_id(HIAI_MAX_CARD_NUM, HIAI_MAX_DEVICE_NUM))

    def test_is_greater_than_or_equal_int32(self):
        """
        Tests the is_greater_than_or_equal_int32 function
        for comparison with the maximum signed 32-bit integer value.
        """
        self.assertTrue(is_greater_than_or_equal_int32(MAX_INT32))
        self.assertTrue(is_greater_than_or_equal_int32(MAX_INT32 + 1))
        self.assertFalse(is_greater_than_or_equal_int32(MAX_INT32 - 1))

    def test_uchar_array_to_str(self):
        """
        Tests the uchar_array_to_str function for converting byte arrays to strings.
        """
        arr = (c_ubyte * 4)(65, 66, 67, 0)
        self.assertEqual(uchar_array_to_str(arr), "ABC")

        arr_str = (c_ubyte * 5)(72, 101, 108, 108, 111)
        self.assertEqual(uchar_array_to_str(arr_str), "Hello")

        arr_empty = (c_ubyte * 1)(0)
        self.assertEqual(uchar_array_to_str(arr_empty), "")

    def test_is_valid_chip_info(self):
        """
        Tests the is_valid_chip_info function for validating chip information.
        """
        valid_chip = {"chip_name": "910B4", "chip_type": "Ascend", "chip_version": "v1"}
        valid_chip_2 = {"chip_name": "", "chip_type": "Ascend", "chip_version": "v1"}
        invalid_chip = {"chip_name": "", "chip_type": "", "chip_version": ""}
        self.assertTrue(is_valid_chip_info(valid_chip))
        self.assertTrue(is_valid_chip_info(valid_chip_2))
        self.assertFalse(is_valid_chip_info(invalid_chip))

    def test_is_valid_board_info(self):
        """
        Tests the is_valid_board_info function for validating board information.
        """
        valid_board = {"board_id": 0xAB, "pcb_id": 5678, "bom_id": 1, "slot_id": 1}
        valid_board_2 = {"board_id": INVALID_ID, "pcb_id": 5678, "bom_id": 1, "slot_id": INVALID_ID}
        invalid_board = {"board_id": INVALID_ID, "pcb_id": INVALID_ID, "bom_id": INVALID_ID, "slot_id": INVALID_ID}
        self.assertTrue(is_valid_board_info(valid_board))
        self.assertTrue(is_valid_board_info(valid_board_2))
        self.assertFalse(is_valid_chip_info(invalid_board))

    def test_is_910a3_chip(self):
        """
        Tests the is_910a3_chip function for identifying the ASCEND_910A3 chip.
        """
        valid_board_id = 0xb0
        invalid_board_id = 0xb5
        self.assertTrue(is_910a3_chip(valid_board_id))
        self.assertFalse(is_910a3_chip(invalid_board_id))

    def test_get_device_type_by_chip_name(self):
        """
        Tests the get_device_type_by_chip_name function for retrieving device type by chip name.
        """
        self.assertEqual(get_device_type_by_chip_name("310P"), ASCEND_310P)
        self.assertEqual(get_device_type_by_chip_name("310B"), ASCEND_310B)
        self.assertEqual(get_device_type_by_chip_name("310"), ASCEND_310)
        self.assertEqual(get_device_type_by_chip_name("910B4"), ASCEND_910B)
        self.assertEqual(get_device_type_by_chip_name("910A"), ASCEND_910)
        self.assertEqual(get_device_type_by_chip_name("NonExistentChip"), "")

    def test_get_device_type(self):
        """
        Tests the get_device_type function for retrieving device type
        based on chip name and board ID.
        """
        self.assertEqual(get_device_type("Ascend910A3",
                                         0xb0), ASCEND_910A3)
        self.assertEqual(get_device_type("910B4",
                                         0x34), ASCEND_910B)

    def test_is_training_card(self):
        """
        Tests the is_training_card function for checking whether a device is a training card.
        """
        self.assertFalse(is_training_card(ASCEND_310, 0xAB))
        self.assertFalse(is_training_card(ASCEND_310B, 0xAB))
        self.assertFalse(is_training_card(ASCEND_310P, 0xAB))
        self.assertFalse(is_training_card(ASCEND_910B, A300I_A2_Board_Id))
        self.assertFalse(is_training_card(ASCEND_910B, A300I_A2_Board_Id))
        self.assertTrue(is_training_card(ASCEND_910B, 0x34))


class TestGetDeviceLogicId(unittest.TestCase):
    """
    TestGetDeviceLogicId exercises `get_device_logic_id` for invalid card/device IDs,
    missing symbol, error return, success, and invalid logic ID written by DCMI.
    """

    def test_invalid_card_or_device_id(self):
        """
        Tests get_device_logic_id when card_id or device_id is invalid.
        """
        logic_id, err = get_device_logic_id(-1, 0)
        self.assertEqual(logic_id, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid cardID", str(err))

        logic_id, err = get_device_logic_id(0, -1)
        self.assertEqual(logic_id, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_device_logic_id when DCMI instance or dcmi_get_device_logic_id is missing.
        """
        mock_get_dcmi_instance.return_value = None
        logic_id, err = get_device_logic_id(0, 0)
        self.assertEqual(logic_id, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("dcmi_get_device_logic_id", str(err))
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_device_logic_id when dcmi_get_device_logic_id returns a non-zero code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance
        mock_dcmi_instance.dcmi.dcmi_get_device_logic_id.return_value = -8001

        logic_id, err = get_device_logic_id(0, 0)
        self.assertEqual(logic_id, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests get_device_logic_id when the logic ID is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(logic_id_ref, card_id, device_id):  # pylint: disable=unused-argument
            ctypes.cast(logic_id_ref, ctypes.POINTER(c_int)).contents.value = 5
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_logic_id.side_effect = fake_func

        logic_id, err = get_device_logic_id(0, 1)
        self.assertEqual(logic_id, 5)
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_logic_id_written(self, mock_get_dcmi_instance):
        """
        Tests get_device_logic_id when DCMI writes an invalid logic ID (out of valid range).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        invalid_logic_id = HIAI_MAX_CARD_NUM * HIAI_MAX_DEVICE_NUM

        def fake_func(logic_id_ref, card_id, device_id):  # pylint: disable=unused-argument
            ctypes.cast(logic_id_ref, ctypes.POINTER(c_int)).contents.value = invalid_logic_id
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_logic_id.side_effect = fake_func

        logic_id, err = get_device_logic_id(0, 0)
        self.assertEqual(logic_id, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid logicID", str(err))


class TestGetDeviceNumInCard(unittest.TestCase):
    """
    TestGetDeviceNumInCard exercises `get_device_num_in_card` for invalid card ID,
    missing symbol, error return, success, and invalid device num written by DCMI.
    """

    def test_invalid_card_id(self):
        """
        Tests get_device_num_in_card when card_id is invalid.
        """
        num, err = get_device_num_in_card(-1)
        self.assertEqual(num, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid cardID", str(err))

        num, err = get_device_num_in_card(HIAI_MAX_CARD_NUM)
        self.assertEqual(num, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid cardID", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_device_num_in_card when DCMI instance or dcmi_get_device_num_in_card is missing.
        """
        mock_get_dcmi_instance.return_value = None
        num, err = get_device_num_in_card(0)
        self.assertEqual(num, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("dcmi_get_device_num_in_card", str(err))
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_device_num_in_card when dcmi_get_device_num_in_card returns a non-zero code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance
        mock_dcmi_instance.dcmi.dcmi_get_device_num_in_card.return_value = -8001

        num, err = get_device_num_in_card(0)
        self.assertEqual(num, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests get_device_num_in_card when the device count is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card_id, device_num_ref):  # pylint: disable=unused-argument
            ctypes.cast(device_num_ref, ctypes.POINTER(c_int)).contents.value = 4
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_num_in_card.side_effect = fake_func

        num, err = get_device_num_in_card(0)
        self.assertEqual(num, 4)
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_device_num_written(self, mock_get_dcmi_instance):
        """
        Tests get_device_num_in_card when DCMI writes an invalid device num (0 or > HIAI_MAX_DEVICE_NUM).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card_id, device_num_ref):  # pylint: disable=unused-argument
            ctypes.cast(device_num_ref, ctypes.POINTER(c_int)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_num_in_card.side_effect = fake_func

        num, err = get_device_num_in_card(0)
        self.assertEqual(num, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid device num", str(err))


class TestGetCardList(unittest.TestCase):
    """
    TestGetCardList exercises `get_card_list` for missing symbol, error return,
    invalid card quantity, all invalid IDs, and success.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when DCMI instance or dcmi_get_card_list is missing.
        """
        mock_get_dcmi_instance.return_value = None
        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, RET_NO_FUNCTION)
        self.assertEqual(ids, [])
        self.assertEqual(invalid_ids, [])
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("dcmi_get_card_list", str(err))
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when dcmi_get_card_list returns a non-zero code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance
        mock_dcmi_instance.dcmi.dcmi_get_card_list.return_value = -8001

        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, -8001)
        self.assertEqual(ids, [])
        self.assertEqual(invalid_ids, [])
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_invalid_card_quantity(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when DCMI writes an invalid card count (<=0 or > HIAI_MAX_CARD_NUM).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(num_ref, buf, max_size):  # pylint: disable=unused-argument
            ctypes.cast(num_ref, ctypes.POINTER(c_int)).contents.value = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_card_list.side_effect = fake_func

        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, RET_INVALID_VALUE)
        self.assertEqual(ids, [])
        self.assertEqual(invalid_ids, [])
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid card quantity", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_all_invalid_card_ids(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when all returned card IDs are negative (invalid).
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(num_ref, buf, max_size):  # pylint: disable=unused-argument
            ctypes.cast(num_ref, ctypes.POINTER(c_int)).contents.value = 2
            buf[0] = -1
            buf[1] = -2
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_card_list.side_effect = fake_func

        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, RET_INVALID_VALUE)
        self.assertEqual(ids, [])
        self.assertEqual(invalid_ids, [-1, -2])
        self.assertIsInstance(err, ValueError)
        self.assertIn("all returned card IDs are invalid", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when card list is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(num_ref, buf, max_size):  # pylint: disable=unused-argument
            ctypes.cast(num_ref, ctypes.POINTER(c_int)).contents.value = 3
            buf[0] = 0
            buf[1] = 1
            buf[2] = 2
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_card_list.side_effect = fake_func

        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, 3)
        self.assertEqual(ids, [0, 1, 2])
        self.assertEqual(invalid_ids, [])
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_with_some_invalid_ids(self, mock_get_dcmi_instance):
        """
        Tests get_card_list when some returned card IDs are negative; valid and invalid lists are split.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(num_ref, buf, max_size):  # pylint: disable=unused-argument
            ctypes.cast(num_ref, ctypes.POINTER(c_int)).contents.value = 4
            buf[0] = 0
            buf[1] = -1
            buf[2] = 1
            buf[3] = -2
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_card_list.side_effect = fake_func

        count, ids, invalid_ids, err = get_card_list()
        self.assertEqual(count, 4)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(invalid_ids, [-1, -2])
        self.assertIsNone(err)


class TestGetChipInfo(unittest.TestCase):
    """
    TestGetChipInfo tests `get_chip_info` via v2/legacy APIs, including empty-content validation.
    """

    def test_invalid_ids(self):
        """
        Tests get_chip_info behavior when invalid card and device IDs are provided.
        """
        info, code, err = get_chip_info(HIAI_MAX_CARD_NUM, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_functions_missing(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None
        info, code, err = get_chip_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_v2_success(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when using the v2 API and the function is successful.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def v2_ok(card, device, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiChipInfoV2Struct)).contents
            _write_c_ubyte_text(s.chip_type, "Ascend")
            _write_c_ubyte_text(s.chip_name, "910B4")
            _write_c_ubyte_text(s.chip_ver, "v1")
            _write_c_ubyte_text(s.npu_name, "NPU910")
            s.aicore_cnt = 32
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info_v2.side_effect = v2_ok

        _ = mock_dcmi_instance.dcmi.dcmi_get_device_chip_info

        info, code, err = get_chip_info(0, 0)
        self.assertIsNone(code)
        self.assertIsNone(err)
        self.assertEqual(info["chip_type"], "Ascend")
        self.assertEqual(info["chip_name"], "910B4")
        self.assertEqual(info["chip_version"], "v1")
        self.assertEqual(info["aicore_cnt"], 32)
        self.assertEqual(info["npu_name"], "NPU910")

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_v2_fail_old_success(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when the v2 API fails but the legacy API succeeds.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info_v2.return_value = -8001

        def old_ok(card, device, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiChipInfoStruct)).contents
            _write_c_ubyte_text(s.chip_type, "Ascend")
            _write_c_ubyte_text(s.chip_name, "910B4")
            _write_c_ubyte_text(s.chip_ver, "v1")
            s.aicore_cnt = 32
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info.side_effect = old_ok

        info, code, err = get_chip_info(0, 0)
        self.assertIsNone(code)
        self.assertIsNone(err)
        self.assertEqual(info["chip_type"], "Ascend")
        self.assertEqual(info["chip_name"], "910B4")
        self.assertEqual(info["chip_version"], "v1")
        self.assertEqual(info["aicore_cnt"], 32)
        self.assertNotIn("npu_name", info)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_v2_fail_old_fail(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when both the v2 and legacy APIs fail.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info_v2.return_value = -8001
        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info.return_value = -8001

        info, code, err = get_chip_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_v2_success_but_invalid_content(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when the v2 API succeeds but returns invalid content.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def v2_ok_but_empty(card, device, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiChipInfoV2Struct)).contents
            _write_c_ubyte_text(s.chip_type, "")
            _write_c_ubyte_text(s.chip_name, "")
            _write_c_ubyte_text(s.chip_ver, "")
            _write_c_ubyte_text(s.npu_name, "")
            s.aicore_cnt = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info_v2.side_effect = v2_ok_but_empty
        _ = mock_dcmi_instance.dcmi.dcmi_get_device_chip_info

        info, code, err = get_chip_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("chip info is empty", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_old_success_but_invalid_content(self, mock_get_dcmi_instance):
        """
        Tests get_chip_info behavior when the legacy API succeeds but returns invalid content.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info_v2.return_value = -1

        def old_ok_but_empty(card, device, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiChipInfoStruct)).contents
            _write_c_ubyte_text(s.chip_type, "")
            _write_c_ubyte_text(s.chip_name, "")
            _write_c_ubyte_text(s.chip_ver, "")
            s.aicore_cnt = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_chip_info.side_effect = old_ok_but_empty

        info, code, err = get_chip_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("chip info is empty", str(err))


class TestInitDeviceInfo(unittest.TestCase):
    """
    TestInitDeviceInfo validates initialization of device info from a logic ID and requested fields.
    """

    def test_init_device_info_success(self):
        """
        Tests the init_device_info function for successful initialization of device info.
        """
        fields = [
            "node_name", "card_id", "device_id",
            "npu_chip_info_hbm_total_memory", "npu_chip_info_hbm_frequency"
        ]
        result = init_device_info(node_name="master", phy_id=1, card_id=1, device_id=2, fields=fields)

        self.assertEqual(result["node_name"], "master")
        self.assertEqual(result["phy_id"], 1)
        self.assertEqual(result["card_id"], 1)
        self.assertEqual(result["device_id"], 2)
        self.assertEqual(result["npu_chip_info_hbm_total_memory"], "NA")
        self.assertEqual(result["npu_chip_info_hbm_frequency"], "NA")
        self.assertIn("errors", result)
        self.assertEqual(result["errors"], [])


class TestGetPhyId(unittest.TestCase):
    """
    TestGetPhyId exercises `get_phy_id` for invalid input,
    missing symbol, error code, and success cases.
    """

    def test_invalid_logic_id(self):
        """
        Tests get_phy_id behavior when the logic ID is invalid.
        """
        phy_id, err = get_phy_id(-1)
        self.assertEqual(phy_id, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid logicID", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_phy_id behavior when the function is missing.
        """
        mock_get_dcmi_instance.return_value = None
        phy_id, err = get_phy_id(1)
        self.assertEqual(phy_id, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_phy_id behavior when the function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_phyid_from_logicid = Mock(return_value=-8001)

        phy_id, err = get_phy_id(1)
        self.assertEqual(phy_id, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests get_phy_id behavior when the PHY ID is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(logic_id, phy_ref):  # pylint: disable=unused-argument
            ctypes.cast(phy_ref, ctypes.POINTER(c_uint)).contents.value = 1
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_phyid_from_logicid.side_effect = fake_func

        phy_id, err = get_phy_id(1)
        self.assertEqual(phy_id, 1)
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_phy_id_written(self, mock_get_dcmi_instance):
        """
        Tests get_phy_id behavior when an invalid PHY ID is written.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        invalid_phy = HIAI_MAX_CARD_NUM * HIAI_MAX_DEVICE_NUM

        def fake_func(logic_id, phy_ref):  # pylint: disable=unused-argument
            ctypes.cast(phy_ref, ctypes.POINTER(c_uint)).contents.value = invalid_phy
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_phyid_from_logicid.side_effect = fake_func

        phy_id, err = get_phy_id(1)
        self.assertEqual(phy_id, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid phyID", str(err))


class TestGetUtilizationRate(unittest.TestCase):
    """
    TestGetUtilizationRate covers `get_utilization_rate` with bad IDs, missing symbol, error returns, and value checks.
    """

    def test_invalid_card_or_device_id(self):
        """
        Tests get_utilization_rate behavior when the card ID or device ID is invalid.
        """
        rate, err = get_utilization_rate(-1, 0, 0)
        self.assertEqual(rate, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid cardID", str(err))

        rate, err = get_utilization_rate(0, -1, 0)
        self.assertEqual(rate, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_utilization_rate behavior when the function is missing.
        """
        mock_get_dcmi_instance.return_value = None
        rate, err = get_utilization_rate(0, 0, 6)
        self.assertEqual(rate, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_utilization_rate behavior when the function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_utilization_rate.return_value = -8001

        rate, err = get_utilization_rate(0, 0, 6)
        self.assertEqual(rate, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_valid_rate(self, mock_get_dcmi_instance):
        """
        Tests get_utilization_rate behavior when a valid utilization rate is retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card_id, device_id, dev_type, rate_ref):  # pylint: disable=unused-argument
            ctypes.cast(rate_ref, ctypes.POINTER(c_uint)).contents.value = 85
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_utilization_rate.side_effect = fake_func

        rate, err = get_utilization_rate(0, 0, 0)
        self.assertEqual(rate, 85)
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_rate_written(self, mock_get_dcmi_instance):
        """
        Tests get_utilization_rate behavior when an invalid utilization rate is written.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card_id, device_id, dev_type, rate_ref):  # pylint: disable=unused-argument
            ctypes.cast(rate_ref, ctypes.POINTER(c_uint)).contents.value = PERCENT + 1
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_utilization_rate.side_effect = fake_func

        rate, err = get_utilization_rate(0, 0, 0)
        self.assertEqual(rate, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)
        self.assertIn("invalid utilization rate", str(err))


def _write_c_ubyte_text(arr, text: str):
    data = text.encode("utf-8")
    n = min(len(arr) - 1, len(data))
    for i in range(n):
        arr[i] = data[i]
    arr[n] = 0


class TestGetBoardInfo(unittest.TestCase):
    """
    TestGetBoardInfo tests `get_board_info` for API presence, error returns, and payload validity.
    """

    def test_invalid_ids(self):
        """
        Tests get_board_info behavior when invalid card and device IDs are provided.
        """
        info, code, err = get_board_info(-1, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_missing(self, mock_get_dcmi_instance):
        """
        Tests get_board_info behavior when the DcmiManager instance is missing.
        """
        mock_get_dcmi_instance.return_value = None
        info, code, err = get_board_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, RET_NO_FUNCTION)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("function not found", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_function_return_error(self, mock_get_dcmi_instance):
        """
        Tests get_board_info behavior when the Dcmi function returns an error code.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_get_device_board_info.return_value = -8001
        info, code, err = get_board_info(0, 0)
        self.assertEqual(info, {})
        self.assertEqual(code, -8001)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("ret: -8001", str(err))

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success(self, mock_get_dcmi_instance):
        """
        Tests get_board_info behavior when board information is successfully retrieved.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiBoardInfoStruct)).contents
            s.board_id = 0xAB
            s.pcb_id = 5678
            s.bom_id = 0
            s.slot_id = 0
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_board_info.side_effect = fake_func

        info, code, err = get_board_info(0, 0)
        self.assertIsNone(code)
        self.assertIsNone(err)
        self.assertEqual(
            info,
            {"board_id": 0xAB, "pcb_id": 5678, "bom_id": 0, "slot_id": 0},
        )

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_success_but_invalid_board_info(self, mock_get_dcmi_instance):
        """
        Tests get_board_info behavior when the board information is invalid.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_func(card, dev, out_ptr):  # pylint: disable=unused-argument
            s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiBoardInfoStruct)).contents
            s.board_id = c_uint(INVALID_ID).value
            s.pcb_id = c_uint(INVALID_ID).value
            s.bom_id = c_uint(INVALID_ID).value
            s.slot_id = c_uint(INVALID_ID).value
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_board_info.side_effect = fake_func

        info, code, err = get_board_info(0, 0)

        self.assertEqual(info, {})
        self.assertEqual(code, RET_INVALID_VALUE)
        self.assertIsInstance(err, ValueError)


class TestFetchDeviceInfoAndType(unittest.TestCase):
    """
    TestFetchDeviceInfoAndType validates `fetch_device_info_and_type` composition
    and error propagation.
    """

    def test_chip_info_error(self):
        """
        Tests fetch_device_info_and_type behavior when chip information retrieval fails.
        """
        with patch("common.npu_metrics.utils.get_chip_info",
                   return_value=({}, RET_NO_FUNCTION, RuntimeError("not found"))):
            chip_info, board_info, device_type, code, err = fetch_device_info_and_type(0, 0)
            self.assertEqual(chip_info, {})
            self.assertEqual(board_info, {})
            self.assertIsNone(device_type)
            self.assertEqual(code, RET_NO_FUNCTION)
            self.assertIsInstance(err, RuntimeError)
            self.assertIn("not found", str(err))

    def test_board_info_error(self):
        """
        Tests fetch_device_info_and_type behavior when board information retrieval fails.
        """
        with patch("common.npu_metrics.utils.get_chip_info",
                   return_value=({"chip_name": "910B4", "chip_type": "Ascend", "chip_version": "v1"}, None, None)), \
                patch("common.npu_metrics.utils.get_board_info",
                      return_value=({}, RET_NO_FUNCTION, RuntimeError("board err"))):
            chip_info, board_info, device_type, code, err = fetch_device_info_and_type(0, 0)
            self.assertEqual(chip_info["chip_name"], "910B4")
            self.assertEqual(board_info, {})
            self.assertIsNone(device_type)
            self.assertEqual(code, RET_NO_FUNCTION)
            self.assertIsInstance(err, RuntimeError)
            self.assertIn("board err", str(err))

    def test_device_type_empty(self):
        """
        Tests fetch_device_info_and_type behavior when the device type is empty.
        """
        with patch("common.npu_metrics.utils.get_chip_info",
                   return_value=({"chip_name": "UNKNOWN", "chip_type": "Ascend", "chip_version": "v1"}, None, None)), \
                patch("common.npu_metrics.utils.get_board_info",
                      return_value=({"board_id": 0xAB, "pcb_id": 5678, "bom_id": 0, "slot_id": 0}, None, None)), \
                patch("common.npu_metrics.utils.get_device_type", return_value=""):
            chip, board, dtype, code, err = fetch_device_info_and_type(1, 2)
            self.assertEqual(chip["chip_name"], "UNKNOWN")
            self.assertEqual(board["board_id"], 0xAB)
            self.assertIsNone(dtype)
            self.assertEqual(code, RET_INVALID_VALUE)
            self.assertIsInstance(err, ValueError)
            self.assertIn("device type is empty", str(err))

    def test_success(self):
        """
        Tests fetch_device_info_and_type behavior when all device info is successfully retrieved.
        """
        with patch("common.npu_metrics.utils.get_chip_info",
                   return_value=({"chip_name": "910B4", "chip_type": "Ascend", "chip_version": "v1"}, None, None)), \
                patch("common.npu_metrics.utils.get_board_info",
                      return_value=({"board_id": 0xb0, "pcb_id": 123, "bom_id": 0, "slot_id": 0}, None, None)), \
                patch("common.npu_metrics.utils.get_device_type", return_value="ASCEND_910B"):
            chip_info, board_info, device_type, code, err = fetch_device_info_and_type(0, 0)
            self.assertEqual(chip_info, {"chip_name": "910B4", "chip_type": "Ascend", "chip_version": "v1"})
            self.assertEqual(board_info, {"board_id": 0xb0, "pcb_id": 123, "bom_id": 0, "slot_id": 0})
            self.assertEqual(device_type, "ASCEND_910B")
            self.assertIsNone(code)
            self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
