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
This module contains unit tests for NPU utility functions that wrap DCMI calls.
"""

import ctypes

from unittest import TestCase
from unittest.mock import patch, Mock

from hardware_diagnosis.recovery.hardware_strategies.npu.utils import get_device_error_code_string, set_device_reset


class TestGetDeviceErrorCodeString(TestCase):
    """
    TestGetDeviceErrorCodeString tests `get_device_error_code_string()` and `set_device_reset()`.
    """

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_get_device_error_code_string_success(self, mock_get_dcmi_instance):
        """
        Tests the successful retrieval of an error code string.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        def fake_dcmi_func(card, device, error_code, buf, buf_size):  # pylint: disable=unused-argument
            mock_error = b"Mock error message"
            if buf_size.value > len(mock_error):
                ctypes.memmove(ctypes.addressof(buf), mock_error, len(mock_error))
                buf[len(mock_error)] = b"\x00"
            return 0

        mock_dcmi_instance.dcmi.dcmi_get_device_errorcode_string.side_effect = fake_dcmi_func

        msg, error_code, err = get_device_error_code_string(0, 0, 0, int(0x8C2FA001))

        self.assertEqual(msg, "Mock error message")
        self.assertIsNone(error_code)
        self.assertIsNone(err)

    @patch("common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance")
    def test_set_device_reset_success(self, mock_get_dcmi_instance):
        """
        Tests the successful execution of a device reset command.
        """
        mock_dcmi_instance = Mock()
        mock_get_dcmi_instance.return_value = mock_dcmi_instance

        mock_dcmi_instance.dcmi.dcmi_set_device_reset.return_value = 0

        ret, err = set_device_reset(0, 0, 0)

        self.assertEqual(ret, 0)
        self.assertIsNone(err)
