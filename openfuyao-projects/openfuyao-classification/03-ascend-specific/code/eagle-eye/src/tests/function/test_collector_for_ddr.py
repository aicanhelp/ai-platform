# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# YOU MAY obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ctypes
import pytest

from unittest.mock import MagicMock

from hardware_monitor.collector.metrics.npu.collector_for_ddr import DDRCollector
from common.npu_metrics.dcmi.dcmi_wrapper import (
    DcmiMemoryInfoStruct,
    DcmiChipInfoV2Struct,
    DcmiBoardInfoStruct,
)
from common.npu_metrics.constants import RET_INVALID_VALUE


def _write_c_ubyte_text(arr, text: str) -> None:
    data = text.encode("utf-8")
    n = min(len(arr) - 1, len(data))
    for i in range(n):
        arr[i] = data[i]
    arr[n] = 0


@pytest.fixture
def mock_dcmi_instance_fixture(mocker):
    mock_dcmi_instance = MagicMock()
    mock_dcmi = MagicMock()
    mock_dcmi_instance.dcmi = mock_dcmi

    mocker.patch(
        "common.npu_metrics.dcmi.dcmi.DcmiManager.get_dcmi_instance",
        return_value=mock_dcmi_instance,
    )

    return mock_dcmi_instance


def _setup_collector_with_single_device() -> DDRCollector:
    collector = DDRCollector()
    fake_cache = MagicMock()

    def _fake_get(key):
        if key in ("logic_id", "phy_id", "card_id", "device_id"):
            return [0]
        return []

    fake_cache.get.side_effect = _fake_get
    collector.cache = fake_cache
    return collector


@pytest.mark.parametrize(
    "scenario, chip_name, memory_size, memory_available, "
    "expect_error_code, expect_error_message, "
    "expect_total, expect_used, expect_freq, expect_util",
    [
        # 场景一：910B（无DDR模块）
        (
            "no_ddr_module",
            "910B4",
            1024,   # 实际不会用到
            512,    # 实际不会用到
            RET_INVALID_VALUE,
            "LogicID(0): Ascend910B does not have ddr module",
            "NA", "NA", "NA", "NA",
        ),
        # 场景二：310 + 内存信息非法（total=0）
        (
            "invalid_memory",
            "310",
            0,
            512,
            RET_INVALID_VALUE,
            "Invalid memory info for logicID(0), total(0) < available(512) or total is 0",
            "NA", "NA", "NA", "NA",
        ),
        # 场景三：310 + 正常内存信息
        (
            "success",
            "310",
            1024,
            512,
            None,
            None,
            1024, 512, 1000, 50,
        ),
    ],
    ids=["no_ddr_module", "invalid_memory", "success"],
)
def test_ddr_collect_impl(
    mock_dcmi_instance_fixture,
    monkeypatch,
    scenario,
    chip_name,
    memory_size,
    memory_available,
    expect_error_code,
    expect_error_message,
    expect_total,
    expect_used,
    expect_freq,
    expect_util,
):
    monkeypatch.setenv("NODE_NAME", "node-1")

    def fake_get_chip_info(card_id, device_id, out_ptr):  # pylint: disable=unused-argument
        s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiChipInfoV2Struct)).contents
        _write_c_ubyte_text(s.chip_type, "Ascend")
        _write_c_ubyte_text(s.chip_name, chip_name)
        _write_c_ubyte_text(s.chip_ver, "v1")
        _write_c_ubyte_text(s.npu_name, "NPU" + chip_name)
        s.aicore_cnt = 64
        return 0

    def fake_get_board_info(card_id, device_id, out_ptr):  # pylint: disable=unused-argument
        s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiBoardInfoStruct)).contents
        s.board_id = 0xAB
        s.pcb_id = 5678
        s.bom_id = 0
        s.slot_id = 0
        return 0

    def fake_get_device_memory_info(card, dev, out_ptr):  # pylint: disable=unused-argument
        s = ctypes.cast(out_ptr, ctypes.POINTER(DcmiMemoryInfoStruct)).contents
        s.memory_size = memory_size
        s.memory_available = memory_available
        s.freq = 1000
        s.hugepagesize = 4096
        s.hugepages_total = 128
        s.hugepages_free = 64
        s.utiliza = 50
        return 0

    dcmi = mock_dcmi_instance_fixture.dcmi
    dcmi.dcmi_get_device_chip_info_v2.side_effect = fake_get_chip_info
    dcmi.dcmi_get_device_board_info.side_effect = fake_get_board_info
    dcmi.dcmi_get_device_memory_info_v3.side_effect = fake_get_device_memory_info

    collector = _setup_collector_with_single_device()
    result = collector._collect_impl()
    info = result["0"]

    assert info["node_name"] == "node-1"
    assert info["card_id"] == 0
    assert info["device_id"] == 0

    errors = info.get("errors")

    if expect_error_code is None:
        assert errors == []
    else:
        assert len(errors) == 1
        err = errors[0]
        assert err["module"] == "mem_info"
        assert err["error_code"] == expect_error_code
        assert expect_error_message in err["error_message"]

    if expect_total == "NA":
        assert info["npu_chip_info_total_memory"] == "NA"
        assert info["npu_chip_info_used_memory"] == "NA"
        assert info["npu_chip_info_memory_frequency"] == "NA"
        assert info["npu_chip_info_memory_utilization"] == "NA"
    elif expect_total is not None:
        assert info["npu_chip_info_total_memory"] == expect_total
        assert info["npu_chip_info_used_memory"] == expect_used
        assert info["npu_chip_info_memory_frequency"] == expect_freq
        assert info["npu_chip_info_memory_utilization"] == expect_util
