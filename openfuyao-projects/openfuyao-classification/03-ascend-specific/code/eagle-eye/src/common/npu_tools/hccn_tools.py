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
hccn_tool related utilities for NPU network metrics collection.
Extracted from utils.py and collector_for_network.py
"""

import logging
import re
import os
import subprocess
import threading

from typing import Any, Tuple, Union

from common.npu_metrics.constants import RET_INVALID_VALUE, LIMIT_SIZE
from common.base.utils import check_path, LimitedWriter

module_name = os.getenv("MODULE_NAME", "default")
logger = logging.getLogger(module_name)

STAT_PATTERN = re.compile(r"(\w+):(\d+)")
SPEED_PATTERN = re.compile(r"Speed:\s*(\d+)\s*Mb/s", re.IGNORECASE)
BANDWIDTH_PATTERN = re.compile(
    r"Bandwidth\s+(TX|RX):\s*([0-9]*\.?[0-9]+)\s*MB/sec", re.IGNORECASE
)
LINK_UP_COUNT_PATTERN = re.compile(r"\[devid\s+\d+\]link up count\s+:\s+(\d+)")


def get_npu_packet_statistics(logic_id: int, phy_id: int) -> dict[str, Any]:
    """
    Retrieves packet statistics for a given NPU logic ID and physical ID.
    Returns statistics needed to calculate packet loss rate and retransmit rate.
    """
    args = ["-i", str(int(phy_id)), "-stat", "-g"]
    out, ret_code, err = get_info_from_hccn_tool(*args)
    if err is not None:
        msg = f"Failed to get hccn_tool command exec result from logicID({logic_id}): {err}"
        logger.error("[hccn_tools] %s", msg)
        return {"error_code": ret_code, "error_message": msg}

    # Parse statistics from output
    stats = {}

    for line in out.splitlines():
        m = STAT_PATTERN.search(line)
        if m:
            key = m.group(1)
            try:
                value = int(m.group(2))
                stats[key] = value
            except ValueError:
                continue

    # Return available statistics for packet loss and retransmit rate calculation
    fields_of_interest = ["roce_rx_err_pkt_num", "roce_rx_all_pkt_num", "roce_new_pkt_rty_num", "roce_tx_all_pkt_num"]
    result = {field: stats[field] for field in fields_of_interest if field in stats}

    if not result:
        msg = f"No relevant packet statistics fields found for logicID({logic_id})"
        logger.warning("[hccn_tools] %s", msg)
        return {"error_code": RET_INVALID_VALUE, "error_message": msg}

    return result


def get_npu_link_speed(logic_id: int, phy_id: int) -> dict[str, Any]:
    """
    Retrieves the link speed for a given NPU logic ID and physical ID.
    Returns speed in MB/s (converted from Mb/s).
    """
    args = ["-i", str(int(phy_id)), "-speed", "-g"]
    out, ret_code, err = get_info_from_hccn_tool(*args)
    if err is not None:
        msg = f"Failed to get hccn_tool command exec result from logicID({logic_id}): {err}"
        logger.error("[hccn_tools] %s", msg)
        return {"error_code": ret_code, "error_message": msg}

    for line in out.splitlines():
        m = SPEED_PATTERN.search(line)
        if m:
            try:
                speed_mbps = int(m.group(1))
                # Convert Mb/s to MB/s (1 Mb/s = 0.125 MB/s)
                speed_mb_per_sec = speed_mbps * 0.125
                return {"speed": speed_mb_per_sec}
            except ValueError:
                continue

    msg = f"Failed to get npu link speed from logicID({logic_id})"
    logger.warning("[hccn_tools] %s", msg)
    return {"error_code": RET_INVALID_VALUE, "error_message": msg}


def get_npu_interface_traffic(logic_id: int, phy_id: int) -> dict[str, Any]:
    """
    Retrieves the network interface traffic (TX/RX bandwidth)
    for a given NPU logic ID and physical ID.
    """
    args = ["-i", str(int(phy_id)), "-bandwidth", "-g"]
    out, ret_code, err = get_info_from_hccn_tool(*args)
    if err is not None:
        msg = f"Failed to get hccn_tool command exec result from logicID({logic_id}): {err}"
        logger.error("[hccn_tools] %s", msg)
        return {"error_code": ret_code, "error_message": msg}

    tx = float(RET_INVALID_VALUE)
    rx = float(RET_INVALID_VALUE)

    for line in out.splitlines():
        m = BANDWIDTH_PATTERN.search(line)
        if not m:
            continue
        kind = m.group(1).upper()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if kind == "TX":
            tx = val
        elif kind == "RX":
            rx = val

    if tx == float(RET_INVALID_VALUE) or rx == float(RET_INVALID_VALUE):
        msg = f"TX/RX traffic not fully available for logicID({logic_id}), tx={tx}, rx={rx}"
        logger.warning("[hccn_tools] %s", msg)
        return {"error_code": RET_INVALID_VALUE, "error_message": msg}

    return {
        "tx": tx,
        "rx": rx
    }


def get_npu_link_up_count(logic_id: int, phy_id: int) -> dict[str, Any]:
    """
    Retrieves the link up count for a given NPU logic ID and physical ID.
    """
    args = ["-i", str(int(phy_id)), "-link_stat", "-g"]
    out, ret_code, err = get_info_from_hccn_tool(*args)
    if err is not None:
        msg = f"Failed to get hccn_tool command exec result from logicID({logic_id}): {err}"
        logger.error("[hccn_tools] %s", msg)
        return {"error_code": ret_code, "error_message": msg}

    for line in out.splitlines():
        m = LINK_UP_COUNT_PATTERN.search(line)
        if m:
            try:
                count = int(m.group(1))
                return {"link_up_count": count}
            except ValueError:
                continue

    msg = f"Failed to get npu link up count from logicID({logic_id})"
    logger.warning("[hccn_tools] %s", msg)
    return {"error_code": RET_INVALID_VALUE, "error_message": msg}


def get_npu_link_status(logic_id: int, phy_id: int) -> dict[str, Any]:
    """
    Retrieves the link status for a given NPU logic ID and physical ID.
    """
    args = ["-i", str(int(phy_id)), "-link", "-g"]
    out, ret_code, err = get_info_from_hccn_tool(*args)
    if err is not None:
        msg = f"Failed to get hccn_tool command exec result from logicID({logic_id}): {err}"
        logger.error("[hccn_tools] %s", msg)
        return {"error_code": ret_code, "error_message": msg}

    replaced = out.replace("\r", "").replace("\n", "")
    parts = replaced.split()
    if len(parts) == 3 and parts[0].lower() == "link" and parts[1].lower() == "status:":
        status = parts[2].upper()
        return {"status": status}
    msg = f"Failed to get npu link status from logicID({logic_id})"
    logger.error("[hccn_tools] %s", msg)
    return {"error_code": RET_INVALID_VALUE, "error_message": msg}


def get_info_from_hccn_tool(*args: str) -> Tuple[str, Union[int, None], Union[Exception, None]]:
    """
    Runs the hccn_tool command to fetch information, capturing both standard output and error.
    """
    hccn_tool = "/usr/local/Ascend/driver/tools/hccn_tool"
    _, err = check_path(hccn_tool)
    if err is not None:
        return "", RET_INVALID_VALUE, err

    lw_out = LimitedWriter(LIMIT_SIZE)
    lw_err = LimitedWriter(LIMIT_SIZE)

    def drain(stream, sink: LimitedWriter):
        try:
            for chunk in iter(lambda: stream.read(65536), b""):
                sink.write(chunk)
        except Exception:  # pylint:disable=broad-except
            pass

    try:
        with subprocess.Popen(
                [hccn_tool, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
        ) as proc:
            assert proc.stdout is not None and proc.stderr is not None

            t_out = threading.Thread(target=drain, args=(proc.stdout, lw_out), daemon=True)
            t_err = threading.Thread(target=drain, args=(proc.stderr, lw_err), daemon=True)
            t_out.start()
            t_err.start()

            rc = proc.wait()
            t_out.join()
            t_err.join()

            if rc != 0:
                return "", rc, RuntimeError(
                    f"hccn_tool exit code {rc}, stderr={lw_err.get_buffer_bytes().decode(errors='ignore')}"
                )

    except FileNotFoundError:
        return "", RET_INVALID_VALUE, FileNotFoundError(f"hccn_tool not found: {hccn_tool}")
    except Exception as e:  # pylint: disable=broad-except
        return "", RET_INVALID_VALUE, e

    return lw_out.get_buffer_bytes().decode(errors="ignore"), None, None
