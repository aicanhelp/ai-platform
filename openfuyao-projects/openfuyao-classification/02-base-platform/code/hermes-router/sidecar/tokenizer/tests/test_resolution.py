# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from providers.resolution import _is_remote_model


def test_is_remote_model_recognizes_huggingface_style_id() -> None:
    assert _is_remote_model("Qwen/Qwen3-8B") is True


def test_is_remote_model_recognizes_existing_local_path(tmp_path: Path) -> None:
    local_dir = tmp_path / "local-model"
    local_dir.mkdir()

    assert _is_remote_model(str(local_dir)) is False


def test_is_remote_model_rejects_unsupported_uri_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported URI scheme"):
        _is_remote_model("s3://bucket/model")


def test_is_remote_model_recognizes_absolute_path(tmp_path: Path) -> None:
    local_dir = tmp_path / "absolute-model"
    local_dir.mkdir()
    absolute_path = local_dir.resolve()

    assert _is_remote_model(str(absolute_path)) is False


def test_is_remote_model_recognizes_relative_path_dot_slash() -> None:
    assert _is_remote_model("./local-model") is False
    assert _is_remote_model("../parent-model") is False


def test_is_remote_model_recognizes_home_expanded_path() -> None:
    assert _is_remote_model("~/models/my-model") is False


def test_is_remote_model_recognizes_nonexistent_local_directory(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent-model"
    assert not nonexistent.exists()

    assert _is_remote_model(str(nonexistent)) is False


def test_is_remote_model_treats_single_segment_name_as_local_like() -> None:
    assert _is_remote_model("Qwen3-8B") is False


def test_is_remote_model_treats_org_slash_name_as_remote() -> None:
    assert _is_remote_model("deepseek-ai/DeepSeek-V3") is True


def test_is_remote_model_handles_symlink(tmp_path: Path) -> None:
    real_dir = tmp_path / "real-model"
    real_dir.mkdir()
    symlink_path = tmp_path / "symlink-model"
    symlink_path.symlink_to(real_dir)

    assert _is_remote_model(str(symlink_path)) is False


def test_is_remote_model_handles_file_uri(tmp_path: Path) -> None:
    local_dir = tmp_path / "file-uri-model"
    local_dir.mkdir()

    with pytest.raises(ValueError, match="unsupported URI scheme"):
        _is_remote_model(f"file://{local_dir}")
