/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// DecodePluginParameters decodes a plugin parameters JSON object with unknown
// field rejection while preserving the empty-raw-message default contract.
func DecodePluginParameters(raw json.RawMessage, params any) error {
	if len(raw) == 0 {
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(params); err != nil {
		return fmt.Errorf("invalid parameters: %w", err)
	}
	return nil
}

// ValidateEmptyPluginParameters accepts an omitted or empty JSON object and
// rejects explicit null and unknown fields for parameterless plugins.
func ValidateEmptyPluginParameters(raw json.RawMessage) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil
	}
	if bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("invalid parameters: explicit null is not allowed")
	}
	var params struct{}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&params); err != nil {
		return fmt.Errorf("invalid parameters: %w", err)
	}
	return nil
}
