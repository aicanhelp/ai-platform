// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package lm

import (
	"fmt"

	"github.com/urfave/cli/v2"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

// NewLabelers generates the necessary labelers for the Kubernetes node based on its hardware configuration.
func NewLabelers(c *cli.Context) (common.Labeler, error) {
	ascendLabeler, err := NewAscendLabeler(c)
	if err != nil {
		return nil, fmt.Errorf("error creating Ascend labeler: %v", err)
	}
	if ascendLabeler == nil {
		return nil, fmt.Errorf("error creating Ascend labeler: nil labeler")
	}
	return ascendLabeler, nil
}
