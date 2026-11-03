/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package hooks

import (
	"encoding/json"
	"fmt"
	"os"
)

type config struct {
	// 添加：
	// hook action的enable
	// hook fail policy
}

// hook config
func readHookActionConfigFile(configPath string) (*config, error) {
	if configPath == "" {
		return nil, fmt.Errorf("credential provider config path is empty")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("unable to read  configuration from %q: %w", configPath, err)
	}
	cf := &config{}

	err = json.Unmarshal(data, &cf)
	if err != nil {
		return nil, fmt.Errorf("error decoding config %s: %w", configPath, err)
	}

	return cf, nil
}

func validateHookActionConfigFile(config *config) []error {
	var allErrs []error

	return allErrs
}
