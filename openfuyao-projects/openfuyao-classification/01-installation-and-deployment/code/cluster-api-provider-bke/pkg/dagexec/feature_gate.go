/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

// WithFeatureGateExecutors applies feature-gate injection rules to Config.
// When enabled is false, Yaml/Helm executors are cleared (Legacy for those types).
// When enabled is true, the provided executors are set (nil still means not registered).
func WithFeatureGateExecutors(cfg Config, enabled bool, yamlExecutor, helmExecutor ComponentExecutor) Config {
	if !enabled {
		cfg.YamlExecutor = nil
		cfg.HelmExecutor = nil
		return cfg
	}
	cfg.YamlExecutor = yamlExecutor
	cfg.HelmExecutor = helmExecutor
	return cfg
}
