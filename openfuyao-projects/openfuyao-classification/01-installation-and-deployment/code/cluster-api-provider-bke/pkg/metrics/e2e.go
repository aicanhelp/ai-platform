/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
)

const (
	e2eDataSavePath = "/bke2e/metrics"
	// DirMode is the directory permission mode (read, write, execute for owner; read, execute for group and others)
	DirMode = 0755
	// FileMode is the file permission mode (read, write for owner; read for group and others)
	FileMode = 0644
)

// gatherMetrics collects metrics for the given key and metric names
func (r *BKEClusterMetricRegister) gatherMetrics(key string) (map[string][]map[string]string, error) {
	return r.Gather(key, PhaseDurationSeconds, NodeBootstrapDurationSeconds)
}

// processMetrics converts metrics to httpExportResponse format
func processMetrics(metricsGather map[string][]map[string]string) []httpExportResponse {
	return ProcessMetrics(metricsGather)
}

// serializeMetrics serializes metrics to JSON format
func serializeMetrics(resp []httpExportResponse) ([]byte, error) {
	return json.MarshalIndent(resp, "", " ")
}

// prepareFilePath prepares the file path for metrics data
func prepareFilePath(key string) string {
	key = strings.Replace(key, "/", "_", -1)
	return fmt.Sprintf("%s/%s.json", e2eDataSavePath, key)
}

// ensureDirectory ensures the directory exists
func ensureDirectory(filePath string) error {
	if !utils.Exists(path.Dir(filePath)) {
		if err := os.MkdirAll(path.Dir(filePath), DirMode); err != nil {
			return err
		}
	}
	return nil
}

// writeMetricsToFile writes metrics data to the specified file
func writeMetricsToFile(metricsPath string, metricsData []byte) error {
	return os.WriteFile(metricsPath, metricsData, FileMode)
}

// ensureBaseDirectory ensures the base directory exists
func ensureBaseDirectory() error {
	if !utils.Exists(e2eDataSavePath) {
		if err := os.MkdirAll(e2eDataSavePath, DirMode); err != nil {
			return err
		}
	}
	return nil
}

// E2EDataGather gathers E2E metrics data and saves to file
func (r *BKEClusterMetricRegister) E2EDataGather(key string) error {
	// Ensure base directory exists
	if err := ensureBaseDirectory(); err != nil {
		return err
	}

	// Gather metrics
	metricsGather, err := r.gatherMetrics(key)
	if err != nil {
		return err
	}

	// Process metrics
	resp := processMetrics(metricsGather)

	// Serialize metrics
	metricsData, err := serializeMetrics(resp)
	if err != nil {
		return err
	}

	// Prepare file path
	metricsPath := prepareFilePath(key)

	// Ensure directory exists
	if err := ensureDirectory(metricsPath); err != nil {
		return err
	}

	// Write metrics to file
	return writeMetricsToFile(metricsPath, metricsData)
}
