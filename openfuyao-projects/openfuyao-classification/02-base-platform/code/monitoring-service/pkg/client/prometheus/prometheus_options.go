/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package prometheus

import "os"

// Options defines configuration settings for the application.
type Options struct {
	Endpoint           string `json:"endpoint,omitempty" yaml:"endpoint,omitempty"`
	InsecureSkipVerify bool   `json:"insecureSkipVerify" yaml:"insecureSkipVerify"`
}

// NewPrometheusOptions initializes and returns an instance of Options with Prometheus endpoint.
func NewPrometheusOptions() *Options {
	return &Options{
		Endpoint:           getPrometheusEndpoint(),
		InsecureSkipVerify: getPrometheusInsecureSkipVerify(),
	}
}

// getPrometheusEndpoint retrieves the Prometheus service endpoint from environment variables.
func getPrometheusEndpoint() string {

	prometheusEndpoint := os.Getenv("PROMETHEUS_HOST")
	return prometheusEndpoint
}

func getPrometheusInsecureSkipVerify() bool {
	return os.Getenv("PROMETHEUS_INSECURE_SKIP_VERIFY") == "true"
}
