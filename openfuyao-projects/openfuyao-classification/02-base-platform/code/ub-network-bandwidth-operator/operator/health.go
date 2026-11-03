/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package operator

import (
	"net/http"
	"os"

	"k8s.io/klog/v2"
)

const (
	ubseSocketPath = "/var/run/ubse/ubse.sock"
)

// StartHealthServer starts HTTP health and readiness endpoints.
func StartHealthServer(port string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		files := []string{
			"/var/run/ubse/ubse.sock",
			"/var/run/ubse/ubse_ubm.socket",
		}

		var missingFiles []string
		for _, file := range files {
			if _, err := os.Stat(file); os.IsNotExist(err) {
				missingFiles = append(missingFiles, file)
			}
		}

		if len(missingFiles) > 0 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, err := w.Write([]byte(`{"status": "unhealthy", "message": "Missing socket files: ` + missingFiles[0] + `"}`))
			if err != nil {
				return
			}
		} else {
			w.WriteHeader(http.StatusOK)
			_, err := w.Write([]byte(`{"status": "healthy", "message": "All socket files exist"}`))
			if err != nil {
				return
			}
		}
	})

	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte(`{"status": "ready", "message": "Bandwidth operator is ready"}`))
		if err != nil {
			return
		}
	})

	go func() {
		klog.Infof("health server started on port %s", port)
		if err := http.ListenAndServe(":"+port, mux); err != nil {
			klog.Errorf("health server failed: %v", err)
		}
	}()
}
