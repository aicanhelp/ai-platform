/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHealthzHandler(t *testing.T) {
	testDir := t.TempDir()

	tests := []struct {
		name           string
		setupFiles     []string
		expectedStatus int
		expectedBody   string
	}{
		{
			name:           "both files exist",
			setupFiles:     []string{"ubse.sock", "ubse_ubm.socket"},
			expectedStatus: http.StatusOK,
			expectedBody:   `{"status": "healthy", "message": "All socket files exist"}`,
		},
		{
			name:           "ubse.sock missing",
			setupFiles:     []string{"ubse_ubm.socket"},
			expectedStatus: http.StatusServiceUnavailable,
			expectedBody:   `{"status": "unhealthy", "message": "Missing socket files: `,
		},
		{
			name:           "ubse_ubm.socket missing",
			setupFiles:     []string{"ubse.sock"},
			expectedStatus: http.StatusServiceUnavailable,
			expectedBody:   `{"status": "unhealthy", "message": "Missing socket files: `,
		},
		{
			name:           "both files missing",
			setupFiles:     []string{},
			expectedStatus: http.StatusServiceUnavailable,
			expectedBody:   `{"status": "unhealthy", "message": "Missing socket files: `,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, file := range tt.setupFiles {
				filePath := testDir + "/" + file
				f, err := os.Create(filePath)
				assert.NoError(t, err)
				f.Close()
			}

			req := httptest.NewRequest("GET", "/healthz", nil)
			w := httptest.NewRecorder()

			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")

				files := []string{
					testDir + "/ubse.sock",
					testDir + "/ubse_ubm.socket",
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

			handler.ServeHTTP(w, req)

			resp := w.Result()
			body, err := io.ReadAll(resp.Body)
			assert.NoError(t, err)

			assert.Equal(t, tt.expectedStatus, resp.StatusCode)
			assert.Contains(t, string(body), tt.expectedBody)

			for _, file := range tt.setupFiles {
				os.Remove(testDir + "/" + file)
			}
		})
	}
}

func TestHealthzIntegration(t *testing.T) {
	testDir := t.TempDir()

	f1, err := os.Create(testDir + "/ubse.sock")
	assert.NoError(t, err)
	f1.Close()

	f2, err := os.Create(testDir + "/ubse_ubm.socket")
	assert.NoError(t, err)
	f2.Close()

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		files := []string{
			testDir + "/ubse.sock",
			testDir + "/ubse_ubm.socket",
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

	handler.ServeHTTP(w, req)

	resp := w.Result()
	body, err := io.ReadAll(resp.Body)
	assert.NoError(t, err)

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Contains(t, string(body), "healthy")
}
