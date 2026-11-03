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

package httpserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestOutputResponse(t *testing.T) {
	testOutputResponseNormalCases(t)
	testOutputResponseNilReview(t)
	testOutputResponseInvalidJSON(t)
}

func testOutputResponseNormalCases(t *testing.T) {
	testTokenReview := createTestTokenReview()

	tests := []struct {
		name           string
		review         *v1.TokenReview
		status         int
		expectedStatus int
	}{
		{
			name:           "success response",
			review:         testTokenReview,
			status:         http.StatusOK,
			expectedStatus: http.StatusOK,
		},
		{
			name: "unauthorized response",
			review: &v1.TokenReview{
				Status: v1.TokenReviewStatus{Authenticated: false},
			},
			status:         http.StatusUnauthorized,
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "internal server error response",
			review:         &v1.TokenReview{},
			status:         http.StatusInternalServerError,
			expectedStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			OutputResponse(rr, tt.review, tt.status)

			assert.Equal(t, tt.expectedStatus, rr.Code)

			if tt.review != nil {
				var response v1.TokenReview
				err := json.Unmarshal(rr.Body.Bytes(), &response)
				assert.NoError(t, err)
				assert.Equal(t, tt.review, &response)
			}
		})
	}
}

func testOutputResponseNilReview(t *testing.T) {
	t.Run("nil review", func(t *testing.T) {
		rr := httptest.NewRecorder()
		OutputResponse(rr, nil, http.StatusOK)

		assert.Equal(t, http.StatusOK, rr.Code)
		assert.Equal(t, "null\n", rr.Body.String())
	})
}

func testOutputResponseInvalidJSON(t *testing.T) {
	t.Run("normal review should encode successfully", func(t *testing.T) {
		rr := httptest.NewRecorder()
		review := &v1.TokenReview{
			Status: v1.TokenReviewStatus{
				Authenticated: true,
			},
		}

		OutputResponse(rr, review, http.StatusOK)

		assert.Equal(t, http.StatusOK, rr.Code)
		assert.NotEmpty(t, rr.Body.String())
	})
}

func createTestTokenReview() *v1.TokenReview {
	return &v1.TokenReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "TokenReview",
			APIVersion: "authentication.k8s.io/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-token-review",
		},
		Spec: v1.TokenReviewSpec{
			Token: "test-token",
		},
		Status: v1.TokenReviewStatus{
			Authenticated: true,
			User: v1.UserInfo{
				Username: "test-user",
				UID:      "test-uid",
				Groups:   []string{"test-group"},
			},
		},
	}
}
