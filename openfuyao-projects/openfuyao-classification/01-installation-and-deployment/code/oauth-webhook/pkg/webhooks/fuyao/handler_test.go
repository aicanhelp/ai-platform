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

package fuyao

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/api/authentication/v1"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
)

var testValidator = NewJWTValidator([]byte("a-valid-string-secret-that-is-at-least-512-bits-long-which-is-very-long"))

func TestNewInnerFuyaoWebhook(t *testing.T) {
	hook := NewInnerFuyaoWebhook([]byte("signing-key"))
	if hook == nil || hook.validator == nil {
		t.Fatal("expected webhook validator to be initialized")
	}
}

// TestWebhookHandlerMethodNotAllowed test cases when the http method is not post
func TestWebhookHandlerMethodNotAllowed(t *testing.T) {
	req, err := http.NewRequest("GET", "/oauth/tokenauth/fuyao", nil)
	if err != nil {
		t.Fatal(err)
	}
	h := &InnerFuyaoWebhook{validator: testValidator}
	rr := httptest.NewRecorder()
	h.WebhookHandler(rr, req)

	// Check the status code
	if rr.Code != http.StatusBadRequest {
		t.Errorf("Expected status code %d; got %d", http.StatusBadRequest, rr.Code)
	}

	// Check the response body
	var respReview v1.TokenReview
	err = json.Unmarshal(rr.Body.Bytes(), &respReview)
	if err != nil {
		t.Fatalf("Error decoding response body: %v", err)
	}

	// Assert on the fields of respReview
	if respReview.Status.Authenticated {
		t.Errorf("Expected authenticated status to be false, got true")
	}
	if respReview.Status.Error != fuyaoerrors.ErrStrRequestMethodNotAllowed {
		t.Errorf(
			"Expected error message '%s', got '%s'",
			fuyaoerrors.ErrStrRequestMethodNotAllowed,
			respReview.Status.Error,
		)
	}
}

// TestWebhookHandlerParamsCheckFailed test the cases that the input body is incomplete
func TestWebhookHandlerParamsCheckFailed(t *testing.T) {
	// Prepare randomBytes for error json decoding
	randomBytes := make([]byte, 100)

	req, err := http.NewRequest("POST", "/oauth/tokenauth/fuyao", bytes.NewBuffer(randomBytes))
	if err != nil {
		t.Fatal(err)
	}
	h := &InnerFuyaoWebhook{validator: testValidator}
	rr := httptest.NewRecorder()
	h.WebhookHandler(rr, req)

	// Check the status code
	if rr.Code != http.StatusInternalServerError {
		t.Errorf("Expected status code %d; got %d", http.StatusInternalServerError, rr.Code)
	}

	// Check the response body
	var respReview v1.TokenReview
	err = json.Unmarshal(rr.Body.Bytes(), &respReview)
	if err != nil {
		t.Fatalf("Error decoding response body: %v", err)
	}

	// Assert on the fields of respReview
	if respReview.Status.Authenticated {
		t.Errorf("Expected authenticated status to be false, got true")
	}
	if respReview.Status.Error != fuyaoerrors.ErrStrFailToDecodeTokenReview {
		t.Errorf(
			"Expected error message '%s', got '%s'",
			fuyaoerrors.ErrStrRequestMethodNotAllowed,
			respReview.Status.Error,
		)
	}
}

// TestWebhookHandlerValidateFailed test the case that the token in invalid
func TestWebhookHandlerValidateFailed(t *testing.T) {
	// Prepare a valid token review request
	tokenReviewReq := v1.TokenReview{
		Spec: v1.TokenReviewSpec{
			Token: "invalid_token",
		},
	}

	body, err := json.Marshal(tokenReviewReq)
	if err != nil {
		t.Fatal(err)
	}

	request, err := http.NewRequest("POST", "/oauth/tokenauth/fuyao", bytes.NewBuffer(body))
	if err != nil {
		t.Fatal(err)
	}
	h := &InnerFuyaoWebhook{validator: testValidator}
	rr := httptest.NewRecorder()
	h.WebhookHandler(rr, request)

	// Check the status code
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("Expected status code %d; got %d", http.StatusUnauthorized, rr.Code)
	}

	// Check the response body
	var respReview v1.TokenReview
	err = json.Unmarshal(rr.Body.Bytes(), &respReview)
	if err != nil {
		t.Fatalf("Error decoding response body: %v", err)
	}

	// Assert on the fields of respReview
	if respReview.Status.Authenticated {
		t.Errorf("Expected authenticated status to be false, got true")
	}
	if respReview.Status.Error != "token contains an invalid number of segments" {
		t.Errorf(
			"Expected error message token contains an invalid number of segments, got '%s'",
			respReview.Status.Error,
		)
	}
}

// TestWebhookHandlerSuccessfullyHandled is the successful example for testing
func TestWebhookHandlerSuccessfullyHandled(t *testing.T) {
	// Prepare a valid token review request
	tokenReviewReq := v1.TokenReview{
		Spec: v1.TokenReviewSpec{
			Token: "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTkyNDkyMDAwMCwic3ViIjo" +
				"iYWRtaW4ifQ.49rrmIRdFXfhigt70993BThscDv6fZECMQYb6g1c9JPVpXYL7ghKJ_OCOp8eKsisZBlgsm0jAXpo0ffJzfuaYQ",
		},
	}

	body, err := json.Marshal(tokenReviewReq)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "/oauth/tokenauth/fuyao", bytes.NewBuffer(body))
	if err != nil {
		t.Fatal(err)
	}
	h := &InnerFuyaoWebhook{validator: testValidator}
	rr := httptest.NewRecorder()
	h.WebhookHandler(rr, req)

	// Check the status code
	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code %d; got %d", http.StatusOK, rr.Code)
	}

	// Check the response body
	var respReview v1.TokenReview
	err = json.Unmarshal(rr.Body.Bytes(), &respReview)
	if err != nil {
		t.Fatalf("Error decoding response body: %v", err)
	}

	// Assert on the fields of respReview
	if !respReview.Status.Authenticated {
		t.Errorf("Expected authenticated status to be true, got false")
	}
	if respReview.Status.Error != "" {
		t.Errorf("Expected no err, but got '%s'", respReview.Status.Error)
	}
	if respReview.Status.User.Username != "admin" {
		t.Errorf("Expected user admin, but got '%s'", respReview.Status.User.Username)
	}
}
