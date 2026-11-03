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

// Package fuyao defines webhook handlers for fuyao internal oauth2server
package fuyao

import (
	"encoding/json"
	"net/http"

	"k8s.io/api/authentication/v1"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/httpserver"
	"openfuyao/oauth-webhook/pkg/zlog"
)

// InnerFuyaoWebhook is the struct for fuyao internal oauth2server webhook handler
type InnerFuyaoWebhook struct {
	// init a validator
	validator Validator
}

// NewInnerFuyaoWebhook init func for InnerFuyaoWebhook
func NewInnerFuyaoWebhook(signingKey []byte) *InnerFuyaoWebhook {
	return &InnerFuyaoWebhook{validator: NewJWTValidator(signingKey)}
}

// WebhookHandler fuyao internal oauth2server webhook handler
func (h *InnerFuyaoWebhook) WebhookHandler(w http.ResponseWriter, r *http.Request) {
	var req v1.TokenReview

	// only allow post requests
	if r.Method != http.MethodPost {
		zlog.LogErrorf(fuyaoerrors.ErrStrRequestMethodNotAllowed)
		req.Status = v1.TokenReviewStatus{
			Authenticated: false,
			Error:         fuyaoerrors.ErrStrRequestMethodNotAllowed,
		}
		httpserver.OutputResponse(w, &req, http.StatusBadRequest)
		return
	}

	// fetch and check all params
	decoder := json.NewDecoder(r.Body)
	err := decoder.Decode(&req)
	if err != nil {
		zlog.LogErrorf("%s", fuyaoerrors.ErrStrFailToDecodeTokenReview)
		req.Status = v1.TokenReviewStatus{
			Authenticated: false,
			Error:         fuyaoerrors.ErrStrFailToDecodeTokenReview,
		}
		httpserver.OutputResponse(w, &req, http.StatusInternalServerError)
		return
	}

	// specific handler
	tokenReview, ok := h.validator.Validate(req)

	// wrap http return
	if !ok {
		httpserver.OutputResponse(w, tokenReview, http.StatusUnauthorized)
		return
	}

	httpserver.OutputResponse(w, tokenReview, http.StatusOK)
	return
}
