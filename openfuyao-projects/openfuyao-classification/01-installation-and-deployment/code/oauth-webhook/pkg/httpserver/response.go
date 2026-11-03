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

// Package httpserver defines the httpserver options, middlewares, and responses
package httpserver

import (
	"encoding/json"
	"net/http"

	"k8s.io/api/authentication/v1"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/zlog"
)

// OutputResponse outputs the TokenReview with status and possible fuyaoerrors
func OutputResponse(w http.ResponseWriter, review *v1.TokenReview, status int) {
	w.WriteHeader(status)
	err := json.NewEncoder(w).Encode(review)
	if err != nil {
		zlog.LogErrorf("%s", fuyaoerrors.ErrStrFailToEncodeTokenReviewResponse)
	}

	return
}
