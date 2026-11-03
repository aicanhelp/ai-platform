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

// Package server hosts the HTTP API for cache-indexer. The handler is the
// only entry into the query plane and MUST only call the score service.
package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	"gitcode.com/openFuyao/cache-indexer/pkg/score"
)

// Wire types keep field names and casing verbatim;
// upstream Hermes-router depends on them.
type hitRateRequest struct {
	ServerIP []string           `json:"server_ip"`
	Body     hitRateRequestBody `json:"body"`
}

type hitRateRequestBody struct {
	TokenIDs  []int32 `json:"token_ids"`
	CacheSalt string  `json:"cache_salt"`
	BlockSize int64   `json:"block_size"`
}

type hitRateResponse struct {
	ServerScoreList []serverScoreDTO `json:"server_score_list"`
	Message         string           `json:"message"`
	Status          int              `json:"status"`
}

type serverScoreDTO struct {
	ServerIP   string  `json:"server_ip"`
	L1HitRatio float64 `json:"l1_hit_ratio"`
	L3HitRatio float64 `json:"l3_hit_ratio"`
}

type errorResponse struct {
	Message string `json:"message"`
	Status  int    `json:"status"`
}

type healthResponse struct {
	Status string `json:"status"`
}

func newMux(log logr.Logger, scoreSvc score.Service, cfg config.HTTPConfig) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(log, w, http.StatusOK, healthResponse{Status: "ok"})
	})

	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(log, w, http.StatusOK, healthResponse{Status: "ok"})
	})

	mux.HandleFunc("/kv-cache/hit-rate", hitRateHandler(log.WithName("hit-rate"), scoreSvc, cfg.MaxHitRateBodyBytes))

	return mux
}

func hitRateHandler(log logr.Logger, scoreSvc score.Service, maxBodyBytes int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		handleHitRate(log, scoreSvc, maxBodyBytes, w, r)
	}
}

func handleHitRate(
	log logr.Logger,
	scoreSvc score.Service,
	maxBodyBytes int64,
	w http.ResponseWriter,
	r *http.Request,
) {
	if !allowPostMethod(log, w, r) {
		return
	}
	log.V(1).Info("hit-rate request received", "remoteAddr", r.RemoteAddr)

	req, ok := decodeHitRateRequest(log, w, r, maxBodyBytes)
	if !ok || !validateHitRateRequest(log, w, req) {
		return
	}

	scores, ok := computeHitRateScores(log, scoreSvc, w, r, req)
	if !ok {
		return
	}

	respBody := buildHitRateResponse(scores)
	logHitRateCompletion(log, req, respBody)
	writeJSON(log, w, http.StatusOK, respBody)
}

func allowPostMethod(log logr.Logger, w http.ResponseWriter, r *http.Request) bool {
	if r.Method == http.MethodPost {
		return true
	}
	writeJSON(log, w, http.StatusMethodNotAllowed, errorResponse{
		Message: "method not allowed",
		Status:  http.StatusMethodNotAllowed,
	})
	return false
}

func decodeHitRateRequest(
	log logr.Logger,
	w http.ResponseWriter,
	r *http.Request,
	maxBodyBytes int64,
) (hitRateRequest, bool) {
	var req hitRateRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		switch {
		case errors.As(err, &maxErr):
			log.Info("hit-rate request rejected: body too large", "error", err.Error())
			writeJSON(log, w, http.StatusRequestEntityTooLarge, errorResponse{
				Message: "request body too large",
				Status:  http.StatusRequestEntityTooLarge,
			})
		case errors.Is(err, io.EOF):
			log.Info("hit-rate request rejected: empty body", "error", err.Error())
			writeJSON(log, w, http.StatusBadRequest, errorResponse{
				Message: "empty request body",
				Status:  http.StatusBadRequest,
			})
		default:
			log.Info("hit-rate request rejected: invalid JSON", "error", err.Error())
			writeJSON(log, w, http.StatusBadRequest, errorResponse{
				Message: "invalid JSON: " + err.Error(),
				Status:  http.StatusBadRequest,
			})
		}
		return hitRateRequest{}, false
	}
	return req, true
}

func validateHitRateRequest(log logr.Logger, w http.ResponseWriter, req hitRateRequest) bool {
	if len(req.ServerIP) == 0 {
		log.Info("hit-rate request rejected: server_ip is required")
		writeJSON(log, w, http.StatusBadRequest, errorResponse{
			Message: "server_ip is required",
			Status:  http.StatusBadRequest,
		})
		return false
	}
	if req.Body.BlockSize > 0 {
		return true
	}
	log.Info("hit-rate request rejected: body.block_size must be > 0",
		"blockSize", req.Body.BlockSize)
	writeJSON(log, w, http.StatusBadRequest, errorResponse{
		Message: "body.block_size must be > 0",
		Status:  http.StatusBadRequest,
	})
	return false
}

func computeHitRateScores(
	log logr.Logger,
	scoreSvc score.Service,
	w http.ResponseWriter,
	r *http.Request,
	req hitRateRequest,
) ([]apis.ServerScore, bool) {
	scores, err := scoreSvc.Compute(r.Context(), requestModelContext(req.Body), req.ServerIP, req.Body.TokenIDs)
	if err == nil {
		return scores, true
	}

	log.Error(err, "hit-rate request failed: compute",
		"servers", len(req.ServerIP), "tokenIDs", len(req.Body.TokenIDs))
	writeJSON(log, w, http.StatusInternalServerError, errorResponse{
		Message: "score evaluation failed",
		Status:  http.StatusInternalServerError,
	})
	return nil, false
}

func requestModelContext(body hitRateRequestBody) apis.ModelContext {
	// ModelName / LoraName stay zero until multi-model routing lands.
	// Ingest uses the same zero ModelContext, so L1 partitions align.
	return apis.ModelContext{
		BlockSize: body.BlockSize,
		CacheSalt: body.CacheSalt,
	}
}

func buildHitRateResponse(scores []apis.ServerScore) hitRateResponse {
	dtos := make([]serverScoreDTO, len(scores))
	for i, s := range scores {
		dtos[i] = serverScoreDTO{
			ServerIP:   s.ServerIP,
			L1HitRatio: s.L1HitRatio,
			L3HitRatio: s.L3HitRatio,
		}
	}
	return hitRateResponse{ServerScoreList: dtos}
}

func logHitRateCompletion(log logr.Logger, req hitRateRequest, respBody hitRateResponse) {
	respJSON, err := json.Marshal(respBody)
	if err != nil {
		log.Error(err, "hit-rate response marshal failed for logging",
			"servers", len(req.ServerIP), "tokenIDs", len(req.Body.TokenIDs))
		return
	}
	log.Info("hit-rate request completed",
		"servers", len(req.ServerIP),
		"tokenIDs", len(req.Body.TokenIDs),
		"responseBody", string(respJSON))
}

func writeJSON(log logr.Logger, w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Error(err, "write JSON response failed", "status", code)
	}
}
