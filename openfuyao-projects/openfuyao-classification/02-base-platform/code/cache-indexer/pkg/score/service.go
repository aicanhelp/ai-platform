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

// Package score implements query scoring through Builder, registry, and indexes.
// HTTP handlers should call this service instead of reaching into dependencies.
package score

import (
	"context"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/blockkey"
	indexl1 "gitcode.com/openFuyao/cache-indexer/pkg/index/l1"
	indexl3 "gitcode.com/openFuyao/cache-indexer/pkg/index/l3"
)

// EndpointResolver maps vLLM Pod IPs to Mooncake transport_endpoint values.
// discovery.Registry implements it without making score import discovery.
type EndpointResolver interface {
	ResolveTransportEndpoint(serverIP string) (apis.TransportEndpoint, bool)
}

// Service is the single entry point used by the HTTP handler.
type Service interface {
	// Compute computes L1/L3 prefix hit ratios for the given servers.
	Compute(
		ctx context.Context,
		modelCtx apis.ModelContext,
		serverIPs []string,
		tokenIDs []int32,
	) ([]apis.ServerScore, error)
}

type service struct {
	log      logr.Logger
	builder  blockkey.Builder
	l1       indexl1.Indexer
	l3       indexl3.Indexer
	resolver EndpointResolver
}

// New returns a service that tolerates missing cache dependencies.
// Nil builder/l1 disable L1; nil l3/resolver makes L3 ratio 0.
func New(
	log logr.Logger,
	builder blockkey.Builder,
	l1 indexl1.Indexer,
	l3 indexl3.Indexer,
	resolver EndpointResolver,
) Service {
	return &service{log: log, builder: builder, l1: l1, l3: l3, resolver: resolver}
}

// Compute builds BlockHash values once, then checks L1/L3 per server.
// Scores are returned in input order.
func (s *service) Compute(
	_ context.Context,
	modelCtx apis.ModelContext,
	serverIPs []string,
	tokenIDs []int32,
) ([]apis.ServerScore, error) {
	out := initServerScores(serverIPs)
	keys, ready, err := s.buildRequestKeys(modelCtx, tokenIDs, len(serverIPs))
	if err != nil {
		return nil, err
	}
	if !ready {
		return out, nil
	}
	for i := range out {
		out[i] = s.computeServerScore(modelCtx, out[i].ServerIP, keys)
	}
	s.log.V(1).Info("score: computed",
		"servers", len(serverIPs), "totalBlocks", len(keys))
	return out, nil
}

func initServerScores(serverIPs []string) []apis.ServerScore {
	out := make([]apis.ServerScore, len(serverIPs))
	for i, ip := range serverIPs {
		out[i] = apis.ServerScore{ServerIP: ip}
	}
	return out
}

func (s *service) buildRequestKeys(
	modelCtx apis.ModelContext,
	tokenIDs []int32,
	serverCount int,
) ([]apis.BlockHash, bool, error) {
	if s.builder == nil {
		s.log.V(1).Info("score: builder not wired; returning zero scores",
			"servers", serverCount)
		return nil, false, nil
	}

	keys, err := s.builder.BuildFromRequest(modelCtx, tokenIDs)
	if err != nil {
		return nil, false, err
	}
	if len(keys) == 0 {
		s.log.V(1).Info("score: token_ids shorter than one block",
			"tokenIDs", len(tokenIDs), "blockSize", modelCtx.BlockSize)
		return nil, false, nil
	}

	s.log.V(1).Info("score: computed block keys",
		"total", len(keys), "firstKey", string(keys[0]))
	return keys, true, nil
}

func (s *service) computeServerScore(
	modelCtx apis.ModelContext,
	serverIP string,
	keys []apis.BlockHash,
) apis.ServerScore {
	result := apis.ServerScore{
		ServerIP:    serverIP,
		TotalBlocks: len(keys),
	}
	s.applyL1Score(&result, modelCtx, serverIP, keys)
	s.applyL3Score(&result, modelCtx, serverIP, keys)
	s.log.V(1).Info("hit-rate per-server result",
		"server_ip", serverIP,
		"l1_hit_rate", result.L1HitRatio,
		"l3_hit_rate", result.L3HitRatio)
	return result
}

func (s *service) applyL1Score(
	result *apis.ServerScore,
	modelCtx apis.ModelContext,
	serverIP string,
	keys []apis.BlockHash,
) {
	if s.l1 == nil {
		return
	}

	res, err := s.l1.MatchPrefix(modelCtx, serverIP, keys)
	if err != nil {
		s.log.Error(err, "L1 MatchPrefix failed", "server_ip", serverIP)
		return
	}
	if res == nil {
		return
	}

	result.L1MatchedBlocks = res.MatchedBlocks
	result.L1HitRatio = float64(res.MatchedBlocks) / float64(result.TotalBlocks)
}

func (s *service) applyL3Score(
	result *apis.ServerScore,
	modelCtx apis.ModelContext,
	serverIP string,
	keys []apis.BlockHash,
) {
	if s.l3 == nil || s.resolver == nil {
		return
	}

	ep, ok := s.resolver.ResolveTransportEndpoint(serverIP)
	if !ok {
		s.log.V(1).Info("L3 endpoint not resolved; L3 ratio = 0",
			"server_ip", serverIP)
		return
	}

	res, err := s.l3.MatchPrefix(l3LookupContext(modelCtx), ep, keys)
	if err != nil {
		s.log.Error(err, "L3 MatchPrefix failed",
			"server_ip", serverIP, "endpoint", ep)
		return
	}
	if res == nil {
		return
	}

	result.L3MatchedBlocks = res.MatchedBlocks
	result.L3HitRatio = float64(res.MatchedBlocks) / float64(result.TotalBlocks)
}

func l3LookupContext(modelCtx apis.ModelContext) apis.ModelContext {
	return apis.ModelContext{
		ModelName: modelCtx.ModelName,
		LoraName:  modelCtx.LoraName,
	}
}
