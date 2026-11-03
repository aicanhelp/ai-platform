/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 */

package agentssh

import (
	"testing"

	"github.com/stretchr/testify/assert"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
)

func TestBinaryArtifactName(t *testing.T) {
	cfg := bkeinit.BkeConfig{}
	assert.Equal(t, DefaultBKEAgentArtifact, BinaryArtifactName(cfg, ""))
	assert.Equal(t, "bkeagent-1.2.3-linux-{.arch}", BinaryArtifactName(cfg, "v1.2.3"))
}

func TestServiceCandidates(t *testing.T) {
	cfg := bkeinit.BkeConfig{}
	names := ServiceCandidates(cfg, "v2.0.0")
	assert.Equal(t, []string{"bkeagent-2.0.0.service", defaultServiceName}, names)

	cfg.CustomExtra = map[string]string{"bkeagent-service": "my-agent.service"}
	assert.Equal(t, []string{"my-agent.service"}, ServiceCandidates(cfg, "v2.0.0"))
}

func TestBinaryURLForArch(t *testing.T) {
	params := ArtifactParams{
		BaseURL:        "http://repo/files",
		BinaryArtifact: "bkeagent-latest-linux-{.arch}",
	}
	assert.Equal(t, "http://repo/files/bkeagent-latest-linux-arm64", BinaryURLForArch(params, "arm64"))
}
