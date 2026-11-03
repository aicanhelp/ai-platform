/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 ******************************************************************/

package phaseutil

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// RFC 2606 reserved names for unit tests (SCA-safe fixtures).
const (
	testNTPServer       = "ntp.invalid"
	testAgentHealthPort = "9443"
)

func TestRenderBKEAgentServiceContent(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					NTPServer:       testNTPServer,
					AgentHealthPort: testAgentHealthPort,
				},
			},
		},
	}
	raw := []byte("ExecStart=bkeagent --health-port= --ntpserver=\n")
	rendered := string(RenderBKEAgentServiceContent(cluster, raw))
	assert.Contains(t, rendered, "--ntpserver="+testNTPServer)
	assert.Contains(t, rendered, "--health-port="+testAgentHealthPort)
	assert.NotContains(t, rendered, "--log-level=")
}

func TestRenderBKEAgentServiceContent_AgentLogLevel(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					NTPServer:       testNTPServer,
					AgentHealthPort: testAgentHealthPort,
				},
				CustomExtra: map[string]string{
					AgentLogLevelCustomExtraKey: "DEBUG",
				},
			},
		},
	}
	raw := []byte("ExecStart=/usr/local/bin/bkeagent --kubeconfig=/etc/openFuyao/bkeagent/config --health-port= --ntpserver=\n")
	rendered := string(RenderBKEAgentServiceContent(cluster, raw))
	assert.Contains(t, rendered, "--ntpserver="+testNTPServer)
	assert.Contains(t, rendered, "--health-port="+testAgentHealthPort)
	assert.Contains(t, rendered, "--log-level=DEBUG")
}

func TestRenderBKEAgentServiceContent_AgentLogLevelAllowlist(t *testing.T) {
	raw := []byte("ExecStart=bkeagent --health-port= --ntpserver= --log-level=\n")

	t.Run("normalize case", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					CustomExtra: map[string]string{
						AgentLogLevelCustomExtraKey: "warning",
					},
				},
			},
		}
		rendered := string(RenderBKEAgentServiceContent(cluster, raw))
		assert.Contains(t, rendered, "--log-level=WARNING")
		assert.Equal(t, 1, strings.Count(rendered, "--log-level="))
	})

	t.Run("reject invalid", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					CustomExtra: map[string]string{
						AgentLogLevelCustomExtraKey: "INFO --evil",
					},
				},
			},
		}
		rendered := string(RenderBKEAgentServiceContent(cluster, raw))
		assert.NotContains(t, rendered, "--log-level=")
	})
}

func TestRenderBKEAgentServiceContent_AgentLogLevelPlaceholder(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				CustomExtra: map[string]string{
					AgentLogLevelCustomExtraKey: "TRACE",
				},
			},
		},
	}
	raw := []byte("ExecStart=bkeagent --health-port= --ntpserver= --log-level=\n")
	rendered := string(RenderBKEAgentServiceContent(cluster, raw))
	assert.Contains(t, rendered, "--log-level=TRACE")
	assert.Equal(t, 1, strings.Count(rendered, "--log-level="))
}

func TestRenderBKEAgentServiceContent_StripEmptyLogLevelPlaceholder(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
	raw := []byte("ExecStart=bkeagent --health-port= --ntpserver= --log-level=\n")
	rendered := string(RenderBKEAgentServiceContent(cluster, raw))
	assert.NotContains(t, rendered, "--log-level=")
}

func TestAppendFlagToExecStart_MultilineContinuation(t *testing.T) {
	content := "[Service]\n" +
		"ExecStart=/usr/local/bin/bkeagent \\\n" +
		"  --kubeconfig=/etc/openFuyao/bkeagent/config \\\n" +
		"  --health-port=9443\n" +
		"Restart=always\n"
	got := appendFlagToExecStart(content, "--log-level=INFO")

	assert.Contains(t, got, "ExecStart=/usr/local/bin/bkeagent \\\n")
	assert.Contains(t, got, "  --kubeconfig=/etc/openFuyao/bkeagent/config \\\n")
	assert.Contains(t, got, "  --health-port=9443 --log-level=INFO\n")
	assert.NotContains(t, got, "\\ --log-level=")
	assert.Equal(t, 1, strings.Count(got, "--log-level=INFO"))
}

func TestAppendFlagToExecStart_SingleLine(t *testing.T) {
	content := "ExecStart=/usr/local/bin/bkeagent --health-port=9443\n"
	got := appendFlagToExecStart(content, "--log-level=DEBUG")
	assert.Equal(t, "ExecStart=/usr/local/bin/bkeagent --health-port=9443 --log-level=DEBUG\n", got)
}

func TestAppendFlagToExecStart_NoExecStart(t *testing.T) {
	content := "[Service]\nRestart=always\n"
	got := appendFlagToExecStart(content, "--log-level=INFO")
	assert.Equal(t, content, got)
}

func TestRenderBKEAgentServiceContent_MultilineExecStart(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					NTPServer:       testNTPServer,
					AgentHealthPort: testAgentHealthPort,
				},
				CustomExtra: map[string]string{
					AgentLogLevelCustomExtraKey: "INFO",
				},
			},
		},
	}
	raw := []byte("[Service]\n" +
		"ExecStart=/usr/local/bin/bkeagent \\\n" +
		"  --kubeconfig=/etc/openFuyao/bkeagent/config \\\n" +
		"  --health-port= --ntpserver=\n")
	rendered := string(RenderBKEAgentServiceContent(cluster, raw))
	assert.Contains(t, rendered, "--ntpserver="+testNTPServer)
	assert.Contains(t, rendered, "--health-port="+testAgentHealthPort)
	assert.Contains(t, rendered, "  --health-port="+testAgentHealthPort+" --ntpserver="+testNTPServer+" --log-level=INFO\n")
	assert.NotContains(t, rendered, "\\ --log-level=")
}

func TestRenderBKEAgentServiceFile(t *testing.T) {
	if _, err := os.Stat("/bkeagent.service.tmpl"); err != nil {
		t.Skip("provider template not available in test environment")
	}

	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					NTPServer:       testNTPServer,
					AgentHealthPort: testAgentHealthPort,
				},
			},
		},
	}

	out := filepath.Join(t.TempDir(), "bkeagent.service")
	require.NoError(t, RenderBKEAgentServiceFile(cluster, out))
	data, err := os.ReadFile(out)
	require.NoError(t, err)
	content := string(data)
	assert.Contains(t, content, "--ntpserver="+testNTPServer)
	assert.Contains(t, content, "--health-port="+testAgentHealthPort)
}
