/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package postprocess

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/scriptutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/testutils"
)

type fakeExecutor struct {
	lastCommand string
	lastArgs    []string
	output      string
	err         error
}

func (f *fakeExecutor) WithContext(_ context.Context) exec.Executor {
	return f
}

func (f *fakeExecutor) ExecuteCommand(command string, arg ...string) error {
	f.lastCommand = command
	f.lastArgs = arg
	return f.err
}

func (f *fakeExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	f.lastCommand = command
	f.lastArgs = arg
	return f.err
}

func (f *fakeExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	f.lastCommand = command
	f.lastArgs = arg
	return f.output, f.err
}

func (f *fakeExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	f.lastCommand = command
	f.lastArgs = arg
	return f.output, f.err
}

func (f *fakeExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return f.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (f *fakeExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	f.lastCommand = command
	f.lastArgs = append([]string{outfileArg}, arg...)
	return f.output, f.err
}

func (f *fakeExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	f.lastCommand = command
	f.lastArgs = append([]string{outfileArg}, arg...)
	return f.output, f.err
}

func (f *fakeExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	f.lastCommand = command
	f.lastArgs = arg
	return f.output, f.err
}

func (f *fakeExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	f.lastCommand = command
	f.lastArgs = arg
	return f.err
}

func TestValidateParams(t *testing.T) {
	p := &PostprocessPlugin{}

	t.Run("valid params", func(t *testing.T) {
		err := p.validateParams(map[string]string{
			"HTTP_REPO": "http://repo.example.com",
			"VERSION":   "v1.0.0",
			"ROLE":      "master",
		})
		require.NoError(t, err)
	})

	t.Run("invalid name", func(t *testing.T) {
		err := p.validateParams(map[string]string{
			"1BAD": "ok",
		})
		require.Error(t, err)
	})

	t.Run("invalid value", func(t *testing.T) {
		err := p.validateParams(map[string]string{
			"HTTP_REPO": "http://repo.example.com?bad=$",
		})
		require.Error(t, err)
	})

	t.Run("too long", func(t *testing.T) {
		err := p.validateParams(map[string]string{
			"LONG": strings.Repeat("a", 4097),
		})
		require.Error(t, err)
	})
}

func TestRenderScriptWithParams(t *testing.T) {
	p := &PostprocessPlugin{}
	script := "node=${NODE_IP}, role=${ROLE}, missing=${MISSING}"
	out, err := p.renderScriptWithParams(script, scriptutil.ScriptConfig{
		ScriptName: "test.sh",
		Params: map[string]string{
			"ROLE": "master",
		},
	}, "10.0.0.1")
	require.NoError(t, err)
	require.Equal(t, "node=10.0.0.1, role=master, missing=${MISSING}", out)
}

func TestParseConfig(t *testing.T) {
	p := &PostprocessPlugin{}

	t.Run("missing config.json", func(t *testing.T) {
		_, err := p.parseConfig(&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "cm1", Namespace: "user-system"},
			Data:       map[string]string{},
		})
		require.Error(t, err)
	})
}

func TestLoadConfigPriority(t *testing.T) {
	nodeIP := "10.0.0.1"
	globalCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-all-config",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"global.sh","order":1,"params":{"ROLE":"master"}}]}`,
		},
	}
	batchMapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `{"10.0.0.1":"001"}`,
		},
	}
	batchCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-batch-001",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"batch.sh","order":1,"params":{"ROLE":"worker"}}]}`,
		},
	}

	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, globalCM, batchMapping, batchCM)
	p := &PostprocessPlugin{k8sClient: fakeClient}

	cfg, err := p.loadConfig(nodeIP)
	require.NoError(t, err)
	require.Equal(t, "global.sh", cfg.Scripts[0].ScriptName)
}

func TestLoadConfigBatchAndNode(t *testing.T) {
	nodeIP := "10.0.0.1"
	batchMapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `{"10.0.0.1":"002"}`,
		},
	}
	batchCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-batch-002",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"batch.sh","order":1}]}`,
		},
	}
	nodeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-node-10.0.0.1",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"node.sh","order":1}]}`,
		},
	}

	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, batchMapping, batchCM, nodeCM)
	p := &PostprocessPlugin{k8sClient: fakeClient}

	cfg, err := p.loadConfig(nodeIP)
	require.NoError(t, err)
	require.Equal(t, "batch.sh", cfg.Scripts[0].ScriptName)
}

func TestLoadConfigNodeHit(t *testing.T) {
	nodeIP := "10.0.0.2"
	nodeCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "postprocess-config-node-10.0.0.2",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"node.sh","order":1}]}`,
		},
	}

	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, nodeCM)
	p := &PostprocessPlugin{k8sClient: fakeClient}

	cfg, err := p.loadConfig(nodeIP)
	require.NoError(t, err)
	require.Equal(t, "node.sh", cfg.Scripts[0].ScriptName)
}

func TestLoadConfigMiss(t *testing.T) {
	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil)
	p := &PostprocessPlugin{k8sClient: fakeClient}

	_, err := p.loadConfig("10.0.0.3")
	require.Error(t, err)
}

func TestGetAllScriptsAndScriptExists(t *testing.T) {
	cm1 := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "script-a.sh",
			Namespace: "user-system",
			Labels:    map[string]string{"bke.postprocess.script": "true"},
		},
	}
	cm2 := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "script-b.sh",
			Namespace: "user-system",
			Labels:    map[string]string{"bke.postprocess.script": "true"},
		},
	}

	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, cm1, cm2)
	p := &PostprocessPlugin{k8sClient: fakeClient}

	scripts, err := p.getAllScripts()
	require.NoError(t, err)
	require.Len(t, scripts, 2)
	require.True(t, p.scriptExists(scripts, "script-a.sh"))
	require.True(t, p.scriptExists(scripts, "script-b.sh"))
	require.False(t, p.scriptExists(scripts, "script-c.sh"))
}

func TestExecuteRenderedScript(t *testing.T) {
	exec := &fakeExecutor{output: "ok"}
	p := &PostprocessPlugin{exec: exec}
	out, err := p.executeRenderedScript("/tmp/test.sh")
	require.NoError(t, err)
	require.Equal(t, "ok", out)
	require.Equal(t, "/bin/sh", exec.lastCommand)
	require.Equal(t, []string{"/tmp/test.sh"}, exec.lastArgs)
}

func TestExecuteRenderedScriptError(t *testing.T) {
	exec := &fakeExecutor{output: "boom", err: context.DeadlineExceeded}
	p := &PostprocessPlugin{exec: exec}
	_, err := p.executeRenderedScript("/tmp/test.sh")
	require.Error(t, err)
}

func TestWriteRenderedScriptToDisk(t *testing.T) {
	p := &PostprocessPlugin{}
	path, err := p.writeRenderedScriptToDisk("test.sh", "10.0.0.1", "echo ok")
	if err != nil {
		if os.IsPermission(err) || strings.Contains(strings.ToLower(err.Error()), "permission") {
			t.Skipf("skip write test due to permission: %v", err)
		}
		require.NoError(t, err)
	}
	_ = os.Remove(path)
}

func TestSanitizeFileNameAndPreview(t *testing.T) {
	sanitized := scriptutil.SanitizeFileName("10.0.0.1:22 /test\\path")
	require.Equal(t, "10.0.0.1_22__test_path", sanitized)
	require.Equal(t, "abc", scriptutil.PreviewScript("abc", 10))
	require.Equal(t, "ab ... (truncated)", scriptutil.PreviewScript("abcdef", 2))
}

func TestExecuteScriptParamValidationSkip(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "install-test.sh",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"install-test.sh": "#!/bin/sh\necho ok\n",
		},
	}
	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, cm)
	p := &PostprocessPlugin{k8sClient: fakeClient, exec: &fakeExecutor{}}

	out, err := p.executeScript(scriptutil.ScriptConfig{
		ScriptName: "install-test.sh",
		Order:      1,
		Params: map[string]string{
			"BAD": "value$",
		},
	}, "10.0.0.1")
	require.NoError(t, err)
	require.Contains(t, out, "skipped")
}

func TestExecuteScriptConfigMapMissing(t *testing.T) {
	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil)
	p := &PostprocessPlugin{k8sClient: fakeClient, exec: &fakeExecutor{}}

	_, err := p.executeScript(scriptutil.ScriptConfig{
		ScriptName: "missing.sh",
		Order:      1,
		Params:     map[string]string{},
	}, "10.0.0.1")
	require.Error(t, err)
}

func TestExecuteScriptContentMissing(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "empty.sh",
			Namespace: "user-system",
		},
		Data: map[string]string{},
	}
	fakeClient, _ := testutils.TestGetRuntimeFakeClient(nil, cm)
	p := &PostprocessPlugin{k8sClient: fakeClient, exec: &fakeExecutor{}}

	_, err := p.executeScript(scriptutil.ScriptConfig{
		ScriptName: "empty.sh",
		Order:      1,
		Params:     map[string]string{},
	}, "10.0.0.1")
	require.Error(t, err)
}
