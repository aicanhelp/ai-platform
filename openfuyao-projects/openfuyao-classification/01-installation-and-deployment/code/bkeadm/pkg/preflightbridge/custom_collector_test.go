/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
	customcollector "gopkg.openfuyao.cn/preflight/collectors/custom"
)

func TestCustomHostExecutorReadsLocalFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "hosts")
	require.NoError(t, os.WriteFile(path, []byte("127.0.0.1 localhost\n"), 0o600))
	executor := &customHostExecutor{}
	output, err := executor.ReadFile(context.Background(), api.Subject{ID: "bootstrap"}, path, 1024)
	require.NoError(t, err)
	assert.Equal(t, "127.0.0.1 localhost\n", string(output.Data))
	assert.Contains(t, output.Digest.String(), "sha256:")
	_, err = executor.ReadFile(context.Background(), api.Subject{ID: "bootstrap"}, "relative", 1024)
	assert.ErrorContains(t, err, "absolute")
}

func TestCustomHostExecutorRunsSecureLocalScript(t *testing.T) {
	path := filepath.Join(t.TempDir(), "check")
	script := "#!/bin/sh\nprintf '%s' " +
		"'{\"apiVersion\":\"preflight.openfuyao.io/fact/v1\",\"available\":true," +
		"\"type\":\"bool\",\"value\":true}'\n"
	require.NoError(t, os.WriteFile(path, []byte(script), 0o700))
	executor := &customHostExecutor{}
	output, err := executor.Run(
		context.Background(), api.Subject{ID: "bootstrap"}, path, nil, customcollector.MaxOutputBytes,
	)
	require.NoError(t, err)
	assert.Contains(t, string(output.Data), `"value":true`)

	require.NoError(t, os.Chmod(path, 0o722))
	_, err = executor.Run(context.Background(), api.Subject{ID: "bootstrap"}, path, nil, 1024)
	assert.ErrorContains(t, err, "group/world writable")
}

func TestBoundedOutputAndScriptModes(t *testing.T) {
	_, err := boundedOutput([]byte("too large"), 2)
	assert.ErrorContains(t, err, "exceeds")
	assert.ErrorContains(t, validateScriptMode(0o100600), "executable")
	assert.ErrorContains(t, validateScriptMode(0o040700), "regular")
	assert.NoError(t, validateScriptMode(0o100700))
}

func TestCustomSpecMapsPublicContract(t *testing.T) {
	check := preflightconfig.CustomCheck{Key: "health", Collector: preflightconfig.CustomCollector{
		Type: "exec", Path: "/opt/check", ResultType: "integerList", PublicFields: []string{"value"},
	}}
	spec := customSpec(api.OperationClusterCreate, check)
	assert.Equal(t, api.ValueTypeSet, spec.ResultType)
	assert.Equal(t, "integerList", spec.ResultKind)
	assert.Equal(t, "custom.cluster-create.health", spec.FactKey)
}

func TestResultValueType(t *testing.T) {
	tests := map[string]api.ValueType{
		"bool": api.ValueTypeBool, "integer": api.ValueTypeInteger,
		"string": api.ValueTypeString, "stringList": api.ValueTypeSet,
		"integerList": api.ValueTypeSet, "object": api.ValueTypeObject,
		"unknown": "",
	}
	for input, expected := range tests {
		assert.Equal(t, expected, resultValueType(input), input)
	}
}

type fakeCustomRemote struct {
	commands   []api.SSHReadCommand
	custom     bool
	statExit   int
	statStdout string
	catExit    int
	catStderr  string
	customExit int
	readErr    error
}

func (f *fakeCustomRemote) RunRead(
	_ context.Context,
	_ api.Subject,
	command api.SSHReadCommand,
) (api.SSHReadResult, error) {
	f.commands = append(f.commands, command)
	if f.readErr != nil {
		return api.SSHReadResult{}, f.readErr
	}
	if command.Name == "stat" {
		stdout := f.statStdout
		if stdout == "" {
			stdout = "81c0\n"
		}
		return api.SSHReadResult{Stdout: stdout, ExitCode: f.statExit}, nil
	}
	return api.SSHReadResult{Stdout: "script bytes", ExitCode: f.catExit, Stderr: f.catStderr}, nil
}

func (f *fakeCustomRemote) RunCustom(
	_ context.Context,
	_ api.Subject,
	_ string,
	_ []string,
) (api.SSHReadResult, error) {
	f.custom = true
	return api.SSHReadResult{Stdout: `{"apiVersion":"preflight.openfuyao.io/fact/v1"}`, ExitCode: f.customExit}, nil
}

func TestCustomHostExecutorUsesRemoteWithoutUploading(t *testing.T) {
	remote := &fakeCustomRemote{}
	executor := &customHostExecutor{remote: remote}
	output, err := executor.Run(
		context.Background(), api.Subject{ID: "node-1"}, "/opt/check", []string{"--json"}, 1024,
	)
	require.NoError(t, err)
	assert.True(t, remote.custom)
	assert.Equal(t, "sha256:", output.Digest.String()[:7])
	require.Len(t, remote.commands, 2)
	assert.Equal(t, "stat", remote.commands[0].Name)
	assert.Equal(t, "cat", remote.commands[1].Name)
}

// TestCustomHostExecutorRemoteErrorPaths freezes remote failure handling:
// non-zero stat exit, unreadable mode output, non-zero script exit and
// remote read errors each surface as actionable errors.
func TestCustomHostExecutorRemoteErrorPaths(t *testing.T) {
	subject := api.Subject{ID: "node-1"}

	statFails := &fakeCustomRemote{}
	statFails.statExit = 1
	executor := &customHostExecutor{remote: statFails}
	_, err := executor.Run(context.Background(), subject, "/opt/check", nil, 1024)
	assert.ErrorContains(t, err, "stat remote script")

	badMode := &fakeCustomRemote{}
	badMode.statStdout = "not-hex"
	executor = &customHostExecutor{remote: badMode}
	_, err = executor.Run(context.Background(), subject, "/opt/check", nil, 1024)
	assert.ErrorContains(t, err, "parse remote script mode")

	insecureMode := &fakeCustomRemote{}
	insecureMode.statStdout = "81e2" // 0o100762: executable but group-writable
	executor = &customHostExecutor{remote: insecureMode}
	_, err = executor.Run(context.Background(), subject, "/opt/check", nil, 1024)
	assert.ErrorContains(t, err, "group/world writable")

	scriptFails := &fakeCustomRemote{}
	scriptFails.customExit = 3
	executor = &customHostExecutor{remote: scriptFails}
	_, err = executor.Run(context.Background(), subject, "/opt/check", nil, 1024)
	assert.ErrorContains(t, err, "exited 3")

	readFails := &fakeCustomRemote{}
	readFails.readErr = assert.AnError
	executor = &customHostExecutor{remote: readFails}
	_, err = executor.ReadFile(context.Background(), subject, "/etc/hosts", 1024)
	assert.ErrorIs(t, err, assert.AnError)

	catFails := &fakeCustomRemote{}
	catFails.catExit = 1
	catFails.catStderr = "permission denied"
	executor = &customHostExecutor{remote: catFails}
	_, err = executor.ReadFile(context.Background(), subject, "/etc/hosts", 1024)
	assert.ErrorContains(t, err, "permission denied")
}

// TestCustomHostExecutorLocalErrorPaths freezes local failure handling:
// missing files, non-regular files and oversized output.
func TestCustomHostExecutorLocalErrorPaths(t *testing.T) {
	executor := &customHostExecutor{}
	_, err := executor.ReadFile(context.Background(), api.Subject{ID: "bootstrap"}, "/missing/file", 1024)
	assert.ErrorContains(t, err, "open file")

	_, err = executor.Run(context.Background(), api.Subject{ID: "bootstrap"}, "/missing/script", nil, 1024)
	assert.ErrorContains(t, err, "stat script")

	dir := t.TempDir()
	_, err = executor.Run(context.Background(), api.Subject{ID: "bootstrap"}, dir, nil, 1024)
	assert.ErrorContains(t, err, "regular file")

	path := filepath.Join(dir, "check")
	require.NoError(t, os.WriteFile(path, []byte("#!/bin/sh\ntrue\n"), 0o700))
	_, err = executor.Run(context.Background(), api.Subject{ID: "bootstrap"}, path, nil, 2)
	assert.ErrorContains(t, err, "exceeds")

	_, err = executor.Run(context.Background(), api.Subject{ID: "bootstrap"}, "relative/path", nil, 1024)
	assert.ErrorContains(t, err, "absolute")
}

// TestBoundedWriterRejectsOversizedWrites freezes the streaming bound.
func TestBoundedWriterRejectsOversizedWrites(t *testing.T) {
	var sink bytes.Buffer
	writer := &boundedWriter{writer: &sink, remaining: 4}
	n, err := writer.Write([]byte("ab"))
	require.NoError(t, err)
	assert.Equal(t, 2, n)
	_, err = writer.Write([]byte("abcdef"))
	assert.ErrorContains(t, err, "exceeds")
	assert.Equal(t, "ab", sink.String())
}

// TestValidCustomPath freezes the clean-absolute-path rule.
func TestValidCustomPath(t *testing.T) {
	assert.True(t, validCustomPath("/opt/check"))
	assert.False(t, validCustomPath("opt/check"))
	assert.False(t, validCustomPath("/opt/../check"))
}

// TestCustomRouteMatchesOperationKeys freezes the router key namespace.
func TestCustomRouteMatchesOperationKeys(t *testing.T) {
	route, err := customRoute(api.OperationInit, preflightconfig.OperationPolicy{}, nil)
	require.NoError(t, err)
	assert.True(t, route.Match(api.Subject{}, "custom.init.hosts"))
	assert.False(t, route.Match(api.Subject{}, "custom.cluster-create.hosts"))
}
