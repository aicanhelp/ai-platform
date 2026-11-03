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
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
	customcollector "gopkg.openfuyao.cn/preflight/collectors/custom"
	"gopkg.openfuyao.cn/preflight/collectors/router"
)

const (
	secureFileModeMask = 18
	regularFileModeBit = 32768
	fileTypeModeMask   = 61440
	executableModeMask = 73
)

type customHostExecutor struct {
	remote customRemoteExecutor
}

type customRemoteExecutor interface {
	RunRead(context.Context, api.Subject, api.SSHReadCommand) (api.SSHReadResult, error)
	RunCustom(context.Context, api.Subject, string, []string) (api.SSHReadResult, error)
}

func customRoute(
	operation api.Operation,
	policy preflightconfig.OperationPolicy,
	remote customRemoteExecutor,
) (router.Route, error) {
	specs := make([]customcollector.Spec, 0, len(policy.CustomChecks))
	for _, check := range policy.CustomChecks {
		specs = append(specs, customSpec(operation, check))
	}
	collector, err := customcollector.New(&customHostExecutor{remote: remote}, specs)
	if err != nil {
		return router.Route{}, err
	}
	return router.Route{Match: func(_ api.Subject, key string) bool {
		return strings.HasPrefix(key, "custom."+string(operation)+".")
	}, Collector: collector}, nil
}

func customSpec(operation api.Operation, check preflightconfig.CustomCheck) customcollector.Spec {
	return customcollector.Spec{
		FactKey: customcollector.FactKey(operation, check.Key),
		Type:    check.Collector.Type, Path: check.Collector.Path,
		Args:         append([]string(nil), check.Collector.Args...),
		ResultType:   resultValueType(check.Collector.ResultType),
		ResultKind:   check.Collector.ResultType,
		PublicFields: append([]string(nil), check.Collector.PublicFields...),
	}
}

func resultValueType(resultType string) api.ValueType {
	switch resultType {
	case "bool":
		return api.ValueTypeBool
	case "integer":
		return api.ValueTypeInteger
	case "string":
		return api.ValueTypeString
	case "stringList", "integerList":
		return api.ValueTypeSet
	case "object":
		return api.ValueTypeObject
	default:
		return ""
	}
}

func (e *customHostExecutor) ReadFile(
	ctx context.Context,
	subject api.Subject,
	path string,
	limit int64,
) (customcollector.Output, error) {
	if !validCustomPath(path) {
		return customcollector.Output{}, fmt.Errorf("custom path must be a clean absolute path")
	}
	if e.isRemote(subject) {
		result, err := e.remote.RunRead(ctx, subject, api.SSHReadCommand{Name: "cat", Args: []string{"--", path}})
		if err != nil {
			return customcollector.Output{}, err
		}
		if result.ExitCode != 0 {
			return customcollector.Output{}, fmt.Errorf("read remote file failed: %s", strings.TrimSpace(result.Stderr))
		}
		return boundedOutput([]byte(result.Stdout), limit)
	}
	file, err := os.Open(path)
	if err != nil {
		return customcollector.Output{}, fmt.Errorf("open file: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(file, limit+1))
	closeErr := file.Close()
	if readErr != nil {
		return customcollector.Output{}, fmt.Errorf("read file: %w", readErr)
	}
	if closeErr != nil {
		return customcollector.Output{}, fmt.Errorf("close file: %w", closeErr)
	}
	return boundedOutput(data, limit)
}

func (e *customHostExecutor) Run(
	ctx context.Context,
	subject api.Subject,
	path string,
	args []string,
	limit int64,
) (customcollector.Output, error) {
	if !validCustomPath(path) {
		return customcollector.Output{}, fmt.Errorf("custom script path must be a clean absolute path")
	}
	if e.isRemote(subject) {
		if err := e.validateRemoteScript(ctx, subject, path); err != nil {
			return customcollector.Output{}, err
		}
		script, err := e.ReadFile(ctx, subject, path, limit)
		if err != nil {
			return customcollector.Output{}, fmt.Errorf("hash remote script: %w", err)
		}
		result, err := e.remote.RunCustom(ctx, subject, path, args)
		if err != nil {
			return customcollector.Output{}, err
		}
		if result.ExitCode != 0 {
			return customcollector.Output{}, fmt.Errorf("remote script exited %d", result.ExitCode)
		}
		output, err := boundedOutput([]byte(result.Stdout), limit)
		output.Digest = script.Digest
		return output, err
	}
	if err := validateLocalScript(path); err != nil {
		return customcollector.Output{}, err
	}
	script, err := e.ReadFile(ctx, subject, path, limit)
	if err != nil {
		return customcollector.Output{}, fmt.Errorf("hash script: %w", err)
	}
	command := exec.CommandContext(ctx, path, args...)
	command.Dir = "/"
	command.Env = []string{"PATH=/usr/sbin:/usr/bin:/sbin:/bin", "LANG=C", "LC_ALL=C"}
	var stdout bytes.Buffer
	command.Stdout = &boundedWriter{writer: &stdout, remaining: limit}
	command.Stderr = io.Discard
	if err := command.Run(); err != nil {
		return customcollector.Output{}, fmt.Errorf("script execution failed: %w", err)
	}
	return customcollector.Output{Data: stdout.Bytes(), Digest: script.Digest}, nil
}

func (e *customHostExecutor) isRemote(subject api.Subject) bool {
	return e.remote != nil && subject.ID != "bootstrap" && subject.ID != "cluster"
}

func (e *customHostExecutor) validateRemoteScript(ctx context.Context, subject api.Subject, path string) error {
	result, err := e.remote.RunRead(ctx, subject, api.SSHReadCommand{Name: "stat", Args: []string{"-c", "%f", path}})
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("stat remote script failed")
	}
	mode, err := strconv.ParseUint(strings.TrimSpace(result.Stdout), 16, 32)
	if err != nil {
		return fmt.Errorf("parse remote script mode: %w", err)
	}
	return validateScriptMode(uint32(mode))
}

func validateLocalScript(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat script: %w", err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("script must be a regular file")
	}
	mode := uint32(info.Mode().Perm()) | regularFileModeBit
	return validateScriptMode(mode)
}

func validateScriptMode(mode uint32) error {
	if mode&fileTypeModeMask != regularFileModeBit {
		return fmt.Errorf("script must be a regular file")
	}
	if mode&executableModeMask == 0 {
		return fmt.Errorf("script must be executable")
	}
	if mode&secureFileModeMask != 0 {
		return fmt.Errorf("script must not be group/world writable")
	}
	return nil
}

func boundedOutput(data []byte, limit int64) (customcollector.Output, error) {
	if int64(len(data)) > limit {
		return customcollector.Output{}, fmt.Errorf(
			"custom collector output exceeds the %d-byte limit; reduce the script output or file size", limit)
	}
	digest := sha256.Sum256(data)
	return customcollector.Output{Data: data, Digest: api.Digest(fmt.Sprintf("sha256:%x", digest[:]))}, nil
}

type boundedWriter struct {
	writer    io.Writer
	remaining int64
}

func (w *boundedWriter) Write(data []byte) (int, error) {
	if int64(len(data)) > w.remaining {
		return 0, fmt.Errorf(
			"custom collector output exceeds the %d-byte limit; reduce the script output", w.remaining)
	}
	written, err := w.writer.Write(data)
	w.remaining -= int64(written)
	return written, err
}

func validCustomPath(path string) bool {
	return filepath.IsAbs(path) && filepath.Clean(path) == path
}

var _ customcollector.Executor = (*customHostExecutor)(nil)
