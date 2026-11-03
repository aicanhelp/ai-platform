/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package shell

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

type Shell interface {
	// Execute runs shell commands. When timeout > 0, the process is killed on deadline
	// (SIGINT then SIGKILL). When timeout <= 0, behavior matches the legacy unbounded Execute.
	Execute(ctx context.Context, timeout time.Duration, execCommands []string) ([]string, error)
}

type Task struct {
	Exec exec.Executor
}

// Execute Run the command
func (t *Task) Execute(ctx context.Context, timeout time.Duration, execCommands []string) ([]string, error) {
	var result []string
	if len(execCommands) < 1 {
		return result, errors.New("The execution instruction is null ")
	}
	script := strings.Join(execCommands, " ")
	ex := exec.Bound(t.Exec, ctx)
	var (
		s   string
		err error
	)
	if timeout > 0 {
		s, err = ex.ExecuteCommandWithTimeout(timeout, "/bin/sh", "-c", script)
	} else {
		s, err = ex.ExecuteCommandWithOutput("/bin/sh", "-c", script)
	}
	if err != nil {
		return result, errors.New(fmt.Sprintf("{code:%s, reason: %s}", err.Error(), s))
	}
	result = append(result, s)
	return result, nil
}
