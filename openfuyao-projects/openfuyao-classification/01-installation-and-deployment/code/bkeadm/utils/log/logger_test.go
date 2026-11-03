/*
   Copyright @ 2021 bocloud <fushaosong@beyondcent.com>.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package log

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

func TestDebug(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "debug message",
			format: false,
			args:   []any{"test debug message"},
		},
		{
			name:   "debug message with format",
			format: true,
			args:   []any{"test debug message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Debugf(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Debug(tt.args...)
				})
			}
		})
	}
}

func TestIsPreflightCommand(t *testing.T) {
	assert.True(t, isPreflightCommand([]string{"bke", "preflight", "init"}))
	assert.True(t, isPreflightCommand([]string{"bke", "--kubeconfig", "/tmp/k", "preflight", "cluster-create"}))
	assert.True(t, isPreflightCommand([]string{"bke", "--kubeconfig=/tmp/k", "preflight", "cluster-create"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init", "--file", "preflight-cluster.yaml"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init", "--domain", "preflight-prod"}))
}

func TestInfo(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "info message",
			format: false,
			args:   []any{"test info message"},
		},
		{
			name:   "info message with format",
			format: true,
			args:   []any{"test info message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Infof(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Info(tt.args...)
				})
			}
		})
	}
}

func TestWarn(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "warn message",
			format: false,
			args:   []any{"test warn message"},
		},
		{
			name:   "warn message with format",
			format: true,
			args:   []any{"test warn message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Warnf(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Warn(tt.args...)
				})
			}
		})
	}
}

func TestError(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "error message",
			format: false,
			args:   []any{"test error message"},
		},
		{
			name:   "error message with format",
			format: true,
			args:   []any{"test error message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Errorf(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Error(tt.args...)
				})
			}
		})
	}
}

func TestTrace(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "trace message",
			format: false,
			args:   []any{"test trace message"},
		},
		{
			name:   "trace message with format",
			format: true,
			args:   []any{"test trace message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Tracef(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Trace(tt.args...)
				})
			}
		})
	}
}

func TestFatal(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		args   []any
	}{
		{
			name:   "fatal message",
			format: false,
			args:   []any{"test fatal message"},
		},
		{
			name:   "fatal message with format",
			format: true,
			args:   []any{"test fatal message %d", 42},
		},
	}

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.Exit, func(code int) {
		assert.Equal(t, 1, code)
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					Fatalf(tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					Fatal(tt.args...)
				})
			}
		})
	}
}

func TestSteppedInfo(t *testing.T) {
	tests := []struct {
		name   string
		format bool
		step   string
		args   []any
	}{
		{
			name:   "stepped info message",
			format: false,
			step:   "step.1",
			args:   []any{"test stepped info message"},
		},
		{
			name:   "stepped info message with format",
			format: true,
			step:   "step.1",
			args:   []any{"test stepped info message %d", 42},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.format {
				assert.NotPanics(t, func() {
					SteppedInfof(tt.step, tt.args[0].(string), tt.args[1:]...)
				})
			} else {
				assert.NotPanics(t, func() {
					SteppedInfo(tt.step, tt.args...)
				})
			}
		})
	}
}
