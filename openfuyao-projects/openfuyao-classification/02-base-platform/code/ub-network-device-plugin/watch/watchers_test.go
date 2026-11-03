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

package watch

import (
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFiles(t *testing.T) {
	tests := []struct {
		name    string
		files   []string
		wantErr bool
	}{
		{
			name:    "成功创建监听器",
			files:   []string{"."},
			wantErr: false,
		},
		{
			name:    "监听多个文件",
			files:   []string{".", ".."},
			wantErr: false,
		},
		{
			name:    "空文件列表",
			files:   []string{},
			wantErr: false,
		},
		{
			name:    "不存在的文件",
			files:   []string{"/nonexistent/file"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			watcher, err := Files(tt.files...)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, watcher)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, watcher)
				if watcher != nil {
					watcher.Close()
				}
			}
		})
	}
}

func TestSignals(t *testing.T) {
	tests := []struct {
		name    string
		signals []os.Signal
	}{
		{
			name:    "监听中断信号",
			signals: []os.Signal{os.Interrupt},
		},
		{
			name:    "监听终止信号",
			signals: []os.Signal{syscall.SIGTERM},
		},
		{
			name:    "监听多个信号",
			signals: []os.Signal{os.Interrupt, syscall.SIGTERM},
		},
		{
			name:    "空信号列表",
			signals: []os.Signal{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signalCh := Signals(tt.signals...)
			assert.NotNil(t, signalCh)

			if len(tt.signals) > 0 && signalCh != nil {
				select {
				case sig := <-signalCh:
					if sig != nil {
						t.Errorf("不应立即收到信号: %v", sig)
					}
				default:
				}
			}
		})
	}
}

func TestFiles_AddAfterCreation(t *testing.T) {
	watcher, err := Files(".")
	assert.NoError(t, err)
	assert.NotNil(t, watcher)
	defer watcher.Close()

	err = watcher.Add("..")
	assert.NoError(t, err)
}

func TestSignals_ChannelBuffer(t *testing.T) {
	signalCh := Signals(os.Interrupt)

	if signalCh != nil {
		select {
		case sig := <-signalCh:
			if sig != nil {
				t.Errorf("不应立即收到信号: %v", sig)
			}
		default:
		}
	}
}

func TestFiles_Close(t *testing.T) {
	watcher, err := Files(".")
	assert.NoError(t, err)
	assert.NotNil(t, watcher)

	err = watcher.Close()
	assert.NoError(t, err)
}

func TestSignals_SendSignal(t *testing.T) {
	if testing.Short() {
		t.Skip("跳过短测试中的信号发送测试")
	}

	t.Skip("跳过信号发送测试，在不同平台上行为不一致")
}
