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

package util

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"console-service/pkg/constant"
	"console-service/pkg/server/config"
)

// TestClearByte checks if all bytes in the slice are set to zero.
func TestClearByte(t *testing.T) {
	// Create a byte slice with non-zero values.
	data := []byte{1, 2, 3, 4, 5}

	// Call ClearByte to zero out the slice.
	ClearByte(data)

	// Check each byte to ensure it's been set to zero.
	for i, b := range data {
		if b != 0 {
			t.Errorf("byte at index %d is not zero, got %d", i, b)
		}
	}

	var emptyData []byte
	ClearByte(emptyData) // This should not cause any issue or panic.
}

// MockFile implement multipart.File interface for testing
type MockFile struct {
	Data   []byte
	Offset int
	Err    error // 可以被设置为模拟读取错误
}

func (m *MockFile) Read(p []byte) (int, error) {
	if m.Err != nil {
		return 0, m.Err
	}
	if m.Offset >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[m.Offset:])
	m.Offset += n
	return n, nil
}

func (m *MockFile) Close() error {
	return nil
}

func (m *MockFile) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func (m *MockFile) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("negative offset")
	}
	if int(off) >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

const perm = 0755

func createTempConfigFile(t *testing.T) {
	tempDir := "./temp"
	tempFileServerName := fmt.Sprintf("%s/%s", tempDir, serverName)
	err := os.MkdirAll("./temp", perm)
	if err != nil {
		t.Fatalf("Failed to create dir %s: %v", tempDir, err)
	}
	f, err := os.OpenFile(tempFileServerName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
	if err != nil {
		t.Fatalf("Failed top create file %s: %v", tempFileServerName, err)
	}
	_, err = f.WriteString("example-server")
	if err != nil {
		t.Fatalf("Failed to write file %s: %v", tempFileServerName, err)
	}
	err = f.Close()
	if err != nil {
		t.Fatalf("Failed to close file %s: %v", tempFileServerName, err)
	}

	patchConfigDir := gomonkey.ApplyGlobalVar(&configDir, "./temp")

	t.Cleanup(func() {
		err := os.RemoveAll(tempDir)
		if err != nil {
			t.Fatalf("Failed to remove dir %s: %v", tempDir, err)
		}
		patchConfigDir.Reset()
	})
}

func TestGetConsoleServiceConfig(t *testing.T) {
	client := fake.NewSimpleClientset()

	t.Run("ConfigMapNitExistingAndReadingPodFile", func(t *testing.T) {
		createTempConfigFile(t)

		got, err := GetConsoleServiceConfig(client)
		if err != nil {
			t.Errorf("Should return nil error, but got %v", err)
		}
		want := &config.ConsoleServiceConfig{
			ServerName: "example-server",
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("GetConsoleServiceConfig() = %v, expected %v", got, want)
		}
	})

	testConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.ConsoleServiceConfigmap,
			Namespace: constant.ConsoleServiceDefaultNamespace,
		},
		Data: map[string]string{
			serverName: "example-server",
		},
	}
	_, err := client.CoreV1().ConfigMaps(constant.ConsoleServiceDefaultNamespace).Create(
		context.TODO(), testConfigMap, metav1.CreateOptions{})
	if err != nil {
		return
	}
	t.Run("ConfigMapAndPodFileExisting", func(t *testing.T) {
		got, err := GetConsoleServiceConfig(client)
		if err != nil {
			t.Errorf("Should return error, but got nil")
		}
		want := &config.ConsoleServiceConfig{
			ServerName: "example-server",
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("GetConsoleServiceConfig() = %v, expected %v", got, want)
		}
	})
}
