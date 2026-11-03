/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package k8s

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	testUnixRoot = "/tmp"
	testAbsPath  = "c:/tmp/test"
)

func newScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	return scheme
}

type mockExecutor struct {
	output string
	err    error
}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return m.err
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return m.err
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return m.err
}

func TestExecuteInvalidCommandFormat(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"invalid:format"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Command format error")
	assert.Empty(t, result)
}

func TestExecuteUnsupportedResourceType(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"unsupported:ns/name:ro:/tmp/test"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Not with configMap or Secret resources")
	assert.Empty(t, result)
}

func TestExecuteUnsupportedOperation(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:ns/name:invalid:/tmp/test"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Unsupported operation types")
	assert.Empty(t, result)
}

func TestExecuteInvalidNamespaceKey(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:invalid@key:ro:/tmp/test"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestExecuteDefaultNamespace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
			"key2": "value2",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	testFile := path.Join(testUnixRoot, "config.json")
	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:test-config:ro:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestExecuteConfigMapRead(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
			"key2": "value2",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	testFile := path.Join(testUnixRoot, "config.json")
	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:default/test-config:ro:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestExecuteSecretRead(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"username": []byte("admin"),
			"password": []byte("secret"),
		},
		Type: corev1.SecretTypeOpaque,
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret).Build()

	testFile := path.Join(testUnixRoot, "secret.json")
	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"secret:default/test-secret:ro:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestExecuteRelativePathError(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:ns/name:ro:relative/path"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
	assert.Empty(t, result)
}

func TestExecuteResourceNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	testFile := path.Join(testUnixRoot, "test.json")
	commands := []string{"configmap:default/nonexistent:ro:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Unable to get the specified resource")
	assert.Empty(t, result)
}

func TestExecuteConfigMapWrite(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("test content for configmap"), nil
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:default/test-config:rw:/tmp/test.json"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestExecuteSecretWrite(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("secret data"), nil
	})

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
		Data: map[string][]byte{},
		Type: corev1.SecretTypeOpaque,
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"secret:default/test-secret:rw:/tmp/secret.json"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestExecuteWriteToNonExistentFile(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:ns/name:rw:/nonexistent/path/file.txt"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestExecuteRxConfigMap(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	testFile := filepath.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	commands := []string{"configmap:default/test-script:rx:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, result)
}

func TestExecuteRxSecret(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"script.sh": []byte("echo hello"),
		},
		Type: corev1.SecretTypeOpaque,
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	testFile := filepath.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	commands := []string{"secret:default/test-script:rx:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, result)
}

func TestExecuteRxWithoutAbsPath(t *testing.T) {
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	commands := []string{"configmap:default/test-script:rx:relative/path"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, result)
}

func TestExecuteRxWithCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "",
		err:    assert.AnError,
	}

	testFile := filepath.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	commands := []string{"configmap:default/test-script:rx:" + testFile}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestExecuteMultipleCommands(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script1.sh": "echo hello",
			"script2.sh": "echo world",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "output",
		err:    nil,
	}

	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	commands := []string{"configmap:default/test-script:rx:" + filepath.Join(testUnixRoot, "output.txt")}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Equal(t, []string{"output", "output"}, result)
}

func TestHandleReadWriteCreateNew(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("new content"), nil
	})

	client := fake.NewClientBuilder().WithScheme(newScheme()).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"configmap:default/new-config:rw:/tmp/newfile.json"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)

	obj := &corev1.ConfigMap{}
	err = client.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "new-config"}, obj)
	assert.NoError(t, err)
	assert.Equal(t, "new content", obj.Data["content"])
}

func TestHandleReadWriteSecretCreateNew(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("new secret content"), nil
	})

	client := fake.NewClientBuilder().WithScheme(newScheme()).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	commands := []string{"secret:default/new-secret:rw:/tmp/newsecret.json"}
	result, err := task.Execute(context.Background(), 0, commands)
	assert.NoError(t, err)
	assert.Empty(t, result)

	obj := &corev1.Secret{}
	err = client.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "new-secret"}, obj)
	assert.NoError(t, err)
}

func TestGetResourceObjectConfigMap(t *testing.T) {
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}

	obj, err := task.getResourceObject("configmap", "default", "test-config")
	assert.NoError(t, err)
	assert.NotNil(t, obj)
}

func TestGetResourceObjectSecret(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"key1": []byte("value1"),
		},
		Type: corev1.SecretTypeOpaque,
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}

	obj, err := task.getResourceObject("secret", "default", "test-secret")
	assert.NoError(t, err)
	assert.NotNil(t, obj)
}

func TestGetResourceObjectNotFound(t *testing.T) {
	client := fake.NewClientBuilder().WithScheme(newScheme()).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}

	obj, err := task.getResourceObject("configmap", "default", "nonexistent")
	assert.Error(t, err)
	assert.Nil(t, obj)
	assert.Contains(t, err.Error(), "Unable to get the specified resource")
}

func TestWriteResourceToFileConfigMap(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "config.json")
	configMap := &corev1.ConfigMap{
		Data: map[string]string{
			"key1": "value1\n",
			"key2": "value2\n",
		},
	}

	err := writeResourceToFile(configMap, tmpFile)
	assert.NoError(t, err)

}

func TestWriteResourceToFileSecret(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "secret.bin")
	secret := &corev1.Secret{
		Data: map[string][]byte{
			"key1": []byte{0x01, 0x02, 0x03},
			"key2": []byte{0x04, 0x05, 0x06},
		},
		Type: corev1.SecretTypeOpaque,
	}

	err := writeResourceToFile(secret, tmpFile)
	assert.NoError(t, err)
}

func TestWriteResourceToFileUnsupportedType(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "test.bin")

	err := writeResourceToFile(&corev1.Pod{}, tmpFile)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported resource type")
}

func TestWriteResourceToFileOpenError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return nil, assert.AnError
	})

	configMap := &corev1.ConfigMap{
		Data: map[string]string{
			"key1": "value1",
		},
	}

	err := writeResourceToFile(configMap, "/invalid/path/file")
	assert.Error(t, err)
}

func TestExtractScriptFromResourceConfigMap(t *testing.T) {
	configMap := &corev1.ConfigMap{
		Data: map[string]string{
			"script1.sh": "echo hello",
			"script2.sh": "echo world",
		},
	}

	extractScriptFromResource(configMap)
}

func TestExtractScriptFromResourceSecret(t *testing.T) {
	secret := &corev1.Secret{
		Data: map[string][]byte{
			"script1.sh": []byte("echo hello"),
			"script2.sh": []byte("echo world"),
		},
		Type: corev1.SecretTypeOpaque,
	}

	extractScriptFromResource(secret)
}

func TestExtractScriptFromResourceUnsupported(t *testing.T) {
	script := extractScriptFromResource(&corev1.Pod{})
	assert.Empty(t, script)
}

func TestEnsureDirExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	tmpDir := t.TempDir()
	testPath := filepath.Join(tmpDir, "subdir", "nested", "file.txt")
	err := ensureDirExists(testPath)
	assert.NoError(t, err)
}

func TestEnsureDirExistsAlreadyExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	tmpDir := t.TempDir()
	err := ensureDirExists(tmpDir)
	assert.NoError(t, err)
}

func TestEnsureDirExistsMkdirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return assert.AnError
	})

	tmpDir := t.TempDir()
	testPath := filepath.Join(tmpDir, "subdir", "file.txt")
	err := ensureDirExists(testPath)
	assert.Error(t, err)
}

func TestHandleReadOnly(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	testFile := path.Join(testUnixRoot, "config.json")
	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadOnly("configmap", "default", "test-config", testFile)
	assert.NoError(t, err)
}

func TestHandleReadOnlyRelativePath(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadOnly("configmap", "default", "test-config", "relative/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
}

func TestHandleReadOnlyEnsureDirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return assert.AnError
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{
			"key1": "value1",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	testFile := filepath.Join(testUnixRoot, "config.json")
	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadOnly("configmap", "default", "test-config", testFile)
	assert.Error(t, err)
}

func TestHandleExecute(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	testFile := filepath.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	result, err := task.handleExecute(0, "configmap", "default", "test-script", testFile)
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, result)
}

func TestHandleExecuteNoAbsPath(t *testing.T) {
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	result, err := task.handleExecute(0, "configmap", "default", "test-script", "relative/path")
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, result)
}

func TestHandleExecuteEnsureDirError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return assert.AnError
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	testFile := path.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	result, err := task.handleExecute(0, "configmap", "default", "test-script", testFile)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestHandleExecuteWriteFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.WriteFile, func(name string, data []byte, perm os.FileMode) error {
		return assert.AnError
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-script",
			Namespace: "default",
		},
		Data: map[string]string{
			"script.sh": "echo hello",
		},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	mockExec := &mockExecutor{
		output: "hello",
		err:    nil,
	}

	testFile := path.Join(testUnixRoot, "output.txt")
	task := &Task{
		K8sClient: client,
		Exec:      mockExec,
	}
	result, err := task.handleExecute(0, "configmap", "default", "test-script", testFile)
	assert.Error(t, err)
	assert.Empty(t, result)
}

func TestHandleReadWrite(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("test content"), nil
	})

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Data: map[string]string{},
	}
	client := fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(configMap).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadWrite("configmap", "default", "test-config", "/tmp/test.json")
	assert.NoError(t, err)
}

func TestHandleReadWriteRelativePath(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadWrite("configmap", "default", "test-config", "relative/path")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absolute path")
}

func TestHandleReadWriteFileNotExist(t *testing.T) {
	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadWrite("configmap", "default", "test-config", "/nonexistent/file")
	assert.Error(t, err)
}

func TestHandleReadWriteReadFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(path.IsAbs, func(p string) bool {
		return true
	})

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return nil, assert.AnError
	})

	task := &Task{
		K8sClient: fake.NewClientBuilder().Build(),
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadWrite("configmap", "default", "test-config", "/tmp/test")
	assert.Error(t, err)
}

func TestHandleReadWriteUpdateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.IsFile, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.ReadFile, func(name string) ([]byte, error) {
		return []byte("test content"), nil
	})

	client := fake.NewClientBuilder().WithScheme(newScheme()).Build()

	task := &Task{
		K8sClient: client,
		Exec:      &exec.CommandExecutor{},
	}
	err := task.handleReadWrite("configmap", "default", "test-config", "/tmp/test")
	assert.NoError(t, err)
}

func TestK8sInterface(t *testing.T) {
	var _ K8s = &Task{}
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "configmap", configmap)
	assert.Equal(t, "secret", secret)
	assert.Equal(t, "ro", ro)
	assert.Equal(t, "rw", rw)
	assert.Equal(t, "rx", rx)
	assert.Equal(t, 4, resourceLength)
}

func TestConstantsValues(t *testing.T) {
	assert.Equal(t, uint32(0666), uint32(openfilePermission))
	assert.Equal(t, uint32(0644), uint32(RwRR))
}
