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
package webterminal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client"
	faker "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	"openfuyao.com/web-terminal-service/api/v1beta1"
)

func CreateFakeClient() client.Client {
	Scheme := runtime.NewScheme()
	v1beta1.AddToScheme(Scheme)
	v1.AddToScheme(Scheme)
	fakeMgrClient := faker.NewClientBuilder().WithScheme(Scheme).Build()
	return fakeMgrClient
}

func TestNewTerminal(t *testing.T) {
	type args struct {
		client    kubernetes.Interface
		config    *rest.Config
		mgrclient client.Client
	}
	fakeMgrClient := CreateFakeClient()
	config := &rest.Config{Host: "hw"}
	client := fake.NewSimpleClientset()
	tests := []struct {
		name string
		args args
		want HandleInterface
	}{
		{
			name: "success",
			args: args{
				client:    client,
				config:    config,
				mgrclient: fakeMgrClient,
			},
			want: &terminaler{
				client:    client,
				config:    config,
				MgrClient: fakeMgrClient,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewTerminal(tt.args.client, tt.args.config, tt.args.mgrclient); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewTerminal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsPodReady(t *testing.T) {
	type args struct {
		pod *v1.Pod
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "success",
			args: args{
				pod: &v1.Pod{
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			want: true,
		},
		{
			name: "false",
			args: args{
				pod: &v1.Pod{
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodInitialized,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isPodReady(tt.args.pod); got != tt.want {
				t.Errorf("isPodReady() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTerminalerCreateUserPod(t1 *testing.T) {
	type fields struct {
		client    kubernetes.Interface
		config    *rest.Config
		MgrClient client.Client
	}
	type args struct {
		ctx  context.Context
		user string
	}
	tests := []struct {
		name     string
		fields   fields
		args     args
		mockErr  error
		mockData string
	}{
		{
			name: "success",
			fields: fields{
				client:    fake.NewSimpleClientset(),
				config:    &rest.Config{},
				MgrClient: CreateFakeClient(),
			},
			args: args{
				ctx:  context.TODO(),
				user: "user",
			},
			mockData: "./",
			mockErr:  nil,
		},
	}
	for _, tt := range tests {
		t1.Run(tt.name, func(t1 *testing.T) {
			patch := gomonkey.ApplyFunc(getImagePath, func(path string) (string, error) {
				if tt.mockErr != nil {
					return "", tt.mockErr
				}
				return (tt.mockData), nil
			})
			defer patch.Reset()
			t := &terminaler{
				client:    tt.fields.client,
				config:    tt.fields.config,
				MgrClient: tt.fields.MgrClient,
			}
			t.CreateUserPod(tt.args.ctx, tt.args.user)
		})
	}
}

func TestGetImagePath(t *testing.T) {
	type args struct {
		filePath string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "succeed",
			args: args{
				filePath: "",
			},
			want:    "Hello, this is a test file!",
			wantErr: false,
		},
		{
			name: "fail",
			args: args{
				filePath: "",
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Step 1: 创建临时文件
			tmpFile, err := ioutil.TempFile("", "testfile-*.txt")
			if err != nil {
				t.Fatalf("Failed to create temp file: %v", err)
			}
			defer os.Remove(tmpFile.Name()) // 在测试结束时删除临时文件

			// Step 2: 向临时文件写入内容
			content := "Hello, this is a test file!"
			_, err = tmpFile.WriteString(content)
			if err != nil {
				t.Fatalf("Failed to write to temp file: %v", err)
			}

			// Step 3: 确保文件已关闭
			err = tmpFile.Close()
			if err != nil {
				t.Fatalf("Failed to close temp file: %v", err)
			}
			var got string
			// Step 4: 调用 getImagePath 读取临时文件
			if tt.name == "succeed" {
				got, err = getImagePath(tmpFile.Name())
			} else {
				got, err = getImagePath(tt.args.filePath)
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("getImagePath() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			// Step 5: 验证读取的内容
			if got != tt.want {
				t.Errorf("getImagePath() got = %v, want %v", got, tt.want)
			}
		})
	}
}

type mockExecutor struct{}

func (m *mockExecutor) StreamWithContext(ctx context.Context, options remotecommand.StreamOptions) error {
	// 模拟命令输出
	if options.Stdout != nil {
		options.Stdout.Write([]byte("mock output"))
	}
	return nil
}

func (m *mockExecutor) Stream(options remotecommand.StreamOptions) error {
	// 模拟命令输出
	if options.Stdout != nil {
		options.Stdout.Write([]byte("mock output"))
	}
	return nil
}

type streamExecutor struct {
	write []byte
	err   error
}

func (s *streamExecutor) StreamWithContext(ctx context.Context, options remotecommand.StreamOptions) error {
	if options.Stdout != nil && len(s.write) > 0 {
		_, _ = options.Stdout.Write(s.write)
	}
	return s.err
}

func (s *streamExecutor) Stream(options remotecommand.StreamOptions) error {
	if options.Stdout != nil && len(s.write) > 0 {
		_, _ = options.Stdout.Write(s.write)
	}
	return s.err
}

type mockPersuo struct {
	buf bytes.Buffer
}

func (m *mockPersuo) Read(p []byte) (int, error) {
	return m.buf.Read(p)
}

func (m *mockPersuo) Write(p []byte) (int, error) {
	return m.buf.Write(p)
}

func (m *mockPersuo) Next() *remotecommand.TerminalSize {
	return &remotecommand.TerminalSize{Width: 80, Height: 24}
}

type stubClient struct {
	client.Client
	getFn func(ctx context.Context, key types.NamespacedName, obj client.Object) error
}

func (s stubClient) Get(ctx context.Context, key types.NamespacedName, obj client.Object, opts ...client.GetOption) error {
	if s.getFn != nil {
		return s.getFn(ctx, key, obj)
	}
	return s.Client.Get(ctx, key, obj, opts...)
}

func TestTerminalerStartSessionWithPing(t1 *testing.T) {
	conn := setupWebSockerServer(t1)
	defer conn.Close()

	type fields struct {
		client    kubernetes.Interface
		config    *rest.Config
		MgrClient client.Client
	}
	type args struct {
		ctx           context.Context
		namespace     string
		podName       string
		containerName string
		conn          *websocket.Conn
		mockErr       error
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "ok",
			fields: fields{
				client:    fake.NewSimpleClientset(),
				config:    &rest.Config{},
				MgrClient: CreateFakeClient(),
			},
			args: args{
				ctx:       context.TODO(),
				namespace: "default",
				conn:      conn,
			},
		},
		{
			name: "mock err",
			fields: fields{
				client:    fake.NewSimpleClientset(),
				config:    &rest.Config{},
				MgrClient: CreateFakeClient(),
			},
			args: args{
				ctx:       context.TODO(),
				namespace: "system",
				conn:      conn,
			},
		},
	}
	for _, tt := range tests {
		t1.Run(tt.name, func(t1 *testing.T) {
			t := &terminaler{
				client:    tt.fields.client,
				config:    tt.fields.config,
				MgrClient: tt.fields.MgrClient,
			}
			patch := gomonkey.ApplyMethod(reflect.TypeOf(&terminaler{}), "HandleTerminal",
				func(_ *terminaler, ctx context.Context,
					namespace string, podName string, containerName string, conn *websocket.Conn) {
					fmt.Println("success")
				})
			defer patch.Reset()
			t.startSessionWithPing(tt.args.ctx, tt.args.namespace, tt.args.podName, tt.args.containerName, tt.args.conn)
		})
	}
}

func TestTerminalerGetShell(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "shellExists",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName, shell string) bool {
			return shell == "bash"
		})
	defer patch.Reset()

	got := term.getShell(context.Background(), "ns", "pod", "container")
	assert.Equal(t, "bash", got)
}

func TestTerminalerGetShellFallback(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "shellExists",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName, shell string) bool {
			return shell == "sh"
		})
	defer patch.Reset()

	got := term.getShell(context.Background(), "ns", "pod", "container")
	assert.Equal(t, "sh", got)
}

func TestTerminalerGetShellNone(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "shellExists",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName, shell string) bool {
			return false
		})
	defer patch.Reset()

	got := term.getShell(context.Background(), "ns", "pod", "container")
	assert.Equal(t, "", got)
}

func TestTerminalerShellExistsExecError(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "executePodExec",
		func(_ *terminaler, options execOptions) (remotecommand.Executor, error) {
			return nil, errors.New("exec error")
		})
	defer patch.Reset()

	ok := term.shellExists(context.Background(), "ns", "pod", "container", "bash")
	assert.False(t, ok)
}

func TestTerminalerShellExistsStreamError(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "executePodExec",
		func(_ *terminaler, options execOptions) (remotecommand.Executor, error) {
			return &streamExecutor{err: errors.New("stream error")}, nil
		})
	defer patch.Reset()

	ok := term.shellExists(context.Background(), "ns", "pod", "container", "bash")
	assert.False(t, ok)
}

func TestTerminalerShellExistsOutput(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "executePodExec",
		func(_ *terminaler, options execOptions) (remotecommand.Executor, error) {
			return &streamExecutor{write: []byte("/bin/bash")}, nil
		})
	defer patch.Reset()

	ok := term.shellExists(context.Background(), "ns", "pod", "container", "bash")
	assert.True(t, ok)
}

func TestTerminalerStartProcessExecError(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "executePodExec",
		func(_ *terminaler, options execOptions) (remotecommand.Executor, error) {
			return nil, errors.New("exec error")
		})
	defer patch.Reset()

	err := term.startProcess(context.Background(), execOptions{persuo: &mockPersuo{}})
	assert.Error(t, err)
}

func TestTerminalerStartProcessStreamError(t *testing.T) {
	term := &terminaler{}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "executePodExec",
		func(_ *terminaler, options execOptions) (remotecommand.Executor, error) {
			return &streamExecutor{err: errors.New("stream error")}, nil
		})
	defer patch.Reset()

	err := term.startProcess(context.Background(), execOptions{persuo: &mockPersuo{}})
	assert.Error(t, err)
}

func TestHandleTerminalNoShell(t *testing.T) {
	term := &terminaler{}
	conn := &websocket.Conn{}

	var wrote []byte
	patch1 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "getShell",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string) string {
			return ""
		})
	patch2 := gomonkey.ApplyMethod(reflect.TypeOf(&websocket.Conn{}), "WriteMessage",
		func(_ *websocket.Conn, messageType int, data []byte) error {
			wrote = append([]byte{}, data...)
			return nil
		})
	patch3 := gomonkey.ApplyMethod(reflect.TypeOf(&Window{}), "Close",
		func(_ *Window, reason string) {})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	term.HandleTerminal(context.Background(), "ns", "pod", "container", conn)
	assert.Contains(t, string(wrote), "No valid shell found")
}

func TestHandleTerminalStartProcessError(t *testing.T) {
	term := &terminaler{}
	conn := &websocket.Conn{}

	var closedReason string
	patch1 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "getShell",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string) string {
			return "bash"
		})
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "startProcess",
		func(_ *terminaler, ctx context.Context, options execOptions) error {
			return errors.New("boom")
		})
	patch3 := gomonkey.ApplyMethod(reflect.TypeOf(&Window{}), "Close",
		func(_ *Window, reason string) {
			closedReason = reason
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	term.HandleTerminal(context.Background(), "ns", "pod", "container", conn)
	assert.Equal(t, "boom", closedReason)
}

func TestHandleTerminalStartProcessSuccess(t *testing.T) {
	term := &terminaler{}
	conn := &websocket.Conn{}

	var closedReason string
	patch1 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "getShell",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string) string {
			return "bash"
		})
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "startProcess",
		func(_ *terminaler, ctx context.Context, options execOptions) error {
			return nil
		})
	patch3 := gomonkey.ApplyMethod(reflect.TypeOf(&Window{}), "Close",
		func(_ *Window, reason string) {
			closedReason = reason
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	term.HandleTerminal(context.Background(), "ns", "pod", "container", conn)
	assert.Equal(t, "Process finished", closedReason)
}

func TestHandleCusterTerminal_NotFoundCreatesAndStarts(t *testing.T) {
	term := &terminaler{MgrClient: CreateFakeClient()}
	ctx := context.WithValue(context.Background(), "username", "user1")

	var created bool
	var started bool
	patch1 := gomonkey.ApplyMethod(reflect.TypeOf(term.MgrClient), "Get",
		func(_ client.Client, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			return errors.New("not found")
		})
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "CreateUserPod",
		func(_ *terminaler, ctx context.Context, user string) {
			created = true
		})
	patch3 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "startSessionWithPing",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string, conn *websocket.Conn) {
			started = true
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	term.HandleCusterTerminal(ctx, "user1", &websocket.Conn{})
	assert.True(t, created)
	assert.True(t, started)
}

func TestHandleCusterTerminal_ExistingNotDeletingStarts(t *testing.T) {
	term := &terminaler{MgrClient: CreateFakeClient()}
	ctx := context.WithValue(context.Background(), "username", "user1")

	var created bool
	var started bool
	patch1 := gomonkey.ApplyMethod(reflect.TypeOf(term.MgrClient), "Get",
		func(_ client.Client, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if wt, ok := obj.(*v1beta1.WebterminalTemplate); ok {
				wt.Name = key.Name
				wt.Namespace = key.Namespace
			}
			return nil
		})
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "CreateUserPod",
		func(_ *terminaler, ctx context.Context, user string) {
			created = true
		})
	patch3 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "startSessionWithPing",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string, conn *websocket.Conn) {
			started = true
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	term.HandleCusterTerminal(ctx, "user1", &websocket.Conn{})
	assert.False(t, created)
	assert.True(t, started)
}

func TestHandleCusterTerminal_DeletingPollErrorStops(t *testing.T) {
	getClient := stubClient{
		Client: CreateFakeClient(),
		getFn: func(ctx context.Context, key types.NamespacedName, obj client.Object) error {
			if wt, ok := obj.(*v1beta1.WebterminalTemplate); ok {
				now := time.Now()
				wt.ObjectMeta.DeletionTimestamp = &metav1.Time{Time: now}
				wt.ObjectMeta.Finalizers = []string{Finalizer}
			}
			return nil
		},
	}
	term := &terminaler{MgrClient: getClient}
	ctx := context.WithValue(context.Background(), "username", "user1")

	var created bool
	var started bool
	patch1 := gomonkey.ApplyFunc(controllerutil.ContainsFinalizer,
		func(obj client.Object, finalizer string) bool {
			return true
		})
	patch2 := gomonkey.ApplyFunc(wait.PollUntilContextTimeout,
		func(ctx context.Context, interval, timeout time.Duration, immediate bool, condition wait.ConditionWithContextFunc) error {
			return errors.New("poll error")
		})
	patch3 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "CreateUserPod",
		func(_ *terminaler, ctx context.Context, user string) {
			created = true
		})
	patch4 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(&terminaler{}), "startSessionWithPing",
		func(_ *terminaler, ctx context.Context, namespace, podName, containerName string, conn *websocket.Conn) {
			started = true
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()
	defer patch4.Reset()

	term.HandleCusterTerminal(ctx, "user1", &websocket.Conn{})
	assert.False(t, created)
	assert.False(t, started)
}
