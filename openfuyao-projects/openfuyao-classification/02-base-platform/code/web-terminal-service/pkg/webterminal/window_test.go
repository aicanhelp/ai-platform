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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func setupWebSockerServer(t *testing.T) *websocket.Conn {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				break
			}
			conn.WriteMessage(websocket.TextMessage, msg)
		}

	}))
	t.Cleanup(server.Close)
	wsURL := strings.Replace(server.URL, "http", "ws", 1)

	clientConn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	return clientConn
}

func TestWindowClose(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()

	type args struct {
		reason string
	}
	tests := []struct {
		name string
		conn *websocket.Conn
		args args
	}{
		{
			name: "success",
			conn: conn,
			args: args{
				reason: "Process finished",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Window{
				conn:     tt.conn,
				sizeChan: make(chan remotecommand.TerminalSize),
			}
			w.Close(tt.args.reason)
		})
	}
}

func TestWindowNext(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()

	tests := []struct {
		name string
		conn *websocket.Conn
		want *remotecommand.TerminalSize
	}{
		{
			name: "success",
			conn: conn,
			want: &remotecommand.TerminalSize{Width: 80, Height: 24},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Window{
				conn:     tt.conn,
				sizeChan: make(chan remotecommand.TerminalSize, 1),
			}
			w.sizeChan <- *tt.want
			if got := w.Next(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Window.Next() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWindowNextZeroSize(t *testing.T) {
	w := &Window{
		sizeChan: make(chan remotecommand.TerminalSize, 1),
	}
	w.sizeChan <- remotecommand.TerminalSize{Width: 0, Height: 0}
	assert.Nil(t, w.Next())
}

func TestWindowRead(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()

	type args struct {
		buffer []byte
	}
	tests := []struct {
		name      string
		conn      *websocket.Conn
		args      args
		msg       Message
		ctx       context.Context
		want      int
		wantErr   bool
		wantRenew bool
	}{
		{
			name: "stdin message",
			conn: conn,
			args: args{
				buffer: make([]byte, 1024),
			},
			msg: Message{
				Op:   "stdin",
				Data: "test stdin",
			},
			ctx:       context.WithValue(context.Background(), "path", "test path"),
			want:      len("test stdin"),
			wantErr:   false,
			wantRenew: false,
		},
		{
			name: "resize message",
			conn: conn,
			args: args{
				buffer: make([]byte, 1024),
			},
			msg: Message{
				Op:   "resize",
				Rows: 20,
				Cols: 40,
			},
			ctx:       context.WithValue(context.Background(), "path", "test path"),
			want:      0,
			wantErr:   false,
			wantRenew: false,
		},
		{
			name: "unknown message",
			conn: conn,
			args: args{
				buffer: make([]byte, 1024),
			},
			msg: Message{
				Op:   "unknow",
				Data: "some data",
			},
			ctx:       context.WithValue(context.Background(), "path", "test path"),
			want:      len(endOfWindow),
			wantErr:   true,
			wantRenew: false,
		},
		{
			name: "stdin message with KubectlAPI in path",
			conn: conn,
			args: args{
				buffer: make([]byte, 1024),
			},
			msg: Message{
				Op:   "stdin",
				Data: "test message with KubectlAPI",
			},
			ctx:       context.WithValue(context.Background(), "path", KubectlApi),
			want:      len("test message with KubectlAPI"),
			wantErr:   false,
			wantRenew: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Window{
				conn:       tt.conn,
				sizeChan:   make(chan remotecommand.TerminalSize, 1),
				ctx:        tt.ctx,
				terminaler: &terminaler{},
			}

			msgBytes, err := json.Marshal(tt.msg)
			require.NoError(t, err)
			err = conn.WriteMessage(websocket.TextMessage, msgBytes)
			require.NoError(t, err)

			if tt.msg.Data == "test message with KubectlAPI" {
				w.ctx = context.WithValue(w.ctx, "username", "testuser")
				fakeClient := fake.NewClientBuilder().Build()
				w.terminaler.MgrClient = fakeClient
			}

			got, err := w.Read(tt.args.buffer)
			if (err != nil) != tt.wantErr {
				t.Errorf("Window.Read() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Window.Read() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWindowToast(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()
	type args struct {
		buffer string
	}
	tests := []struct {
		name    string
		conn    *websocket.Conn
		args    args
		wantErr bool
	}{
		{
			name: "success",
			conn: conn,
			args: args{
				buffer: "Test toast message",
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Window{
				conn:       tt.conn,
				sizeChan:   make(chan remotecommand.TerminalSize, 1),
				ctx:        context.Background(),
				terminaler: &terminaler{},
			}
			if err := w.Toast(tt.args.buffer); (err != nil) != tt.wantErr {
				t.Errorf("Window.Toast() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestWindowWrite(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()

	type args struct {
		buffer []byte
	}
	tests := []struct {
		name            string
		conn            *websocket.Conn
		args            args
		path            string
		ctx             context.Context
		setDeadlineErr  error
		writeMessageErr error
		want            int
		wantErr         bool
	}{
		{
			name: "kubeapipath",
			conn: conn,
			args: args{
				buffer: []byte("test message"),
			},
			path:    KubectlApi,
			ctx:     context.WithValue(context.Background(), "path", KubectlApi),
			want:    len("test message"),
			wantErr: false,
		},
		{
			name: "testpath",
			conn: conn,
			args: args{
				buffer: []byte("test message"),
			},
			path:    "testpath",
			ctx:     context.WithValue(context.Background(), "path", "testpath"),
			want:    len("test message"),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().Build()

			w := &Window{
				conn:       tt.conn,
				sizeChan:   make(chan remotecommand.TerminalSize, 1),
				ctx:        tt.ctx,
				terminaler: &terminaler{MgrClient: fakeClient},
			}
			if tt.path == KubectlApi {
				w.ctx = context.WithValue(w.ctx, "username", "testuser")
				w.Renewtime()
			}

			got, err := w.Write(tt.args.buffer)
			if (err != nil) != tt.wantErr {
				t.Errorf("Window.Write() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Window.Write() = %v, want %v", got, tt.want)
			}

		})
	}
}

func TestWindowWriteConcurrent(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()

	w := &Window{
		conn:       conn,
		sizeChan:   make(chan remotecommand.TerminalSize, 1),
		ctx:        context.WithValue(context.Background(), "path", "testpath"),
		terminaler: &terminaler{},
	}

	const writers = 10
	const writesPerWriter = 5

	var wg sync.WaitGroup
	errCh := make(chan error, writers*writesPerWriter)
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < writesPerWriter; j++ {
				_, err := w.Write([]byte("test message"))
				errCh <- err
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		require.NoError(t, err)
	}
}

func TestWindowRenewtime(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
	}{
		{
			name: "success",
			ctx:  context.WithValue(context.Background(), "username", "testuser"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().Build()
			w := &Window{
				sizeChan:   make(chan remotecommand.TerminalSize, 1),
				ctx:        tt.ctx,
				terminaler: &terminaler{MgrClient: fakeClient},
			}

			w.Renewtime()
		})
	}
}

func TestWindowSendMessage(t *testing.T) {
	conn := setupWebSockerServer(t)
	defer conn.Close()
	tests := []struct {
		name string
		conn *websocket.Conn
	}{
		{
			name: "success",
			conn: conn,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Window{
				conn:       tt.conn,
				sizeChan:   make(chan remotecommand.TerminalSize, 1),
				ctx:        context.Background(),
				terminaler: &terminaler{},
			}
			w.sendMessage()
		})
	}
}
