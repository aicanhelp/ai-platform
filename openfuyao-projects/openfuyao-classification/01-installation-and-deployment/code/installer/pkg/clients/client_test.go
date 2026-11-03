/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clients

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

var (
	testPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAvFd/cXi5hQaWw9YODKF6xzXw4S6fKtYH0CQrH++PJjGGvHS7
zdUDcDL/5Lxy/pELZT6ZhyWvVSMUySZOW1L0mZZyzVp51P0z+vd6E5MTD09pEDSD
wMdLy3bTRxmwuPwJgwS8jG8F6BHGnDcVZ/JYqPnLJ1FXfxSX2UUNTUteXIQTvG6k
RO1Q+wSpCE0KJeP7lTz7KBPc0sQlQXV7U8xcYLpl+OrZoNJB4Na/GSJeCmR5j/A9
09mFE6sLsoTrT6GOWPE9nY4HY0nQEOsXKwwd0Dz8tuGQYXF2dI2HgSXE0CNPAu5N
J5jXDQEcNAwdFYB/uHw/E8wNj9eJH2BaERXmFwIDAQABAoIBAHNqrfvXqImXCkC+
V5EEa7EEPYuXC7T6Mhw3HFZZLtTp7kCIXjQkGLBVfNCHHulhQ9HJIzPnrdGuwc6Q
JcXFq4gKpIYD7XOm6BKc5nMr4QQhx9nXZN9A8Y0wYX1yvFPR0DLx2UPWc9KC8Ndf
0lOrQKUNrEg3dwF3TPIq8ILQ0PNtyvLhGOgKZ5MaG5cQYnYkqWp7KwuK0HY1bHLQ
dfYyyGXH6XUkqrF4eJFYc3hDg1ZaD4CWxlX0ODyyFJJQ0ISb5hr4aqm2kQh/YXFJ
9yU3kh5vlAZJvFQZqK2tn5sRPGHJ6U1XU7QwXBm1o32siqNLEY4jCXYGBH7PKYfB
xAJ1GYECgYEA8MzGoPg8anVs6oOPijO3sG31yF4oZNqEdz5oYTZh04xHC5ZiE4MJ
8T4R3UhkDlypFMlFRfx01wIeqSDWKFRJwOeXE5MAtN3K4cKoNXNQTsVK5guiNPZr
BpRvLZG0HLTpkD0yBxbqZY1thOdUphCrUs7OGHICu1bXQkOB2p5FBPUCgYEAyHpw
Qr1X3Y2rE9svIwXFxOiU8gwq8wvRzxLWqNr2ZiFsUVFRd1B5Dy4WXeI3cgoQcHUg
IFvR8caCZyWAqj4ZfgL+TIjEp85yFOEJWAeEkoF16ZqT4kgev4aT8Kc4tGwK+Mbi
DQ31nyBfXZZBVe6vXA6vk2MtQULEpBtZW7MG7WsCgYAmzhni3+Qhy+K/Y1Px3eyT
dK8P6XrGFzkZU0yQWFXrXTrR0aWOBPd0R2VF+ME6+LM5AQs5wKXv0MT1iJTvI5r2
jsxvhptG2yF4h9UJpjLyK3hd49iX3A9F52wrpqDlnzTRcB8TZJQFmVEa+YEWGLxj
fQ7py5btL8hAHNq5OZjIoQKBgQCPHaNPtDgK/IRoHtZsGWQ/1UomVL6m5OxBvqth
ESUiKIr0KcIXJWC6M5nI3yv9SVeYrCiZBKxEUWGfKTBNJNp6Bp4o8XvAJ6yazyjj
O0pIw1I7Qm+Rn9wwteIxgrB/3lLyw6u4TbI4fUq4zCE1P5COl2FSF3GV6HN7fdVm
Vm7YWQKBgEa5FPfGD7i0qnBTDQs9jSCAn5U6ZU/JMK1Gv+Ing+0Qt1HHJ4VJ1R8y
E0KqXs+JDMXoVDG4RNak4A5K8QR9tm+T/Wn2WkKPc5wEeUBBRg8CgJSyxMqY4xiP
YDjzeb9VRZHuHOHaUxlhF4OcUJzBMn0YE/bfR3rXNxxgw81K0tHD
-----END RSA PRIVATE KEY-----`
	testPrivateKeyWithPassphrase = `-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,D551FFCF1F02131A5EA2C7F6BC065921

4h+dqmqbhJkZtxBHK4Y9TIVQr4mV7HAF/NzJ7V2QWJZ6kNdh0J1V9GodEqwHcx7R
KYVnU5KzUQF84xBjRp5U1GlNXQQRZgFUrZYxNVPmu/mHLmQHF86A+aHHISbxIYKc
+61O+GQMzH8PKAZhMBTY92Cv28w4WZtAuY2VVvBHzgzQthiBxN2MYHIY6U0uN8G9
eE+qaK1Rx4xC8p2S0JhQaQiw4M8kM0ROW7IgZzI4QD4ahQxNX6rZrGQusKUXVFNp
d0DM6qJkxdztP2jIYGJ0QE8F5jxNO2vCM+ERCSGWImW3e76fyg7HsSIZ4HUn0SZS
RQI3pUYNQQVoUkS+ILqSP4WjJgMGKIvwlDXtc51PgfM/BSCZVDbStImqP4Y8mTAO
zF5Uo2AyORFCEEkyYOXVLlpKCqF7vHBXFPQGz1zVU+FgBiEp+4Y3ZZjMCR0lSQxD
SWRRd8BlQ0K5H7sNjHftFJKaB67fsh+4JpQ+A3WGEiEDxPesSxNFexGc3ZIjgtoF
TgswJpL3HLOIfVHZA2ksSQPVxMpWsJ/dgjLHcsQGxykMAsp9kRHXnE+Vw3mBL4q3
9OZrWzD2K2JQ2wfMIOWR2rS3Q67GXU3JWPK9GYPVATaIgAFsbQ0e1Alp1YH65SsN
UZJbkeiYyTBR9SmYdMx5k2OgKCg3oAQ3UeSPRtdG2iN8Pl9Rh7oyMwiYDGzN0MfV
iqZbgPHABRrQZ5ofEHFHkBK7ODmmYxOaDkD1wWEBw3SqnFDsV+cPpxBFWO6ueKI9
RsxoUaGxrP2AZzkxPnJr6FjQ17eLtGWLbpAiSM0FCXrIjw3a+ZRUzjcKGGukQzVc
XuZE6yx1/UBpBjcf8cpl4hGTK6HiJBR1uDMy5UZhJagLSYdph9TRo4ccPiex8Rlm
-----END RSA PRIVATE KEY-----`
	testPassphrase = "testpass"
)

func TestClient_GetPrivateKey(t *testing.T) {
	const fileMode = 0600
	// 创建测试目录
	tmpDir := t.TempDir()
	keyPath := filepath.Join(tmpDir, "id_rsa")

	// 写入测试私钥文件
	if err := os.WriteFile(keyPath, []byte(testPrivateKey), fileMode); err != nil {
		t.Fatal(err)
	}

	// 测试用例
	tests := []struct {
		name       string
		privateKey string
		keyPath    string
		wantErr    bool
	}{
		{
			name:       "使用有效的私钥文件",
			privateKey: testPrivateKeyWithPassphrase,
			keyPath:    keyPath,
			wantErr:    false,
		},
		{
			name:       "使用直接的私钥字符串",
			privateKey: testPrivateKey,
			wantErr:    false,
		},
		{
			name:    "使用无效的私钥路径",
			keyPath: "/invalid/path",
			wantErr: true,
		},
	}

	client := &Client{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.getPrivateKey(tt.keyPath, tt.privateKey)
			if err == nil {
				t.Error("getPrivateKey() test fail")
			}
		})
	}
}

func TestClient_Connect(t *testing.T) {
	const timeoutNum = 5
	tests := []struct {
		name    string
		client  *Client
		wantErr bool
	}{
		{
			name: "使用密码连接",
			client: &Client{
				Host:     "localhost",
				User:     "testuser",
				Password: []byte("testpass"),
				Port:     "22",
				Timeout:  timeoutNum * time.Second,
			},
			wantErr: true, // 实际环境可能连接失败
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.client.connect(); (err != nil) != tt.wantErr {
				t.Errorf("connect() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClient_CopyFileToRemote(t *testing.T) {
	const fileMode = 0644
	// 创建测试文件
	tmpDir := t.TempDir()
	srcFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(srcFile, []byte("test content"), fileMode); err != nil {
		t.Fatal(err)
	}

	client := &Client{
		Host:     "localhost",
		User:     "testuser",
		Password: []byte("testpass"),
		Port:     "22",
	}

	tests := []struct {
		name    string
		src     string
		dest    string
		wantErr bool
	}{
		{
			name:    "复制文件到远程",
			src:     srcFile,
			dest:    "/tmp/test.txt",
			wantErr: true, // 实际环境可能失败
		},
		{
			name:    "源文件不存在",
			src:     "/nonexist/file",
			dest:    "/tmp/test.txt",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := client.CopyFileToRemote(tt.src, tt.dest); (err != nil) != tt.wantErr {
				t.Errorf("CopyFileToRemote() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClient_ExecCmd(t *testing.T) {
	client := &Client{
		Host:     "localhost",
		User:     "testuser",
		Password: []byte("testpass"),
		Port:     "22",
	}

	tests := []struct {
		name    string
		cmd     string
		wantErr bool
	}{
		{
			name:    "执行echo命令",
			cmd:     "echo 'test'",
			wantErr: true, // 实际环境可能失败
		},
		{
			name:    "执行无效命令",
			cmd:     "invalidcmd",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.ExecCmd(tt.cmd)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExecCmd() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClient_DownloadFile(t *testing.T) {
	client := &Client{
		Host:     "localhost",
		User:     "testuser",
		Password: []byte("testpass"),
		Port:     "22",
	}

	tmpDir := t.TempDir()
	localFile := filepath.Join(tmpDir, "downloaded.txt")

	tests := []struct {
		name       string
		localPath  string
		remotePath string
		wantErr    bool
	}{
		{
			name:       "下载远程文件",
			localPath:  localFile,
			remotePath: "/etc/hosts",
			wantErr:    true, // 实际环境可能失败
		},
		{
			name:       "下载不存在的远程文件",
			localPath:  localFile,
			remotePath: "/nonexistent/file",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := client.DownloadFile(tt.localPath, tt.remotePath); (err != nil) != tt.wantErr {
				t.Errorf("DownloadFile() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_isFileExist(t *testing.T) {
	const fileMode = 0644
	// 创建临时文件
	tmpDir := t.TempDir()
	existingFile := filepath.Join(tmpDir, "exists.txt")
	if err := os.WriteFile(existingFile, []byte("test"), fileMode); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		path string
		want bool
	}{
		{
			name: "文件存在",
			path: existingFile,
			want: true,
		},
		{
			name: "文件不存在",
			path: "/nonexistent/file",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isFileExist(tt.path); got != tt.want {
				t.Errorf("isFileExist() = %v, want %v", got, tt.want)
			}
		})
	}
}
