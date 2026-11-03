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

package remote

import (
	"bufio"
	"io"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	gossh "golang.org/x/crypto/ssh"
)

func TestNewWithOutPassSSHClient(t *testing.T) {
	client, err := NewWithOutPassSSHClient("key")
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if client != nil {
		t.Error("expected nil client")
	}
}

func TestSsh_Close(t *testing.T) {
	ssh := &Ssh{
		sshClient: nil,
		alive:     false,
	}
	err := ssh.Close()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestSsh_Exec_NilClient(t *testing.T) {
	ssh := &Ssh{
		sshClient: nil,
		alive:     true,
	}
	_, _, err := ssh.Exec("ls")
	if err == nil {
		t.Error("expected error when ssh client is nil")
	}
}

func TestSsh_Exec_NotAlive(t *testing.T) {
	ssh := &Ssh{
		sshClient: nil,
		alive:     false,
	}
	_, _, err := ssh.Exec("ls")
	if err == nil {
		t.Error("expected error when ssh not alive")
	}
}

func TestNewSSHClient_WithSSHKey(t *testing.T) {
	host := &Host{
		SSHKey: "test-key",
	}
	client, err := NewSSHClient(host)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if client == nil {
		t.Error("expected non-nil client")
	}
	if !client.alive {
		t.Error("expected client to be alive")
	}
}

func TestSsh_Exec_EmptyCmd(t *testing.T) {
	ssh := &Ssh{
		sshClient: nil,
		alive:     true,
	}
	stdErrs, stdOuts, err := ssh.Exec("")
	// When sshClient is nil, Exec returns error before checking empty cmd
	if err == nil {
		t.Error("expected error when sshClient is nil")
	}
	if len(stdErrs) != 0 {
		t.Error("expected empty stdErrs")
	}
	if len(stdOuts) != 0 {
		t.Error("expected empty stdOuts")
	}
}

func TestSsh_readPipe(t *testing.T) {
	ssh := &Ssh{}

	reader := bufio.NewReader(strings.NewReader("test line"))
	line, err := ssh.readPipe(reader)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if line != "test line" {
		t.Errorf("expected 'test line', got %s", line)
	}

	reader = bufio.NewReader(strings.NewReader(""))
	_, err = ssh.readPipe(reader)
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
}

func TestSsh_readOutputAndError(t *testing.T) {
	ssh := &Ssh{}

	outReader := bufio.NewReader(strings.NewReader("output\n"))
	errReader := bufio.NewReader(strings.NewReader("error\n"))

	stdErrs, stdOuts, err := ssh.readOutputAndError(outReader, errReader)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(stdOuts) == 0 {
		t.Error("expected stdout")
	}
	if len(stdErrs) == 0 {
		t.Error("expected stderr")
	}
}

func TestNewNormalSSHClient_Mock(t *testing.T) {
	patches := gomonkey.ApplyFunc(gossh.Dial, func(network, addr string, config *gossh.ClientConfig) (*gossh.Client, error) {
		return &gossh.Client{}, nil
	})
	defer patches.Reset()

	client, err := NewNormalSSHClient("root", "pass", "192.168.1.1", "22")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if client == nil {
		t.Error("expected non-nil client")
	}
}

func TestNewSSHClient_WithUserAndAddress(t *testing.T) {
	patches := gomonkey.ApplyFunc(NewNormalSSHClient, func(user, password, host, port string) (*gossh.Client, error) {
		return &gossh.Client{}, nil
	})
	defer patches.Reset()

	host := &Host{
		User:     "root",
		Password: "pass",
		Address:  "192.168.1.1",
		Port:     "22",
	}

	client, err := NewSSHClient(host)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if client == nil {
		t.Error("expected non-nil client")
	}
}
