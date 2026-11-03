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
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
	gossh "golang.org/x/crypto/ssh"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type Ssh struct {
	sshClient *gossh.Client
	mu        sync.RWMutex
	alive     bool
}

// NewSSHClient new ssh client
func NewSSHClient(host *Host) (*Ssh, error) {
	var (
		sshClient *gossh.Client
		err       error
	)

	switch {
	case host.User != "" && host.Address != "":
		if sshClient, err = NewNormalSSHClient(host.User, host.Password, host.Address, host.Port); err != nil {
			return nil, err
		}
		break
	case host.SSHKey != "":
		if sshClient, err = NewWithOutPassSSHClient(host.SSHKey); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("some fields are blank")
	}

	return &Ssh{
		sshClient: sshClient,
		alive:     true,
	}, nil
}

// NewNormalSSHClient new ssh client with username and password
func NewNormalSSHClient(user string, password string, host string, port string) (*gossh.Client, error) {
	config := &gossh.ClientConfig{
		User:            user,
		Auth:            []gossh.AuthMethod{gossh.Password(password)},
		Timeout:         30 * time.Second,
		HostKeyCallback: func(hostname string, remote net.Addr, key gossh.PublicKey) error { return nil },
	}
	config.SetDefaults()

	address := fmt.Sprintf("%s:%s", host, port)
	client, err := gossh.Dial("tcp", address, config)
	if err != nil {
		return nil, err
	}
	log.Infof("ssh client for %s created", address)
	return client, nil
}

// NewWithOutPassSSHClient new ssh client with ssh key
func NewWithOutPassSSHClient(sshKey interface{}) (*gossh.Client, error) {
	return nil, nil
}

// Exec	command on remote host
// just return stderr and error
func (s *Ssh) Exec(cmd string) ([]string, []string, error) {
	var stdErrs []string
	var stdOuts []string

	s.mu.RLock()
	client := s.sshClient
	alive := s.alive
	s.mu.RUnlock()

	if client == nil {
		return stdErrs, stdOuts, errors.New("Before run, have to new a ssh client")
	}

	// 不执行命令直接返回
	if cmd == "" {
		return stdErrs, stdOuts, nil
	}

	// 判断ssh连接是否关闭
	if !alive {
		return stdErrs, stdOuts, errors.New("ssh client is not alive，skip this command")
	}

	session, err := client.NewSession()
	if err != nil {
		return stdErrs, stdOuts, errors.Wrap(err, "Create session failed")
	}

	defer session.Close()

	r, _ := session.StdoutPipe()
	e, _ := session.StderrPipe()

	if err := session.Run(cmd); err != nil {
		return stdErrs, stdOuts, errors.Wrap(err, "run command failed")
	}

	outReader := bufio.NewReader(r)
	errReader := bufio.NewReader(e)

	return s.readOutputAndError(outReader, errReader)
}

// readOutputAndError reads both stdout and stderr from the readers
func (s *Ssh) readOutputAndError(outReader, errReader *bufio.Reader) ([]string, []string, error) {
	var stdErrs []string
	var stdOuts []string

	errReadFlag := false
	outReadFlag := false
	var errs []string
	for {
		if errReadFlag && outReadFlag {
			break
		}
		if !errReadFlag {
			stderr, err := s.readPipe(errReader)
			if stderr != "" {
				stdErrs = append(stdErrs, stderr)
			}
			if err != nil {
				errReadFlag = true
				if err != io.EOF {
					errs = append(errs, err.Error())
				}
			}
		}

		if !outReadFlag {
			stdout, err := s.readPipe(outReader)
			if stdout != "" {
				stdOuts = append(stdOuts, stdout)
			}
			if err != nil {
				outReadFlag = true
				if err != io.EOF {
					errs = append(errs, err.Error())
				}
			}
		}
	}

	if len(errs) > 0 {
		return stdErrs, stdOuts, errors.New(strings.Join(errs, ";"))
	}
	return stdErrs, stdOuts, nil
}

// readPipe read pipe buffer
func (s *Ssh) readPipe(reader *bufio.Reader) (string, error) {
	line, _, err := reader.ReadLine()
	if err == io.EOF {
		return "", err
	}
	if err != nil && err != io.EOF {
		return "", errors.Wrap(err, "Read pipe buffer failed")
	}
	return string(line), nil
}

func (s *Ssh) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.sshClient == nil || !s.alive {
		return nil
	}
	s.alive = false
	return s.sshClient.Close()
}
