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
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type HostRemoteClient struct {
	SshClient  *Ssh
	SftpClient *Sftp
	host       *Host
	log        *log.Logger
}

// NewRemoteClient returns a new remote client with ssh and sftp client
func NewRemoteClient(h *Host) (*HostRemoteClient, error) {
	h, err := h.Validate()
	if err != nil {
		return nil, errors.Wrap(err, "host validation failed")
	}
	c := &HostRemoteClient{
		host: h,
		log:  log.With("host", h.Address),
	}

	c.SshClient, err = NewSSHClient(c.host)
	if err != nil {
		return nil, fmt.Errorf("Failed to create %s ssh client, err: %w", c.host.Address, err)
	}
	if c.SshClient == nil {
		return nil, fmt.Errorf("Failed to create %s ssh client,err: %w", c.host.Address, err)
	}

	c.SftpClient, err = NewSFTPClient(c.SshClient.sshClient)
	if err != nil {
		return nil, fmt.Errorf("Failed to create %s sftp client, err: %w", c.host.Address, err)
	}
	if c.SftpClient == nil {
		return nil, fmt.Errorf("Failed to create %s sftp client, err: %w", c.host.Address, err)
	}
	return c, nil
}

// SetLogger sets the logger for the remote client.
func (c *HostRemoteClient) SetLogger(logger *log.Logger) {
	c.log = logger
}

// Exec executes a command on the remote client.
func (cli *HostRemoteClient) Exec(ctx context.Context, cmd Command, stdErrChan chan CombineOut, stdOutChan chan CombineOut) {

	hostIp := cli.host.Address

	defer func() {
		if err := recover(); err != nil {
			stdErrChan <- NewCombineOut(hostIp, "", fmt.Sprintf("%v", err))
		}
		return
	}()

	go func() {
		for {
			select {
			case <-ctx.Done():
				if err := cli.CloseRemoteCli(); err != nil {
					stdErrChan <- NewCombineOut(hostIp, "", fmt.Sprintf("%v", err))
				}
				return
			}
		}
	}()

	for _, file := range cmd.FileUp {
		cli.log.Infof("Upload file %q to %q %q", file.Src, hostIp, file.Dst)
		if err := cli.SftpClient.UploadFile(file.Src, file.Dst); err != nil {
			errInfo := fmt.Sprintf("Failed to upload file %q to %q %q, err: %s", file.Src, hostIp, file.Dst, err.Error())
			cli.log.Debug(errInfo)
			stdErrChan <- NewCombineOut(hostIp, "", errInfo)
			continue
		}
		cli.log.Debug(fmt.Sprintf("Upload file %q to %q %q, success", file.Src, hostIp, file.Dst))
	}

	for _, c := range cmd.List() {
		cli.log.Infof("Execute command %q on %q", c, hostIp)
		stderr, stdout, err := cli.SshClient.Exec(c)
		if len(stderr) != 0 {
			errInfo := fmt.Sprintf("Failed to execute command %q on %q, stderr: %v", c, hostIp, stderr)
			cli.log.Debug(errInfo)
			stdErrChan <- NewCombineOut(hostIp, c, errInfo)
		}
		if err != nil {
			errInfo := fmt.Sprintf("Failed to execute command %q on %q, err: %s", c, hostIp, err.Error())
			cli.log.Debug(errInfo)
			stdErrChan <- NewCombineOut(hostIp, c, errInfo)
		}
		if len(stdout) != 0 {
			cli.log.Debug(fmt.Sprintf("Execute command %q on %q, stdout: %s", c, hostIp, stdout))
			stdOutChan <- NewCombineOut(hostIp, c, strings.Join(stdout, "\n"))
		}
		if len(stderr) == 0 && err == nil {
			cli.log.Debugf("Execute command %q on %q, success", c, hostIp)
		}
	}

}

// CloseRemoteCli closes the remote client.
func (cli *HostRemoteClient) CloseRemoteCli() error {
	var errs []string
	if err := cli.SftpClient.Close(); err != nil && err != io.EOF {
		errs = append(errs, fmt.Errorf("Failed to close %s sftp client, err: %w", cli.host.Address, err).Error())
	}
	if err := cli.SshClient.Close(); err != nil && err != io.EOF {
		errs = append(errs, fmt.Errorf("Failed to close %s ssh client, err: %w", cli.host.Address, err).Error())
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, ";"))
	}
	return nil
}
