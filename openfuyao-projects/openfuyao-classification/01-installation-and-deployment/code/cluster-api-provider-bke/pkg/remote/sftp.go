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
	"io"
	"os"
	"path"
	"sync"

	"github.com/pkg/errors"
	gosftp "github.com/pkg/sftp"
	gossh "golang.org/x/crypto/ssh"
)

type Sftp struct {
	sftpClient *gosftp.Client
	mu         sync.RWMutex
	alive      bool
}

// NewSFTPClient 以ssh客户端为基础建立sftp连接客户端
func NewSFTPClient(sshClient *gossh.Client) (*Sftp, error) {
	sftpClient, err := gosftp.NewClient(sshClient)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to new sftp client")
	}
	return &Sftp{
		sftpClient: sftpClient,
		alive:      true,
	}, nil
}

func (s *Sftp) UploadFile(localFilePath string, remoteDirPath string) error {
	s.mu.RLock()
	alive := s.alive
	client := s.sftpClient
	s.mu.RUnlock()

	if client == nil || !alive {
		return errors.New("sftp client is not alive")
	}

	// 打开本地文件
	localFile, err := os.Open(localFilePath)
	defer localFile.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to open local file %s", localFilePath)
	}

	// 创建远程文件
	remoteFileName := path.Base(localFilePath)
	remoteFilePath := path.Join(remoteDirPath, remoteFileName)

	// 有时因为远程文件已经存在，s.sftpClient.Create会报错，所以默认先删除在重新创建。
	_ = client.Remove(remoteFilePath)

	// 递归创建上层目录（如果已经存在则无操作）
	if err = client.MkdirAll(remoteDirPath); err != nil {
		return errors.Wrapf(err, "failed to create remote directory %s", remoteDirPath)
	}

	remoteFile, err := client.Create(remoteFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create remote file %s", remoteFilePath)
	}
	defer remoteFile.Close()

	// 用io.Copy的方式上传文件，速度更快
	_, err = io.Copy(remoteFile, localFile)
	if err != nil {
		return errors.Wrapf(err, "failed to copy local file to remote file %s", remoteFilePath)
	}

	statRemoteFile, err := remoteFile.Stat()
	if err != nil {
		return errors.Wrapf(err, "failed to stat remote file %s", remoteFilePath)
	}
	statLocalFile, err := localFile.Stat()
	if err != nil {
		return errors.Wrapf(err, "failed to stat local file %s", localFilePath)
	}

	if statRemoteFile.Size() != statLocalFile.Size() {
		if err := client.Remove(path.Join(remoteDirPath, remoteFileName)); err != nil {
			return errors.Wrapf(err, "failed to remove damaged remote file %s", remoteFilePath)
		}
		return errors.Errorf("file size mismatch after upload: local=%d, remote=%d, file=%s",
			statLocalFile.Size(), statRemoteFile.Size(), remoteFilePath)
	}

	return nil
}

func (s *Sftp) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.sftpClient == nil || !s.alive {
		return nil
	}
	s.alive = false
	return s.sftpClient.Close()
}
