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
	"bufio"
	"fmt"
	"io"
	"net"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"

	"installer/pkg/constants"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

const (
	ptyRequestMsgWidth  = 50
	ptyRequestMsgHeight = 100
)

// Client 提供ssh客户端
type Client struct {
	Host           string
	User           string
	Password       []byte
	Port           string
	PrivateKey     string
	PrivateKeyPath string
	Timeout        time.Duration
	sftpClient     *sftp.Client
	SshClient      *ssh.Client
}

// getPrivateKey 获取PrivateKey
func (c *Client) getPrivateKey(privateKeyPath string, privateKey string) (ssh.AuthMethod, error) {
	var signer ssh.Signer
	var err error

	if isFileExist(privateKeyPath) {
		data, err := os.ReadFile(filepath.Clean(privateKeyPath))
		if err != nil {
			return nil, err
		}
		if privateKey == "" {
			signer, err = ssh.ParsePrivateKey(data)
			if err != nil {
				return nil, err
			}
		} else {
			phrase := []byte(privateKey)
			signer, err = ssh.ParsePrivateKeyWithPassphrase(data, phrase)
			if err != nil {
				return nil, err
			}
		}
	} else {
		signer, err = ssh.ParsePrivateKey([]byte(privateKey))
		if err != nil {
			return nil, err
		}
	}

	return ssh.PublicKeys(signer), nil
}

func (c *Client) getAuthSlice(key, keyPath string, password []byte) ([]ssh.AuthMethod, error) {
	var auths []ssh.AuthMethod

	if len(password) != 0 {
		auths = append(auths, ssh.Password(string(password)))
	}
	if key != "" || keyPath != "" {
		auth, err := c.getPrivateKey(keyPath, key)
		if err != nil {
			return auths, err
		}
		auths = append(auths, auth)
	}
	return auths, nil
}

// connect 根据提供的信息创建ssh连接
func (c *Client) connect() error {
	auths, err := c.getAuthSlice(c.PrivateKey, c.PrivateKeyPath, c.Password)
	if err != nil {
		return fmt.Errorf("SSH key could not be parsed: %v", err)
	}

	sshConfig := &ssh.ClientConfig{
		Auth:            auths,
		User:            c.User,
		Timeout:         c.Timeout,
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		HostKeyAlgorithms: []string{ssh.KeyAlgoED25519, ssh.KeyAlgoECDSA256, ssh.KeyAlgoRSASHA512,
			ssh.KeyAlgoRSASHA256},
	}
	point := net.JoinHostPort(c.Host, c.Port)
	sshClient, err := ssh.Dial(constants.TCP, point, sshConfig)
	if err != nil {
		return fmt.Errorf("establish connection to %s fail: %v", point, err)
	}
	c.SshClient = sshClient
	return nil
}

func (c *Client) session() (*ssh.Session, error) {
	if c.SshClient == nil {
		return nil, fmt.Errorf("connection closed")
	}

	session, err := c.SshClient.NewSession()
	if err != nil {
		return nil, err
	}

	terminalModes := ssh.TerminalModes{
		ssh.ECHO:          0,
		ssh.TTY_OP_ISPEED: 14400,
		ssh.TTY_OP_OSPEED: 14400,
	}
	if err := session.RequestPty(constants.Xterm, ptyRequestMsgHeight, ptyRequestMsgWidth, terminalModes); err != nil {
		return nil, err
	}
	return session, nil
}

// Close 关闭ssh连接
func (c *Client) Close() {
	if c.SshClient == nil && c.sftpClient == nil {
		return
	}

	if c.SshClient != nil {
		if err := c.SshClient.Close(); err != nil {
			zlog.Warn("Failed to closed SshClient")
		}
		c.SshClient = nil
	}
	if c.sftpClient != nil {
		if err := c.sftpClient.Close(); err != nil {
			zlog.Warn("Failed to closed sftpClient")
		}
		c.sftpClient = nil
	}
}

// SudoExecCmd 以root权限执行命令
func (c *Client) SudoExecCmd(cmd string) (string, error) {
	if err := c.connect(); err != nil {
		return "", fmt.Errorf("failed to connect %s: %v", c.Host, err)
	}
	defer c.SshClient.Close()

	session, err := c.session()
	if err != nil {
		return "", fmt.Errorf("failed create ssh session with %s : %v", c.Host, err)
	}
	defer session.Close()

	cmd = fmt.Sprintf("sudo -E /bin/bash <<EOF\n%s\nEOF", cmd)

	stdin, err := session.StdinPipe()
	if err != nil {
		return "", err
	}

	stdout, err := session.StdoutPipe()
	if err != nil {
		return "", err
	}

	zlog.Infof("exec command: %s on %s", cmd, c.Host)
	if err := session.Start(cmd); err != nil {
		return "", err
	}

	output := c.execCmdAutoInputPasswd(stdin, stdout)
	outputStr := strings.TrimPrefix(string(output), fmt.Sprintf("[sudo] password for %s:", c.User))
	zlog.Infof("command %s output is: %s", c.Host, outputStr)
	if err := session.Wait(); err != nil {
		zlog.Errorf("failed to exec cmd %s on %s, err is %s", cmd, c.Host, err.Error())
		return strings.TrimSpace(outputStr), fmt.Errorf("%s: %v", strings.TrimSpace(outputStr), err)
	}
	return strings.TrimSpace(outputStr), nil
}

// SudoExecCmdNoLog 以root权限执行命令,不打印log
func (c *Client) SudoExecCmdNoLog(cmd string) (string, error) {
	if err := c.connect(); err != nil {
		return "", fmt.Errorf("failed to connect %s: %v", c.Host, err)
	}
	defer c.SshClient.Close()

	session, err := c.session()
	if err != nil {
		return "", fmt.Errorf("failed create ssh session with %s : %v", c.Host, err)
	}
	defer session.Close()

	cmd = fmt.Sprintf("sudo -E /bin/bash <<EOF\n%s\nEOF", cmd)

	stdin, err := session.StdinPipe()
	if err != nil {
		return "", err
	}

	stdout, err := session.StdoutPipe()
	if err != nil {
		return "", err
	}

	if err := session.Start(cmd); err != nil {
		return "", err
	}

	output := c.execCmdAutoInputPasswd(stdin, stdout)
	outputStr := strings.TrimPrefix(string(output), fmt.Sprintf("[sudo] password for %s:", c.User))
	if err := session.Wait(); err != nil {
		zlog.Errorf("failed to exec cmd %s on %s, err is %s", cmd, c.Host, err.Error())
		return strings.TrimSpace(outputStr), fmt.Errorf("%s: %v", strings.TrimSpace(outputStr), err)
	}
	return strings.TrimSpace(outputStr), nil
}

func (c *Client) execCmdAutoInputPasswd(stdin io.WriteCloser, stdout io.Reader) []byte {
	var reader, singleLine = bufio.NewReader(stdout), ""
	var output []byte
	var password []byte
	password = append(c.Password, []byte("\n")...)

	for {
		singleByte, err := reader.ReadByte()
		if err != nil {
			break
		}

		output = append(output, singleByte)

		if singleByte == byte('\n') {
			singleLine = ""
			continue
		}

		singleLine += string(singleByte)

		coverPasswd := strings.HasPrefix(singleLine, "[sudo] password for") || strings.HasPrefix(singleLine, "Password")
		if coverPasswd && strings.HasSuffix(singleLine, ": ") {
			if _, err := stdin.Write(password); err != nil {
				break
			}
		}
	}
	password = []byte("")
	return output
}

// ExecCmd 执行一个命令
func (c *Client) ExecCmd(cmd string) (string, error) {
	err := c.connect()
	if err != nil {
		return "", fmt.Errorf("failed to connect %s: %v", c.Host, err)
	}

	session, err := c.session()
	if err != nil {
		return "", fmt.Errorf("failed create ssh session with %s : %v", c.Host, err)
	}

	defer session.Close()
	defer c.SshClient.Close()

	zlog.Infof("exec command %s on %s", cmd, c.Host)
	out, err := session.CombinedOutput(cmd)
	zlog.Infof("command %s output is %s on %s", cmd, string(out), c.Host)
	if err != nil {
		return "", fmt.Errorf("failed exec command on %s: %v", c.Host, err)
	}
	return strings.TrimSuffix(string(out), "\r\n"), err
}

// ConnectSftpClient 创建ftp链接
func (c *Client) ConnectSftpClient(opts ...sftp.ClientOption) error {
	var client *sftp.Client
	var err error

	client, err = sftp.NewClient(c.SshClient, opts...)
	c.sftpClient = client
	return err
}

// RemoteFileExist 检查在远端机器上是否有文件
func (c *Client) RemoteFileExist(remote string) (bool, error) {
	err := c.connect()
	if err != nil {
		return false, fmt.Errorf("failed to connect %s: %v", c.Host, err)
	}
	defer c.SshClient.Close()

	return c.isRemoteFileExist(remote)
}

// CopyFileToRemote 拷贝单个文件到远端机器，在远端机器自动创建目标文件
func (c *Client) CopyFileToRemote(srcFile string, dest string) error {
	err := c.connect()
	if err != nil {
		return fmt.Errorf("failed to create connect ssh client with %s: %v", c.Host, err)
	}

	err = c.ConnectSftpClient()
	if err != nil {
		return fmt.Errorf("failed to create sftp client with %s: %v", c.Host, err)
	}
	defer c.SshClient.Close()
	defer c.sftpClient.Close()

	err = c.copyFileToRemote(srcFile, dest)
	if err != nil {
		return fmt.Errorf("failed to copy file to %s, %v", c.Host, err)
	}
	return nil
}

func (c *Client) isRemoteFileExist(filePath string) (bool, error) {
	fileName := path.Base(filePath)
	fileDirName := path.Dir(filePath)

	checkFileCmd := fmt.Sprintf("ls %s/%s 2>/dev/null | wc -l", fileDirName, fileName)
	output, err := c.SudoExecCmd(checkFileCmd)
	if err != nil {
		return false, err
	}
	count, err := strconv.Atoi(strings.TrimSpace(output))
	if err != nil {
		return false, err
	}
	return count != 0, nil
}

// CopyFilesToRemote 拷贝当前文件夹下的所有文件到标文件夹
func (c *Client) CopyFilesToRemote(src string, dst string) error {
	if err := c.connect(); err != nil {
		return fmt.Errorf("failed to create ssh client with %s, %v", c.Host, err)
	}
	if err := c.ConnectSftpClient(); err != nil {
		return fmt.Errorf("failed to connect sftp client %s : %v", c.Host, err)
	}
	defer c.SshClient.Close()
	defer c.sftpClient.Close()

	command := fmt.Sprintf("mkdir -p -m %o %s", constants.RwxRwxRwx, dst)
	_, err := c.SudoExecCmd(command)
	if err != nil {
		return fmt.Errorf("failed to create remote floder: %v", dst)
	}

	srcFiles, err := utils.GetDirFiles(src)
	if err != nil {
		return fmt.Errorf("obtain file name %v", err)
	}

	for _, srcFile := range srcFiles {
		PathSlice := strings.Split(srcFile, constants.Slash)
		dstFile := filepath.Join(dst, PathSlice[len(PathSlice)-1])
		if err := c.copyFileToRemote(srcFile, dstFile); err != nil {
			return err
		}
	}
	return nil
}

// copyFileToRemote 拷贝本地文件到远端机器
func (c *Client) copyFileToRemote(local string, dest string) error {
	zlog.Info(fmt.Sprintf("copy %s to %s", local, dest))
	cleanLocal := filepath.Clean(local)
	localFile, err := os.Open(cleanLocal)
	if err != nil {
		return fmt.Errorf("open local file %s failed: %v", cleanLocal, err)
	}
	defer localFile.Close()

	stat, err := localFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file stat: %v", err)
	}
	if stat.IsDir() {
		return fmt.Errorf("the source %s is not a file", cleanLocal)
	}

	// permission 0666
	destFile, err := c.sftpClient.Create(dest)
	if err != nil {
		return fmt.Errorf("failed to create remote file %s on %s： %v", dest, c.Host, err)
	}
	if err := destFile.Chmod(stat.Mode()); err != nil {
		return fmt.Errorf("chmod remote file failed %v", err)
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, localFile)
	if err != nil {
		return fmt.Errorf("failed to copy %s to remote %s on %s", cleanLocal, dest, c.Host)
	}
	return nil
}

// DownloadFile 从远端拷贝文件到当前机器
func (c *Client) DownloadFile(local string, remote string) error {
	zlog.Infof("download %s to %s", remote, local)
	err := c.connect()
	if err != nil {
		return fmt.Errorf("failed to connect ssh client with %s: %v", c.Host, err)
	}

	err = c.ConnectSftpClient()
	if err != nil {
		return fmt.Errorf("failed to connect sftp client with %s: %v", c.Host, err)
	}
	defer c.SshClient.Close()
	defer c.sftpClient.Close()

	exist, err := c.RemoteFileExist(remote)
	if err != nil {
		return fmt.Errorf("failed to ehck remote file on %s: %v", c.Host, err)
	}
	if !exist {
		return fmt.Errorf("remote file is not exist")
	}

	if err := os.MkdirAll(filepath.Dir(local), constants.RwxRxRx); err != nil {
		return err
	}

	// 打开远端文件
	remoteFile, err := c.sftpClient.Open(remote)
	if err != nil {
		return fmt.Errorf("failed to open remote file: %v, remote path: %s", err, remote)
	}
	defer func() {
		if err := remoteFile.Close(); err != nil {
			zlog.Error("Failed to close file", err)
		}
	}()
	remoteInfo, err := remoteFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to get remote file stat: %v", err)
	}

	localFile, err := os.Create(filepath.Clean(local))
	if err != nil {
		return fmt.Errorf("failed to create local file %v", err)
	}
	defer func() {
		if err := localFile.Close(); err != nil {
			zlog.Error("Failed to close file:", err)
		}
	}()
	err = localFile.Chmod(remoteInfo.Mode())
	if err != nil {
		return fmt.Errorf("failed to set file permission: %v", err)
	}

	_, err = remoteFile.WriteTo(localFile)
	return err
}

func isFileExist(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	} else if os.IsExist(err) {
		return true
	}
	return false
}
