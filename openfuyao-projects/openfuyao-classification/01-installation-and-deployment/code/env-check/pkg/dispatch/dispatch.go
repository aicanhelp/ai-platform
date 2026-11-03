/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dispatch

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/sshutil"
)

const (
	numThree = 3
	num256K  = 256 * 1024

	defaultSSHTimeoutSec   = 30
	defaultTimeoutSec      = 600
	defaultPollIntervalSec = 15
	uploadRetryIntervalSec = 2
)

// Runner node dispatcher
type Runner struct {
	cfg          *config.DispatchConfig
	appCfg       *config.AppConfig
	logger       logger.Logger
	sshClients   map[string]*SSHClient
	newSSHClient func(host config.Host) (*SSHClient, error)
}

// sshConn abstracts ssh.Client for testing
type sshConn interface {
	NewSession() (sshSession, error)
	Close() error
}

// sshSession abstracts ssh.Session for testing
type sshSession interface {
	Close() error
	Output(cmd string) ([]byte, error)
	Run(cmd string) error
	SetStdin(r io.Reader)
}

// sftpClient abstracts sftp.Client for testing
type sftpClient interface {
	Close() error
	Stat(path string) (os.FileInfo, error)
	Remove(path string) error
	Create(path string) (sftpFile, error)
	Open(path string) (sftpFile, error)
}

// sftpFile abstracts sftp.File for testing
type sftpFile interface {
	io.Writer
	io.Reader
	Close() error
}

// SSHClient wraps SSH and SFTP connection
type SSHClient struct {
	client     sshConn
	sftpClient sftpClient
	host       config.Host
}

type sshClientWrapper struct {
	*ssh.Client
}

func (w *sshClientWrapper) NewSession() (sshSession, error) {
	s, err := w.Client.NewSession()
	if err != nil {
		return nil, err
	}
	return &sshSessionWrapper{Session: s}, nil
}

type sshSessionWrapper struct {
	*ssh.Session
}

func (w *sshSessionWrapper) SetStdin(r io.Reader) {
	w.Session.Stdin = r
}

type sftpClientWrapper struct {
	*sftp.Client
}

func (w *sftpClientWrapper) Create(path string) (sftpFile, error) {
	f, err := w.Client.Create(path)
	if err != nil {
		return nil, err
	}
	return &sftpFileWrapper{File: f}, nil
}

func (w *sftpClientWrapper) Open(path string) (sftpFile, error) {
	f, err := w.Client.Open(path)
	if err != nil {
		return nil, err
	}
	return &sftpFileWrapper{File: f}, nil
}

type sftpFileWrapper struct {
	*sftp.File
}

// nodeExecRequest contains all input needed to start checks on one node.
type nodeExecRequest struct {
	ip         string
	client     *SSHClient
	nodeResult *config.NodeResult
	checkArg   string
	mu         *sync.Mutex
}

// collectErrChan collects errors from channel and joins them into one error.
func collectErrChan(errChan <-chan error) error {
	var errs []string
	for err := range errChan {
		errs = append(errs, err.Error())
	}
	if len(errs) == 0 {
		return nil
	}
	return fmt.Errorf("%s", strings.Join(errs, "; "))
}

// waitAndCollectErr waits for goroutines, closes error channel, and collects errors.
func waitAndCollectErr(wg *sync.WaitGroup, errChan chan error) error {
	wg.Wait()
	close(errChan)
	return collectErrChan(errChan)
}

func setCompleted(completed map[string]bool, ip string) {
	if completed != nil {
		completed[ip] = true
	}
}

func isCompleted(completed map[string]bool, ip string) bool {
	if completed == nil {
		return false
	}
	return completed[ip]
}

// removeRemoteFileIfExists removes a remote file and ignores not-exist errors.
func (d *Runner) removeRemoteFileIfExists(client *SSHClient, remotePath string) error {
	err := client.sftpClient.Remove(remotePath)
	if err == nil || os.IsNotExist(err) || errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return fmt.Errorf("failed to remove remote file %s: %v", remotePath, err)
}

// NewDispatcher creates a new runner
func NewDispatcher(cfg *config.DispatchConfig, appCfg *config.AppConfig, log logger.Logger) *Runner {
	d := &Runner{
		cfg:        cfg,
		appCfg:     appCfg,
		logger:     log,
		sshClients: make(map[string]*SSHClient),
	}
	d.newSSHClient = d.createSSHClient
	return d
}

// Execute executes dispatch operation
func (d *Runner) Execute() (*config.DispatchResult, error) {
	startTime := time.Now()
	d.logger.Info("dispatch check start......")

	result := newResult()
	if len(d.appCfg.Hosts) == 0 {
		return nil, fmt.Errorf("no hosts configured")
	}

	if err := d.executeSteps(result); err != nil {
		return nil, err
	}

	result.Duration = time.Since(startTime).String()
	d.summarizeResults(result)

	d.logger.Info(fmt.Sprintf("dispatch check completed: result=%s", result.Summary.Result))
	return result, nil
}

// newResult creates a new dispatch result
func newResult() *config.DispatchResult {
	return &config.DispatchResult{
		Timestamp: time.Now(),
		Nodes:     make([]config.NodeResult, 0),
		Summary:   config.DispatchSummary{},
	}
}

// executeSteps runs all dispatch steps
func (d *Runner) executeSteps(result *config.DispatchResult) error {
	d.logger.Info("establishing SSH connections...")
	if err := d.establishConnections(); err != nil {
		return fmt.Errorf("failed to establish SSH connections: %v", err)
	}
	defer d.closeAllConnections()

	d.logger.Info("detecting node architectures...")
	if err := d.detectNodeArch(); err != nil {
		d.logger.Warning(fmt.Sprintf("failed to detect some node arch: %v", err))
	}

	d.logger.Info("cleaning remote environment...")
	if err := d.cleanRemoteEnv(); err != nil {
		d.logger.Warning(fmt.Sprintf("failed to clean some remote env: %v", err))
	}

	d.logger.Info("dispatching files to nodes...")
	if err := d.transferFiles(); err != nil {
		return fmt.Errorf("failed to dispatch files: %v", err)
	}

	d.logger.Info("executing checks on remote nodes...")
	nodeStatuses := d.executeRemotely()

	d.logger.Info("collecting results...")
	d.collectResults(nodeStatuses, result)

	return nil
}

// establishConnections establishes SSH connections to all hosts
func (d *Runner) establishConnections() error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(d.appCfg.Hosts))
	semaphore := make(chan struct{}, d.cfg.ConcurrentLimit)

	for _, host := range d.appCfg.Hosts {
		wg.Add(1)
		go func(h config.Host) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			client, err := d.newSSHClient(h)
			if err != nil {
				errChan <- fmt.Errorf("failed to connect to %s: %v", h.IP, err)
				return
			}
			d.sshClients[h.IP] = client
			d.logger.Info(fmt.Sprintf("connected to %s", h.IP))
		}(host)
	}

	return waitAndCollectErr(&wg, errChan)
}

// createSSHClient creates SSH and SFTP connection to a host
func (d *Runner) createSSHClient(host config.Host) (*SSHClient, error) {
	sshConfig := sshutil.BuildClientConfig(host, defaultSSHTimeoutSec*time.Second)
	address := sshutil.Address(host)
	client, err := ssh.Dial("tcp", address, sshConfig)
	if err != nil {
		return nil, err
	}

	sftpClient, err := sftp.NewClient(
		client,
		sftp.UseConcurrentWrites(true),
		sftp.UseConcurrentReads(true),
	)
	if err != nil {
		if closeErr := client.Close(); closeErr != nil {
			return nil, fmt.Errorf("failed to create sftp client: %v (also failed to close ssh client: %v)", err, closeErr)
		}
		return nil, fmt.Errorf("failed to create sftp client: %v", err)
	}

	return &SSHClient{
		client:     &sshClientWrapper{Client: client},
		sftpClient: &sftpClientWrapper{Client: sftpClient},
		host:       host,
	}, nil
}

// closeAllConnections closes all SSH and SFTP connections
func (d *Runner) closeAllConnections() {
	for ip, client := range d.sshClients {
		if client.sftpClient != nil {
			if err := client.sftpClient.Close(); err != nil {
				d.logger.Warning(fmt.Sprintf("failed to close sftp connection to %s: %v", ip, err))
			}
		}
		if err := client.client.Close(); err != nil {
			d.logger.Warning(fmt.Sprintf("failed to close ssh connection to %s: %v", ip, err))
		}
	}
}

// detectNodeArch detects architecture of all remote nodes
func (d *Runner) detectNodeArch() error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(d.sshClients))
	semaphore := make(chan struct{}, d.cfg.ConcurrentLimit)

	for ip, client := range d.sshClients {
		wg.Add(1)
		go func(clientIP string, sshClient *SSHClient) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			stdout, _, err := d.executeCommandWithOutput(sshClient,
				"uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/'")
			if err != nil {
				errChan <- fmt.Errorf("failed to detect arch for %s: %v", clientIP, err)
				return
			}

			arch := strings.TrimSpace(string(stdout))
			if arch == "" {
				arch = "amd64"
			}

			host := sshClient.host
			host.Arch = arch
			sshClient.host = host

			d.logger.Info(fmt.Sprintf("detected arch for %s: %s", clientIP, arch))
		}(ip, client)
	}

	return waitAndCollectErr(&wg, errChan)
}

// executeCommandWithOutput executes command and returns output
func (d *Runner) executeCommandWithOutput(client *SSHClient, cmd string) ([]byte, []byte, error) {
	session, err := client.client.NewSession()
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()

	stdout, err := session.Output(cmd)
	return stdout, nil, err
}

// cleanRemoteEnv cleans remote environment
func (d *Runner) cleanRemoteEnv() error {
	commands := []string{
		fmt.Sprintf("mkdir -p %s", d.cfg.WorkDir),
		fmt.Sprintf("chmod 777 %s", d.cfg.WorkDir),
		fmt.Sprintf("rm -rf %s/*", d.cfg.WorkDir),
	}

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, d.cfg.ConcurrentLimit)

	for ip, client := range d.sshClients {
		wg.Add(1)
		go func(clientIP string, sshClient *SSHClient) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			for _, cmd := range commands {
				if err := d.executeCommand(sshClient, cmd); err != nil {
					d.logger.Warning(fmt.Sprintf("failed to execute on %s: %v", clientIP, err))
				}
			}
		}(ip, client)
	}

	wg.Wait()
	return nil
}

// transferFiles dispatches binary and config files to nodes
func (d *Runner) transferFiles() error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(d.sshClients))
	semaphore := make(chan struct{}, d.cfg.ConcurrentLimit)

	for ip, client := range d.sshClients {
		wg.Add(1)
		go func(clientIP string, sshClient *SSHClient) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			if err := d.transferToNode(clientIP, sshClient); err != nil {
				errChan <- err
			}
		}(ip, client)
	}

	return waitAndCollectErr(&wg, errChan)
}

// transferToNode dispatches files to a single node
func (d *Runner) transferToNode(clientIP string, sshClient *SSHClient) error {
	localBinary := d.getBinaryPathForHost(sshClient.host)
	remoteBinary := fmt.Sprintf("%s/envCheck", d.cfg.WorkDir)

	needUpload := true
	localStat, err := os.Stat(localBinary)
	if err == nil {
		remoteStat, err := sshClient.sftpClient.Stat(remoteBinary)
		if err == nil && remoteStat.Size() == localStat.Size() {
			needUpload = false
			d.logger.Info(fmt.Sprintf("binary already exists on %s, skipping upload", clientIP))
		}
	}

	if needUpload {
		// Use compressed upload for large binaries to reduce transfer time
		if err := d.uploadFileCompressed(sshClient, localBinary, remoteBinary); err != nil {
			return fmt.Errorf("failed to upload binary to %s: %v", clientIP, err)
		}
	}

	chmodCmd := fmt.Sprintf("chmod +x %s", remoteBinary)
	if err := d.executeCommand(sshClient, chmodCmd); err != nil {
		return fmt.Errorf("failed to chmod binary on %s: %v", clientIP, err)
	}

	remoteConfig := fmt.Sprintf("%s/config.json", d.cfg.WorkDir)
	nodeConfig, err := d.generateNodeConfig(sshClient.host.Role)
	if err != nil {
		return fmt.Errorf("failed to generate config for %s: %v", clientIP, err)
	}
	if err := d.uploadConfigBytes(sshClient, nodeConfig, remoteConfig); err != nil {
		return fmt.Errorf("failed to upload config to %s: %v", clientIP, err)
	}

	archInfo := ""
	if sshClient.host.Arch != "" {
		archInfo = fmt.Sprintf(" (arch: %s)", sshClient.host.Arch)
	}
	d.logger.Info(fmt.Sprintf("files dispatched to %s%s", clientIP, archInfo))
	return nil
}

// getBinaryPathForHost returns local binary path for the given host
func (d *Runner) getBinaryPathForHost(host config.Host) string {
	if host.Arch != "" {
		paths := []string{
			fmt.Sprintf("./build/envCheck_%s", host.Arch),
			fmt.Sprintf("./envCheck_%s", host.Arch),
		}
		for _, path := range paths {
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
	}

	exePath, err := os.Executable()
	if err != nil {
		d.logger.Warning(fmt.Sprintf("failed to get executable path: %v", err))
		return "./envCheck"
	}
	return exePath
}

// generateNodeConfig generates a config for a single node with its roles
func (d *Runner) generateNodeConfig(roles []string) ([]byte, error) {
	data, err := json.Marshal(d.appCfg)
	if err != nil {
		return nil, err
	}

	var cfg config.AppConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	d.logger.Info(fmt.Sprintf("generating config for roles: %v", roles))

	// Filter ports to only include the node's roles
	if cfg.PortCheck != nil && cfg.PortCheck.Ports != nil {
		filteredPorts := make(map[string][]string)
		for _, role := range roles {
			if ports, ok := cfg.PortCheck.Ports[role]; ok {
				filteredPorts[role] = ports
				d.logger.Info(fmt.Sprintf("role %s matched with ports: %v", role, ports))
			} else {
				d.logger.Info(fmt.Sprintf("role %s not found in ports config", role))
			}
		}
		cfg.PortCheck.Ports = filteredPorts
	}

	return json.Marshal(cfg)
}

// uploadConfigBytes uploads config bytes directly to remote node via SFTP
func (d *Runner) uploadConfigBytes(client *SSHClient, data []byte, remotePath string) error {
	if err := d.removeRemoteFileIfExists(client, remotePath); err != nil {
		d.logger.Warning(err.Error())
	}

	remoteFile, err := client.sftpClient.Create(remotePath)
	if err != nil {
		return fmt.Errorf("failed to create remote file %s: %v", remotePath, err)
	}
	defer remoteFile.Close()

	if _, err := remoteFile.Write(data); err != nil {
		return fmt.Errorf("failed to write config to remote: %v", err)
	}

	return nil
}

// uploadFile uploads a file via SFTP with retry and size verification
func (d *Runner) uploadFile(client *SSHClient, localPath, remotePath string) error {
	var lastErr error
	for i := 0; i < numThree; i++ {
		if i > 0 {
			d.logger.Info(fmt.Sprintf("retry upload %s -> %s (%d/%d)", localPath, remotePath, i+1, numThree))
			time.Sleep(uploadRetryIntervalSec * time.Second)
		}
		lastErr = d.uploadViaSFTP(client, localPath, remotePath)
		if lastErr == nil {
			return nil
		}
	}
	return lastErr
}

// uploadViaSFTP performs a single upload attempt using SFTP
func (d *Runner) uploadViaSFTP(client *SSHClient, localPath, remotePath string) error {
	localFile, err := os.Open(localPath)
	if err != nil {
		return fmt.Errorf("failed to open local file %s: %v", localPath, err)
	}
	defer localFile.Close()

	if err := d.removeRemoteFileIfExists(client, remotePath); err != nil {
		d.logger.Warning(err.Error())
	}

	remoteFile, err := client.sftpClient.Create(remotePath)
	if err != nil {
		return fmt.Errorf("failed to create remote file %s: %v", remotePath, err)
	}

	bufWriter := bufio.NewWriterSize(remoteFile, num256K)
	if _, err = io.Copy(bufWriter, localFile); err != nil {
		if closeErr := remoteFile.Close(); closeErr != nil {
			d.logger.Warning(fmt.Sprintf("failed to close remote file %s after copy failure: %v", remotePath, closeErr))
		}
		return fmt.Errorf("failed to copy data to remote file: %v", err)
	}

	if err = bufWriter.Flush(); err != nil {
		if closeErr := remoteFile.Close(); closeErr != nil {
			d.logger.Warning(fmt.Sprintf("failed to close remote file %s after flush failure: %v", remotePath, closeErr))
		}
		return fmt.Errorf("failed to flush remote file: %v", err)
	}

	if err = remoteFile.Close(); err != nil {
		return fmt.Errorf("failed to close remote file: %v", err)
	}

	localStat, err := localFile.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat local file: %v", err)
	}
	remoteStat, err := client.sftpClient.Stat(remotePath)
	if err != nil {
		return fmt.Errorf("failed to stat remote file: %v", err)
	}
	if localStat.Size() != remoteStat.Size() {
		if err := d.removeRemoteFileIfExists(client, remotePath); err != nil {
			return fmt.Errorf("file size mismatch after upload: local=%d remote=%d; cleanup failed: %v",
				localStat.Size(), remoteStat.Size(), err)
		}
		return fmt.Errorf("file size mismatch after upload: local=%d remote=%d", localStat.Size(), remoteStat.Size())
	}
	return nil
}

// uploadFileCompressed uploads a file using gzip compression via SSH
func (d *Runner) uploadFileCompressed(client *SSHClient, localPath, remotePath string) error {
	localFile, err := os.Open(localPath)
	if err != nil {
		return fmt.Errorf("failed to open local file: %v", err)
	}
	defer localFile.Close()

	var buf bytes.Buffer
	gzWriter := gzip.NewWriter(&buf)
	if _, err := io.Copy(gzWriter, localFile); err != nil {
		return fmt.Errorf("failed to compress local file: %v", err)
	}
	if err := gzWriter.Close(); err != nil {
		return fmt.Errorf("failed to close gzip writer: %v", err)
	}

	session, err := client.client.NewSession()
	if err != nil {
		return fmt.Errorf("failed to create SSH session: %v", err)
	}
	defer session.Close()

	session.SetStdin(bytes.NewReader(buf.Bytes()))
	remoteDir := filepath.Dir(remotePath)
	decompressCmd := fmt.Sprintf("mkdir -p %s && gunzip > %s", remoteDir, remotePath)
	if err := session.Run(decompressCmd); err != nil {
		return fmt.Errorf("remote decompress failed: %v", err)
	}

	return nil
}

// executeCommand executes a command on remote host
func (d *Runner) executeCommand(client *SSHClient, cmd string) error {
	session, err := client.client.NewSession()
	if err != nil {
		return err
	}
	defer session.Close()

	return session.Run(cmd)
}

// executeRemotely executes checks on all nodes
func (d *Runner) executeRemotely() map[string]*config.NodeResult {
	results := make(map[string]*config.NodeResult)
	var mu sync.Mutex

	checks := d.getChecksToRun()
	checkArg := strings.Join(checks, ",")

	for ip, client := range d.sshClients {
		nodeResult := &config.NodeResult{
			IP:        ip,
			Role:      client.host.Role,
			Status:    "running",
			StartTime: time.Now(),
		}
		results[ip] = nodeResult
		d.executeOnNode(nodeExecRequest{
			ip:         ip,
			client:     client,
			nodeResult: nodeResult,
			checkArg:   checkArg,
			mu:         &mu,
		})
	}

	return results
}

// executeOnNode executes check on a single node
func (d *Runner) executeOnNode(req nodeExecRequest) {
	roles := strings.Join(req.client.host.Role, ",")
	cmd := fmt.Sprintf("cd %s && ./envCheck run-local --checks %s --roles %s > %s/check.log 2>&1 & echo $!",
		d.cfg.WorkDir, req.checkArg, roles, d.cfg.WorkDir)

	session, err := req.client.client.NewSession()
	if err != nil {
		req.mu.Lock()
		req.nodeResult.Status = "failed"
		req.nodeResult.Error = fmt.Sprintf("failed to create session: %v", err)
		req.mu.Unlock()
		return
	}

	output, err := session.Output(cmd)
	closeErr := session.Close()

	if err != nil {
		req.mu.Lock()
		req.nodeResult.Status = "failed"
		req.nodeResult.Error = fmt.Sprintf("failed to start check: %v", err)
		req.mu.Unlock()
		return
	}
	if closeErr != nil {
		d.logger.Warning(fmt.Sprintf("failed to close start session on %s: %v", req.ip, closeErr))
	}

	pid := strings.TrimSpace(string(output))
	d.logger.Info(fmt.Sprintf("started check on %s with PID %s", req.ip, pid))
}

// getChecksToRun returns list of checks to run
func (d *Runner) getChecksToRun() []string {
	checks := []string{"kernel", "port", "disk", "clock", "fileQuery", "programCheck", "route", "iptables"}
	if len(d.cfg.Checks) > 0 {
		checks = d.cfg.Checks
	}

	if len(d.cfg.SkipChecks) == 0 {
		return checks
	}

	skipMap := make(map[string]bool, len(d.cfg.SkipChecks))
	for _, skip := range d.cfg.SkipChecks {
		skipMap[skip] = true
	}

	var filteredChecks []string
	for _, check := range checks {
		if !skipMap[check] {
			filteredChecks = append(filteredChecks, check)
		}
	}
	return filteredChecks
}

// collectResults polls and collects results from nodes
func (d *Runner) collectResults(nodeStatuses map[string]*config.NodeResult, result *config.DispatchResult) {
	timeout := time.Duration(d.cfg.Timeout) * time.Second
	if timeout == 0 {
		timeout = defaultTimeoutSec * time.Second
	}

	pollInterval := time.Duration(d.cfg.PollInterval) * time.Second
	if pollInterval == 0 {
		pollInterval = defaultPollIntervalSec * time.Second
	}

	startTime := time.Now()
	completed := make(map[string]bool)

	for len(completed) < len(nodeStatuses) {
		if time.Since(startTime) > timeout {
			d.logger.Warning("collection timeout reached")
			break
		}

		for ip, nodeResult := range nodeStatuses {
			if isCompleted(completed, ip) {
				continue
			}
			if d.collectFromNode(ip, nodeResult, result, completed) {
				continue
			}
		}

		time.Sleep(pollInterval)
	}

	d.markTimeoutFailures(nodeStatuses, completed, result)
}

// collectFromNode collects result from a single node, returns true if handled
func (d *Runner) collectFromNode(ip string, nodeResult *config.NodeResult,
	result *config.DispatchResult, completed map[string]bool) bool {
	if nodeResult == nil {
		return true
	}
	if completed == nil {
		nodeResult.Status = "failed"
		nodeResult.Error = "internal error: completed map is nil"
		if result != nil {
			result.Nodes = append(result.Nodes, *nodeResult)
		}
		return true
	}

	client, ok := d.sshClients[ip]
	if !ok {
		nodeResult.Status = "failed"
		nodeResult.Error = "SSH client not available"
		setCompleted(completed, ip)
		result.Nodes = append(result.Nodes, *nodeResult)
		return true
	}

	resultPath := fmt.Sprintf("%s/result.json", d.cfg.WorkDir)
	errorPath := fmt.Sprintf("%s/error.log", d.cfg.WorkDir)

	if d.checkFileExists(client, resultPath) {
		resultData, err := d.downloadFile(client, resultPath)
		if err == nil && len(resultData) > 0 {
			nodeResult.Status = "success"
			nodeResult.EndTime = time.Now()
			nodeResult.ResultFile = string(resultData)
			nodeResult.Error = ""
			nodeResult.ErrorFile = ""

			var checkResults []config.CheckResult
			if err := json.Unmarshal(resultData, &checkResults); err == nil {
				nodeResult.Results = checkResults
			}
		} else {
			nodeResult.Status = "failed"
			nodeResult.Error = "failed to download result"
		}
		setCompleted(completed, ip)
		result.Nodes = append(result.Nodes, *nodeResult)
		return true
	}

	errorData, err := d.readRemoteFile(client, errorPath)
	if err == nil && len(errorData) > 0 {
		nodeResult.Status = "failed"
		nodeResult.EndTime = time.Now()
		nodeResult.ErrorFile = string(errorData)
		nodeResult.Error = string(errorData)
		setCompleted(completed, ip)
		result.Nodes = append(result.Nodes, *nodeResult)
		return true
	}

	return false
}

// markTimeoutFailures marks unfinished nodes as timeout failures
func (d *Runner) markTimeoutFailures(nodeStatuses map[string]*config.NodeResult,
	completed map[string]bool, result *config.DispatchResult) {
	for ip, nodeResult := range nodeStatuses {
		if !isCompleted(completed, ip) {
			nodeResult.Status = "failed"
			nodeResult.Error = "timeout waiting for result"
			result.Nodes = append(result.Nodes, *nodeResult)
		}
	}
}

// checkFileExists checks if a remote file exists
func (d *Runner) checkFileExists(client *SSHClient, path string) bool {
	session, err := client.client.NewSession()
	if err != nil {
		return false
	}
	defer session.Close()

	err = session.Run(fmt.Sprintf("test -f %s", path))
	return err == nil
}

// readRemoteFile reads a remote file, returns empty on error
func (d *Runner) readRemoteFile(client *SSHClient, path string) ([]byte, error) {
	session, err := client.client.NewSession()
	if err != nil {
		return nil, err
	}
	defer session.Close()

	return session.Output(fmt.Sprintf("cat %s 2>/dev/null", path))
}

// downloadFile downloads a file from remote host via SFTP
func (d *Runner) downloadFile(client *SSHClient, remotePath string) ([]byte, error) {
	remoteFile, err := client.sftpClient.Open(remotePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open remote file %s: %v", remotePath, err)
	}
	defer remoteFile.Close()

	return io.ReadAll(remoteFile)
}

// summarizeResults updates summary and evaluates node results
func (d *Runner) summarizeResults(result *config.DispatchResult) {
	for i := range result.Nodes {
		node := &result.Nodes[i]
		normalizeNodeStatus(node)
		accumulateSummary(&result.Summary, node)
	}

	updateSummaryResult(&result.Summary)
}

func normalizeNodeStatus(node *config.NodeResult) {
	if node.Status != "success" {
		return
	}
	failedChecks := failedCheckNames(node.Results)
	if len(failedChecks) > 0 {
		node.Status = "failed"
		node.Error = fmt.Sprintf("failed checks: %s", strings.Join(failedChecks, ", "))
		return
	}
}

func failedCheckNames(results []config.CheckResult) []string {
	var failedChecks []string
	for _, result := range results {
		if result.Status != "fail" {
			continue
		}
		checkType := result.CheckType
		if checkType == "" {
			checkType = "unknown"
		}
		failedChecks = append(failedChecks, checkType)
	}
	return failedChecks
}

func accumulateSummary(summary *config.DispatchSummary, node *config.NodeResult) {
	summary.TotalNodes++
	switch node.Status {
	case "success":
		summary.SuccessNodes++
	case "failed":
		summary.FailedNodes++
	case "running":
		summary.RunningNodes++
	default: // unknown status, ignore
		return
	}
}

func updateSummaryResult(summary *config.DispatchSummary) {
	if summary.FailedNodes > 0 {
		summary.Result = "FAIL"
		return
	}
	summary.Result = "PASS"
}

// SaveResults saves results to file
func SaveResults(results []config.CheckResult, resultPath string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(resultPath)
	if err := os.MkdirAll(dir, config.DirectoryMode); err != nil {
		return err
	}

	return os.WriteFile(resultPath, data, config.FileMode)
}
