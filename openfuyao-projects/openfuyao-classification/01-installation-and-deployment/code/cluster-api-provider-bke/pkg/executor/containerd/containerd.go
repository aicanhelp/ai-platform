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

package containerd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cio"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/namespaces"
	srvconfig "github.com/containerd/containerd/services/server/config"
	"github.com/moby/sys/signal"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	pb "k8s.io/cri-api/pkg/apis/runtime/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// ContainerdClient is an interface for containerd operations
type ContainerdClient interface {
	// Pull pulls an image from the containerd registry
	Pull(Image ImageRef) error
	// Close closes the containerd client connection
	Close()
	// Stop stops a container
	Stop(containerId string) error
	// Delete deletes a container
	Delete(containerId string) error
	// Run creates and starts a container
	Run(cs ContainerSpec) error
	// EnsureImageExists checks if an image exists and pulls it if not
	EnsureImageExists(image ImageRef) error
	// EnsureContainerRun ensures a container is running
	EnsureContainerRun(containerId string) (bool, error)
	// ContainerList lists containers with optional filters
	ContainerList(filters ...string) ([]containerd.Container, error)
	// Ping checks if containerd service is running
	Ping() error
}

// ImageRef represents a container image reference with authentication
type ImageRef struct {
	// Image is the image name
	Image string `json:"image"`
	// Username is the username for authentication
	Username string `json:"username"`
	// Password is the password for authentication
	Password string `json:"password"`
}

type Client struct {
	imageClient pb.ImageServiceClient
	condClient  *containerd.Client
	ctx         context.Context
	cancel      context.CancelFunc
}

var (
	containerdSock      = "unix:///var/run/containerd/containerd.sock"
	containerdNamespace = "k8s.io"
	containerdSockLinux = "/var/run/containerd/containerd.sock"
)

// ContainerdConfigFilePath is the path to containerd configuration file
const ContainerdConfigFilePath = "/etc/containerd/config.toml"

// SockAddrHashLength is the length of socket address hash
const SockAddrHashLength = 8

// ContainerdReadyTimeoutMinutes is the timeout duration in minutes for waiting containerd to be ready
const ContainerdReadyTimeoutMinutes = 2

// ContainerdReadyPollIntervalSeconds is the polling interval in seconds for checking containerd readiness
const ContainerdReadyPollIntervalSeconds = 5

// NewContainedClient creates a new containerd client
func NewContainedClient() (ContainerdClient, error) {
	if !utils.Exists(containerdSockLinux) {
		return nil, errors.New("containerd service does not exist. ")
	}
	imageClient, _, err := getImageClient(containerdSock)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	ctx = namespaces.WithNamespace(ctx, containerdNamespace)
	condClient, err := containerd.New(containerdSockLinux)
	if err != nil {
		return nil, err
	}
	var cancel context.CancelFunc
	ctx, cancel = context.WithCancel(ctx)

	c := &Client{
		imageClient: imageClient,
		condClient:  condClient,
		ctx:         ctx,
		cancel:      cancel,
	}
	if err := c.Ping(); err != nil {
		return nil, errors.Wrapf(err, "failed to ping containerd service")
	}

	return c, nil
}

// Close closes the containerd client connection
func (c *Client) Close() {
	if c.condClient != nil {
		c.condClient.Close()
	}
	return
}

// Ping checks if containerd service is running
func (c *Client) Ping() error {
	v, err := c.condClient.Version(c.ctx)
	if err == nil {
		log.Debugf("containerd version: %s", v.Version)
	}
	return err
}

// Stop stops a container
func (c *Client) Stop(id string) error {
	container, err := c.condClient.LoadContainer(c.ctx, id)
	if err != nil {
		return err
	}

	if err := setStopLabelWithContext(container, c.ctx); err != nil {
		return err
	}

	timeout, err := getStopTimeoutWithContext(container, c.ctx)
	if err != nil {
		return err
	}

	task, status, err := getTaskAndStatus(container, c.ctx)
	if err != nil {
		return err
	}

	if status.Status == containerd.Created || status.Status == containerd.Stopped {
		return nil
	}

	return stopTask(task, container, c.ctx, status, timeout)
}

// getTaskAndStatus gets the task and its status
func getTaskAndStatus(container containerd.Container, ctx context.Context) (containerd.Task, containerd.Status, error) {
	task, err := container.Task(ctx, nil)
	if err != nil {
		return nil, containerd.Status{}, err
	}
	status, err := task.Status(ctx)
	if err != nil {
		return nil, containerd.Status{}, err
	}
	return task, status, nil
}

// stopTask handles the task stopping logic
func stopTask(task containerd.Task, container containerd.Container, ctx context.Context, status containerd.Status, timeout *time.Duration) error {
	paused := status.Status == containerd.Paused || status.Status == containerd.Pausing

	exitCh, err := task.Wait(ctx)
	if err != nil {
		return err
	}

	l, err := container.Labels(ctx)
	if err != nil {
		return err
	}

	sig, err := getStopSignal(l)
	if err != nil {
		return err
	}

	if err := task.Kill(ctx, sig); err != nil {
		return err
	}

	if paused {
		if err := resumeTask(task, container, ctx); err != nil {
			return err
		}
	}

	sigtermCtx, sigtermCtxCancel := context.WithTimeout(ctx, *timeout)
	defer sigtermCtxCancel()

	err = waitContainerStop(sigtermCtx, exitCh, container.ID())
	if err == nil {
		return nil
	}

	return ctx.Err()
}

// setStopLabelWithContext sets the stop label on the container
func setStopLabelWithContext(container containerd.Container, ctx context.Context) error {
	opt := containerd.WithAdditionalContainerLabels(map[string]string{
		"containerd.io/restart.explicitly-stopped": strconv.FormatBool(true),
	})
	return container.Update(ctx, containerd.UpdateContainerOpts(opt))
}

// getStopTimeoutWithContext gets the stop timeout from container labels
func getStopTimeoutWithContext(container containerd.Container, ctx context.Context) (*time.Duration, error) {
	l, err := container.Labels(ctx)
	if err != nil {
		return nil, err
	}
	t, ok := l["nerdctl/stop-timeout"]
	if !ok {
		// Default is 10 seconds.
		t = "10"
	}
	td, err := time.ParseDuration(t + "s")
	if err != nil {
		return nil, err
	}
	timeout := &td
	return timeout, nil
}

// getStopSignal gets the stop signal from labels
func getStopSignal(labels map[string]string) (syscall.Signal, error) {
	sig, err := signal.ParseSignal("SIGTERM")
	if err != nil {
		return 0, err
	}
	if stopSignal, ok := labels[containerd.StopSignalLabel]; ok {
		sig, err = signal.ParseSignal(stopSignal)
		if err != nil {
			return 0, err
		}
	}
	return sig, nil
}

// resumeTask resumes the task if paused
func resumeTask(task containerd.Task, container containerd.Container, ctx context.Context) error {
	if err := task.Resume(ctx); err != nil {
		return fmt.Errorf("unpause container %s: %w", container.ID(), err)
	}
	return nil
}

// Delete deletes a container.
// todo 该方法存在bug 未完全解决，暂且删除容器使用exec的方式
// Delete deletes a container by name
func (c *Client) Delete(name string) error {
	container, labels, err := c.findContainerByName(name)
	if err != nil {
		return err
	}

	var delOpts []containerd.DeleteOpts
	if _, err := container.Image(c.ctx); err == nil {
		delOpts = append(delOpts, containerd.WithSnapshotCleanup)
	}

	id := container.ID()
	defer func() {
		c.cleanupNerdctlFiles(id, labels)
	}()

	task, err := container.Task(c.ctx, cio.Load)
	if err != nil {
		if errdefs.IsNotFound(err) {
			if container.Delete(c.ctx, containerd.WithSnapshotCleanup) != nil {
				return container.Delete(c.ctx)
			}
		}
		return err
	}

	if err := c.handleTaskDeletion(task, id); err != nil {
		return err
	}

	if err := container.Delete(c.ctx, delOpts...); err != nil {
		return err
	}
	return nil
}

// findContainerByName finds a container by name (io.kubernetes.pod.name or nerdctl/name)
func (c *Client) findContainerByName(name string) (containerd.Container, map[string]string, error) {
	containers, err := c.ContainerList()
	if err != nil {
		return nil, nil, err
	}

	var container containerd.Container
	var labels = make(map[string]string)
	for _, con := range containers {
		labels, err = con.Labels(c.ctx)
		if err != nil {
			return nil, nil, err
		}
		if labels["io.kubernetes.pod.name"] == name {
			container = con
			break
		}
		if labels["nerdctl/name"] == name {
			container = con
			break
		}
	}
	if container == nil {
		return nil, nil, fmt.Errorf("container %s not found", name)
	}
	return container, labels, nil
}

// cleanupNerdctlFiles cleans up nerdctl-related files for a container.
// Best-effort only (called from defer); failures are Debug so the library layer does not emit business Error.
func (c *Client) cleanupNerdctlFiles(id string, labels map[string]string) {
	nerdctlDir := filepath.Join("/var/lib/nerdctl", sockAddrHash())
	dataStoreDir := filepath.Join(nerdctlDir, "datastore")
	nameStoreDir := filepath.Join(dataStoreDir, "names", containerdNamespace)
	hostsDir := filepath.Join(dataStoreDir, "etchosts", containerdNamespace, id)
	volumeDir := filepath.Join(dataStoreDir, "volumes", containerdNamespace, id)
	// 处理nerdctl的容器
	// 删除nerdctl的容器目录
	if v, ok := labels["nerdctl/state-dir"]; ok {
		if err := os.RemoveAll(v); err != nil {
			log.Debugf("failed to remove container state dir %s: %v", v, err)
		}
	}

	if v, ok := labels["nerdctl/name"]; ok {
		if err := os.RemoveAll(filepath.Join(nameStoreDir, v)); err != nil {
			if !os.IsNotExist(err) {
				log.Debugf("failed to remove container name %s: %v", v, err)
			}
		}
	}
	// 删除nerdctl的hosts文件
	if err := os.RemoveAll(hostsDir); err != nil {
		if !os.IsNotExist(err) {
			log.Debugf("failed to remove container hosts dir %s: %v", hostsDir, err)
		}
	}
	// 删除nerdctl的容器卷
	if v, ok := labels["nerdctl/anonymous-volumes"]; ok {
		var anonVolumes []string
		if err := json.Unmarshal([]byte(v), &anonVolumes); err != nil {
			log.Debugf("failed to unmarshal anonymous volumes %s: %v", v, err)
			return
		}
		for _, name := range anonVolumes {
			if err := os.RemoveAll(filepath.Join(volumeDir, name)); err != nil {
				log.Debugf("failed to remove anonymous volume %s: %v", name, err)
			}
		}
	}
}

// handleTaskCreatedStopped handles deletion for Created or Stopped tasks
func (c *Client) handleTaskCreatedStopped(task containerd.Task, id string) error {
	if _, err := task.Delete(c.ctx); err != nil && !errdefs.IsNotFound(err) {
		return fmt.Errorf("failed to delete task %v: %w", id, err)
	}
	return nil
}

// handleTaskPaused handles deletion for Paused tasks
func (c *Client) handleTaskPaused(task containerd.Task, id string) error {
	_, err := task.Delete(c.ctx, containerd.WithProcessKill)
	if err != nil && !errdefs.IsNotFound(err) {
		return fmt.Errorf("failed to delete task %v: %w", id, err)
	}
	return nil
}

// handleTaskRunning handles deletion for Running tasks
func (c *Client) handleTaskRunning(task containerd.Task, id string) error {
	if err := task.Kill(c.ctx, syscall.SIGKILL); err != nil {
		return fmt.Errorf("send SIGKILL to task %s: %w", id, err)
	}

	es, err := task.Wait(c.ctx)
	if err == nil && es != nil {
		return c.handleTaskExitStatus(es, id)
	}

	_, err = task.Delete(c.ctx, containerd.WithProcessKill)
	if err != nil && !errdefs.IsNotFound(err) {
		return fmt.Errorf("delete task %s: %w", id, err)
	}
	return nil
}

// handleTaskExitStatus handles the exit status of a task
func (c *Client) handleTaskExitStatus(es <-chan containerd.ExitStatus, id string) error {
	for {
		select {
		case exitStatus := <-es:
			if exitStatus.ExitCode() != 0 {
				return fmt.Errorf("wait task %s exit: exited with code %d", id, exitStatus.ExitCode())
			}
			return nil
		}
	}
}

// handleTaskDeletion handles the deletion of a container's task based on its status
func (c *Client) handleTaskDeletion(task containerd.Task, id string) error {
	status, err := task.Status(c.ctx)
	if err != nil {
		if errdefs.IsNotFound(err) {
			return nil
		}
		return err
	}
	switch status.Status {
	case containerd.Created, containerd.Stopped:
		return c.handleTaskCreatedStopped(task, id)
	case containerd.Paused:
		return c.handleTaskPaused(task, id)
	default:
		return c.handleTaskRunning(task, id)
	}
}

// Pull pulls an image from the containerd registry
func (c *Client) Pull(image ImageRef) error {
	request := &pb.PullImageRequest{
		Image: &pb.ImageSpec{
			Image: image.Image,
		},
	}
	if image.Username != "" && image.Password != "" {
		request.Auth = &pb.AuthConfig{
			Username: image.Username,
			Password: image.Password,
		}
	}

	log.Debugf("pulling containerd image %s", image.Image)
	resp, err := c.imageClient.PullImage(context.Background(), request)
	if err != nil {
		return fmt.Errorf("pull containerd image %s: %w", image.Image, err)
	}
	log.Debugf("pulled containerd image %s, imageRef=%s", image.Image, resp.GetImageRef())
	return nil
}

// EnsureImageExists checks if an image exists and pulls it if not
func (c *Client) EnsureImageExists(image ImageRef) error {
	request := &pb.ImageStatusRequest{
		Image: &pb.ImageSpec{
			Image: image.Image,
		},
	}
	status, err := c.imageClient.ImageStatus(context.Background(), request)
	if err != nil {
		return fmt.Errorf("get containerd image status for %s: %w", image.Image, err)
	}
	if status.Image == nil {
		log.Infof("Image %s not found, pulling...", image.Image)
		if err := c.Pull(image); err != nil {
			return fmt.Errorf("ensure containerd image %s exists: %w", image.Image, err)
		}
	}
	return nil
}

// ContainerList lists containers with optional filters
func (c *Client) ContainerList(filters ...string) ([]containerd.Container, error) {
	return c.condClient.Containers(c.ctx, filters...)
}

// EnsureContainerRun ensures a container is running
func (c *Client) EnsureContainerRun(containerId string) (bool, error) {
	container, err := c.condClient.LoadContainer(c.ctx, containerId)
	if err != nil {
		return false, err
	}
	t, err := container.Task(c.ctx, nil)
	if err != nil {
		return false, err
	}
	s, err := t.Status(c.ctx)
	if err != nil {
		return false, err
	}
	return s.Status == containerd.Running, nil
}

func waitContainerStop(ctx context.Context, exitCh <-chan containerd.ExitStatus, id string) error {
	select {
	case <-ctx.Done():
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("wait container %v: %w", id, err)
		}
		return nil
	case status := <-exitCh:
		return status.Error()
	}
}

func sockAddrHash() string {
	d := digest.SHA256.FromString(containerdSockLinux)
	return d.Encoded()[0:SockAddrHashLength]
}

// GetContainerdConfig retrieves the containerd configuration from the specified path
func GetContainerdConfig(path string) (*srvconfig.Config, error) {
	if path == "" {
		path = ContainerdConfigFilePath
	}
	cfg := &srvconfig.Config{}
	err := srvconfig.LoadConfig(path, cfg)
	if err != nil {
		return nil, err
	}
	return cfg, nil
}

// WaitContainerdReady waits until containerd is ready
func WaitContainerdReady() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(ContainerdReadyTimeoutMinutes)*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(time.Duration(ContainerdReadyPollIntervalSeconds)*time.Second, func() (bool, error) {
		log.Debugf("Waiting for containerd to be ready")
		_, err := NewContainedClient()
		if err == nil {
			return true, nil
		}
		log.Debugf("containerd is not available: %v", err)
		return false, nil
	}, ctx.Done())
	if err != nil {
		return fmt.Errorf("wait containerd available: %w", err)
	}
	return nil
}
