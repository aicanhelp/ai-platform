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
	"github.com/sirupsen/logrus"
	"k8s.io/apimachinery/pkg/util/wait"
	pb "k8s.io/cri-api/pkg/apis/runtime/v1"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

type ContainerdClient interface {
	Pull(Image ImageRef) error
	//Push(Image ImageRef) (readClose io.ReadCloser, err error)
	Close()
	Stop(containerId string) error
	Delete(containerId string) error
	Run(cs ContainerSpec) error
	EnsureImageExists(image ImageRef) error
	EnsureContainerRun(containerId string) (bool, error)
	ContainerList(filters ...string) ([]containerd.Container, error)
	Ping() error
}

type ImageRef struct {
	Image    string `json:"image"`
	Username string `json:"username"`
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

const ContainerdConfigFilePath = "/etc/containerd/config.toml"

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

func (c *Client) Close() {
	if c.condClient != nil {
		_ = c.condClient.Close()
	}
}

func (c *Client) Ping() error {
	v, err := c.condClient.Version(c.ctx)
	if err == nil {
		log.Debugf("containerd version: %s", v.Version)
	}
	return err
}

func (c *Client) Stop(id string) error {
	container, err := c.condClient.LoadContainer(c.ctx, id)
	if err != nil {
		return err
	}

	// set label
	opt := containerd.WithAdditionalContainerLabels(map[string]string{
		"containerd.io/restart.explicitly-stopped": strconv.FormatBool(true),
	})
	err = container.Update(c.ctx, containerd.UpdateContainerOpts(opt))
	if err != nil {
		return err
	}
	// set timeout
	var timeout *time.Duration
	if err != nil {
		return err
	}
	l, err := container.Labels(c.ctx)
	t, ok := l["nerdctl/stop-timeout"]
	if !ok {
		// Default is 10 seconds.
		t = "10"
	}
	td, err := time.ParseDuration(t + "s")
	if err != nil {
		return err
	}
	timeout = &td

	// get task
	task, err := container.Task(c.ctx, nil)
	if err != nil {
		return err
	}
	//log.Info("stop container: ", id)
	//err = task.Pause(c.ctx)
	//if err != nil {
	//	return err
	//}
	//_, err = task.Delete(c.ctx, containerd.WithProcessKill)
	//if err != nil {
	//	log.Infof("delete container failed: ", id)
	//	return err
	//}
	status, err := task.Status(c.ctx)
	if err != nil {
		return err
	}
	paused := false

	switch status.Status {
	case containerd.Created, containerd.Stopped:
		return nil
	case containerd.Paused, containerd.Pausing:
		paused = true
	default:
	}

	exitCh, err := task.Wait(c.ctx)
	if err != nil {
		return err
	}

	sig, err := signal.ParseSignal("SIGTERM")
	if err != nil {
		return err
	}
	if stopSignal, ok := l[containerd.StopSignalLabel]; ok {
		sig, err = signal.ParseSignal(stopSignal)
		if err != nil {
			return err
		}
	}

	if err := task.Kill(c.ctx, sig); err != nil {
		return err
	}

	// signal will be sent once resume is finished
	if paused {
		if err := task.Resume(c.ctx); err != nil {
			logrus.Warnf("Cannot unpause container %s: %s", container.ID(), err)
		} else {
			// no need to do it again when send sigkill signal
			paused = false
		}
	}

	sigtermCtx, sigtermCtxCancel := context.WithTimeout(c.ctx, *timeout)
	defer sigtermCtxCancel()

	err = waitContainerStop(sigtermCtx, exitCh, container.ID())
	if err == nil {
		return nil
	}

	return c.ctx.Err()
}

// Delete deletes a container.
// todo 该方法存在bug 未完全解决，暂且删除容器使用exec的方式
func (c *Client) Delete(name string) error {
	containers, err := c.ContainerList()
	if err != nil {
		return err
	}

	var container containerd.Container
	var labels = make(map[string]string)
	for _, con := range containers {
		labels, err = con.Labels(c.ctx)
		if err != nil {
			return err
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
		return fmt.Errorf("container %s not found", name)
	}

	id := container.ID()
	defer func() {
		nerdctlDir := filepath.Join("/var/lib/nerdctl", sockAddrHash())
		dataStoreDir := filepath.Join(nerdctlDir, "datastore")
		nameStoreDir := filepath.Join(dataStoreDir, "names", containerdNamespace)
		hostsDir := filepath.Join(dataStoreDir, "etchosts", containerdNamespace, id)
		volumeDir := filepath.Join(dataStoreDir, "volumes", containerdNamespace, id)
		// 处理nerdctl的容器
		// 删除nerdctl的容器目录
		if v, ok := labels["nerdctl/state-dir"]; ok {
			if err := os.RemoveAll(v); err != nil {
				log.Errorf("failed to remove container state dir: %s, err:%v", v, err)
			}
		}
		// 删除nerdctl记录的容器名

		if v, ok := labels["nerdctl/name"]; ok {
			if err := os.RemoveAll(filepath.Join(nameStoreDir, v)); err != nil {
				if !os.IsNotExist(err) {
					log.Errorf("failed to remove container name: %s, err:%v", v, err)
				}
			}
		}
		// 删除nerdctl的hosts文件
		if err := os.RemoveAll(hostsDir); err != nil {
			if !os.IsNotExist(err) {
				log.Errorf("failed to remove container hosts dir: %s, err:%v", hostsDir, err)
			}
		}
		// 删除nerdctl的容器卷
		if v, ok := labels["nerdctl/anonymous-volumes"]; ok {
			var anonVolumes []string
			if err := json.Unmarshal([]byte(v), &anonVolumes); err != nil {
				log.Errorf("failed to unmarshal anonymous volumes: %s, err:%v", v, err)
			}
			for _, name := range anonVolumes {
				err := os.RemoveAll(filepath.Join(volumeDir, name))
				if err != nil {
					log.Errorf("failed to remove anonymous volume: %s, err:%v", name, err)
				}
			}
		}
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

	status, err := task.Status(c.ctx)
	if err != nil {
		if errdefs.IsNotFound(err) {
			return nil
		}
		return err
	}
	switch status.Status {
	case containerd.Created, containerd.Stopped:
		if _, err := task.Delete(c.ctx); err != nil && !errdefs.IsNotFound(err) {
			return fmt.Errorf("failed to delete task %v: %w", id, err)
		}
	case containerd.Paused:
		_, err := task.Delete(c.ctx, containerd.WithProcessKill)
		if err != nil && !errdefs.IsNotFound(err) {
			return fmt.Errorf("failed to delete task %v: %w", id, err)
		}
	// default is the case, when status.Status = containerd.Running
	default:

		if err := task.Kill(c.ctx, syscall.SIGKILL); err != nil {
			log.Error(err, "failed to send SIGKILL")
		}
		es, err := task.Wait(c.ctx)
		if err == nil {
			for {
				select {
				case exitStatus := <-es:
					if exitStatus.ExitCode() != 0 {
						log.Error(err, "failed to delete task %v", id)
					}
					return nil
				}
			}
		}

		_, err = task.Delete(c.ctx, containerd.WithProcessKill)
		if err != nil && !errdefs.IsNotFound(err) {
			log.Error(err, "failed to delete task %v", id)
		}
	}
	var delOpts []containerd.DeleteOpts
	if _, err := container.Image(c.ctx); err == nil {
		delOpts = append(delOpts, containerd.WithSnapshotCleanup)
	}

	if err := container.Delete(c.ctx, delOpts...); err != nil {
		return err
	}
	return nil
}
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

	log.Debugf("PullImageRequest: %v", request)
	resp, err := c.imageClient.PullImage(context.Background(), request)
	log.Debugf("PullImageResponse: %v", resp)
	return err
}

func (c *Client) EnsureImageExists(image ImageRef) error {
	request := &pb.ImageStatusRequest{
		Image: &pb.ImageSpec{
			Image: image.Image,
		},
	}
	status, err := c.imageClient.ImageStatus(context.Background(), request)
	if err != nil {
		return err
	}
	if status.Image == nil {
		log.Infof("Image %s not found, pulling...", image.Image)
		if err := c.Pull(image); err != nil {
			return err
		}
	}
	return nil
}
func (c *Client) ContainerList(filters ...string) ([]containerd.Container, error) {
	return c.condClient.Containers(c.ctx, filters...)
}

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
	return d.Encoded()[0:8]
}

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

func WaitContainerdReady() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(5*time.Second, func() (bool, error) {
		log.Infof("Waiting for containerd to be ready")
		_, err := NewContainedClient()
		if err == nil {
			return true, nil
		}
		log.Warnf("containerd is not available: %v", err)
		return false, nil
	}, ctx.Done())
	if err != nil {
		log.Errorf("Failed to wait containerd available: %v", err)
		return errors.Wrapf(err, "failed to wait containerd available")
	}
	return nil
}

//func (c *Client) Push(image ImageRef) (readCloser io.ReadCloser, err error) {
//	named, err := refdocker.ParseDockerRef(image.Image)
//	if err != nil {
//		return nil, err
//	}
//	// 默认不跳过仓库身份验证
//	insecure := false
//
//	ref := named.String()
//	refDomain := refdocker.Domain(named)
//
//	if image.Password == "" || image.Username == "" {
//		insecure = true
//	}
//
//	singlePlatform := platforms.DefaultStrict()
//	singlePlatformRef := ref
//	singlePlatformImg, err := converter.Convert(c.ctx, c.condClient, singlePlatformRef, ref,
//		converter.WithPlatform(singlePlatform))
//	if err != nil {
//		return nil, errors.Wrapf(err, "failed to create a tmp single-platform image %q", singlePlatformRef)
//	}
//
//	defer c.condClient.ImageService().Delete(context.TODO(), singlePlatformImg.Name)
//	logrus.Infof("pushing as a single-platform image (%s, %s)", singlePlatformImg.Target.MediaType, singlePlatformImg.Target.Digest)
//
//	pushFunc := func(r remotes.Resolver) error {
//		return push.Push(c.ctx, c.condClient, r, os.Stdout, singlePlatformRef, ref, singlePlatform, true)
//	}
//	//var dOpts []dockerconfigresolver.Opt
//	//if insecure {
//	//	logrus.Warnf("skipping verifying HTTPS certs for %q", refDomain)
//	//	dOpts = append(dOpts, dockerconfigresolver.WithSkipVerifyCerts(true))
//	//}
//	var dOpts []Opt
//	if insecure {
//		logrus.Warnf("skipping verifying HTTPS certs for %q", refDomain)
//		dOpts = append(dOpts, WithSkipVerifyCerts(true))
//	}
//	//resolver, err := dockerconfigresolver.New(refDomain, dOpts...)
//	resolver, err := New(refDomain, image.Username, image.Password, dOpts...)
//	if err != nil {
//		return nil, err
//	}
//	if err = pushFunc(resolver); err != nil {
//		if insecure {
//			logrus.WithError(err).Warnf("server %q does not seem to support HTTPS, falling back to plain HTTP", refDomain)
//			dOpts = append(dOpts, WithPlainHTTP(true))
//			resolver, err = New(refDomain, "", "", dOpts...)
//			if err != nil {
//				return nil, err
//			}
//			err := pushFunc(resolver)
//			if err != nil {
//				return nil, err
//			}
//
//		} else {
//			logrus.WithError(err).Errorf("server %q does not seem to support HTTPS", refDomain)
//			return nil, err
//		}
//	}
//	return nil, err
//}
//
//type opts struct {
//	plainHTTP       bool
//	skipVerifyCerts bool
//}
//
//// Opt for New
//type Opt func(*opts)
//
//// refHostname is like "docker.io".
//
//func New(refHostname, username, password string, optFuncs ...Opt) (remotes.Resolver, error) {
//	var o opts
//	for _, of := range optFuncs {
//		of(&o)
//	}
//	var authzOpts []docker.AuthorizerOpt
//	var insecureClient *http.Client
//	if o.skipVerifyCerts {
//		insecureClient = newInsecureClient()
//		authzOpts = append(authzOpts, docker.WithAuthClient(insecureClient))
//	}
//	if authCreds, err := NewAuthCreds(refHostname, username, password); err != nil {
//		return nil, err
//	} else {
//		authzOpts = append(authzOpts, docker.WithAuthCreds(authCreds))
//	}
//	authz := docker.NewDockerAuthorizer(authzOpts...)
//	plainHTTPFunc := docker.MatchLocalhost
//	if o.plainHTTP {
//		plainHTTPFunc = docker.MatchAllHosts
//	}
//	regOpts := []docker.RegistryOpt{
//		docker.WithAuthorizer(authz),
//		docker.WithPlainHTTP(plainHTTPFunc),
//	}
//	if o.skipVerifyCerts {
//		regOpts = append(regOpts, docker.WithClient(insecureClient))
//	}
//	resovlerOpts := docker.ResolverOptions{
//		Hosts: docker.ConfigureDefaultRegistries(regOpts...),
//	}
//	resolver := docker.NewResolver(resovlerOpts)
//	return resolver, nil
//}
//func newInsecureClient() *http.Client {
//	tr := &http.Transport{
//		TLSClientConfig: &tls.Config{
//			InsecureSkipVerify: true,
//		},
//	}
//	return &http.Client{
//		Transport: tr,
//	}
//}
//
//// AuthCreds is for docker.WithAuthCreds
//type AuthCreds func(string) (string, string, error)
//
//// AuthCreds can be nil.
//
//func NewAuthCreds(refHostname, username, password string) (AuthCreds, error) {
//	var credFunc AuthCreds
//	authConfigHostnames := []string{refHostname}
//	if refHostname == "docker.io" || refHostname == "registry-1.docker.io" {
//		// "docker.io" appears as ""https://index.docker.io/v1/" in ~/.docker/config.json .
//		// GetAuthConfig takes the hostname part as the argument: "index.docker.io"
//		authConfigHostnames = append([]string{"index.docker.io"}, refHostname)
//	}
//
//	for _, authConfigHostname := range authConfigHostnames {
//		// GetAuthConfig does not raise an error on ENOENT
//		ac := dockercliconfigtypes.AuthConfig{}
//
//		if authConfigHostname == refHostname {
//			ac.Username = username
//			ac.Password = password
//			ac.ServerAddress = refHostname
//			credFunc = func(credFuncArg string) (string, string, error) {
//				return ac.Username, ac.Password, nil
//			}
//			return credFunc, nil
//		}
//	}
//	// credsFunc can be nil here
//	return credFunc, nil
//}
//
//// WithSkipVerifyCerts skips verifying TLS certs
//func WithSkipVerifyCerts(b bool) Opt {
//	return func(o *opts) {
//		o.skipVerifyCerts = b
//	}
//}
//
//// WithPlainHTTP enables insecure plain HTTP
//func WithPlainHTTP(b bool) Opt {
//	return func(o *opts) {
//		o.plainHTTP = b
//	}
//}
