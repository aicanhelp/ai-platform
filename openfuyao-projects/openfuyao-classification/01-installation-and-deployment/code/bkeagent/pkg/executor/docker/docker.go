package docker

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/registry"
	dockerapi "github.com/docker/docker/client"
	dockerConfig "github.com/docker/docker/daemon/config"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

type DockerClient interface {
	Pull(image ImageRef) error
	Push(image ImageRef) (readCloser io.ReadCloser, err error)
	Run(cs ContainerSpec) error
	EnsureImageExists(image ImageRef) error
	EnsureContainerRun(containerId string) (bool, error)
	RemoveContainer(id string) error
	Ping() error
}

type Client struct {
	Client *dockerapi.Client
	ctx    context.Context
}

type ImageRef struct {
	Image    string `json:"image"`
	Username string `json:"username"`
	Password string `json:"password"`
}

type ContainerSpec struct {
	ContainerConfig  *container.Config
	HostConfig       *container.HostConfig
	NetworkingConfig *network.NetworkingConfig
	Platform         *specs.Platform
	ContainerName    string
}

const (
	dockerSock                 = "/var/run/docker.sock"
	DockerDaemonConfigFilePath = "/etc/docker/daemon.json"

	OverrideDockerConfig = `
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd
`
)

var (
	runtimeKeyPathMap = map[string]string{
		"runc":     "/usr/local/sbin/runc",
		"richrunc": "/usr/local/beyondvm/runc",
		"kata":     "",
	}
	//go:embed tlscert.sh
	tlsCertScript string
)

func NewDockerClient() (DockerClient, error) {
	if !utils.Exists(dockerSock) {
		return nil, errors.New("Docker service does not exist. ")
	}

	ctx := context.Background()
	cli, err := dockerapi.NewClientWithOpts(dockerapi.FromEnv, dockerapi.WithAPIVersionNegotiation())
	if err != nil {
		log.Error("get container runtime client err:", err)
		return nil, err
	}

	c := &Client{
		Client: cli,
		ctx:    ctx,
	}
	if c.Ping() != nil {
		return nil, errors.New("docker service is not running")
	}

	return c, nil
}

func (c *Client) Close() {
	_ = c.Client.Close()
}

func (c *Client) Ping() error {
	p, err := c.Client.Ping(c.ctx)
	if err == nil {
		log.Debugf("docker api version: %s", p.APIVersion)
	}
	return err
}

// Pull pulls an image from the Docker registry.
func (c *Client) Pull(img ImageRef) error {
	imagePullOptions := image.PullOptions{}
	if len(img.Username) != 0 && len(img.Password) != 0 {
		authConfig := registry.AuthConfig{
			Username: img.Username,
			Password: img.Password,
		}
		encodedJSON, err := json.Marshal(authConfig)
		if err != nil {
			log.Errorf(" encoded docker RegistryAuth err: %v", err)
			return err
		}
		authStr := base64.URLEncoding.EncodeToString(encodedJSON)
		imagePullOptions.RegistryAuth = authStr
	}

	reader, err := c.Client.ImagePull(c.ctx, img.Image, imagePullOptions)
	if err != nil {
		log.Errorf("docker pull image %s error %v", img.Image, err)
		return err
	}
	io.Copy(os.Stdout, reader)

	return nil
}

// Push pushes an image to the Docker registry.
func (c *Client) Push(img ImageRef) (io.ReadCloser, error) {
	imagePushOptions := image.PushOptions{}
	if len(img.Username) != 0 && len(img.Password) != 0 {
		authConfig := registry.AuthConfig{
			Username: img.Username,
			Password: img.Password,
		}
		encodedJSON, err := json.Marshal(authConfig)
		if err != nil {
			log.Errorf(" encoded docker RegistryAuth err: %v", err)
			return nil, err
		}
		authStr := base64.URLEncoding.EncodeToString(encodedJSON)
		imagePushOptions.RegistryAuth = authStr
	}

	closer, err := c.Client.ImagePush(c.ctx, img.Image, imagePushOptions)
	if err != nil {
		return nil, err
	}
	return closer, nil
}

func (c *Client) Run(cs ContainerSpec) error {
	resp, err := c.Client.ContainerCreate(c.ctx, cs.ContainerConfig, cs.HostConfig, cs.NetworkingConfig, cs.Platform, cs.ContainerName)
	if err != nil {
		log.Error(err)
	}
	if err := c.Client.ContainerStart(c.ctx, resp.ID, container.StartOptions{}); err != nil {
		if err != nil {
			log.Error(err)
		}
	}
	log.Infof("container ID %s", resp.ID)
	return nil
}

func (c *Client) EnsureImageExists(image ImageRef) error {
	imageInspect, _, _ := c.Client.ImageInspectWithRaw(c.ctx, image.Image)
	if imageInspect.ID == "" {
		log.Infof("Image %s not found, pulling...", image.Image)
		err := c.Pull(image)
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) EnsureContainerRun(containerId string) (bool, error) {
	containerInfo, _ := c.Client.ContainerInspect(c.ctx, containerId)
	// Check whether the mirror warehouse already exists
	if containerInfo.ContainerJSONBase != nil {
		if containerInfo.State.Running {
			return true, nil
		}
		err := c.Client.ContainerStart(c.ctx, containerInfo.ID, container.StartOptions{})
		if err == nil {
			return true, nil
		}
		err = c.Client.ContainerRemove(c.ctx, containerInfo.ID, container.RemoveOptions{Force: true})
		if err != nil {
			return false, err
		}
	}
	return false, nil
}

func (c *Client) RemoveContainer(id string) error {
	err := c.Client.ContainerRemove(c.ctx, id, container.RemoveOptions{Force: true})
	if err != nil {
		return err
	}
	return nil
}

func GetDockerDaemonConfig(path string) (*dockerConfig.Config, error) {
	if path == "" {
		path = DockerDaemonConfigFilePath
	}
	cfg := &dockerConfig.Config{}
	if !utils.Exists(path) {
		if !utils.Exists(filepath.Dir(path)) {
			err := os.MkdirAll(filepath.Dir(path), 0755)
			if err != nil {
				return nil, errors.Wrapf(err, "create docker daemon config dir %s failed", path)
			}
		}
		_, err := os.OpenFile(path, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return nil, errors.Wrapf(err, "create docker daemon config file %s failed", path)
		}
		return cfg, nil

	}
	f, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "read docker daemon config file %s failed", path)
	}
	err = json.Unmarshal(f, &cfg)
	if err != nil {
		return nil, errors.Wrapf(err, "unmarshal docker daemon config file %s failed", path)
	}
	if err = dockerConfig.Validate(cfg); err != nil {
		return nil, errors.Wrapf(err, "validate docker daemon config file %s failed", path)
	}
	return cfg, nil
}

func ConfigDockerDaemon(cgroupDriver, lowLevelRuntime string, dataRoot string, enableTls bool, tlsHost string, insecureRegis ...string) (err error) {
	if err := OverrideDockerService(); err != nil {
		return errors.New("override docker service failed")
	}
	if !utils.Exists("/etc/docker") {
		if err = os.MkdirAll("/etc/docker", 0755); err != nil {
			return errors.Errorf("create /etc/docker dir err: %v", err)
		}
	}

	if err = BaseConfig(); err != nil {
		return errors.Errorf("config docker base err: %v", err)
	}

	if len(insecureRegis) > 0 || insecureRegis != nil {
		if err = ConfigInsecureRegistries(insecureRegis); err != nil {
			return errors.Errorf("config docker insecure registries err: %v", err)
		}
	}

	if cgroupDriver != "" {
		if err = ConfigCgroupDriver(cgroupDriver); err != nil {
			return errors.Errorf("config docker cgroup driver err: %v", err)
		}
	}

	if lowLevelRuntime != "" {
		if err = ConfigRuntime(lowLevelRuntime); err != nil {
			return errors.Errorf("config docker low level runtime err: %v", err)
		}
	}

	if dataRoot != "" {
		if !utils.Exists(dataRoot) {
			if err = os.MkdirAll(dataRoot, 0711); err != nil {
				return err
			}
		}
		if err = ConfigDataRoot(dataRoot); err != nil {
			return errors.Errorf("config docker data root err: %v", err)
		}
	}

	if enableTls {
		if err = ConfigDockerTls(tlsHost); err != nil {
			return errors.Errorf("config docker tls err: %v", err)
		}
	}

	return nil
}

func OverrideDockerService() error {
	log.Info("override docker service")
	if !utils.Exists("/etc/systemd/system/docker.service.d") {
		err := os.MkdirAll("/etc/systemd/system/docker.service.d", 0755)
		if err != nil {
			return err
		}
	}
	err := os.WriteFile("/etc/systemd/system/docker.service.d/docker.conf", []byte(OverrideDockerConfig), 0644)
	if err != nil {
		return err
	}
	return nil
}

func ConfigInsecureRegistries(registries []string) error {
	for i, registry := range registries {
		if registry == "" {
			registries = append(registries[:i], registries[i+1:]...)
			continue
		}
	}
	if len(registries) == 0 {
		return nil
	}
	log.Infof("config docker insecure registries, registries: %v", registries)
	var cfg interface{}
	// if daemon.json does not exist, create it,and add insecure-registries、exec-opts
	if !utils.Exists(DockerDaemonConfigFilePath) {
		// create daemon.json
		if !utils.Exists(filepath.Dir(DockerDaemonConfigFilePath)) {
			err := os.MkdirAll(filepath.Dir(DockerDaemonConfigFilePath), 0755)
			if err != nil {
				return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
			}
		}
		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		cfg = map[string]interface{}{
			"insecure-registries": registries,
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"insecure-registries": registries,
		}
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}
	if v, ok := cfg.(map[string]interface{}); ok {
		if _, ok := v["insecure-registries"]; ok {
			if vr, ok := v["insecure-registries"].([]interface{}); ok {
				configRegistries := []string{}
				for _, r := range vr {
					t := r.(string)
					configRegistries = append(configRegistries, t)
				}
				registries = append(registries, configRegistries...)
				registries = utils.UniqueStringSlice(registries)
				v["insecure-registries"] = registries
			}
		} else {
			v["insecure-registries"] = registries
		}
	}
	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(DockerDaemonConfigFilePath, b, 0644); err != nil {
		return err
	}
	return nil
}

func ConfigCgroupDriver(driver string) error {
	log.Infof("config docker cgroup driver, driver: %s", driver)
	exceptDriver := fmt.Sprintf("native.cgroupdriver=%s", driver)

	log.Infof("ensure docker cgroup driver is %s, if not, change it in %s", driver, DockerDaemonConfigFilePath)
	var cfg interface{}
	if !utils.Exists(DockerDaemonConfigFilePath) {
		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		cfg = map[string]interface{}{
			"exec-opts": []string{exceptDriver},
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}

	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"exec-opts": []string{exceptDriver},
		}
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}
	if v, ok := cfg.(map[string]interface{}); ok {
		if opts, ok := v["exec-opts"].([]interface{}); ok {
			for i, opt := range opts {
				t := opt.(string)
				if t == exceptDriver {
					log.Debugf("docker cgroup driver is %s, no need to change", driver)
					return nil
				}
				if strings.Contains(t, "native.cgroupdriver") {
					op := strings.Split(t, "=")
					op[1] = driver
					opts[i] = strings.Join(op, "=")
				}
			}
		} else {
			v["exec-opts"] = []string{exceptDriver}
		}
	}
	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(DockerDaemonConfigFilePath, b, 0644); err != nil {
		return err
	}
	return nil
}

func ConfigRuntime(runtime string) error {
	log.Infof("config docker runtime, runtime: %s", runtime)
	var cfg interface{}
	log.Infof("ensure docker default runtime is %s, if not, change it in %s", runtime, DockerDaemonConfigFilePath)
	if !utils.Exists(DockerDaemonConfigFilePath) {
		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		if runtime == "runc" {
			cfg = map[string]interface{}{
				"default-runtime": runtime,
			}
		} else {
			cfg = map[string]interface{}{
				"default-runtime": runtime,
				"runtimes": map[string]interface{}{
					runtime: map[string]interface{}{
						"path": runtimeKeyPathMap[runtime],
					},
				},
			}
		}

		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}

	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		if runtime == "runc" {
			cfg = map[string]interface{}{
				"default-runtime": runtime,
			}
		} else {
			cfg = map[string]interface{}{
				"default-runtime": runtime,
				"runtimes": map[string]interface{}{
					runtime: map[string]interface{}{
						"path": runtimeKeyPathMap[runtime],
					},
				},
			}
		}
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}

	if v, ok := cfg.(map[string]interface{}); ok {
		if defaultRuntime, ok := v["default-runtime"]; ok {
			if defaultRuntime == runtime {
				log.Debugf("docker runtime is %s, no need to change", runtime)
			} else {
				v["default-runtime"] = runtime
			}
		} else {
			v["default-runtime"] = runtime
		}

		if runtimes, ok := v["runtimes"].(map[string]interface{}); ok {
			if _, ok := runtimes[runtime]; ok {
				log.Debugf("runtime %s is exists, no need to add", runtime)
			}
			if runtime != "runc" {
				runtimes[runtime] = map[string]interface{}{
					"path": runtimeKeyPathMap[runtime],
				}
			}
		} else {
			if runtime != "runc" {
				v["runtimes"] = map[string]interface{}{
					runtime: map[string]interface{}{
						"path": runtimeKeyPathMap[runtime],
					},
				}
			}
		}
	}
	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(DockerDaemonConfigFilePath, b, 0644); err != nil {
		return err
	}
	return nil
}

func ConfigDataRoot(dataRoot string) error {
	log.Infof("config docker data root, data root: %s", dataRoot)
	var cfg interface{}
	// if daemon.json does not exist, create it,and add insecure-registries、exec-opts
	if !utils.Exists(DockerDaemonConfigFilePath) {
		// create daemon.json
		if !utils.Exists(filepath.Dir(DockerDaemonConfigFilePath)) {
			err := os.MkdirAll(filepath.Dir(DockerDaemonConfigFilePath), 0755)
			if err != nil {
				return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
			}
		}

		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		cfg = map[string]interface{}{
			"data-root": dataRoot,
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"data-root": dataRoot,
		}
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}
	if v, ok := cfg.(map[string]interface{}); ok {
		if root, ok := v["data-root"]; ok {
			if root == dataRoot {
				log.Debugf("docker data-root is %s, no need to change", dataRoot)
			}
		} else {
			v["data-root"] = dataRoot
		}
	}
	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(DockerDaemonConfigFilePath, b, 0644); err != nil {
		return err
	}
	return nil
}

func ConfigDockerTls(tlsHost string) error {
	log.Infof("config docker tls, tls host: %s", tlsHost)
	if tlsHost == "" {
		tlsHost = "127.0.0.1"
	}
	var cfg interface{}
	// save tlscert.sh to /etc/docker/certs/tlscert.sh
	if !utils.Exists("/etc/docker/certs") {
		err := os.MkdirAll("/etc/docker/certs", 0755)
		if err != nil {
			return errors.Wrapf(err, "create docker certs dir failed")
		}
	}
	err := os.WriteFile("/etc/docker/certs/tlscert.sh", []byte(tlsCertScript), 0755)
	if err != nil {
		return errors.Wrapf(err, "write tlscert.sh failed")
	}

	executor := &exec.CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "cd /etc/docker/certs && ./tlscert.sh "+tlsHost)
	if err != nil {
		return errors.Wrapf(err, "generate docker tls cert failed, output: %s, err: %v", output, err)
	}

	// export DOCKER_CONFIG=/etc/docker/certs to /etc/profile
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", "echo 'export DOCKER_CONFIG=/etc/docker/certs' >> /etc/profile")
	if err != nil {
		log.Warnf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile failed, output: %s, err: %v", output, err)
	}
	log.Debugf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile, output: %s", output)
	// source /etc/profile use bash
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c", "source /etc/profile")
	if err != nil {
		log.Warnf("source /etc/profile failed, output: %s, err: %v", output, err)
	}

	// if daemon.json does not exist, create it,and add tls config
	if !utils.Exists(DockerDaemonConfigFilePath) {
		// create daemon.json
		if !utils.Exists(filepath.Dir(DockerDaemonConfigFilePath)) {
			err := os.MkdirAll(filepath.Dir(DockerDaemonConfigFilePath), 0755)
			if err != nil {
				return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
			}
		}

		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		cfg := map[string]interface{}{
			"tls":       true,
			"tlsverify": true,
			"tlscacert": "/etc/docker/certs/ca.pem",
			"tlscert":   "/etc/docker/certs/server-cert.pem",
			"tlskey":    "/etc/docker/certs/server-key.pem",
			"hosts":     []string{"tcp://0.0.0.0:2376", "unix:///var/run/docker.sock"},
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"tls":       true,
			"tlsverify": true,
			"tlscacert": "/etc/docker/certs/ca.pem",
			"tlscert":   "/etc/docker/certs/server-cert.pem",
			"tlskey":    "/etc/docker/certs/server-key.pem",
			"hosts":     []string{"tcp://0.0.0.0:2376", "unix:///var/run/docker.sock"},
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}
	if v, ok := cfg.(map[string]interface{}); ok {
		v["tls"] = true
		v["tlsverify"] = true
		v["tlscacert"] = "/etc/docker/certs/ca.pem"
		v["tlscert"] = "/etc/docker/certs/server-cert.pem"
		v["tlskey"] = "/etc/docker/certs/server-key.pem"

		// add unix:///var/run/docker.sock and tcp://0.0.0.0:2376
		if hosts := v["hosts"]; hosts != nil {
			if h, ok := hosts.([]interface{}); ok {
				for _, host := range h {
					if host == "unix:///var/run/docker.sock" {
						continue
					} else {
						v["hosts"] = append(h, "unix:///var/run/docker.sock")
					}
					if host == "tcp://0.0.0.0:2376" {
						continue
					} else {
						v["hosts"] = append(h, "tcp://0.0.0.0:2376")
					}
				}
			}
		} else {
			v["hosts"] = []interface{}{
				"unix:///var/run/docker.sock",
				"tcp://0.0.0.0:2376",
			}
		}
	}
	data, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return errors.Wrapf(err, "marshal docker daemon config failed")
	}
	err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
	if err != nil {
		return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	return nil
}

func BaseConfig() error {
	var cfg interface{}
	// if daemon.json does not exist, create it,and add insecure-registries、exec-opts
	if !utils.Exists(DockerDaemonConfigFilePath) {
		// create daemon.json
		if !utils.Exists(filepath.Dir(DockerDaemonConfigFilePath)) {
			err := os.MkdirAll(filepath.Dir(DockerDaemonConfigFilePath), 0755)
			if err != nil {
				return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
			}
		}
		log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
		_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, 0644)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		cfg = map[string]interface{}{
			"log-driver": "json-file",
			"log-opts": map[string]interface{}{
				"max-size": "100m",
			},
		}
		data, err := json.MarshalIndent(cfg, "", " ")
		if err != nil {
			return errors.Wrapf(err, "marshal docker daemon config failed")
		}
		err = os.WriteFile(DockerDaemonConfigFilePath, data, 0644)
		if err != nil {
			return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
		return nil
	}

	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"log-driver": "json-file",
			"log-opts": map[string]interface{}{
				"max-size": "100m",
			},
		}
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}

	if v, ok := cfg.(map[string]interface{}); ok {
		if _, ok := v["log-driver"]; !ok {
			v["log-driver"] = "json-file"
		}

		if _, ok := v["log-opts"]; !ok {
			v["log-opts"] = map[string]interface{}{
				"max-size": "100m",
			}
		} else {
			if v, ok := v["log-opts"].(map[string]interface{}); ok {
				if _, ok := v["max-size"]; !ok {
					v["max-size"] = "100m"
				}
			}
		}
	}

	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(DockerDaemonConfigFilePath, b, 0644); err != nil {
		return err
	}
	return nil
}

func WaitDockerReady() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(5*time.Second, func() (bool, error) {
		log.Infof("Waiting for Docker to be ready")
		_, err := NewDockerClient()
		if err == nil {
			return true, nil
		}
		log.Warnf("Docker is not available: %v", err)
		return false, nil
	}, ctx.Done())
	if err != nil {
		log.Errorf("Failed to wait Docker available: %v", err)
		return errors.Wrapf(err, "failed to wait Docker available")
	}
	return nil
}
