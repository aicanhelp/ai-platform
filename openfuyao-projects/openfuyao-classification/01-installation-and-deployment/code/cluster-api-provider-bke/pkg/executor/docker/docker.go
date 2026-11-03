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

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// DockerClient is an interface for Docker operations
type DockerClient interface {
	// Pull pulls an image from the Docker registry
	Pull(image ImageRef) error
	// Push pushes an image to the Docker registry
	Push(image ImageRef) (readCloser io.ReadCloser, err error)
	// Run creates and starts a container
	Run(cs ContainerSpec) error
	// EnsureImageExists checks if an image exists and pulls it if not
	EnsureImageExists(image ImageRef) error
	// EnsureContainerRun ensures a container is running
	EnsureContainerRun(containerId string) (bool, error)
	// RemoveContainer removes a container
	RemoveContainer(id string) error
	// Ping checks if Docker service is running
	Ping() error
}

// Client represents a Docker client
type Client struct {
	Client *dockerapi.Client
	ctx    context.Context
}

// ImageRef represents Docker image reference with authentication
type ImageRef struct {
	Image    string `json:"image"`
	Username string `json:"username"`
	Password string `json:"password"`
}

// ContainerSpec represents Docker container specification
type ContainerSpec struct {
	ContainerConfig  *container.Config
	HostConfig       *container.HostConfig
	NetworkingConfig *network.NetworkingConfig
	Platform         *specs.Platform
	ContainerName    string
}

// DockerDaemonConfig represents parameters for ConfigDockerDaemon function
type DockerDaemonConfig struct {
	CgroupDriver       string
	LowLevelRuntime    string
	DataRoot           string
	EnableTls          bool
	TlsHost            string
	InsecureRegistries []string
}

const (
	dockerSock = "/var/run/docker.sock"
	// DockerDaemonConfigFilePath is the path to Docker daemon configuration file
	DockerDaemonConfigFilePath = "/etc/docker/daemon.json"

	// OverrideDockerConfig is the override configuration for Docker service
	OverrideDockerConfig = `
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd
`

	// DefaultDirFileMode is the default file mode for directory creation
	DefaultDirFileMode = 0755

	// DefaultFileMode is the default file mode for file creation
	DefaultFileMode = 0644

	// DataRootDirFileMode is the file mode for data root directory creation
	DataRootDirFileMode = 0711

	// DockerReadyTimeoutMinutes is the timeout for waiting Docker to be ready in minutes
	DockerReadyTimeoutMinutes = 2

	// DockerReadyPollInterval is the interval for polling Docker readiness
	DockerReadyPollInterval = 5 * time.Second

	// MinSplitParts is the minimum number of parts after splitting a string with '=' delimiter
	MinSplitParts = 2
)

var (
	// runtimeKeyPathMap maps runtime names to their binary paths
	runtimeKeyPathMap = map[string]string{
		"runc":     "/usr/local/sbin/runc",
		"richrunc": "/usr/local/beyondvm/runc",
		"kata":     "",
	}
	//go:embed tlscert.sh
	// tlsCertScript contains the TLS certificate generation script
	tlsCertScript string
)

// NewDockerClient creates a new Docker client
func NewDockerClient() (DockerClient, error) {
	if !utils.Exists(dockerSock) {
		return nil, errors.New("Docker service does not exist. ")
	}

	ctx := context.Background()
	cli, err := dockerapi.NewClientWithOpts(dockerapi.FromEnv, dockerapi.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("create docker client: %w", err)
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

// Close closes the Docker client connection
func (c *Client) Close() error {
	return c.Client.Close()
}

// Ping checks if Docker service is running
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
			return fmt.Errorf("encode docker registry auth for image %s: %w", img.Image, err)
		}
		authStr := base64.URLEncoding.EncodeToString(encodedJSON)
		imagePullOptions.RegistryAuth = authStr
	}

	reader, err := c.Client.ImagePull(c.ctx, img.Image, imagePullOptions)
	if err != nil {
		return fmt.Errorf("pull docker image %s: %w", img.Image, err)
	}
	written, err := io.Copy(os.Stdout, reader)
	if err != nil {
		return fmt.Errorf("read docker pull response for image %s: %w", img.Image, err)
	}
	if written < 0 {
		// This is just to use the written variable, though it should never be negative
		return errors.New("unexpected negative byte count")
	}

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
			return nil, fmt.Errorf("encode docker registry auth for image %s: %w", img.Image, err)
		}
		authStr := base64.URLEncoding.EncodeToString(encodedJSON)
		imagePushOptions.RegistryAuth = authStr
	}

	closer, err := c.Client.ImagePush(c.ctx, img.Image, imagePushOptions)
	if err != nil {
		return nil, fmt.Errorf("push docker image %s: %w", img.Image, err)
	}
	return closer, nil
}

// Run creates and starts a container
func (c *Client) Run(cs ContainerSpec) error {
	resp, err := c.Client.ContainerCreate(c.ctx, cs.ContainerConfig, cs.HostConfig,
		cs.NetworkingConfig, cs.Platform, cs.ContainerName)
	if err != nil {
		return fmt.Errorf("create container %s: %w", cs.ContainerName, err)
	}
	if err := c.Client.ContainerStart(c.ctx, resp.ID, container.StartOptions{}); err != nil {
		return fmt.Errorf("start container %s (id=%s): %w", cs.ContainerName, resp.ID, err)
	}
	log.Debugf("container created and started: name=%s id=%s", cs.ContainerName, resp.ID)
	return nil
}

// EnsureImageExists checks if an image exists and pulls it if not
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

// EnsureContainerRun ensures a container is running
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

// RemoveContainer removes a container
func (c *Client) RemoveContainer(id string) error {
	err := c.Client.ContainerRemove(c.ctx, id, container.RemoveOptions{Force: true})
	if err != nil {
		return err
	}
	return nil
}

// GetDockerDaemonConfig retrieves the Docker daemon configuration from the specified path
func GetDockerDaemonConfig(path string) (*dockerConfig.Config, error) {
	if path == "" {
		path = DockerDaemonConfigFilePath
	}
	cfg := &dockerConfig.Config{}
	if !utils.Exists(path) {
		if !utils.Exists(filepath.Dir(path)) {
			err := os.MkdirAll(filepath.Dir(path), DefaultDirFileMode)
			if err != nil {
				return nil, errors.Wrapf(err, "create docker daemon config dir %s failed", path)
			}
		}
		_, err := os.OpenFile(path, os.O_RDONLY|os.O_CREATE, DefaultFileMode)
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

// ConfigDockerDaemon configures the Docker daemon with the specified parameters
func ConfigDockerDaemon(params DockerDaemonConfig) (err error) {
	cgroupDriver := params.CgroupDriver
	lowLevelRuntime := params.LowLevelRuntime
	dataRoot := params.DataRoot
	enableTls := params.EnableTls
	tlsHost := params.TlsHost
	insecureRegis := params.InsecureRegistries

	if err := OverrideDockerService(); err != nil {
		return errors.New("override docker service failed")
	}
	if !utils.Exists("/etc/docker") {
		if err = os.MkdirAll("/etc/docker", DefaultDirFileMode); err != nil {
			return fmt.Errorf("create /etc/docker dir err: %w", err)
		}
	}

	if err = BaseConfig(); err != nil {
		return fmt.Errorf("config docker base err: %w", err)
	}

	if len(insecureRegis) > 0 || insecureRegis != nil {
		if err = ConfigInsecureRegistries(insecureRegis); err != nil {
			return fmt.Errorf("config docker insecure registries err: %w", err)
		}
	}

	if cgroupDriver != "" {
		if err = ConfigCgroupDriver(cgroupDriver); err != nil {
			return fmt.Errorf("config docker cgroup driver err: %w", err)
		}
	}

	if lowLevelRuntime != "" {
		if err = ConfigRuntime(lowLevelRuntime); err != nil {
			return fmt.Errorf("config docker low level runtime err: %w", err)
		}
	}

	if dataRoot != "" {
		if !utils.Exists(dataRoot) {
			if err = os.MkdirAll(dataRoot, DataRootDirFileMode); err != nil {
				return err
			}
		}
		if err = ConfigDataRoot(dataRoot); err != nil {
			return fmt.Errorf("config docker data root err: %w", err)
		}
	}

	if enableTls {
		if err = ConfigDockerTls(tlsHost); err != nil {
			return fmt.Errorf("config docker tls err: %w", err)
		}
	}

	return nil
}

// OverrideDockerService overrides the Docker service configuration
func OverrideDockerService() error {
	log.Info("override docker service")
	if !utils.Exists("/etc/systemd/system/docker.service.d") {
		err := os.MkdirAll("/etc/systemd/system/docker.service.d", DefaultDirFileMode)
		if err != nil {
			return err
		}
	}
	err := os.WriteFile("/etc/systemd/system/docker.service.d/docker.conf", []byte(OverrideDockerConfig), DefaultFileMode)
	if err != nil {
		return err
	}
	return nil
}

// ConfigInsecureRegistries configures Docker insecure registries
func ConfigInsecureRegistries(registries []string) error {
	// Clean up empty registries
	registries = cleanRegistries(registries)
	if len(registries) == 0 {
		return nil
	}

	log.Infof("config docker insecure registries, registries: %v", registries)

	// if daemon.json does not exist, create it
	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createDockerConfigWithRegistries(registries)
	}

	// Update existing configuration
	return updateDockerConfigWithRegistries(registries)
}

// cleanRegistries removes empty entries from the registries slice
func cleanRegistries(registries []string) []string {
	for i := len(registries) - 1; i >= 0; i-- {
		if registries[i] == "" {
			registries = append(registries[:i], registries[i+1:]...)
		}
	}
	return registries
}

// createDockerConfigWithRegistries creates a new Docker configuration with the given registries
func createDockerConfigWithRegistries(registries []string) error {
	if err := ensureDockerConfigFileExists(); err != nil {
		return err
	}
	cfg := map[string]interface{}{
		"insecure-registries": registries,
	}
	return writeDockerConfigToFile(cfg)
}

// updateDockerConfigWithRegistries updates an existing Docker configuration with the given registries
func updateDockerConfigWithRegistries(registries []string) error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}

	var cfg interface{}
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
		// Handle insecure registries
		registries = handleInsecureRegistries(v, registries)
	}

	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}

	if err = os.WriteFile(DockerDaemonConfigFilePath, b, DefaultFileMode); err != nil {
		return err
	}

	return nil
}

// handleInsecureRegistries handles the insecure registries configuration
func handleInsecureRegistries(v map[string]interface{}, registries []string) []string {
	if v == nil {
		return registries
	}

	if _, ok := v["insecure-registries"]; ok {
		return handleExistingInsecureRegistries(v, registries)
	} else {
		v["insecure-registries"] = registries
	}
	return registries
}

// handleExistingInsecureRegistries handles existing insecure registries configuration
func handleExistingInsecureRegistries(v map[string]interface{}, registries []string) []string {
	// 检查v是否为nil，避免空指针解引用
	if v == nil {
		log.Warnf("Received nil map for insecure registries configuration")
		return registries
	}

	if vr, ok := v["insecure-registries"].([]interface{}); ok {
		var configRegistries []string
		for _, r := range vr {
			if t, ok := r.(string); ok {
				configRegistries = append(configRegistries, t)
			} else {
				log.Warnf("Registry configuration is not a string, got: %T", r)
			}
		}
		registries = append(registries, configRegistries...)
		registries = utils.UniqueStringSlice(registries)
		v["insecure-registries"] = registries
	}
	return registries
}

// ConfigCgroupDriver configures Docker cgroup driver
func ConfigCgroupDriver(driver string) error {
	log.Infof("config docker cgroup driver, driver: %s", driver)
	exceptDriver := fmt.Sprintf("native.cgroupdriver=%s", driver)
	log.Infof("ensure docker cgroup driver is %s, if not, change it in %s", driver, DockerDaemonConfigFilePath)

	// if daemon.json does not exist, create it
	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createDockerConfigWithCgroupDriver(exceptDriver)
	}

	// Update existing configuration
	return updateDockerConfigWithCgroupDriver(driver, exceptDriver)
}

// createDockerConfigWithCgroupDriver creates a new Docker configuration with the given cgroup driver
func createDockerConfigWithCgroupDriver(exceptDriver string) error {
	if err := ensureDockerConfigFileExists(); err != nil {
		return err
	}
	cfg := map[string]interface{}{
		"exec-opts": []string{exceptDriver},
	}
	return writeDockerConfig(cfg)
}

// updateDockerConfigWithCgroupDriver updates an existing Docker configuration with the given cgroup driver
func updateDockerConfigWithCgroupDriver(driver, exceptDriver string) error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}

	var cfg interface{}
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
		// Handle exec-opts
		if shouldReturn := handleExecOpts(v, driver, exceptDriver); shouldReturn {
			return nil
		}
	}

	return writeDockerConfigSimple(cfg)
}

// handleExecOpts handles the exec-opts configuration
func handleExecOpts(v map[string]interface{}, driver, exceptDriver string) bool {
	if v == nil {
		return false
	}
	if opts, ok := v["exec-opts"].([]interface{}); ok {
		for i, opt := range opts {
			if shouldReturn := processOpt(opt, opts, i, driver, exceptDriver); shouldReturn {
				return true
			}
		}
	} else {
		v["exec-opts"] = []string{exceptDriver}
	}
	return false
}

// processOpt processes a single option and returns whether to exit early
func processOpt(opt interface{}, opts []interface{}, i int, driver, exceptDriver string) bool {
	if t, ok := opt.(string); ok {
		if t == exceptDriver {
			log.Debugf("docker cgroup driver is %s, no need to change", driver)
			return true
		}
		if strings.Contains(t, "native.cgroupdriver") {
			updateCgroupDriverValue(t, opts, i, driver)
		}
	} else {
		log.Warnf("Docker exec-opts configuration is not a string, got: %T", opt)
	}
	return false
}

// updateCgroupDriverValue updates the cgroup driver value in the option
func updateCgroupDriverValue(t string, opts []interface{}, i int, driver string) {
	op := strings.Split(t, "=")
	if len(op) >= MinSplitParts {
		op[1] = driver
		opts[i] = strings.Join(op, "=")
	}
}

// ConfigRuntime configures the Docker runtime
func ConfigRuntime(runtime string) error {
	log.Infof("config docker runtime, runtime: %s", runtime)
	log.Infof("ensure docker default runtime is %s, if not, change it in %s", runtime, DockerDaemonConfigFilePath)

	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createDockerRuntimeConfig(runtime)
	}

	return updateDockerRuntimeConfig(runtime)
}

// createDockerRuntimeConfig creates the Docker runtime configuration
func createDockerRuntimeConfig(runtime string) error {
	log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
	_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, DefaultFileMode)
	if err != nil {
		return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}

	cfg := buildRuntimeConfig(runtime)

	return writeDockerConfig(cfg)
}

// updateDockerRuntimeConfig updates the existing Docker runtime configuration
func updateDockerRuntimeConfig(runtime string) error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}

	cfg := parseConfigOrBuildNew(runtime, f)

	if v, ok := cfg.(map[string]interface{}); ok {
		updateRuntimeConfiguration(v, runtime)
	}

	return writeDockerConfig(cfg)
}

// buildRuntimeConfig builds the runtime configuration
func buildRuntimeConfig(runtime string) map[string]interface{} {
	if runtime == "runc" {
		return map[string]interface{}{
			"default-runtime": runtime,
		}
	}
	return map[string]interface{}{
		"default-runtime": runtime,
		"runtimes": map[string]interface{}{
			runtime: map[string]interface{}{
				"path": runtimeKeyPathMap[runtime],
			},
		},
	}
}

// parseConfigOrBuildNew parses existing config or builds new one
func parseConfigOrBuildNew(runtime string, f []byte) interface{} {
	var cfg interface{}
	if len(f) == 0 {
		return buildRuntimeConfig(runtime)
	}
	err := json.Unmarshal(f, &cfg)
	if err != nil {
		return buildRuntimeConfig(runtime)
	}
	return cfg
}

// updateRuntimeConfiguration updates the runtime configuration in the config map
func updateRuntimeConfiguration(v map[string]interface{}, runtime string) {
	if v == nil {
		return
	}
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

// writeDockerConfigToFile writes the configuration to the Docker config file
func writeDockerConfigToFile(cfg interface{}) error {
	data, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return errors.Wrapf(err, "marshal docker daemon config failed")
	}
	err = os.WriteFile(DockerDaemonConfigFilePath, data, DefaultFileMode)
	if err != nil {
		return errors.Wrapf(err, "write docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	return nil
}

// writeDockerConfig writes the configuration to the Docker config file
func writeDockerConfig(cfg interface{}) error {
	return writeDockerConfigToFile(cfg)
}

// writeDockerConfigSimple writes the configuration to the Docker config file without error wrapping
func writeDockerConfigSimple(cfg interface{}) error {
	b, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(DockerDaemonConfigFilePath, b, DefaultFileMode); err != nil {
		return err
	}

	return nil
}

// ConfigDataRoot configures the Docker data root directory
func ConfigDataRoot(dataRoot string) error {
	log.Infof("config docker data root, data root: %s", dataRoot)

	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createDockerDataRootConfig(dataRoot)
	}

	return updateDockerDataRootConfig(dataRoot)
}

// createDockerDataRootConfig creates the Docker data root configuration
func createDockerDataRootConfig(dataRoot string) error {
	if err := ensureDockerConfigFileExists(); err != nil {
		return err
	}
	cfg := map[string]interface{}{
		"data-root": dataRoot,
	}
	return writeDockerConfig(cfg)
}

// updateDockerDataRootConfig updates the existing Docker data root configuration
func updateDockerDataRootConfig(dataRoot string) error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}

	cfg, err := parseOrCreateConfig(f, dataRoot)
	if err != nil {
		return err
	}

	// Process the config map to handle data-root
	cfg = processConfigDataRoot(cfg, dataRoot)

	return writeDockerConfigSimple(cfg)
}

// parseOrCreateConfig parses the existing config or creates a new one if file is empty
func parseOrCreateConfig(f []byte, dataRoot string) (interface{}, error) {
	var cfg interface{}
	if len(f) == 0 {
		cfg = map[string]interface{}{
			"data-root": dataRoot,
		}
	} else {
		err := json.Unmarshal(f, &cfg)
		if err != nil {
			return nil, errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}
	return cfg, nil
}

// processConfigDataRoot processes the data-root field in the config
func processConfigDataRoot(cfg interface{}, dataRoot string) interface{} {
	if v, ok := cfg.(map[string]interface{}); ok && v != nil {
		if root, ok := v["data-root"]; ok {
			if root == dataRoot {
				log.Debugf("docker data-root is %s, no need to change", dataRoot)
			}
		} else {
			v["data-root"] = dataRoot
		}
	}
	return cfg
}

// ConfigDockerTls configures TLS for the Docker daemon
func ConfigDockerTls(tlsHost string) error {
	log.Infof("config docker tls, tls host: %s", tlsHost)
	if tlsHost == "" {
		tlsHost = "127.0.0.1"
	}

	// Setup TLS certificate generation
	if err := setupTLSCerts(tlsHost); err != nil {
		return err
	}

	// Configure TLS in Docker daemon config
	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createDockerTLSConfig()
	}

	return updateDockerTLSConfig()
}

// setupTLSCerts sets up TLS certificates
func setupTLSCerts(tlsHost string) error {
	// save tlscert.sh to /etc/docker/certs/tlscert.sh
	if !utils.Exists("/etc/docker/certs") {
		err := os.MkdirAll("/etc/docker/certs", DefaultDirFileMode)
		if err != nil {
			return errors.Wrapf(err, "create docker certs dir failed")
		}
	}
	err := os.WriteFile("/etc/docker/certs/tlscert.sh", []byte(tlsCertScript), DefaultDirFileMode)
	if err != nil {
		return errors.Wrapf(err, "write tlscert.sh failed")
	}

	executor := &exec.CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
		"cd /etc/docker/certs && ./tlscert.sh "+tlsHost)
	if err != nil {
		return fmt.Errorf("generate docker tls cert failed, output: %s: %w", output, err)
	}

	// export DOCKER_CONFIG=/etc/docker/certs to /etc/profile
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
		"echo 'export DOCKER_CONFIG=/etc/docker/certs' >> /etc/profile")
	if err != nil {
		log.Warnf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile failed, output: %s, err: %v", output, err)
	}
	log.Debugf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile, output: %s", output)
	// source /etc/profile use bash
	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c", "source /etc/profile")
	if err != nil {
		log.Warnf("source /etc/profile failed, output: %s, err: %v", output, err)
	}
	return nil
}

// createDockerTLSConfig creates the Docker TLS configuration
func createDockerTLSConfig() error {
	if err := ensureDockerConfigFileExists(); err != nil {
		return err
	}
	cfg := buildTLSConfig()
	return writeDockerConfig(cfg)
}

// updateDockerTLSConfig updates the existing Docker TLS configuration
func updateDockerTLSConfig() error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	var cfg interface{}
	if len(f) == 0 {
		cfg = buildTLSConfig()
		return writeDockerConfig(cfg)
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}

	if v, ok := cfg.(map[string]interface{}); ok {
		updateTLSConfig(v)
	}

	return writeDockerConfigSimple(cfg)
}

// buildTLSConfig builds the TLS configuration
func buildTLSConfig() map[string]interface{} {
	return map[string]interface{}{
		"tls":       true,
		"tlsverify": true,
		"tlscacert": "/etc/docker/certs/ca.pem",
		"tlscert":   "/etc/docker/certs/server-cert.pem",
		"tlskey":    "/etc/docker/certs/server-key.pem",
		"hosts":     []string{"tcp://0.0.0.0:2376", "unix:///var/run/docker.sock"},
	}
}

// updateTLSConfig updates the TLS configuration in the config map
func updateTLSConfig(v map[string]interface{}) {
	if v == nil {
		return
	}
	v["tls"] = true
	v["tlsverify"] = true
	v["tlscacert"] = "/etc/docker/certs/ca.pem"
	v["tlscert"] = "/etc/docker/certs/server-cert.pem"
	v["tlskey"] = "/etc/docker/certs/server-key.pem"

	// add unix:///var/run/docker.sock and tcp://0.0.0.0:2376
	if hosts := v["hosts"]; hosts != nil {
		if h, ok := hosts.([]interface{}); ok {
			newHosts := updateHostsList(h)
			v["hosts"] = newHosts
		}
	} else {
		v["hosts"] = []interface{}{
			"unix:///var/run/docker.sock",
			"tcp://0.0.0.0:2376",
		}
	}
}

// updateHostsList updates the hosts list with required entries
func updateHostsList(hosts []interface{}) []interface{} {
	hasUnixSocket := false
	hasTcpEndpoint := false
	for _, host := range hosts {
		if hostStr, ok := host.(string); ok {
			if hostStr == "unix:///var/run/docker.sock" {
				hasUnixSocket = true
			} else if hostStr == "tcp://0.0.0.0:2376" {
				hasTcpEndpoint = true
			} else {
				// Other host types are ignored for this specific check
			}
		} else {
			log.Warnf("Docker hosts configuration is not a string, got: %T", host)
		}
	}

	// Create a new hosts slice with required entries
	newHosts := hosts
	if !hasUnixSocket {
		newHosts = append(newHosts, "unix:///var/run/docker.sock")
	}
	if !hasTcpEndpoint {
		newHosts = append(newHosts, "tcp://0.0.0.0:2376")
	}
	return newHosts
}

// ensureDockerConfigFileExists ensures the Docker config file exists
func ensureDockerConfigFileExists() error {
	if !utils.Exists(filepath.Dir(DockerDaemonConfigFilePath)) {
		err := os.MkdirAll(filepath.Dir(DockerDaemonConfigFilePath), DefaultDirFileMode)
		if err != nil {
			return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
		}
	}
	log.Infof("docker daemon config file %s not found, create it", DockerDaemonConfigFilePath)
	_, err := os.OpenFile(DockerDaemonConfigFilePath, os.O_RDONLY|os.O_CREATE, DefaultFileMode)
	if err != nil {
		return errors.Wrapf(err, "create docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	return nil
}

// BaseConfig performs base configuration for Docker
func BaseConfig() error {
	if !utils.Exists(DockerDaemonConfigFilePath) {
		return createBaseConfig()
	}

	return updateBaseConfig()
}

// createBaseConfig creates the base Docker configuration
func createBaseConfig() error {
	if err := ensureDockerConfigFileExists(); err != nil {
		return err
	}
	cfg := buildBaseConfig()
	return writeDockerConfig(cfg)
}

// updateBaseConfig updates the existing base Docker configuration
func updateBaseConfig() error {
	// Read the configuration file
	f, err := os.ReadFile(DockerDaemonConfigFilePath)
	if err != nil {
		return errors.Wrapf(err, "read docker daemon config file %s failed", DockerDaemonConfigFilePath)
	}
	var cfg interface{}
	if len(f) == 0 {
		cfg = buildBaseConfig()
	} else {
		err = json.Unmarshal(f, &cfg)
		if err != nil {
			return errors.Wrapf(err, "unmarshal docker daemon config failed")
		}
	}

	if v, ok := cfg.(map[string]interface{}); ok {
		applyBaseConfigDefaults(v)
	}

	return writeDockerConfigSimple(cfg)
}

// buildBaseConfig builds the base Docker configuration
func buildBaseConfig() map[string]interface{} {
	return map[string]interface{}{
		"log-driver": "json-file",
		"log-opts": map[string]interface{}{
			"max-size": "100m",
		},
	}
}

// applyBaseConfigDefaults applies default values to the base configuration
func applyBaseConfigDefaults(v map[string]interface{}) {
	if v == nil {
		return
	}
	if _, ok := v["log-driver"]; !ok {
		v["log-driver"] = "json-file"
	}

	if _, ok := v["log-opts"]; !ok {
		v["log-opts"] = map[string]interface{}{
			"max-size": "100m",
		}
	} else {
		if logOpts, ok := v["log-opts"].(map[string]interface{}); ok {
			if _, ok := logOpts["max-size"]; !ok {
				logOpts["max-size"] = "100m"
			}
		}
	}
}

// WaitDockerReady waits until Docker is ready
func WaitDockerReady() error {
	ctx, cancel := context.WithTimeout(context.Background(), DockerReadyTimeoutMinutes*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(DockerReadyPollInterval, func() (bool, error) {
		log.Debugf("Waiting for Docker to be ready")
		_, err := NewDockerClient()
		if err == nil {
			return true, nil
		}
		log.Debugf("Docker is not available: %v", err)
		return false, nil
	}, ctx.Done())
	if err != nil {
		return fmt.Errorf("wait docker available: %w", err)
	}
	return nil
}
