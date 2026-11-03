/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package preflightbridge is the bkeadm host adapter that turns CLI flags or
// a request YAML into the BKE adapter input, prepares the read-only snapshot,
// and maps the SDK Report to CLI exit codes/output.
//
// It is the ONLY host-side coupling to the SDK: it implements the SDK's
// Snapshot/Credential interfaces and feeds the Adapter. The SDK itself never
// imports bkeadm.
package preflightbridge

import (
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"

	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	"gopkg.openfuyao.cn/bkeadm/utils"
	preflight "gopkg.openfuyao.cn/preflight"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
)

// InitFlagSpec is the shared flag definition for `bke init` and
// `bke preflight init`. Both commands bind flags from the same spec so a
// parameter cannot drift between them. The spec carries name, shorthand,
// default, type and required/execution-only metadata; it does not bind to a
// Cobra command (the caller does that).
type InitFlagSpec struct {
	Name          string
	Shorthand     string
	Default       any
	Usage         string
	Required      bool
	ExecutionOnly bool // e.g. --confirm: parsed for compat, never enters PlanHash
}

// InitFlagSpecs returns the complete, ordered init flag spec table. This is
// the single source of truth shared by `bke init` and `bke preflight init`.
// Adding/removing/renaming a flag here updates both commands in lockstep.
func InitFlagSpecs() []InitFlagSpec {
	return []InitFlagSpec{
		{Name: "file", Shorthand: "f", Default: "", Usage: "bkecluster.yaml"},
		{Name: "domain", Default: configinit.DefaultImageRepo, Usage: "The domain name of the host is customized"},
		{Name: "imageRepoPort", Default: configinit.DefaultImageRepoPort, Usage: "image repository port"},
		{Name: "hostIP", Default: "", Usage: "local kubernetes api server", Required: true},
		{Name: "kubernetesPort", Default: utils.DefaultKubernetesPort, Usage: "local kubernetes port"},
		{Name: "consolePort", Default: utils.DefaultConsolePort, Usage: "bke console port"},
		{Name: "yumRepoPort", Default: configinit.DefaultYumRepoPort, Usage: "yum repository port"},
		{Name: "chartRepoPort", Default: utils.DefaultChartRegistryPort, Usage: "chart repository port"},
		{
			Name:    "ntpServer",
			Default: configinit.DefaultNTPServer,
			Usage:   "value is `local`, an ntp service is started on the host",
		},
		{Name: "runtime", Default: "containerd", Usage: "containerd"},
		{Name: "runtimeStorage", Shorthand: "s", Default: "", Usage: "default `/var/lib/containerd`"},
		{Name: "onlineImage", Default: "", Usage: "Online image to provide binary files and os rpm"},
		{Name: "otherRepo", Default: "", Usage: "Provides the private source address for image mirror installation"},
		{
			Name:    "otherSource",
			Default: "",
			Usage:   "Provides the private source address for system package installation",
		},
		{Name: "otherChart", Default: "", Usage: "Provides the private chart address for helm chart installation"},
		{Name: "clusterAPI", Default: "latest", Usage: "cluster-api-bke version"},
		{Name: "confirm", Default: false, Usage: "Skip the confirmation", ExecutionOnly: true},
		{Name: "oFVersion", Shorthand: "v", Default: "latest", Usage: "openfuyao version"},
		{Name: "versionUrl", Default: utils.DefaultPatchDownURL, Usage: "online openfuyao version config download url"},
		{Name: "installConsole", Default: true, Usage: "Whether to install bkeconsole (default: true)"},
		{Name: "enableNTP", Default: true, Usage: "enable or disable NTP service, default is true"},
		{Name: "imageRepoTLSVerify", Default: true, Usage: "Enable TLS verification for image repository"},
		{Name: "imageRepoCAFile", Default: "", Usage: "CA certificate file for image repository"},
		{Name: "imageRepoUsername", Default: "", Usage: "Username for image repository authentication"},
		{Name: "imageRepoPassword", Default: "", Usage: "Password for image repository authentication"},
		{Name: "imageFilePath", Default: "", Usage: "image file path"},
		{Name: "agentHealthPort", Default: utils.DefaultAgentHealthPort, Usage: "agent listen health port"},
	}
}

// InitOptions is the resolved, pure projection of init flags: defaults applied,
// no DNS lookup, no confirmation, no printing. Both `bke init` (after its
// PreRunE side-effects) and `bke preflight init` produce the same InitOptions
// for the same input, so PlanHash is equivalent.
type InitOptions struct {
	File               string
	HostIP             string
	Domain             string
	KubernetesPort     int
	ConsolePort        int
	ImageRepoPort      int
	YumRepoPort        int
	ChartRepoPort      int
	NtpServer          string
	Runtime            string
	RuntimeStorage     string
	Workspace          string
	OnlineImage        string
	OtherRepo          string
	OtherSource        string
	OtherChart         string
	ClusterAPI         string
	OFVersion          string
	VersionURL         string
	InstallConsole     bool
	EnableNTP          bool
	ImageRepoTLSVerify bool
	ImageRepoCAFile    string
	ImageRepoUsername  string
	ImageRepoPassword  string
	ImageFilePath      string
	AgentHealthPort    int
	ExplicitPorts      map[string]bool
}

// MarkExplicitInitPort records that one init port flag came from the user or
// the init input file rather than from the shared CLI default.
func MarkExplicitInitPort(options *InitOptions, flagName string) {
	if options == nil {
		return
	}
	key := map[string]string{
		"kubernetesPort": "kubernetes",
		"consolePort":    "console",
		"imageRepoPort":  "imageRepo",
		"yumRepoPort":    "yum",
		"chartRepoPort":  "chart",
	}[flagName]
	if key == "" {
		return
	}
	if options.ExplicitPorts == nil {
		options.ExplicitPorts = make(map[string]bool)
	}
	options.ExplicitPorts[key] = true
}

// ResolveInitOptions applies the shared defaulting and pure resolution to the
// raw flag values. It performs NO DNS lookup, NO file read, NO confirmation
// and NO printing — those stay in `bke init`'s PreRunE. It returns an error
// for missing-required or unparseable values (which become INPUT_ERROR).
func ResolveInitOptions(raw map[string]string) (InitOptions, error) {
	options := initOptionsFromRaw(raw)
	applyInitDefaults(&options, raw)
	if err := parseInitNumericFlags(&options, raw); err != nil {
		return InitOptions{}, err
	}
	if err := parseInitBoolFlags(&options, raw); err != nil {
		return InitOptions{}, err
	}
	resolveRuntimeStorage(&options)
	return options, nil
}

func initOptionsFromRaw(raw map[string]string) InitOptions {
	return InitOptions{
		File:              raw["file"],
		HostIP:            raw["hostIP"],
		Domain:            raw["domain"],
		NtpServer:         raw["ntpServer"],
		Runtime:           raw["runtime"],
		RuntimeStorage:    raw["runtimeStorage"],
		OnlineImage:       raw["onlineImage"],
		OtherRepo:         raw["otherRepo"],
		OtherSource:       raw["otherSource"],
		OtherChart:        raw["otherChart"],
		ClusterAPI:        raw["clusterAPI"],
		OFVersion:         raw["oFVersion"],
		VersionURL:        raw["versionUrl"],
		ImageRepoCAFile:   raw["imageRepoCAFile"],
		ImageRepoUsername: raw["imageRepoUsername"],
		ImageRepoPassword: raw["imageRepoPassword"],
		ImageFilePath:     raw["imageFilePath"],
	}
}

func applyInitDefaults(options *InitOptions, raw map[string]string) {
	// Apply defaults from the spec for any unset value.
	for _, s := range InitFlagSpecs() {
		// Explicit non-empty value wins; skip defaulting for this field.
		if v, ok := raw[s.Name]; ok && v != "" {
			continue
		}
		applyInitDefault(options, s.Name)
	}
}

func applyInitDefault(options *InitOptions, name string) {
	switch name {
	case "domain":
		options.Domain = configinit.DefaultImageRepo
	case "imageRepoPort":
		options.ImageRepoPort = atoi(configinit.DefaultImageRepoPort)
	case "kubernetesPort":
		options.KubernetesPort = atoi(utils.DefaultKubernetesPort)
	case "consolePort":
		options.ConsolePort = atoi(utils.DefaultConsolePort)
	case "yumRepoPort":
		options.YumRepoPort = atoi(configinit.DefaultYumRepoPort)
	case "chartRepoPort":
		options.ChartRepoPort = atoi(utils.DefaultChartRegistryPort)
	case "ntpServer":
		options.NtpServer = configinit.DefaultNTPServer
	case "runtime":
		options.Runtime = "containerd"
	case "clusterAPI":
		options.ClusterAPI = "latest"
	case "oFVersion":
		options.OFVersion = "latest"
	case "versionUrl":
		options.VersionURL = utils.DefaultPatchDownURL
	case "installConsole":
		options.InstallConsole = true
	case "enableNTP":
		options.EnableNTP = true
	case "imageRepoTLSVerify":
		options.ImageRepoTLSVerify = true
	case "agentHealthPort":
		options.AgentHealthPort = atoi(utils.DefaultAgentHealthPort)
	default:
		return
	}
}

func parseInitNumericFlags(options *InitOptions, raw map[string]string) error {
	// Parse explicit port values from raw (these were skipped above).
	for name, target := range map[string]*int{
		"kubernetesPort": &options.KubernetesPort, "consolePort": &options.ConsolePort,
		"imageRepoPort": &options.ImageRepoPort, "yumRepoPort": &options.YumRepoPort,
		"chartRepoPort": &options.ChartRepoPort, "agentHealthPort": &options.AgentHealthPort,
	} {
		if v := raw[name]; v != "" {
			parsed, err := strconv.Atoi(v)
			if err != nil {
				return fmt.Errorf("--%s must be an integer: %w", name, err)
			}
			*target = parsed
		}
	}
	return nil
}

func parseInitBoolFlags(options *InitOptions, raw map[string]string) error {
	for name, target := range map[string]*bool{
		"installConsole": &options.InstallConsole, "enableNTP": &options.EnableNTP,
		"imageRepoTLSVerify": &options.ImageRepoTLSVerify,
	} {
		if v := raw[name]; v != "" {
			parsed, err := strconv.ParseBool(v)
			if err != nil {
				return fmt.Errorf("--%s must be a boolean: %w", name, err)
			}
			*target = parsed
		}
	}
	return nil
}

func resolveRuntimeStorage(options *InitOptions) {
	// Runtime storage default depends on runtime.
	if options.RuntimeStorage == "" {
		if options.Runtime == "docker" {
			options.RuntimeStorage = configinit.DefaultCRIDockerDataRootDir
		} else {
			options.RuntimeStorage = configinit.DefaultCRIContainerdDataRootDir
		}
	}
}

// ValidateInitInput checks the resolved options for required fields and range
// constraints. Returns an error describing the first problem (INPUT_ERROR).
func ValidateInitInput(o InitOptions) error {
	if o.HostIP == "" {
		return errors.New("the host IP address must be set")
	}
	if o.KubernetesPort < utils.MinPort || o.KubernetesPort > utils.MaxPort {
		return errors.New("the kubernetes port must be legal")
	}
	if o.ConsolePort <= utils.MinPort || o.ConsolePort > utils.MaxPort {
		return errors.New("the console port must be legal")
	}
	if o.ImageRepoPort < utils.MinPort || o.ImageRepoPort > utils.MaxPort {
		return errors.New("the image server port must be legal")
	}
	if o.ChartRepoPort < utils.MinPort || o.ChartRepoPort > utils.MaxPort {
		return errors.New("the chart server port must be legal")
	}
	if o.YumRepoPort < utils.MinPort || o.YumRepoPort > utils.MaxPort {
		return errors.New("the yum server port must be legal")
	}
	if o.AgentHealthPort < utils.MinPort || o.AgentHealthPort > utils.MaxPort {
		return errors.New("the agent health port must be legal")
	}
	if o.Domain == "" {
		return errors.New("the host domain name is required")
	}
	if o.Runtime != "containerd" {
		return errors.New("the runtime only supports containerd")
	}
	if o.RuntimeStorage == configinit.DefaultCRIDockerDataRootDir {
		return errors.New("the runtime storage does not support the docker data root")
	}
	if o.ImageFilePath != "" {
		if o.InstallConsole {
			return errors.New("local image file and install console are mutually exclusive")
		}
		if o.OtherRepo == "" {
			return errors.New("local image file and offline deploy are mutually exclusive")
		}
	}
	return nil
}

// ResolveInitHostIP mirrors the existing init command's read-only host-IP
// defaulting without invoking its PreRunE (which also performs unrelated I/O
// and confirmation). Explicit input always wins.
func ResolveInitHostIP(options *InitOptions) {
	if options == nil || options.HostIP != "" {
		return
	}
	if ip, _ := utils.GetOutBoundIP(); ip != "" {
		options.HostIP = ip
		return
	}
	if ip, _ := utils.GetIntranetIp(); ip != "" {
		options.HostIP = ip
	}
}

// ApplyInitFileOptions resolves the read-only business defaults that the
// existing bke init command derives from --file. It deliberately performs no
// DNS lookup and no mutation outside options; reachability is evaluated later
// by the endpoint collectors. This keeps the CLI and request entry points on
// the same operation plan without invoking init's side-effectful PreRunE.
func ApplyInitFileOptions(options *InitOptions) error {
	if options == nil || strings.TrimSpace(options.File) == "" {
		return nil
	}
	cluster, err := loadBKEClusterStrict(options.File)
	if err != nil {
		return err
	}
	config := cluster.Spec.ClusterConfig.Cluster
	if repo := config.ImageRepo; repo.Ip != "" && repo.Ip != options.HostIP {
		options.OtherRepo = repoEndpoint(repo.Ip, repo.Port, "443", repo.Prefix, "")
	} else if repo.Ip == "" && repo.Domain != "" {
		options.OtherRepo = repoEndpoint(repo.Domain, repo.Port, "443", repo.Prefix, "")
	} else {
		// Keep the CLI value when the file does not override the image repo.
	}
	if repo := config.HTTPRepo; repo.Ip != "" && repo.Ip != options.HostIP {
		options.OtherSource = repoEndpoint(repo.Ip, repo.Port, "80", "", "http")
	} else if repo.Ip == "" && repo.Domain != "" {
		options.OtherSource = repoEndpoint(repo.Domain, repo.Port, "80", "", "http")
	} else {
		// Keep the CLI value when the file does not override the HTTP repo.
	}
	if repo := config.ChartRepo; repo.Ip != "" {
		options.OtherChart = repoEndpoint(repo.Ip, repo.Port, "443", repo.Prefix, "")
	} else if repo.Domain != "" {
		options.OtherChart = repoEndpoint(repo.Domain, repo.Port, "443", repo.Prefix, "")
	} else {
		// Keep the CLI value when the file does not override the chart repo.
	}
	if config.HTTPRepo.Port != "" {
		port, parseErr := strconv.Atoi(config.HTTPRepo.Port)
		if parseErr != nil {
			return fmt.Errorf("HTTP repository port must be an integer: %w", parseErr)
		}
		options.YumRepoPort = port
		MarkExplicitInitPort(options, "yumRepoPort")
	}
	if config.NTPServer != "" {
		options.NtpServer = config.NTPServer
	}
	return nil
}

func repoEndpoint(host, port, defaultPort, prefix, scheme string) string {
	if port == "" {
		port = defaultPort
	}
	endpoint := net.JoinHostPort(strings.Trim(host, "[]"), port)
	if scheme != "" {
		endpoint = scheme + "://" + endpoint
	}
	if prefix = strings.Trim(prefix, "/"); prefix != "" {
		endpoint += "/" + prefix
	}
	return endpoint
}

// ToAdapterInput converts resolved InitOptions to the BKE host AdapterInput.
// This bridge strips execution-only
// fields and keeps only plan-relevant semantics.
func (o InitOptions) ToAdapterInput() AdapterInput {
	return AdapterInput{
		Operation: api.OperationInit,
		Init: &initadapter.Input{
			HostIP:             o.HostIP,
			Domain:             o.Domain,
			KubernetesPort:     o.KubernetesPort,
			ConsolePort:        o.ConsolePort,
			ImageRepoPort:      o.ImageRepoPort,
			YumRepoPort:        o.YumRepoPort,
			ChartRepoPort:      o.ChartRepoPort,
			NtpServer:          o.NtpServer,
			Runtime:            o.Runtime,
			RuntimeStorage:     o.RuntimeStorage,
			Workspace:          o.Workspace,
			OnlineImage:        o.OnlineImage,
			OtherRepo:          o.OtherRepo,
			OtherSource:        o.OtherSource,
			OtherChart:         o.OtherChart,
			InstallConsole:     o.InstallConsole,
			EnableNTP:          o.EnableNTP,
			ImageFilePath:      o.ImageFilePath,
			AgentHealthPort:    o.AgentHealthPort,
			ExplicitPorts:      copyBoolMap(o.ExplicitPorts),
			ImageRepoTLSVerify: o.ImageRepoTLSVerify,
			ImageRepoCAFile:    o.ImageRepoCAFile,
			ImageRepoAuth:      o.ImageRepoUsername != "" && o.ImageRepoPassword != "",
		},
	}
}

func copyBoolMap(source map[string]bool) map[string]bool {
	if len(source) == 0 {
		return nil
	}
	out := make(map[string]bool, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

// atoi is a tolerant int parser for port defaults.
func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}

// Compile-time assertion that the bridge uses the SDK types.
var _ preflight.Checker = (preflight.Checker)(nil)
