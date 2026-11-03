/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

const (
	LocalKubernetesName     = "kubernetes"
	DefaultKubernetesPort   = "36443"
	DefaultConsolePort      = "30010"
	DefaultLocalK3sRegistry = "rancher/k3s:v1.25.16-k3s4"
	DefaultK3sPause         = "rancher/mirrored-pause:3.6"
	CniPluginPrefix         = "cni-plugins-linux-"
	NerdCtl                 = "/usr/bin/nerdctl"
	KubeCtl                 = "/usr/bin/kubectl"

	DefaultLocalRegistry      = "0.0.0.0:40443/kubernetes/"
	LocalImageRegistryName    = "bocloud_image_registry"
	LocalYumRegistryName      = "bocloud_yum_registry"
	LocalChartRegistryName    = "bocloud_chart_registry"
	LocalNFSRegistryName      = "bocloud_nfs_registry"
	DefaultLocalImageRegistry = "registry:2.8.1"
	DefaultLocalYumRegistry   = "nginx:1.23.0-alpine"
	DefaultLocalChartRegistry = "helm/chartmuseum:v0.16.2"
	DefaultLocalNFSRegistry   = "openebs/nfs-server-alpine:0.9.0"
	PatchImageRegistryName    = "bke-patch-image-registry"

	DefaultThirdMirror = "hub.oepkgs.net/openfuyao"
	DefaultFuyaoMirror = "cr.openfuyao.cn/openfuyao"

	DefaultChartRegistryPort = "38080"

	LocalNTPName         = "local"
	DefaultNTPServerPort = 123

	DefaultExtendManifestsDir = "/etc/openFuyao/addons/manifests"

	DefaultPatchDownURL = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/"

	DefaultAgentHealthPort = "58080"
	MinPort                = 0
	MaxPort                = 65535
)

// Defines the default file location
const (
	ImageFile           = "volumes/registry.image"
	ImageDataFile       = "volumes/image.tar.gz"
	ImageDataDirectory  = "mount/image_registry"
	SourceDataFile      = "volumes/source.tar.gz"
	SourceDataDirectory = "mount/source_registry"
	ChartDataFile       = "mount/source_registry/files/charts.tar.gz"
	PatchDataDirectory  = "mount/source_registry/files/patches"
	ChartDataDirectory  = "mount/charts"
	NFSDataFile         = "mount/source_registry/files/nfsshare.tar.gz"
	NFSDataDirectory    = "mount/nfsshare"
	RPMDataFile         = "rpm.tar.gz"
	ChartFile           = "charts.tar.gz"
	ImageLocalDirectory = "mount/local_image" // 用于operator/调谐器安装的本地镜像目录
	LocalPatchDirectory = "mount/local_image/volumes/patches"
)

// Defines the patch config map data
const (
	// PatchKeyPrefix is prefix for patch key in configmap
	PatchKeyPrefix = "patch."
	// PatchValuePrefix is prefix for patch value in configmap
	PatchValuePrefix = "cm."
	// PatchNameSpace is the namespace of patch config map
	PatchNameSpace = "openfuyao-patch"
)

// KubeOS 不可变 OS 适配
const (
	// ImmutableOSAnnotation BKECluster annotation key
	ImmutableOSAnnotation = "openfuyao.io/immutable-os"
	// ImmutableOSAnnotationValueKubeOS KubeOS 对应值
	ImmutableOSAnnotationValueKubeOS = "kubeos"
)

const (
	KylinDocker = "docker-24.0.7-kylin-{.arch}.tar.gz"
)

const (
	// MaxRetryCount defines the maximum retry count for retryable operations.
	MaxRetryCount = 3
	// DelayTime defines the delay time between retries.
	DelayTime = 1
	// MinDiskSpace defines the minimum disk space required for the init operation.
	MinDiskSpace = 20
	// MinDiskSpaceExisting defines the minimum disk space required when image data already exists.
	MinDiskSpaceExisting = 3
	// HttpUrlFields defines fields num in http url
	HttpUrlFields = 2
	// MatchFields defines field num in reg match
	MatchFields = 2
	// MinRegistryIp defines the lower bound of the registry address range
	MinRegistryIp = 2
	// MaxRegistryIp defines the upper bound of the registry address range
	MaxRegistryIp = 10
	// MinManifestsImageArgs defines the minimum number of args for manifests image command
	MinManifestsImageArgs = 2
	// DefaultImageTags defines the default number of tags for image view
	DefaultImageTags = 3
	// ContainerWaitSeconds defines the wait time in seconds for container operations
	ContainerWaitSeconds = 2
	// HTTPStatusOK defines the HTTP status code for successful requests
	HTTPStatusOK = 200
)

const (
	// DefaultDirPermission Linux Default Dir Permission
	DefaultDirPermission = 0755
	// DefaultFilePermission Linux Default File Permission
	DefaultFilePermission = 0644
	// DefaultReadWritePermission Linux Default Read Write Permission for all users
	DefaultReadWritePermission = 0666
	// SecureFilePermission Linux Secure File Permission
	SecureFilePermission = 0600
	// ReadExecutePermission Linux Read and Execute Permission for special directories
	ReadExecutePermission = 0555
	// ExecutableFilePermission Linux Executable File Permission
	ExecutableFilePermission = 0751
)

const (
	// DefaultMinCheckSeconds min check time
	DefaultMinCheckSeconds = 5
	// DefaultMaxCheckSeconds max check time
	DefaultMaxCheckSeconds = 20
	// DefaultTimeoutSeconds defines the default timeout in seconds
	DefaultTimeoutSeconds = 15
	// DefaultSleepSeconds defines the default sleep time in seconds
	DefaultSleepSeconds = 2
	// ContainerStartWaitSeconds defines the wait time in seconds for container start operations
	ContainerStartWaitSeconds = 5
	// ContainerRemoveWaitSeconds defines the wait time in seconds for container remove operations
	ContainerRemoveWaitSeconds = 3
	// DialTimeoutSeconds defines the timeout in seconds for network dial operations
	DialTimeoutSeconds = 3
	// DockerConnectionMaxRetries defines the maximum retry count for Docker connection
	DockerConnectionMaxRetries = 5
	// DockerConnectionRetrySeconds defines the interval in seconds between Docker connection retries
	DockerConnectionRetrySeconds = 2
)
