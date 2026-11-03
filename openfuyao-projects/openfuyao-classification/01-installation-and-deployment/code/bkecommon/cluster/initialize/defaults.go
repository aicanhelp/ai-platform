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

package initialize

import (
	"fmt"
	"math/big"
	"net"
	"time"

	"gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
)

const (
	// DefaultServiceDNSDomain defines default cluster-internal domain name for Services and Pods
	DefaultServiceDNSDomain = "cluster.local"
	// DefaultServicesSubnet defines default service subnet range
	DefaultServicesSubnet = "10.96.0.0/16"
	// DefaultClusterDNSIP defines default DNS IP
	DefaultClusterDNSIP = "10.96.0.10"
	// DefaultPodSubnet defines default pod subnet range
	DefaultPodSubnet = "10.250.0.0/16"
	// DefaultDNSIPIndex defines the default index for DNS IP calculation in service subnet
	DefaultDNSIPIndex = 10
	// DefaultKubernetesVersion defines default kubernetes version
	DefaultKubernetesVersion = "v1.25.6"
	// DefaultAPIBindPort defines default API port
	DefaultAPIBindPort = 6443
	// DefaultLoadBalancerBindPort defines default load balancer bind port
	DefaultLoadBalancerBindPort = 36443
	// DefaultCertificatesDir defines default certificate directory
	DefaultCertificatesDir = "/etc/kubernetes/pki"
	// DefaultManifestsDir defines default manifests directory
	DefaultManifestsDir = "/etc/kubernetes/manifests"
	// DefaultTimeoutForControlPlane defines default timeout for control plane
	DefaultTimeoutForControlPlane = 4 * time.Minute

	// DefaultEtcdDataDir defines default location of etcd where static pods will save data to
	DefaultEtcdDataDir = "/var/lib/etcd"
	// DefaultProxyBindAddressv4 is the default bind address when the advertise address is v4
	DefaultProxyBindAddressv4 = "0.0.0.0"
	// DefaultProxyBindAddressv6 is the default bind address when the advertise address is v6
	DefaultProxyBindAddressv6 = "::"

	// DefaultKubeletRootDir defines default kubelet root directory
	DefaultKubeletRootDir           = "/var/lib/kubelet"
	DefaultKubeletRootDirVolumeName = "kubelet-root-dir"

	DefaultImageRepo            = "deploy.bocloud.k8s"
	DefaultImageRepoPort        = "40443"
	ImageRegistryKubernetes     = "kubernetes"
	DefaultYumRepo              = "http.bocloud.k8s"
	DefaultYumRepoPort          = "40080"
	DefaultChartRepo            = "cr.openfuyao.cn"
	DefaultOfflineChartRepoPort = "40443"
	DefaultOnlineChartRepoPort  = "443"
	DefaultChartRepoPrefix      = "chart"

	DefaultEtcdImageName              = "etcd"
	DefaultAPIServerImageName         = "kube-apiserver"
	DefaultControllerManagerImageName = "kube-controller-manager"
	DefaultSchedulerImageName         = "kube-scheduler"
	DefaultKubeletImageName           = "kubelet"
	DefaultPauseImageName             = "pause"

	CRIContainerd                   = "containerd"
	CRIDocker                       = "docker"
	DefaultCRIContainerdDataRootDir = "/var/lib/containerd"
	DefaultCRIDockerDataRootDir     = "/var/lib/docker"
	DefaultRuntime                  = "runc"
	DefaultCgroupDriver             = "systemd"

	DefaultEtcdVersion   = "v3.5.21-of.1"
	DefaultEtcdImageTag  = "3.5.21-of.1"
	DefaultPauseImageTag = "3.9"

	//Deprecated
	DefaultK8sV121EtcdImageTag = "3.4.13-0"
	//Deprecated
	DefaultK8sV125EtcdImageTag = "3.5.6-0"
	//Deprecated
	DefaultK8sV121PauseImageTag = "3.6"
	//Deprecated
	DefaultK8sV125PauseImageTag = "3.8"

	DefaultAddonTimeout        = 4 * time.Minute
	DefaultAddonTimeoutString  = "4m0s"
	DefaultAddonInterval       = 2 * time.Second
	DefaultAddonIntervalString = "2s"

	DefaultNTPServer = "cn.pool.ntp.org:123"

	DefaultNodeSSHPort  = "22"
	DefaultNodeUserRoot = "root"
)

// For the subsequent BKE version support for k8s, you need to add it here:
// version   etcd-image-tag  pause-image-tag
// v1.21.1   3.4.13-0        3.6
// v1.23.17  3.5.6-0         3.6
// v1.25.6   3.5.6-0         3.8
// Deprecated: This function is deprecated.
func GetDefaultEtcdK8sVersionImageMap() map[string]string {
	return map[string]string{
		"v1.21.1":  fmt.Sprintf("%s:%s", DefaultEtcdImageName, DefaultK8sV121EtcdImageTag),
		"v1.25.6":  fmt.Sprintf("%s:%s", DefaultEtcdImageName, DefaultK8sV125EtcdImageTag),
		"v1.23.17": fmt.Sprintf("%s:%s", DefaultEtcdImageName, DefaultK8sV125EtcdImageTag),
	}
}

// GetDefaultPauseK8sVersionImageMap returns the pause image map for different k8s versions
func GetDefaultPauseK8sVersionImageMap() map[string]string {
	return map[string]string{
		"v1.21.1":  fmt.Sprintf("%s:%s", DefaultPauseImageName, DefaultK8sV121PauseImageTag),
		"v1.23.17": fmt.Sprintf("%s:%s", DefaultPauseImageName, DefaultK8sV121PauseImageTag),
		"v1.25.6":  fmt.Sprintf("%s:%s", DefaultPauseImageName, DefaultK8sV125PauseImageTag),
	}
}

func SetDefaultBKEConfig(obj *BkeConfig) {
	SetDefaultCluster(obj)
}

func SetDefaultCluster(obj *BkeConfig) {
	SetDefaultCertificatesDir(&obj.Cluster)
	SetDefaultKubernetesVersion(&obj.Cluster)
	SetDefaultEtcdVersion(&obj.Cluster)
	SetDefaultHttpRepo(&obj.Cluster)
	SetDefaultImageRepo(&obj.Cluster)
	SetDefaultChartRepo(&obj.Cluster)
	SetDefaultClusterNetworking(&obj.Cluster)
	SetDefaultEtcd(&obj.Cluster)
	SetDefaultAPIServer(&obj.Cluster)
	SetDefaultControllerManager(&obj.Cluster)
	SetDefaultScheduler(&obj.Cluster)
	SetDefaultKubelet(&obj.Cluster)
	SetDefaultContainerRuntime(&obj.Cluster)
}

func SetDefaultCertificatesDir(obj *v1beta1.Cluster) {
	if obj.CertificatesDir == "" {
		obj.CertificatesDir = DefaultCertificatesDir
	}
}

func SetDefaultKubernetesVersion(obj *v1beta1.Cluster) {
	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = DefaultKubernetesVersion
	}
}

// SetDefaultEtcdVersion sets the default etcd version if the current etcd version is empty
func SetDefaultEtcdVersion(obj *v1beta1.Cluster) {
	if obj.EtcdVersion == "" {
		obj.EtcdVersion = DefaultEtcdVersion
	}
}

func SetDefaultHttpRepo(obj *v1beta1.Cluster) {
	if obj.HTTPRepo.Domain == "" {
		obj.HTTPRepo.Domain = DefaultYumRepo
	}
	if obj.HTTPRepo.Port == "" && obj.HTTPRepo.Domain == DefaultYumRepo {
		obj.HTTPRepo.Port = DefaultYumRepoPort
	}
}

func SetDefaultImageRepo(obj *v1beta1.Cluster) {
	if obj.ImageRepo.Domain == "" {
		obj.ImageRepo.Domain = DefaultImageRepo
	}
	if obj.ImageRepo.Port == "" && obj.ImageRepo.Domain == DefaultImageRepo {
		obj.ImageRepo.Port = DefaultImageRepoPort
	}
	if obj.ImageRepo.Prefix == "" && obj.ImageRepo.Domain == DefaultImageRepo {
		obj.ImageRepo.Prefix = ImageRegistryKubernetes
	}
}

// SetDefaultChartRepo set chart repo default value
func SetDefaultChartRepo(obj *v1beta1.Cluster) {
	if obj.ChartRepo.Domain == "" && obj.ChartRepo.Ip == "" {
		obj.ChartRepo.Domain = DefaultChartRepo
	}
	if obj.ChartRepo.Port == "" && obj.ChartRepo.Domain == DefaultChartRepo {
		obj.ChartRepo.Port = DefaultOnlineChartRepoPort
	}
	if obj.ChartRepo.Port == "" && obj.ChartRepo.Domain != DefaultChartRepo {
		obj.ChartRepo.Port = DefaultOfflineChartRepoPort
	}
	if obj.ChartRepo.Prefix == "" && obj.ChartRepo.Domain == DefaultChartRepo {
		obj.ChartRepo.Prefix = DefaultChartRepoPrefix
	}
}

func SetDefaultClusterNetworking(obj *v1beta1.Cluster) {
	if obj.Networking.DNSDomain == "" {
		obj.Networking.DNSDomain = DefaultServiceDNSDomain
	}
	if obj.Networking.ServiceSubnet == "" {
		obj.Networking.ServiceSubnet = DefaultServicesSubnet
	}
	if obj.Networking.PodSubnet == "" {
		obj.Networking.PodSubnet = DefaultPodSubnet
	}
}

func SetDefaultEtcd(obj *v1beta1.Cluster) {
	if obj.Etcd == nil {
		obj.Etcd = &v1beta1.Etcd{}
	}

	if obj.Etcd.DataDir == "" {
		obj.Etcd.DataDir = DefaultEtcdDataDir
	}
}

func SetDefaultAPIServer(obj *v1beta1.Cluster) {
	if obj.APIServer == nil {
		obj.APIServer = &v1beta1.APIServer{}
	}

	if obj.APIServer.Port == 0 {
		obj.APIServer.Port = int32(DefaultAPIBindPort)
	}

	if obj.APIServer.ExtraArgs == nil {
		obj.APIServer.ExtraArgs = map[string]string{}
	}
	if _, ok := obj.APIServer.ExtraArgs["authorization-mode"]; !ok {
		obj.APIServer.ExtraArgs["authorization-mode"] = "Node,RBAC"
	}
}

func SetDefaultControllerManager(obj *v1beta1.Cluster) {
	if obj.ControllerManager == nil {
		obj.ControllerManager = &v1beta1.ControlPlaneComponent{}
	}
}

func SetDefaultScheduler(obj *v1beta1.Cluster) {
	if obj.Scheduler == nil {
		obj.Scheduler = &v1beta1.ControlPlaneComponent{}
	}
}

func SetDefaultKubelet(obj *v1beta1.Cluster) {
	if obj.Kubelet == nil {
		obj.Kubelet = &v1beta1.Kubelet{}
	}
	if obj.Kubelet.ManifestsDir == "" {
		obj.Kubelet.ManifestsDir = DefaultManifestsDir
	}
	if obj.Kubelet.ExtraArgs == nil {
		obj.Kubelet.ExtraArgs = map[string]string{}
	}
	if obj.Kubelet.ExtraVolumes == nil {
		obj.Kubelet.ExtraVolumes = []v1beta1.HostPathMount{}
	}

	fonudKubeletRootDirVolumeFlag := false
	for _, v := range obj.Kubelet.ExtraVolumes {
		if v.Name == DefaultKubeletRootDirVolumeName {
			fonudKubeletRootDirVolumeFlag = true
			break
		}
	}
	if !fonudKubeletRootDirVolumeFlag {
		obj.Kubelet.ExtraVolumes = append(obj.Kubelet.ExtraVolumes, v1beta1.HostPathMount{
			Name:     DefaultKubeletRootDirVolumeName,
			HostPath: DefaultKubeletRootDir,
		})
	}
}

func SetDefaultContainerRuntime(obj *v1beta1.Cluster) {
	if obj.ContainerRuntime.CRI == "" {
		obj.ContainerRuntime.CRI = CRIContainerd
	}
	if obj.ContainerRuntime.Runtime == "" {
		obj.ContainerRuntime.Runtime = DefaultRuntime
	}
	if obj.ContainerRuntime.Param == nil {
		obj.ContainerRuntime.Param = map[string]string{}
	}

	if _, ok := obj.ContainerRuntime.Param["cgroupDriver"]; !ok {
		obj.ContainerRuntime.Param["cgroupDriver"] = DefaultCgroupDriver
	}

	switch obj.ContainerRuntime.CRI {
	case CRIContainerd:
		if _, ok := obj.ContainerRuntime.Param["data-root"]; !ok {
			obj.ContainerRuntime.Param["data-root"] = DefaultCRIContainerdDataRootDir
		}
	case CRIDocker:
		if _, ok := obj.ContainerRuntime.Param["data-root"]; !ok {
			obj.ContainerRuntime.Param["data-root"] = DefaultCRIDockerDataRootDir
		}
	default:
	}
}

func GetClusterDNSIP(serviceSubnet string) (string, error) {
	if serviceSubnet == DefaultServicesSubnet {
		return DefaultClusterDNSIP, nil
	}
	_, svcSubnet, err := net.ParseCIDR(serviceSubnet)
	if err != nil {
		return "", err
	}
	clusteDNSIP, err := getIndexedIP(svcSubnet, DefaultDNSIPIndex)
	if err != nil {
		return "", err
	}
	return clusteDNSIP.String(), nil
}

// getIndexedIP returns a net.IP that is subnet.IP + index in the contiguous IP space.
func getIndexedIP(subnet *net.IPNet, index int) (net.IP, error) {
	r := big.NewInt(0).Add(big.NewInt(0).SetBytes(subnet.IP.To16()), big.NewInt(int64(index))).Bytes()
	r = append(make([]byte, net.IPv6len), r...)
	ip := net.IP(r[len(r)-net.IPv6len:])
	if !subnet.Contains(ip) {
		return nil, fmt.Errorf("can't generate IP with index %d from subnet. subnet too small. subnet: %q", index, subnet)
	}
	return ip, nil
}
