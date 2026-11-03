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

package kube

import (
	"context"
	"fmt"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/controllers/remote"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// ExpectedErrorCount represents the number of expected error attempts
	ExpectedErrorCount = 2
	// DefaultRestConfigTimeout represents the default timeout for REST config in seconds
	DefaultRestConfigTimeout = 10
)

func init() {
	// Register apiextensions types with the global scheme during process init to
	// avoid runtime data races when controllers and recorders are already running.
	if err := apiextv1.AddToScheme(scheme.Scheme); err != nil {
		panic(err)
	}
	if err := apiextv1beta1.AddToScheme(scheme.Scheme); err != nil {
		panic(err)
	}
}

type RemoteKubeClient interface {
	// InstallAddon install addon to remote cluster
	// bkeNodes parameter is required after BKENode CRD split refactoring
	InstallAddon(bkeCluster *bkev1beta1.BKECluster, addonT *bkeaddon.AddonTransfer, addonRecorder *AddonRecorder, localClient client.Client, bkeNodes bkenode.Nodes) error
	// ApplyYaml apply yaml to remote cluster
	ApplyYaml(task *Task) error
	// NewK8sToken create a new admin K8S token for remote cluster
	NewK8sToken() (string, error)
	// KubeClient return kubernetes clientset and dynamic client
	KubeClient() (*kubernetes.Clientset, dynamic.Interface)
	// Collect remote cluster info
	Collect() (*CollectResult, []error, []error)
	// CheckClusterHealth check cluster health
	CheckClusterHealth(cluster *bkev1beta1.BKECluster, version string, bkeNodes bkev1beta1.BKENodes) error
	// NodeHealthCheck check single node health
	NodeHealthCheck(node *corev1.Node, expectVersion string, log *log.Logger) error
	// CheckComponentHealth check components health
	CheckComponentHealth(node *corev1.Node) error
	// ListNodes list nodes
	ListNodes(option *metav1.ListOptions) (*corev1.NodeList, error)
	// GetPod get pod
	GetPod(namespace, name string) (*corev1.Pod, error)
	// SetLogger set logger
	SetLogger(logger *log.Logger)
	// SetBKELogger set bke logger
	SetBKELogger(bkeLog *bkev1beta1.BKELogger)
}

type Client struct {
	ClientSet     *kubernetes.Clientset
	DynamicClient dynamic.Interface
	RestConfig    *rest.Config
	RESTMapper    meta.RESTMapper
	mapperCache   *perClusterRESTMapper
	Log           *log.Logger
	BKELog        *bkev1beta1.BKELogger
	Ctx           context.Context
}

// 这里有很多生成client的方法，但是几乎适配了任意场景

// NewRemoteClusterClient
// 从集群检索kubeconfig或者token，生成clinet.Client
func NewRemoteClusterClient(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (client.Client, error) {
	var errs []error
	// try use kubeconfig generate restconfig
	restConfig, err := remote.RESTConfig(ctx, "cluster-cache-tracker", c, util.ObjectKey(bkeCluster))
	if err != nil {
		errs = append(errs, err)
	}
	if len(errs) != 0 && bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		restConfig, err = getRestConfigByToken(ctx, c, bkeCluster)
		if err != nil {
			errs = append(errs, err)
		}
	} else {
		errs = append(errs, errors.Errorf("failed use k8s token create remote cluster client, BKECluster %q controlPlaneEndpoint is invalid", utils.ClientObjNS(bkeCluster)))
	}
	ret, err := client.New(restConfig, client.Options{Scheme: c.Scheme()})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create client for Cluster %s/%s", bkeCluster.Namespace, bkeCluster.Name)
	}
	return ret, nil
}

// NewClientFromRestConfig
// 从rest.Config生成RemoteKubeClient
func NewClientFromRestConfig(ctx context.Context, config *rest.Config) (RemoteKubeClient, error) {
	clientSet, err := newKubernetesClientForConfig(config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create cluster clientset")
	}
	dynamicClient, err := newDynamicClientForConfig(config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create remote cluster dynamicClient")
	}
	mapperCache, err := GetDynamicRESTMapper(config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create RESTMapper")
	}
	return &Client{
		ClientSet:     clientSet,
		DynamicClient: dynamicClient,
		RestConfig:    config,
		RESTMapper:    mapperCache.RESTMapper(),
		mapperCache:   mapperCache,
		Log:           log.With("module", "kube"),
		Ctx:           ctx,
	}, nil
}

// NewRemoteClientByCluster
// 使用cluster资源的信息从集群检索kubeconfig或者token，生成RemoteKubeClient
func NewRemoteClientByCluster(ctx context.Context, c client.Client, cluster *clusterv1.Cluster) (RemoteKubeClient, error) {
	config, err := remote.RESTConfig(ctx, "cluster-cache-tracker", c, util.ObjectKey(cluster))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get remote cluster %q config", cluster.Name)
	}
	return NewClientFromRestConfig(ctx, config)
}

// NewRemoteClientByBKECluster
// 使用bkecluster资源的信息从集群检索kubeconfig或者token，生成RemoteKubeClient
func NewRemoteClientByBKECluster(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (RemoteKubeClient, error) {
	var errs []error
	// try use kubeconfig generate restconfig
	config, err := remote.RESTConfig(ctx, "cluster-cache-tracker", c, util.ObjectKey(bkeCluster))
	if err != nil {
		errs = append(errs, err)
	}
	// try use token generate restconfig, if kubeconfig failed
	if len(errs) != 0 && bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		config, err = getRestConfigByToken(ctx, c, bkeCluster)
		if err != nil {
			errs = append(errs, err)
		}
	} else {
		errs = append(errs, errors.Errorf("failed use k8s token create remote cluster client, BKECluster %q controlPlaneEndpoint is invalid", utils.ClientObjNS(bkeCluster)))
	}
	// all failed return error
	if len(errs) == ExpectedErrorCount {
		return nil, kerrors.NewAggregate(errs)
	}

	return NewClientFromRestConfig(ctx, config)
}

// NewClientFromConfig
// 从kubeconfig文件生成RemoteKubeClient
func NewClientFromConfig(kubeConfigPath string) (RemoteKubeClient, error) {
	config, err := clientcmd.BuildConfigFromFlags("", kubeConfigPath)
	if err != nil {
		return nil, err
	}
	return NewClientFromRestConfig(context.Background(), config)
}

// NewClientFromKubeConfig
// 从kubeconfig文件内容生成RemoteKubeClient
func NewClientFromKubeConfig(kubeConfig []byte) (RemoteKubeClient, error) {
	config, err := clientcmd.RESTConfigFromKubeConfig(kubeConfig)
	if err != nil {
		return nil, err
	}
	return NewClientFromRestConfig(context.Background(), config)
}

// NewClientFromK8sToken
// 从token生成RemoteKubeClient
func NewClientFromK8sToken(host, port, token string) (RemoteKubeClient, error) {
	config := &rest.Config{
		BearerToken: token,
		Host:        fmt.Sprintf("https://%s:%s", host, port),
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true, // 设置为true时 不需要CA
		},
	}
	config.UserAgent = remote.DefaultClusterAPIUserAgent("cluster-cache-tracker")
	config.Timeout = DefaultRestConfigTimeout * time.Second
	return NewClientFromRestConfig(context.Background(), config)
}

func (c *Client) KubeClient() (*kubernetes.Clientset, dynamic.Interface) {
	return c.ClientSet, c.DynamicClient
}

// SetLogger sets the logger for the Kubernetes client.
func (c *Client) SetLogger(logger *log.Logger) {
	if logger == nil {
		c.Log = nil
		return
	}
	c.Log = logger.WithCallerSkip(0)

}

// SetBKELogger sets the BKE logger for the Kubernetes client.
func (c *Client) SetBKELogger(bkeLog *bkev1beta1.BKELogger) {
	c.BKELog = bkeLog
}

// GetTargetClusterClient returns target cluster client from bkeCluster
func GetTargetClusterClient(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
	remoteClient, err := GetTargetRemoteClient(ctx, c, bkeCluster)
	if err != nil {
		return nil, nil, err
	}
	cs, dc := remoteClient.KubeClient()
	return cs, dc, nil
}

// GetTargetRemoteClient returns the target cluster remote client from bkeCluster.
func GetTargetRemoteClient(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*Client, error) {
	cluster, err := util.GetOwnerCluster(ctx, c, bkeCluster.ObjectMeta)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get owner cluster for bkeCluster %q", bkeCluster.Name)
	}
	if cluster == nil {
		return nil, errors.Errorf("owner cluster for bkeCluster %q is nil", bkeCluster.Name)
	}
	remoteClient, err := NewRemoteClientByCluster(ctx, c, cluster)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create remote cluster %q client", cluster.Name)
	}
	impl, ok := remoteClient.(*Client)
	if !ok || impl == nil {
		return nil, errors.Errorf("remote cluster %q client is nil", cluster.Name)
	}
	return impl, nil
}

// getRestConfigByToken get rest config by token
func getRestConfigByToken(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*rest.Config, error) {
	secret, err := phaseutil.GetK8sTokenSecret(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}
	token, ok := secret.Data["token"]
	if !ok || string(token) == "" {
		return nil, errors.Errorf("token data in secret %q not found", utils.ClientObjNS(secret))
	}

	config := &rest.Config{
		Host:            fmt.Sprintf("https://%s", bkeCluster.Spec.ControlPlaneEndpoint.String()),
		BearerToken:     string(token),
		BearerTokenFile: "",
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}
	return config, nil
}
