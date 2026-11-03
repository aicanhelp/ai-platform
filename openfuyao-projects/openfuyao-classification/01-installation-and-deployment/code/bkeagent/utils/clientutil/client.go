package clientutil

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
)

// ClientSetFromManagerClusterSecret returns a ready-to-use client from a kubeconfig secret
func ClientSetFromManagerClusterSecret(nsName ...string) (*kubernetes.Clientset, error) {
	path := pkiutil.GetDefaultKubeConfigPath()

	// if ha we need use the ha kubeconfig, to get the clientset
	if len(nsName) == 2 {
		//step 1 get manager cluster clientset
		managerClusterConfig := fmt.Sprintf("%s/%s", utils.Workspace, "config")
		config, err := clientcmd.BuildConfigFromFlags("", managerClusterConfig)
		if err != nil {
			return nil, err
		}
		if err != nil {
			return nil, errors.Errorf("failed to get manger cluster rest config: %v", err)
		}
		clientSet, err := kubernetes.NewForConfig(config)
		if err != nil {
			return nil, err
		}
		// step 2 get ha kubeconfig secret
		secretName := fmt.Sprintf("%s-kubeconfig", nsName[1])

		secret, err := clientSet.CoreV1().Secrets(nsName[0]).Get(context.Background(), secretName, metav1.GetOptions{})
		if err != nil {
			return nil, errors.Errorf("failed to get ha kubeconfig secret: %v", err)
		}

		if err := pkiutil.StoreClusterAPICert(secret, os.TempDir()); err != nil {
			return nil, errors.Errorf("failed to store ha kubeconfig secret to %s, err: %v", os.TempDir(), err)
		}
		path = filepath.Join(os.TempDir(), pkiutil.BKEAdminKubeConfig().BaseName+".conf")
	}

	return ClientSetFromFile(path)
}

// ClientSetFromFile returns a ready-to-use client from a kubeconfig file
func ClientSetFromFile(path string) (*kubernetes.Clientset, error) {
	config, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load admin kubeconfig")
	}
	return ToClientSet(config)
}

// ToClientSet converts a KubeConfig object to a client
func ToClientSet(config *clientcmdapi.Config) (*kubernetes.Clientset, error) {
	overrides := clientcmd.ConfigOverrides{Timeout: "10s"}
	clientConfig, err := clientcmd.NewDefaultClientConfig(*config, &overrides).ClientConfig()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create API client configuration from kubeconfig")
	}

	c, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create API client")
	}
	return c, nil
}

type Client struct {
	ClientSet     *kubernetes.Clientset
	DynamicClient dynamic.Interface
}

var addToScheme sync.Once

func NewKubernetesClient(path string) (*Client, error) {
	config, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load admin kubeconfig")
	}
	overrides := clientcmd.ConfigOverrides{Timeout: "10s"}
	restConfig, err := clientcmd.NewDefaultClientConfig(*config, &overrides).ClientConfig()
	if err != nil {
		log.Infof("get rest config failed, err: %v", err)
		return nil, errors.Errorf("failed to get rest config: %v", err)
	}
	clientSet, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		log.Infof("Failed to initialize kubernetes clientset")
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		log.Infof("Failed to initialize kubernetes dynamic client")
		return nil, err
	}
	// support CRD v1 || v1beta1
	addToScheme.Do(func() {
		if err := apiextv1.AddToScheme(scheme.Scheme); err != nil {
			// This should never happen.
			panic(err)
		}
		if err := apiextv1beta1.AddToScheme(scheme.Scheme); err != nil {
			panic(err)
		}
	})

	return &Client{
		ClientSet:     clientSet,
		DynamicClient: dynamicClient,
	}, nil
}
