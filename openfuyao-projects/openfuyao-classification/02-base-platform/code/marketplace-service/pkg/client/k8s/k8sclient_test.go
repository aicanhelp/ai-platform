package k8s

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient_MissingConfig(t *testing.T) {
	client, err := NewKubernetesClient(&KubernetesCfg{})
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewKubernetesClient_SuccessAndGetters(t *testing.T) {
	patches := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(_ *rest.Config) (kubernetes.Interface, error) {
		return nil, nil
	}).ApplyFunc(snapshotclient.NewForConfig, func(_ *rest.Config) (snapshotclient.Interface, error) {
		return nil, nil
	}).ApplyFunc(apiextensionsclient.NewForConfig, func(_ *rest.Config) (apiextensionsclient.Interface, error) {
		return nil, nil
	})
	defer patches.Reset()

	cfg := &KubernetesCfg{
		KubeConfig: &rest.Config{},
		QPS:        123,
		Burst:      456,
	}
	client, err := NewKubernetesClient(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, client)
	assert.Equal(t, float32(123), client.Config().QPS)
	assert.Equal(t, 456, client.Config().Burst)
	assert.Nil(t, client.Kubernetes())
	assert.Nil(t, client.Snapshot())
	assert.Nil(t, client.ApiExtensions())
}
