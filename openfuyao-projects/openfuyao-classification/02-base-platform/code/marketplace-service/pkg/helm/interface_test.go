package helm

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestHelmClientSetters(t *testing.T) {
	client := &helmClient{}
	client.SetDynamicClient(nil)
	client.SetK8sClient(nil)
	assert.Nil(t, client.dynamicClient)
	assert.Nil(t, client.clientset)
}

func TestNewHelmOperation(t *testing.T) {
	cfg := &rest.Config{}

	patches := gomonkey.ApplyFunc(dynamic.NewForConfig, func(_ *rest.Config) (dynamic.Interface, error) {
		return nil, nil
	}).ApplyFunc(kubernetes.NewForConfig, func(_ *rest.Config) (kubernetes.Interface, error) {
		return nil, nil
	})
	defer patches.Reset()

	op, err := NewHelmOperation(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, op)
}
