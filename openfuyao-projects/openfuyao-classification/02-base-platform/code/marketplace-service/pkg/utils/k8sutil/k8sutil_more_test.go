package k8sutil

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
)

func TestCrExists(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "test.io", Version: "v1", Resource: "samples"}
	scheme := runtime.NewScheme()

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "test.io/v1",
			"kind":       "Sample",
			"metadata": map[string]interface{}{
				"name":      "obj-1",
				"namespace": "default",
			},
		},
	}

	client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, map[schema.GroupVersionResource]string{
		gvr: "SampleList",
	}, obj)

	ok, err := CrExists(client, "obj-1", "default", gvr)
	assert.NoError(t, err)
	assert.True(t, ok)

	ok, err = CrExists(client, "obj-2", "default", gvr)
	assert.NoError(t, err)
	assert.False(t, ok)
}

func TestRegexError(t *testing.T) {
	err := (&RegexError{Message: "bad"}).Error()
	assert.Contains(t, err, "RegexError:")
	assert.Contains(t, err, "bad")
}

func TestCrExistsAdditionalPath(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "test.io", Version: "v1", Resource: "samples"}
	scheme := runtime.NewScheme()
	client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, map[schema.GroupVersionResource]string{
		gvr: "SampleList",
	})

	ok, err := CrExists(client, "obj", "invalid/namespace", gvr)
	assert.NoError(t, err)
	assert.False(t, ok)

	_, _ = client.Resource(gvr).Namespace("default").Create(context.Background(), &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "test.io/v1",
			"kind":       "Sample",
			"metadata": map[string]interface{}{
				"name": "obj-3",
			},
		},
	}, metav1.CreateOptions{})
}
