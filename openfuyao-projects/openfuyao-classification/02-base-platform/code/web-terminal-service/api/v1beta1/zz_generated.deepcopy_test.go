package v1beta1

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestWebterminalTemplateDeepCopy(t *testing.T) {
	now := metav1.NewTime(time.Now())
	orig := &WebterminalTemplate{
		TypeMeta: metav1.TypeMeta{
			Kind:       "WebterminalTemplate",
			APIVersion: "terminal.openfuyao.com/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "wt",
			Namespace: "ns",
			Labels:    map[string]string{"k": "v"},
		},
		Spec: WebterminalTemplateSpec{
			DefaultImage:   "img",
			SessionTimeout: 10,
			ExistsTime:     now,
			RenewTime:      now,
			PodTemplate: PodTemplate{
				ObjectMeta: PodTemplateObjectMeta{
					Name:      "pod",
					Namespace: "ns",
				},
				Spec: PodTemplateSpec{
					Containers: []corev1.Container{{Name: "c1"}},
					Volumes:    []corev1.Volume{{Name: "v1"}},
				},
			},
		},
		Status: WebterminalTemplateStatus{
			Phase: WebTerminalTemplateRunning,
		},
	}

	copyObj := orig.DeepCopy()
	if assert.NotNil(t, copyObj) {
		assert.Equal(t, orig, copyObj)
		assert.NotSame(t, orig, copyObj)
		assert.NotSame(t, &orig.ObjectMeta, &copyObj.ObjectMeta)
		if orig.ObjectMeta.Labels != nil {
			assert.NotSame(t, orig.ObjectMeta.Labels, copyObj.ObjectMeta.Labels)
		}
	}
}

func TestWebterminalTemplateDeepCopyNil(t *testing.T) {
	var orig *WebterminalTemplate
	copyObj := orig.DeepCopy()
	assert.Nil(t, copyObj)
	assert.Nil(t, orig.DeepCopyObject())
}

func TestWebterminalTemplateDeepCopyObject(t *testing.T) {
	orig := &WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "wt"},
	}
	obj := orig.DeepCopyObject()
	assert.NotNil(t, obj)
}

func TestWebterminalTemplateListDeepCopy(t *testing.T) {
	list := &WebterminalTemplateList{
		Items: []WebterminalTemplate{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "one"},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "two"},
			},
		},
	}
	copyList := list.DeepCopy()
	if assert.NotNil(t, copyList) {
		assert.Equal(t, list.Items[0].Name, copyList.Items[0].Name)
		assert.Equal(t, list.Items[1].Name, copyList.Items[1].Name)
		assert.NotSame(t, list, copyList)
		assert.NotSame(t, &list.Items[0], &copyList.Items[0])
	}
}

func TestWebterminalTemplateListDeepCopyNil(t *testing.T) {
	var list *WebterminalTemplateList
	assert.Nil(t, list.DeepCopy())
	assert.Nil(t, list.DeepCopyObject())
}

func TestWebterminalTemplateSpecDeepCopy(t *testing.T) {
	spec := &WebterminalTemplateSpec{DefaultImage: "img"}
	copySpec := spec.DeepCopy()
	assert.NotNil(t, copySpec)
	assert.Equal(t, spec, copySpec)
	assert.NotSame(t, spec, copySpec)
}

func TestWebterminalTemplateStatusDeepCopy(t *testing.T) {
	status := &WebterminalTemplateStatus{Phase: WebTerminalTemplateStopped}
	copyStatus := status.DeepCopy()
	assert.NotNil(t, copyStatus)
	assert.Equal(t, status, copyStatus)
	assert.NotSame(t, status, copyStatus)
}
