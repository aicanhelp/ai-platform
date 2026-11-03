package k8sutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestCreateSecretNil(t *testing.T) {
	secret, err := CreateSecret(fake.NewSimpleClientset(), nil)
	assert.Error(t, err)
	assert.Nil(t, secret)
}

func TestUpdateSecretValidation(t *testing.T) {
	client := fake.NewSimpleClientset()

	secret, err := UpdateSecret(client, nil)
	assert.Error(t, err)
	assert.Nil(t, secret)

	secret, err = UpdateSecret(client, &v1.Secret{})
	assert.Error(t, err)
	assert.Nil(t, secret)

	secret, err = UpdateSecret(client, &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "a",
		},
	})
	assert.Error(t, err)
	assert.Nil(t, secret)
}
