package util

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDecodeBase64String(t *testing.T) {
	input := base64.StdEncoding.EncodeToString([]byte("openfuyao"))
	got, err := DecodeBase64String(input)
	assert.NoError(t, err)
	assert.Equal(t, "openfuyao", got)
}

func TestDecodeBase64String_Invalid(t *testing.T) {
	got, err := DecodeBase64String("not-base64-###")
	assert.Error(t, err)
	assert.Equal(t, "", got)
}

