package util

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/textproto"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/utils/k8sutil"
)

func TestGetFilePartFromMultipart(t *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "demo.txt")
	assert.NoError(t, err)
	_, err = part.Write([]byte("hello"))
	assert.NoError(t, err)
	assert.NoError(t, writer.Close())

	fileHeader := &multipart.FileHeader{
		Header: textproto.MIMEHeader{
			"boundary": []string{writer.Boundary()},
			"Boundary": []string{writer.Boundary()},
		},
	}
	reader, err := getFilePartFromMultipart(&MockFile{Data: body.Bytes()}, fileHeader)
	assert.NoError(t, err)
	if err != nil {
		return
	}
	content, err := io.ReadAll(reader)
	assert.NoError(t, err)
	assert.Equal(t, "hello", string(content))
}

func TestGetFilePartFromMultipartNoFilePart(t *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormField("not-file")
	assert.NoError(t, err)
	_, err = part.Write([]byte("hello"))
	assert.NoError(t, err)
	assert.NoError(t, writer.Close())

	fileHeader := &multipart.FileHeader{
		Header: textproto.MIMEHeader{
			"boundary": []string{writer.Boundary()},
			"Boundary": []string{writer.Boundary()},
		},
	}
	reader, err := getFilePartFromMultipart(&MockFile{Data: body.Bytes()}, fileHeader)
	assert.Error(t, err)
	assert.Nil(t, reader)
}

func TestReadFile(t *testing.T) {
	file := t.TempDir() + "/x.txt"
	assert.NoError(t, os.WriteFile(file, []byte("abc"), 0600))

	content, err := readFile(file)
	assert.NoError(t, err)
	assert.Equal(t, "abc", string(content))

	content, err = readFile(file + ".missing")
	assert.Error(t, err)
	assert.Nil(t, content)
}

func TestGetMarketplaceServiceConfigFallback(t *testing.T) {
	patch := gomonkey.ApplyFunc(k8sutil.GetConfigMap, func(_ kubernetes.Interface, _, _ string) (*v1.ConfigMap, error) {
		return nil, assert.AnError
	})
	defer patch.Reset()

	cfg, err := GetMarketplaceServiceConfig(nil)
	assert.Error(t, err)
	assert.Nil(t, cfg)
}

func TestGetMarketplaceServiceConfigFromConfigMap(t *testing.T) {
	patch := gomonkey.ApplyFunc(k8sutil.GetConfigMap, func(_ kubernetes.Interface, _, _ string) (*v1.ConfigMap, error) {
		return &v1.ConfigMap{
			Data: map[string]string{
				"local-harbor-host":   "host",
				"chart-limit":         "10",
				"marketplace-scenes":  "- scene1\n- scene2",
				"official-harbor-tags-URL": "/tags",
			},
		}, nil
	})
	defer patch.Reset()

	cfg, err := GetMarketplaceServiceConfig(nil)
	assert.NoError(t, err)
	assert.Equal(t, "host", cfg.LocalHarborHost)
	assert.Equal(t, "10", cfg.ChartLimit)
	assert.Equal(t, []string{"scene1", "scene2"}, cfg.MarketplaceScenes)
}

func TestParseConfigWithKnownKeys(t *testing.T) {
	cfg := parseConfig(map[string]string{
		"local-harbor-display-name": "local",
		"local-harbor-host":         "host",
		"local-harbor-project":      "/chartrepo/library",
		"chart-limit":               "100",
		"marketplace-scenes":        "- s1\n- s2",
	})
	assert.Equal(t, "local", cfg.LocalHarborDisplayName)
	assert.Equal(t, "host", cfg.LocalHarborHost)
	assert.Equal(t, "/chartrepo/library", cfg.LocalHarborProject)
	assert.Equal(t, "100", cfg.ChartLimit)
	assert.Equal(t, []string{"s1", "s2"}, cfg.MarketplaceScenes)
	assert.Equal(t, constant.MarketplaceServiceDefaultNamespace, constant.MarketplaceServiceDefaultNamespace)
}
