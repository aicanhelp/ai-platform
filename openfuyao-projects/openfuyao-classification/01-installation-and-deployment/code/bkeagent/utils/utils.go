package utils

import (
	"encoding/base64"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

// Exists 判断所给路径文件/文件夹是否存在
func Exists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	if os.IsNotExist(err) {
		return false
	}
	return false
}

func ContainsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

// SliceRemoveStrings remove all  the given string from the slice.
func SliceRemoveStrings(slice []string, s string) (result []string) {
	for _, item := range slice {
		if item == s {
			continue
		}
		result = append(result, item)
	}
	return
}

// SliceRemoveString removes the first occurrence of the given string from the slice.
func SliceRemoveString(slice []string, s string) (result []string) {
	for _, item := range slice {
		if item == s {
			continue
		}
		result = append(result, item)
	}
	return
}

// SliceContainsSlice returns true if the given dst slice is in the src slice.
func SliceContainsSlice(src []string, dst []string) bool {
	if src == nil {
		return false
	}
	for _, item := range dst {
		if !ContainsString(src, item) {
			return false
		}
	}
	return true
}

func UniqueStringSlice(slice []string) []string {
	m := make(map[string]struct{})
	for _, item := range slice {
		m[item] = struct{}{}
	}
	var result []string
	for k := range m {
		result = append(result, k)
	}
	return result
}

// IsDir 判断所给路径是否为文件夹
func IsDir(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return s.IsDir()
}

// IsFile 判断所给路径是否为文件
func IsFile(path string) bool {
	return !IsDir(path)
}

// SplitNameSpaceName split name space and name
func SplitNameSpaceName(nn string) (string, string, error) {
	ns := strings.Split(nn, ":")
	if len(ns) != 2 {
		return "", "", errors.New("invalid namespace:name format")
	}
	return ns[0], ns[1], nil
}

// ClusterName
// Deprecated
func ClusterName() (string, error) {
	clusterFilePath := filepath.Join(Workspace, "cluster")
	clusterName := ""
	if !Exists(clusterFilePath) {
		return "", errors.New("cluster file not exist")
	}
	b, err := ioutil.ReadFile(clusterFilePath)
	if err != nil {
		log.Warnf("Failed to read file %s", err.Error())
	} else if len(b) > 2 {
		clusterName = strings.Replace(string(b), "\n", "", -1)
	}
	return clusterName, nil
}

func HostName() string {
	// /proc/sys/kernel/hostname,it is an ephemeral hostname
	// "hostname ***" while change this file,but reboot will restore
	hostName, _ := os.Hostname()
	bkeNodeName := ""

	nodeFilePath := filepath.Join(Workspace, "node")
	if !Exists(nodeFilePath) {
		if err := os.WriteFile(nodeFilePath, []byte(hostName), 0644); err != nil {
			log.Warnf("node file not exit and failed to write node file %q err: %s", nodeFilePath, err.Error())
		}
		return hostName
	}

	b, err := os.ReadFile(nodeFilePath)
	if err != nil {
		log.Warnf("Failed to read file %s", err.Error())
	} else if len(b) > 2 {
		bkeNodeName = strings.Replace(string(b), "\n", "", -1)
	} else {
		if err := os.WriteFile(nodeFilePath, []byte(hostName), 0644); err != nil {
			log.Warnf("node file not exit and failed to write node file %q err: %s", nodeFilePath, err.Error())
		}
		return hostName
	}

	switch {
	case bkeNodeName == hostName:
		return bkeNodeName
	case bkeNodeName == "":
		return hostName
	case bkeNodeName != "" && hostName != "":
		return bkeNodeName
	default:
		return hostName
	}
}

func SetNTPServerEnv(server string) error {
	ntpServer, err := FormatNTPServer(server)
	if err != nil {
		return err
	}
	return os.Setenv(NTPServerEnvKey, ntpServer)
}

func GetNTPServerEnv() (string, error) {
	return FormatNTPServer(os.Getenv(NTPServerEnvKey))
}

func FormatNTPServer(server string) (string, error) {
	if server == "" {
		return "", nil
	}
	host, port, err := net.SplitHostPort(server)
	if err != nil {
		return "", err
	}
	if port == "" {
		return net.JoinHostPort(host, NTPServerPort), nil
	}

	return server, nil
}

// B64Encode base64 encode
func B64Encode(s string) string {
	return base64.StdEncoding.EncodeToString([]byte(s))
}

// B64Decode base64 decode
func B64Decode(s string) (string, error) {
	b, err := base64.StdEncoding.DecodeString(s)
	return string(b), err
}
