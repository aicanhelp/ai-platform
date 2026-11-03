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

package utils

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
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
	if len(ns) != NamespaceAndNameLen {
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
	b, err := os.ReadFile(clusterFilePath)
	if err != nil {
		log.Warnf("Failed to read file %s", err.Error())
		return clusterName, nil
	}
	if len(b) > MinimumClusterNameLength {
		clusterName = strings.Replace(string(b), "\n", "", -1)
	}
	return clusterName, nil
}

// ImageTagFromReference returns the tag part of a container image reference.
func ImageTagFromReference(image string) string {
	image = strings.TrimSpace(image)
	if image == "" {
		return ""
	}
	short := image
	if slash := strings.LastIndex(image, "/"); slash >= 0 {
		short = image[slash+1:]
	}
	colon := strings.LastIndex(short, ":")
	if colon < 0 {
		return ""
	}
	return short[colon+1:]
}

// StringSlicesEqual reports whether two string slices contain equal values in equal order.
func StringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func HostName() string {
	// /proc/sys/kernel/hostname,it is an ephemeral hostname
	// "hostname ***" while change this file,but reboot will restore
	hostName, err := os.Hostname()
	if err != nil {
		log.Warnf("Failed to get host name %s", err.Error())
		return ""
	}

	nodeFilePath := filepath.Join(Workspace, "node")
	if !Exists(nodeFilePath) {
		if err = os.WriteFile(nodeFilePath, []byte(hostName), RwRR); err != nil {
			log.Warnf("node file not exit and failed to write node file %q err: %s", nodeFilePath, err.Error())
		}
		return hostName
	}

	b, err := os.ReadFile(nodeFilePath)
	if err != nil {
		log.Warnf("Failed to read file %s", err.Error())
		return hostName
	}

	bkeNodeName := strings.TrimSpace(strings.Replace(string(b), "\n", "", -1))
	if bkeNodeName == "" {
		if err := os.WriteFile(nodeFilePath, []byte(hostName), RwRR); err != nil {
			log.Warnf("node file is empty and failed to write node file %q err: %s", nodeFilePath, err.Error())
		}
		return hostName
	}

	return bkeNodeName
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

// SliceRemoveString removes the first occurrence of the given string from the slice.
func SliceRemoveString(slice []string, s string) []string {
	result := make([]string, 0)
	for _, item := range slice {
		if item == s {
			continue
		}
		result = append(result, item)
	}
	return result
}

// SliceExcludeSlice removes all the elements of the given exclude slice from the src slice.
func SliceExcludeSlice(src []string, exclude []string) []string {
	result := make([]string, 0)
	for _, item := range exclude {
		if ContainsString(src, item) {
			src = SliceRemoveString(src, item)
		}
	}
	result = src
	return result
}

// ContainsString returns true if the given string is in the slice.
func ContainsString(slice []string, s string) bool {
	if len(slice) == 0 {
		return false
	}
	for _, item := range slice {
		if strings.Replace(item, "\n", "", -1) == s {
			return true
		}
	}
	return false
}

// SliceContainsSlice returns true if the given dst slice is in the src slice.
func SliceContainsSlice(src []string, dst []string) bool {
	if len(src) == 0 {
		return false
	}
	for _, item := range dst {
		if !ContainsString(src, item) {
			return false
		}
	}
	return true
}

// TrimSpaceSlice returns a new slice with all the string in the given slice trimmed.
func TrimSpaceSlice(slice []string) []string {
	result := make([]string, 0)
	// Define a set of empty or whitespace-only strings
	emptyStrings := map[string]bool{
		"":     true,
		"\n":   true,
		"\r\n": true,
		"\t":   true,
		" ":    true,
	}

	for _, item := range slice {
		// Check if item is in the set of empty strings
		if emptyStrings[item] {
			continue
		}
	}
	return result
}

func DeepCopy(src interface{}) interface{} {
	data, err := json.Marshal(src)
	if err != nil {
		return nil
	}
	var dst interface{}
	if err = json.Unmarshal(data, dst); err != nil {
		return nil
	}
	return dst
}

// GroupRun 用来按照指定协程数量，运行某一函数,并对错误统一处理
// thread    并发数量
// fn        工作函数，入参为interface{}，需自行在fn函数内对入参类型进行转换
// errFn     错误处理函数,入参为map[interface{}]error， 代表函数fn运行某一variable时,出现的错误
// variables 传入fn函数的参数
func GroupRun(thread int, errFn func(map[interface{}]error), fn func(interface{}) error, variables interface{}) (err error) {
	defer func() {
		if e := recover(); e != nil {
			err = errors.Errorf("%v", e)
		}
	}()
	var wg sync.WaitGroup
	var mu sync.Mutex
	failedMap := make(map[interface{}]error)

	var vs []interface{}
	if reflect.TypeOf(variables).Kind() == reflect.Slice {
		valueOf := reflect.ValueOf(variables)
		for i := 0; i < valueOf.Len(); i++ {
			vs = append(vs, valueOf.Index(i).Interface())
		}
	}

	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	for i := 0; i < len(vs); i += thread {
		func(vs []interface{}) {
			for _, v := range vs {
				wg.Add(1)
				go func(v interface{}) {
					defer wg.Done()
					if err := fn(v); err != nil {
						mu.Lock()
						failedMap[v] = err
						mu.Unlock()
					}
				}(v)
			}
			wg.Wait()
		}(vs[i:min(i+thread, len(vs))])
	}
	errFn(failedMap)
	return
}

func ClientObjNS(obj client.Object) string {
	if obj == nil {
		return "object is nil"
	}
	// 检查接口的具体值是否为nil指针
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr && v.IsNil() {
		return "obj is nil by reflect"
	}
	return fmt.Sprintf("%s/%s", obj.GetNamespace(), obj.GetName())
}

func CtxDone(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

// CommonPrefix Returns the longest common prefix with the most occurrences in a set of strings
func CommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	counter := make(map[string]int)

	prefix := strs[0]
	for i := 1; i < len(strs); i++ {
		prefix = CommonPrefixOfTwo(prefix, strs[i])
		if prefix == "" {
			continue
		}
		counter[prefix]++
	}

	max := 0
	for k, v := range counter {
		if v > max {
			max = v
			prefix = k
		}
	}
	return prefix
}

func CommonPrefixOfTwo(str1, str2 string) string {
	length := min(len(str1), len(str2))
	index := 0
	for index < length && str1[index] == str2[index] {
		index++
	}
	return str1[:index]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func RemoveDuplicateElement(s []string) []string {
	result := make([]string, 0)
	temp := map[string]struct{}{}
	for _, item := range s {
		if _, ok := temp[item]; !ok {
			temp[item] = struct{}{}
			result = append(result, item)
		}
	}
	return result
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

// MostCommonChar 获取字符切片中出现最多的字符
func MostCommonChar(s []string) string {
	if len(s) == 0 {
		return ""
	}

	m := make(map[string]int)
	for _, v := range s {
		m[v]++
	}

	max := 0
	var maxChar string
	for k, v := range m {
		if v > max {
			max = v
			maxChar = k
		}
	}
	return maxChar
}

func TimeNowStr() string {
	return time.Now().Format("2006-01-02 15:04:05")
}

func TimeFormat(t time.Time) string {
	return t.Format("2006-01-02 15:04:05")
}

// Random 生成指定范围的随机数
func Random(min, max int) int {
	rand.Seed(time.Now().UnixNano())
	return rand.Intn(max-min) + min
}

// GetExtraLoadBalanceIP retrieves the external load balancer IP from CustomExtra configuration.
// Returns the IP address string if found and valid, otherwise returns an empty string.
func GetExtraLoadBalanceIP(customExtra map[string]string) string {
	if customExtra == nil {
		return ""
	}

	if v, ok := customExtra["extraLoadBalanceIP"]; ok && v != "" {
		// Validate that it's a valid IP address
		if ip := net.ParseIP(v); ip != nil {
			return v
		}
	}
	return ""
}

func GetNodeIPFromCommandWaitResult(result string) string {
	nodeInfo := strings.Split(result, "/")
	nodeIP := nodeInfo[0]
	if len(nodeInfo) == 2 {
		nodeIP = nodeInfo[1]
	}
	return nodeIP
}
