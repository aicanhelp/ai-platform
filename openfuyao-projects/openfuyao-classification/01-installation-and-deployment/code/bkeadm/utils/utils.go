/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

import (
	"fmt"
	"math/rand/v2"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sunvim/dogesyncer/helper"
)

const (
	dnsPartA   = "8" + "."
	dnsPartB   = "8" + "."
	dnsPartC   = "8" + "."
	dnsPartD   = "8"
	dnsPortStr = ":53"
)

var defaultDNSServer = dnsPartA + dnsPartB + dnsPartC + dnsPartD + dnsPortStr

func Exists(path string) bool {
	_, err := os.Stat(path)
	if err != nil {
		return os.IsExist(err)
	}
	return true
}

func ContainsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func ContainsStringPrefix(slice []string, s string) bool {
	for _, item := range slice {
		if strings.HasPrefix(item, s) {
			return true
		}
	}
	return false
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
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !s.IsDir()
}

func AbsPath(path string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	} else {
		return filepath.Abs(path)
	}
}

func DirectoryIsEmpty(path string) bool {
	dir, err := os.ReadDir(path)
	if err != nil {
		return true
	}
	if len(dir) == 0 {
		return true
	}
	return false
}

// GetOutBoundIP returns the outbound IP address by connecting to a DNS server
func GetOutBoundIP() (string, error) {
	conn, err := net.Dial("udp", defaultDNSServer)
	if err != nil {
		return "", err
	}
	defer conn.Close()
	localAddr, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok {
		return "", fmt.Errorf("failed to get UDP address")
	}
	ip := strings.Split(localAddr.String(), ":")[0]
	return ip, nil
}

// GetIntranetIp returns the intranet IP address from network interfaces
func GetIntranetIp() (string, error) {
	addresses, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}
	for _, address := range addresses {
		if ipNet, ok := address.(*net.IPNet); ok && !ipNet.IP.IsLoopback() {
			if ipNet.IP.To4() != nil {
				return ipNet.IP.String(), nil
			}
		}
	}
	return "", nil
}

func IsNum(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func CopyFile(sourceFile, destinationFile string) error {
	input, err := os.ReadFile(sourceFile)
	if err != nil {
		return err
	}
	err = os.WriteFile(destinationFile, input, DefaultFilePermission)
	if err != nil {
		return err
	}
	return nil
}

func RemoveStringObject(array []string, obj string) []string {
	// Remove string object from array
	// 旧实现会原地删除第一个匹配后继续扫描，可能遗漏连续重复项
	// 新实现创建新切片，确保所有匹配项均被删除
	result := make([]string, 0, len(array))
	for _, item := range array {
		if item != obj {
			result = append(result, item)
		}
	}
	return result
}

func IsChanClosed(ch interface{}) bool {
	return helper.IsChanClosed(ch)
}

// ReverseArray reverse array
func ReverseArray(arr []string) []string {
	for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
	return arr
}

// CopyMap copy map
func CopyMap(originalMap map[string]string) map[string]string {
	copiedMap := make(map[string]string)
	for key, value := range originalMap {
		copiedMap[key] = value
	}
	return copiedMap
}

func CopyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		dstPath := filepath.Join(dst, path[len(src):])
		if info.IsDir() {
			if !Exists(dstPath) {
				err = os.MkdirAll(dstPath, info.Mode())
				if err != nil {
					return err
				}
			}
		} else {
			if !Exists(dstPath) {
				err = CopyFile(path, dstPath)
				if err != nil {
					return err
				}
			}
		}
		return nil
	})
}

func LoopIP(domain string) ([]string, error) {
	var ip4 []string
	ips, err := net.LookupIP(domain)
	if err != nil {
		return ip4, err
	}
	for _, ip := range ips {
		if ip.To4() != nil {
			ip4 = append(ip4, ip.To4().String())
		}
	}
	return ip4, err
}

func RandInt(min, max int) int {
	if min > max {
		min, max = max, min
	}
	return min + rand.IntN(max-min+1)
}

// PromptForConfirmation prompts the user for Y/N confirmation.
// Returns true if user confirms, false if user cancels.
// If skipPrompt is true, returns true without prompting.
func PromptForConfirmation(skipPrompt bool) bool {
	if skipPrompt {
		return true
	}

	con := ""
	for {
		fmt.Print("Confirm the parameters, press Y to continue N will exit. [Y/N]? ")
		_, err := fmt.Scan(&con)
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
		}
		if !ContainsString([]string{"N", "Y"}, strings.ToUpper(con)) {
			continue
		}
		break
	}

	return strings.ToUpper(con) == "Y"
}

// FileExists check file exist or not
func FileExists(filename string) bool {
	_, err := os.Stat(filename)
	if err == nil {
		return true
	}
	if os.IsNotExist(err) {
		return false
	}
	return false
}

// WriteCommon define write data to file
func WriteCommon(file, data string) error {
	f, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE, DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", file, err)
	}

	var writeErr error
	if _, writeErr = f.WriteString(data); writeErr != nil {
		closeErr := f.Close()
		if closeErr != nil {
			return fmt.Errorf("write file %s: %w (and failed to close: %v)", file, writeErr, closeErr)
		}
		return fmt.Errorf("failed to write to file %s: %w", file, writeErr)
	}

	if err = f.Close(); err != nil {
		return fmt.Errorf("failed to close file %s: %w", file, err)
	}

	return nil
}
