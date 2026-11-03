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

package syscompat

import (
	"bufio"
	"os"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	BKEManagedHostsBegin  = "# BEGIN BKEADM MANAGED HOSTS"
	BKEManagedHostsEnd    = "# END BKEADM MANAGED HOSTS"
	BKEManagedSysctlBegin = "# BEGIN BKEADM MANAGED SYSCTL"
	BKEManagedSysctlEnd   = "# END BKEADM MANAGED SYSCTL"
)

func removeManagedBlock(lines []string, begin, end string) []string {
	result := make([]string, 0, len(lines))
	skip := false
	for _, line := range lines {
		switch strings.TrimSpace(line) {
		case begin:
			skip = true
			continue
		case end:
			skip = false
			continue
		default:
			if !skip {
				result = append(result, line)
			}
		}
	}
	return result
}

func removeManagedBlockFromContent(content, begin, end string) string {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	return strings.Join(removeManagedBlock(lines, begin, end), "\n")
}

// SetHosts modifies the /etc/hosts file by updating the IP address associated with a given host.
//
// Parameters:
// - ip: The new IP address to set.
// - host: The host for which to update the IP address.
//
// Returns:
// - error: An error if there was a problem updating the /etc/hosts file, or nil if the update was successful.
func SetHosts(ip, host string) error {
	// Open the /etc/hosts file
	file, err := os.Open("/etc/hosts")
	if err != nil {
		return err
	}
	defer file.Close()

	// Read each line of the file and check if the host exists
	lines := make([]string, 0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, host) {
			continue
		}
		lines = append(lines, line)
	}
	if err = scanner.Err(); err != nil {
		return err
	}

	lines = removeManagedBlock(lines, BKEManagedHostsBegin, BKEManagedHostsEnd)
	lines = append(lines, BKEManagedHostsBegin, ip+"\t"+host, BKEManagedHostsEnd)

	// Open the /etc/hosts file in write mode and truncate its contents
	file, err = os.OpenFile("/etc/hosts", os.O_WRONLY|os.O_TRUNC, utils.DefaultFilePermission)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write the updated lines to the file
	for _, line := range lines {
		_, err = file.WriteString(line + "\n")
		if err != nil {
			return err
		}
	}
	return nil
}
