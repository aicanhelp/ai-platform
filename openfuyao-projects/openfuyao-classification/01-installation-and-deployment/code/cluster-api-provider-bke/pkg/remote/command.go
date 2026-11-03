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

package remote

import (
	"fmt"
	"strings"
)

// Commands defines three types of structures used to describe remote commands and file operations
type Commands []string

type Command struct {
	Cmds   Commands `json:"cmds,omitempty"`   // List of commands to execute
	FileUp []File   `json:"fileUp,omitempty"` // List of files to upload
}
type File struct {
	Src string `json:"src,omitempty"` // Source file path
	Dst string `json:"dst,omitempty"` // Destination file path
}

func (c *Command) String() string {
	return strings.Join(c.Cmds, " && ")
}

func (c *Command) List() []string {
	return c.Cmds
}

func (c *Command) Sudo(user string) {
	if user != "root" {
		for i, cmd := range c.Cmds {
			if strings.HasPrefix(cmd, "sudo") {
				continue
			}
			c.Cmds[i] = fmt.Sprintf("sudo %s", cmd)
		}
	}
}
