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
	"testing"
)

func TestCommand_String(t *testing.T) {
	cmd := Command{
		Cmds: Commands{"ls -la", "pwd", "whoami"},
	}
	result := cmd.String()
	expected := "ls -la && pwd && whoami"
	if result != expected {
		t.Errorf("expected %s, got %s", expected, result)
	}
}

func TestCommand_List(t *testing.T) {
	cmd := Command{
		Cmds: Commands{"ls", "pwd"},
	}
	list := cmd.List()
	if len(list) != 2 {
		t.Errorf("expected 2 commands, got %d", len(list))
	}
}

func TestCommand_Sudo(t *testing.T) {
	tests := []struct {
		name     string
		user     string
		cmds     Commands
		expected []string
	}{
		{
			name:     "root user no sudo",
			user:     "root",
			cmds:     Commands{"ls", "pwd"},
			expected: []string{"ls", "pwd"},
		},
		{
			name:     "non-root user add sudo",
			user:     "user1",
			cmds:     Commands{"ls", "pwd"},
			expected: []string{"sudo ls", "sudo pwd"},
		},
		{
			name:     "already has sudo",
			user:     "user1",
			cmds:     Commands{"sudo ls", "pwd"},
			expected: []string{"sudo ls", "sudo pwd"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := Command{Cmds: tt.cmds}
			cmd.Sudo(tt.user)
			for i, c := range cmd.Cmds {
				if c != tt.expected[i] {
					t.Errorf("expected %s, got %s", tt.expected[i], c)
				}
			}
		})
	}
}
