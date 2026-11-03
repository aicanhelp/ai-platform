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

func TestHost_Validate(t *testing.T) {
	tests := []struct {
		name    string
		host    Host
		wantErr bool
	}{
		{
			name:    "empty user",
			host:    Host{User: "", Password: "pass", Address: "192.168.1.1", Port: "22"},
			wantErr: true,
		},
		{
			name:    "no password and no ssh key",
			host:    Host{User: "root", Password: "", SSHKey: "", Address: "192.168.1.1", Port: "22"},
			wantErr: true,
		},
		{
			name:    "empty address",
			host:    Host{User: "root", Password: "pass", Address: "", Port: "22"},
			wantErr: true,
		},
		{
			name:    "invalid ip",
			host:    Host{User: "root", Password: "pass", Address: "invalid", Port: "22"},
			wantErr: true,
		},
		{
			name:    "empty port",
			host:    Host{User: "root", Password: "pass", Address: "192.168.1.1", Port: ""},
			wantErr: true,
		},
		{
			name:    "valid host",
			host:    Host{User: "root", Password: "pass", Address: "192.168.1.1", Port: "22"},
			wantErr: false,
		},
		{
			name:    "valid with ssh key",
			host:    Host{User: "root", SSHKey: "key", Address: "192.168.1.1", Port: "22"},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.host.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestHost_Fields(t *testing.T) {
	host := Host{
		User:     "root",
		Password: "pass",
		Address:  "192.168.1.1",
		Port:     "22",
		SSHKey:   "key",
	}

	fields := host.Fields()
	if fields.User != "root" {
		t.Errorf("expected user root, got %s", fields.User)
	}
	if fields.Password != "pass" {
		t.Errorf("expected password pass, got %s", fields.Password)
	}
	if fields.Address != "192.168.1.1" {
		t.Errorf("expected address 192.168.1.1, got %s", fields.Address)
	}
	if fields.Port != "22" {
		t.Errorf("expected port 22, got %s", fields.Port)
	}
	if fields.SSHKey != "key" {
		t.Error("expected sshKey key")
	}
}

func TestHosts_Remove(t *testing.T) {
	hosts := Hosts{
		{Address: "192.168.1.1"},
		{Address: "192.168.1.2"},
		{Address: "192.168.1.3"},
	}

	hosts.Remove("192.168.1.2")

	if len(hosts) != 2 {
		t.Errorf("expected 2 hosts, got %d", len(hosts))
	}

	for _, h := range hosts {
		if h.Address == "192.168.1.2" {
			t.Error("host 192.168.1.2 should be removed")
		}
	}
}

func TestHosts_Remove_NonExistent(t *testing.T) {
	hosts := Hosts{
		{Address: "192.168.1.1"},
	}
	hosts.Remove("192.168.1.99")
	if len(hosts) != 1 {
		t.Errorf("expected 1 host, got %d", len(hosts))
	}
}
