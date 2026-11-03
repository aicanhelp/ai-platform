/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package warehouse

import (
	"fmt"
	"os"
	"path"
	"testing"
)

type args struct {
	port     string
	certPath string
	dir      string
	filename string
	content  string
}

type loopStruct struct {
	name    string
	args    args
	wantErr bool
}

func TestSetClientCertificate(t *testing.T) {
	clientCertificateTests := []loopStruct{
		{
			name: "test1",
			args: args{
				port: "40442",
			},
			wantErr: false,
		}, {
			name: "test2",
			args: args{
				port: "40443",
			},
			wantErr: false,
		},
	}
	for _, tt := range clientCertificateTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := SetClientCertificate(tt.args.port); (err != nil) != tt.wantErr {
			}
		})
	}
}

func TestSetClientLocalCertificate(t *testing.T) {

	clientLocalCertificateTests := []loopStruct{
		{
			name: "test3",
			args: args{
				port: "40442",
			},
			wantErr: false,
		}, {
			name: "test4",
			args: args{
				port: "40443",
			},
			wantErr: false,
		},
	}
	for _, tt := range clientLocalCertificateTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := SetClientLocalCertificate(tt.args.port); (err != nil) != tt.wantErr {
			}
		})
	}
}

func TestSetRegistryConfig(t *testing.T) {
	setRegistryConfigTests := []loopStruct{
		{
			name: "test5",
			args: args{
				certPath: fmt.Sprintf("%s/registry", os.TempDir()),
			},
			wantErr: false,
		}, {
			name: "test6",
			args: args{
				certPath: fmt.Sprintf("%s/registrya", os.TempDir()),
			},
			wantErr: false,
		},
	}
	for _, tt := range setRegistryConfigTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := SetRegistryConfig(tt.args.certPath); (err != nil) != tt.wantErr {
				t.Errorf("SetRegistryConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSetServerCertificate(t *testing.T) {
	tests := []loopStruct{
		{
			name: "test7",
			args: args{
				certPath: fmt.Sprintf("%s/registry", os.TempDir()),
			},
			wantErr: false,
		}, {
			name: "test8",
			args: args{
				certPath: fmt.Sprintf("%s/registrya", os.TempDir()),
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := SetServerCertificate(tt.args.certPath); (err != nil) != tt.wantErr {
				t.Errorf("SetServerCertificate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestEnsureDir(t *testing.T) {
	ensureDirTests := []loopStruct{
		{
			name: "test9",
			args: args{
				dir: fmt.Sprintf("%s/registry", os.TempDir()),
			},
			wantErr: false,
		}, {
			name: "test10",
			args: args{
				dir: fmt.Sprintf("%s/registrya", os.TempDir()),
			},
			wantErr: false,
		},
	}
	for _, tt := range ensureDirTests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ensureDir(tt.args.dir); (err != nil) != tt.wantErr {
				t.Errorf("ensureDir() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestWriteContentIfNotExists(t *testing.T) {
	fileMode := 0644
	perm := os.FileMode(fileMode)
	writeContentIfNotExistsTests := []loopStruct{
		{
			name: "test11",
			args: args{
				dir:      os.TempDir(),
				filename: "test.txt",
				content:  "this is test",
			},
			wantErr: false,
		}, {
			name: "test12",
			args: args{
				dir:      os.TempDir(),
				filename: "///test",
				content:  "this is test",
			},
			wantErr: false,
		},
	}
	for _, tt := range writeContentIfNotExistsTests {
		filePath := path.Join(tt.args.dir, tt.args.filename)

		err := os.WriteFile(filePath, []byte("内容"), perm)
		if err != nil {
			t.Error(err)
		}

		t.Run(tt.name, func(t *testing.T) {
			if err := writeContentIfNotExists(tt.args.dir, tt.args.filename, tt.args.content); (err != nil) != tt.wantErr {
				t.Errorf("writeContentIfNotExists() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
