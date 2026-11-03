/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package auth

import (
	"testing"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"
)

func TestAddToContainer(t *testing.T) {
	type args struct {
		c          *restful.Container
		kubeConfig *rest.Config
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			"TestRegisterAuthAPI",
			args{
				c:          restful.NewContainer(),
				kubeConfig: &rest.Config{},
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := AddToContainer(tt.args.c, tt.args.kubeConfig); (err != nil) != tt.wantErr {
				t.Errorf("AddToContainer() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
