/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package apps

import (
	"sync"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"

	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
)

var (
	schemeOnce     sync.Once
	schemeInstance *runtime.Scheme
)

// GetScheme 返回共享的 Scheme 实例
func GetScheme() *runtime.Scheme {
	schemeOnce.Do(func() {
		schemeInstance = runtime.NewScheme()
		utilruntime.Must(scheme.AddToScheme(schemeInstance))
		utilruntime.Must(colocationv1.AddToScheme(schemeInstance))
		schemeInstance.AddUnversionedTypes(v1.SchemeGroupVersion,
			&v1.UpdateOptions{},
			&v1.DeleteOptions{},
			&v1.CreateOptions{})
	})
	return schemeInstance
}
