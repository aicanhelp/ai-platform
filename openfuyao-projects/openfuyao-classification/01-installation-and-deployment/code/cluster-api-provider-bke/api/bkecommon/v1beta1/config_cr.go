/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package v1beta1

import "k8s.io/apimachinery/pkg/runtime/schema"

// KubeletConfigGVK returns the GroupVersionKind for KubeletConfig (KCT).
func KubeletConfigGVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   GVK.Group,
		Version: GVK.Version,
		Kind:    "KubeletConfig",
	}
}

// KubeletConfigGVR returns the GroupVersionResource for KubeletConfig (KCT).
func KubeletConfigGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    GVK.Group,
		Version:  GVK.Version,
		Resource: "kubeletconfigs",
	}
}

// ContainerdConfigGVK returns the GroupVersionKind for ContainerdConfig (CCT).
func ContainerdConfigGVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{
		Group:   GVK.Group,
		Version: GVK.Version,
		Kind:    "ContainerdConfig",
	}
}

// ContainerdConfigGVR returns the GroupVersionResource for ContainerdConfig (CCT).
func ContainerdConfigGVR() schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    GVK.Group,
		Version:  GVK.Version,
		Resource: "containerdconfigs",
	}
}
