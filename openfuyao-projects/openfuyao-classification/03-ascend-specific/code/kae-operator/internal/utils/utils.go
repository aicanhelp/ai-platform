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

// package utils offers some utility functions
package utils

import (
	"fmt"
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

// SetStatus 设置 KAEPolicy 实例的状态和命名空间
func SetStatus(policy *kaev1.KAEPolicy, policyState kaev1.State, namespace string) {
	policy.Status.State = policyState
	policy.Status.Namespace = namespace
}

func getImagePath(repository, image, version, imagePathEnvName string) (string, error) {
	// 1. 通过字符串拼接
	if repository == "" && version == "" {
		if image != "" {
			return image, nil
		}
	} else {
		var imgPath string
		// 如果指定镜像摘要而不是标签，则使用 @
		if strings.HasPrefix(version, "sha256:") {
			imgPath = repository + "/" + image + "@" + version
		} else {
			imgPath = repository + "/" + image + ":" + version
		}
		imgPath = strings.TrimPrefix(imgPath, "/")
		return imgPath, nil
	}

	// 2. 通过环境变量获取
	if envImagePath := os.Getenv(imagePathEnvName); envImagePath != "" {
		return envImagePath, nil
	}

	// 3. 都获取不到
	return "", fmt.Errorf("failed to retrieve image path through both KAEPolicy CR or ENV %s", imagePathEnvName)
}

// Image 为给定的组件类型设置镜像路径
func Image(spec interface{}) (string, error) {
	if driverSpec, ok := spec.(*kaev1.DriverSpec); ok {
		return getImagePath(driverSpec.Repository, driverSpec.Image, driverSpec.Version, "KAE_DRIVER_IMAGE")
	} else if devicePluginSpec, ok := spec.(*kaev1.DevicePluginSpec); ok {
		return getImagePath(devicePluginSpec.Repository, devicePluginSpec.Image, devicePluginSpec.Version,
			"KAE_DEVICE_PLUGIN_IMAGE")
	} else {
		return "", fmt.Errorf("failed to construct image path: %v", spec)
	}
}

// ImagePullPolicy 根据字符串返回对应的镜像拉取策略
func ImagePullPolicy(pullPolicy string) corev1.PullPolicy {
	switch pullPolicy {
	case "IfNotPresent":
		return corev1.PullIfNotPresent
	case "Never":
		return corev1.PullNever
	case "Always":
		return corev1.PullAlways
	default:
		return corev1.PullIfNotPresent
	}
}

// IsDriverEnabled 如果启用了 kae 驱动程序安装，则返回 true。
func IsDriverEnabled(d kaev1.DriverSpec) bool {
	if d.Enabled == nil {
		return false
	}
	return *d.Enabled
}

// IsDriverEnabledCleanup 如果启用了服务结束时清理驱动，则返回 true。
func IsDriverEnabledCleanup(d kaev1.DriverSpec) bool {
	if d.EnabledCleanup == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *d.EnabledCleanup
}

// IsDevicePluginEnabledHPRE 如果启用了 kae hpre 设备插件程序安装，则返回 true
func IsDevicePluginEnabledHPRE(p kaev1.DevicePluginSpec) bool {
	if p.EnabledHPRE == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledHPRE
}

// IsDevicePluginEnabledSEC 如果启用了 kae sec 设备插件程序安装，则返回 true
func IsDevicePluginEnabledSEC(p kaev1.DevicePluginSpec) bool {
	if p.EnabledSEC == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledSEC
}

// IsDevicePluginEnabledZIP 如果启用了 kae zip 设备插件程序安装，则返回 true
func IsDevicePluginEnabledZIP(p kaev1.DevicePluginSpec) bool {
	if p.EnabledZIP == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledZIP
}

// IsPSAEnabled 如果启用 PodSecurityAdmission 配置，则返回 true
func IsPSAEnabled(p kaev1.PSASpec) bool {
	if p.Enabled == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.Enabled
}

// IsOperatorEnabledHPRE Operator 使能 HPRE 设备 返回 true
func IsOperatorEnabledHPRE(p kaev1.OperatorSpec) bool {
	if p.EnabledHPRE == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledHPRE
}

// IsOperatorEnabledSEC Operator 使能 SEC 设备 返回 true，预留接口，当前版本应配置为 false
func IsOperatorEnabledSEC(p kaev1.OperatorSpec) bool {
	if p.EnabledSEC == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledSEC
}

// IsOperatorEnabledZIP Operator 使能 ZIP 设备 返回 true，预留接口，当前版本应配置为 false
func IsOperatorEnabledZIP(p kaev1.OperatorSpec) bool {
	if p.EnabledZIP == nil {
		// 如果用户不指定，则默认为 false
		return false
	}
	return *p.EnabledZIP
}
