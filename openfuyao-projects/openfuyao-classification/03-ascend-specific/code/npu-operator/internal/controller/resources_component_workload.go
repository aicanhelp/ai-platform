// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const (
	inferOperatorDeploymentName       = "infer-operator-manager"
	rdmaDevicePluginDaemonSetName     = "rdma-shared-dp-ds"
	inferOperatorConfigMapName        = "infer-operator-config"
	rdmaDevicePluginConfigMapName     = "rdma-devices"
	componentConfigHashAnnotation     = "openfuyao.com/component-config-hash"
	inferOperatorDefaultHealthPort    = 11254
	rdmaDevicePluginDefaultHealthPort = 11257
)

// A controller-local view lets distinct API specs reuse the workload transforms.
type componentWorkloadConfig struct {
	*v1.ComponentWorkloadSpec
	LogRotate *v1.LogRotate
	Config    map[string]string
}

type componentWorkloadConfigGetter func(*v1.NPUClusterPolicySpec) componentWorkloadConfig

func inferOperatorWorkloadConfig(spec *v1.NPUClusterPolicySpec) componentWorkloadConfig {
	conf := &spec.InferOperator
	return componentWorkloadConfig{
		ComponentWorkloadSpec: &conf.ComponentWorkloadSpec,
		LogRotate:             &conf.LogRotate,
		Config:                conf.Config,
	}
}

func rdmaDevicePluginWorkloadConfig(spec *v1.NPUClusterPolicySpec) componentWorkloadConfig {
	conf := &spec.RDMADevicePlugin
	return componentWorkloadConfig{
		ComponentWorkloadSpec: &conf.ComponentWorkloadSpec,
		LogRotate:             &conf.LogRotate,
		Config:                conf.Config,
	}
}

// Both components use one image for their main process and log setup init container.
func componentWorkloadTransformers[T appsv1.Deployment | appsv1.DaemonSet](
	get componentWorkloadConfigGetter, binary string, healthPort int,
) []componentResourceTransformer[T] {
	return []componentResourceTransformer[T]{
		componentConfigTransformer[T](get, binary),
		componentCommandTransformer[T](get, binary, healthPort),
		newWorkloadPlacementTransformer[T](func(s *v1.NPUClusterPolicySpec) *v1.PlacementSpec {
			return &get(s).PlacementSpec
		}),
		newWorkloadImageTransformer[T](func(s *v1.NPUClusterPolicySpec) []*v1.ImageSpec {
			image := get(s).Image
			initImage := image
			// Pull secrets belong to the Pod; do not add them twice for one image.
			initImage.ImagePullSecrets = nil
			return []*v1.ImageSpec{&image, &initImage}
		}),
		newWorkloadResourceRequirementsTransformer[T](func(s *v1.NPUClusterPolicySpec) *v1.ResourceRequirements {
			return &get(s).Resources
		}),
		newWorkloadVolumesTransformer[T](func(s *v1.NPUClusterPolicySpec) *v1.VolumesSpec {
			return &get(s).VolumeSpec
		}),
		newWorkloadEnvTransformer[T](func(s *v1.NPUClusterPolicySpec) []corev1.EnvVar {
			return get(s).Env
		}),
	}
}

func componentCommandTransformer[T appsv1.Deployment | appsv1.DaemonSet](
	get componentWorkloadConfigGetter, binary string, healthPort int,
) componentResourceTransformer[T] {
	template := newWorkloadPodTemplateSpecGetter[T]()
	return func(_ context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		conf := get(&r.instance.Spec)
		args, err := componentDefaultArgs(binary, healthPort, *conf.LogRotate)
		if err != nil {
			return err
		}
		pod := template(obj)
		for i := range pod.Spec.Containers {
			container := &pod.Spec.Containers[i]
			container.Command = []string{binary}
			container.Args = args
			if len(conf.Command.Command) > 0 {
				container.Command = conf.Command.Command
			}
			if len(conf.Command.Args) > 0 {
				container.Args = conf.Command.Args
			}
		}
		return nil
	}
}

func componentConfigTransformer[T appsv1.Deployment | appsv1.DaemonSet](
	get componentWorkloadConfigGetter, binary string,
) componentResourceTransformer[T] {
	template := newWorkloadPodTemplateSpecGetter[T]()
	return func(_ context.Context, r *NPUClusterPolicyReconciler, obj *T) error {
		config := get(&r.instance.Spec).Config
		if err := validateComponentConfig(binary, config); err != nil {
			return err
		}
		// Roll out configuration changes even if the process only reads config at startup.
		data, err := json.Marshal(config)
		if err != nil {
			return fmt.Errorf("hash %s configuration: %w", binary, err)
		}
		pod := template(obj)
		if pod.Annotations == nil {
			pod.Annotations = map[string]string{}
		}
		pod.Annotations[componentConfigHashAnnotation] = fmt.Sprintf("%x", sha256.Sum256(data))
		return nil
	}
}

func componentDefaultArgs(binary string, healthPort int, logging v1.LogRotate) ([]string, error) {
	if logging.LogLevel == "" {
		logging.LogLevel = v1.LogInfo
	}
	switch logging.LogLevel {
	case v1.LogDebug, v1.LogInfo, v1.LogWarning, v1.LogError, v1.LogCritical:
	default:
		return nil, fmt.Errorf("%s: unsupported log level %q", binary, logging.LogLevel)
	}
	if logging.LogFile == "" {
		logging.LogFile = fmt.Sprintf("/var/log/mindx-dl/%s/%s.log", binary, binary)
	}
	if logging.Rotate == 0 {
		logging.Rotate = 30
	}
	if logging.MaxAge == 0 {
		logging.MaxAge = 7
	}
	args := []string{
		"-logFile=" + logging.LogFile,
		fmt.Sprintf("-logLevel=%d", logging.LogLevel.Value()),
		fmt.Sprintf("-maxBackups=%d", logging.Rotate),
		fmt.Sprintf("-maxAge=%d", logging.MaxAge),
		"--enable-healthz=true",
		fmt.Sprintf("--healthz-address=%d", healthPort),
	}
	if binary == "k8s-rdma-shared-dp" {
		if logging.Compress {
			return nil, fmt.Errorf("RDMA device plugin does not support logRotate.compress")
		}
		args = append(args, "-config-file=/k8s-rdma-shared-dev-plugin/config.json")
	} else {
		args = append(args, "-isCompress="+strconv.FormatBool(logging.Compress))
	}
	return args, nil
}

func validateComponentConfig(binary string, config map[string]string) error {
	switch binary {
	case "infer-operator":
		if value, exists := config["fault_handling"]; exists && value != "true" && value != "false" {
			return fmt.Errorf("inferOperator.config.fault_handling must be true or false")
		}
	case "k8s-rdma-shared-dp":
		if value, exists := config["config.json"]; exists {
			var object map[string]json.RawMessage
			if err := json.Unmarshal([]byte(value), &object); err != nil || object == nil {
				return fmt.Errorf("rdmaDevicePlugin.config[config.json] must be a JSON object")
			}
		}
	}
	return nil
}

func applyComponentConfig(obj *corev1.ConfigMap, binary string, config map[string]string) error {
	if err := validateComponentConfig(binary, config); err != nil {
		return err
	}
	if obj.Data == nil {
		obj.Data = map[string]string{}
	}
	maps.Copy(obj.Data, config)
	return nil
}

func isInferOperatorCRD(obj *apiextensionsv1.CustomResourceDefinition) bool {
	switch obj.Name {
	case "inferservicesets.mindcluster.huawei.com", "inferservices.mindcluster.huawei.com", "instancesets.mindcluster.huawei.com":
		return true
	default:
		return false
	}
}

// Infer CRDs contain user workloads, so neither disabling nor policy garbage
// collection should delete them. Other components retain their existing behavior.
func (h CRDReconcileHooks) canDelete(ctx context.Context, r *NPUClusterPolicyReconciler,
	obj *apiextensionsv1.CustomResourceDefinition,
) (client.Object, bool, error) {
	if isInferOperatorCRD(obj) {
		return nil, false, nil
	}
	return h.defaultResourceReconcileHooks.canDelete(ctx, r, obj)
}
