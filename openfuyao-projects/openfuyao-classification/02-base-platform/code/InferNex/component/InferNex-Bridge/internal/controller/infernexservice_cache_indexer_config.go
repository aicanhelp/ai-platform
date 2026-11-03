/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */


package controller

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	envPodNamespace              = "POD_NAMESPACE"
	cacheIndexerConfigVolumeName = "config"
	cacheIndexerConfigMountPath  = "/etc/cache-indexer"
	cacheIndexerConfigFileKey    = "config.yaml"

	labelOpenFuyaoEngine       = "openfuyao.com/engine"
	labelOpenFuyaoKVManager    = "openfuyao.com/kvmanager"
	openfuyaoKVManagerMooncake = "mooncake"
	openfuyaoEngineValueVLLM   = "vllm"
	openfuyaoPDRoleAggregate     = "aggregate"
)

// cacheIndexerDiscoveryMode selects which label contract cache-indexer uses in ConfigMap.
// Direct InferNexService engines use infernex.io/pdEngineGroup or infernex.io/aggregateGroup
// with engineValue namespace.name (see infernexEngineGroupValue).
// KServe-linked (spec.sourceRef): engineKey is app.kubernetes.io/name (LLMISVC name) for PD and
// aggregate; aggregate LWS worker pods lack the kserve.io/component workload label.
type cacheIndexerDiscoveryMode int

const (
	cacheIndexerDiscoveryInsvcAggregate cacheIndexerDiscoveryMode = iota
	cacheIndexerDiscoveryInsvcPD
	cacheIndexerDiscoveryKServeWorkload
	cacheIndexerDiscoveryKServePD
)

type cacheIndexerDiscoveryProfile struct {
	engineKey    string
	engineValue  string
	pdRoleKey    string
	pdRoleValues []string
}

func cacheIndexerConfigMapName(ownerName string) string {
	return fmt.Sprintf("%s-%s-config", ownerName, cacheIndexerComponent)
}

func (r *InferNexServiceReconciler) resolveCacheIndexerDiscoveryMode(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) (cacheIndexerDiscoveryMode, error) {
	if owner.Spec.SourceRef != nil {
		hasPD, err := r.linkedLLMHasPDMode(ctx, owner)
		if err != nil {
			return 0, err
		}
		if hasPD {
			return cacheIndexerDiscoveryKServePD, nil
		}
		return cacheIndexerDiscoveryKServeWorkload, nil
	}
	if effectiveEngine != nil && EngineIsPDMode(effectiveEngine) {
		return cacheIndexerDiscoveryInsvcPD, nil
	}
	return cacheIndexerDiscoveryInsvcAggregate, nil
}

// kserveAggregateCacheIndexerProfile discovers KServe aggregate engines by
// app.kubernetes.io/name plus Deployment or LWS leader/worker component labels.
func kserveAggregateCacheIndexerProfile(engineValue string) cacheIndexerDiscoveryProfile {
	return cacheIndexerDiscoveryProfile{
		engineKey:   labelAppKubernetesIOName,
		engineValue: engineValue,
		pdRoleKey:   labelAppKubernetesIOComponent,
		pdRoleValues: []string{
			kserveWorkloadComponentDecode,
			kserveWorkloadComponentLeader,
			kserveWorkloadComponentWorker,
		},
	}
}

// discoveryProfileForCacheIndexer maps discovery mode to cache-indexer ConfigMap label keys.
// Insvc PD: prefill only (L1 kv-events). KServe PD: app.kubernetes.io/name + all prefill
// component suffixes (Deployment prefill, LWS leader/worker). KServe aggregate: app.kubernetes.io/name
// + Deployment workload or LWS leader/worker component suffixes.
func discoveryProfileForCacheIndexer(mode cacheIndexerDiscoveryMode, engineValue string) cacheIndexerDiscoveryProfile {
	switch mode {
	case cacheIndexerDiscoveryInsvcPD:
		return cacheIndexerDiscoveryProfile{
			engineKey:    labelInfernexPDEngineGroup,
			engineValue:  engineValue,
			pdRoleKey:    labelOpenFuyaoPDRole,
			pdRoleValues: []string{openfuyaoPDRolePrefill},
		}
	case cacheIndexerDiscoveryInsvcAggregate:
		return cacheIndexerDiscoveryProfile{
			engineKey:    labelInfernexAggregateGroup,
			engineValue:  engineValue,
			pdRoleKey:    labelOpenFuyaoPDRole,
			pdRoleValues: []string{openfuyaoPDRoleAggregate},
		}
	case cacheIndexerDiscoveryKServePD:
		return cacheIndexerDiscoveryProfile{
			engineKey:   labelAppKubernetesIOName,
			engineValue: engineValue,
			pdRoleKey:   labelAppKubernetesIOComponent,
			pdRoleValues: []string{
				kserveWorkloadComponentPrefill,
				kserveWorkloadComponentLeaderPrefill,
				kserveWorkloadComponentWorkerPrefill,
			},
		}
	case cacheIndexerDiscoveryKServeWorkload:
		fallthrough
	default:
		return kserveAggregateCacheIndexerProfile(engineValue)
	}
}

func cacheIndexerPDRoleYAMLFragment(values []string) string {
	var lines strings.Builder
	for _, v := range values {
		lines.WriteString("      - ")
		lines.WriteString(v)
		lines.WriteByte('\n')
	}
	return lines.String()
}

const cacheIndexerConfigYAMLTemplate = `http:
  addr: ":28080"
  shutdownTimeout: 10s
  readHeaderTimeout: 10s
  readTimeout: 30s
  writeTimeout: 30s
  idleTimeout: 120s
  maxHeaderBytes: 1048576
  maxHitRateBodyBytes: 4194304
log:
  level: info
blockKey:
  pythonHashSeed: "0"
  prefixCachingHashAlgo: sha256_cbor
  useIntBlockHashes: true
discovery:
  refreshInterval: 10s
  segmentsFetchTimeout: 2s
  labels:
    engineKey: %s
    engineValue: %s
    pdRoleKey: %s
    pdRoleValue:
%s    kvManagerKey: openfuyao.com/kvmanager
    kvManagerValue: mooncake
  vllm:
    zmqPortName: zmq-pub
  mooncakeMaster:
    httpPortName: http
    rpcPortName: rpc
    httpPort: 9003
    rpcPort: 0
ingest:
  l1:
    backoffInitial: 1s
    backoffMax: 30s
  l3:
    scheme: http
    pollInterval: 10s
    httpTimeout: 5s
`

func renderCacheIndexerConfigYAML(p cacheIndexerDiscoveryProfile) string {
	return fmt.Sprintf(
		cacheIndexerConfigYAMLTemplate,
		p.engineKey, p.engineValue, p.pdRoleKey, cacheIndexerPDRoleYAMLFragment(p.pdRoleValues),
	)
}

func cacheIndexerConfigYAML(mode cacheIndexerDiscoveryMode, namespace, name string) string {
	engineValue := infernexEngineGroupValue(namespace, name)
	if mode == cacheIndexerDiscoveryKServePD || mode == cacheIndexerDiscoveryKServeWorkload {
		// KServe engine pods use app.kubernetes.io/name=<LLMISVC name>, not namespace.name group.
		engineValue = name
	}
	p := discoveryProfileForCacheIndexer(mode, engineValue)
	return renderCacheIndexerConfigYAML(p)
}

func (r *InferNexServiceReconciler) reconcileCacheIndexerConfigMap(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	effectiveEngine *infernexv1alpha1.InferenceEngineSpec,
) (string, error) {
	mode, err := r.resolveCacheIndexerDiscoveryMode(ctx, owner, effectiveEngine)
	if err != nil {
		return "", err
	}
	name := cacheIndexerConfigMapName(owner.Name)
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, cm, func() error {
		if cm.Labels == nil {
			cm.Labels = map[string]string{}
		}
		cm.Labels["infernex.io/owner"] = owner.Name
		cm.Labels["infernex.io/component"] = cacheIndexerComponent
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		workloadName, _ := pdWorkloadIdentity(owner)
		cm.Data[cacheIndexerConfigFileKey] = cacheIndexerConfigYAML(mode, owner.Namespace, workloadName)
		return controllerutil.SetControllerReference(owner, cm, r.Scheme)
	})
	if err != nil {
		return "", err
	}
	return name, nil
}

func (r *InferNexServiceReconciler) deleteCacheIndexerConfigMap(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: cacheIndexerConfigMapName(owner.Name), Namespace: owner.Namespace}}
	if err := r.Delete(ctx, cm); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

func (r *InferNexServiceReconciler) pruneCacheIndexerConfigMap(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	if _, ok := desired[cacheIndexerComponent]; ok {
		return nil
	}
	return r.deleteCacheIndexerConfigMap(ctx, owner)
}

// mergeAggregateInferenceWorkloadPodTemplateLabels adds discovery labels for direct InferNexService
// aggregate engines (not KServe-linked workloads).
func mergeAggregateInferenceWorkloadPodTemplateLabels(tpl *corev1.PodTemplateSpec, namespace, workloadName string) {
	if tpl == nil || strings.TrimSpace(namespace) == "" || strings.TrimSpace(workloadName) == "" {
		return
	}
	mergePodTemplateLabelIfAbsent(tpl, labelInfernexAggregateGroup, infernexEngineGroupValue(namespace, workloadName))
	mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoEngine, openfuyaoEngineValueVLLM)
	mergePodTemplateLabelIfAbsent(tpl, labelOpenFuyaoPDRole, openfuyaoPDRoleAggregate)
}

func applyCacheIndexerPodTemplate(tpl *corev1.PodTemplateSpec, configMapName string) {
	if tpl == nil || strings.TrimSpace(configMapName) == "" {
		return
	}
	c := preferredContainer(tpl, "cache-indexer", "main")
	if c == nil {
		return
	}
	mergeEnvVarFromFieldRef(c, envPodNamespace, "metadata.namespace")
	ensureCacheIndexerProbes(c)
	ensureCacheIndexerConfigVolume(tpl, configMapName)
}

func mergeEnvVarFromFieldRef(c *corev1.Container, name, fieldPath string) {
	if c == nil || strings.TrimSpace(name) == "" || strings.TrimSpace(fieldPath) == "" {
		return
	}
	for i := range c.Env {
		if c.Env[i].Name == name {
			c.Env[i].ValueFrom = &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
			}
			c.Env[i].Value = ""
			return
		}
	}
	c.Env = append(c.Env, corev1.EnvVar{
		Name: name,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
		},
	})
}

func ensureCacheIndexerProbes(c *corev1.Container) {
	if c == nil {
		return
	}
	if c.LivenessProbe == nil {
		c.LivenessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromString("http")},
			},
			InitialDelaySeconds: 5,
			PeriodSeconds:       10,
		}
	}
	if c.ReadinessProbe == nil {
		c.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/readyz", Port: intstr.FromString("http")},
			},
			InitialDelaySeconds: 2,
			PeriodSeconds:       5,
		}
	}
}

func ensureCacheIndexerConfigVolume(tpl *corev1.PodTemplateSpec, configMapName string) {
	volumeFound := false
	for i := range tpl.Spec.Volumes {
		if tpl.Spec.Volumes[i].Name == cacheIndexerConfigVolumeName {
			tpl.Spec.Volumes[i].ConfigMap = &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
			}
			volumeFound = true
			break
		}
	}
	if !volumeFound {
		tpl.Spec.Volumes = append(tpl.Spec.Volumes, corev1.Volume{
			Name: cacheIndexerConfigVolumeName,
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
				},
			},
		})
	}
	c := preferredContainer(tpl, "cache-indexer", "main")
	if c == nil {
		return
	}
	mountFound := false
	for i := range c.VolumeMounts {
		if c.VolumeMounts[i].Name == cacheIndexerConfigVolumeName {
			c.VolumeMounts[i].MountPath = cacheIndexerConfigMountPath
			c.VolumeMounts[i].ReadOnly = true
			mountFound = true
			break
		}
	}
	if !mountFound {
		c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
			Name:      cacheIndexerConfigVolumeName,
			MountPath: cacheIndexerConfigMountPath,
			ReadOnly:  true,
		})
	}
}
