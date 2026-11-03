// Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/util/yaml"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const (
	componentPathPrefix            = "/var/lib/npu-operator/components"
	driverComponent                = componentPathPrefix + "/driver"
	ociRuntimeComponent            = componentPathPrefix + "/oci-runtime"
	vcschedulerComponent           = componentPathPrefix + "/volcano/volcano-scheduler"
	vccontrollerComponent          = componentPathPrefix + "/volcano/volcano-controller"
	volcanoVNPUCRDsComponent       = componentPathPrefix + "/volcano/vnpu-crds"
	volcanoVNPUControllerComponent = componentPathPrefix + "/volcano/vnpu-controller"
	volcanoVNPUSchedulerComponent  = componentPathPrefix + "/volcano/vnpu-scheduler"
	volcanoVNPUNamespaceComponent  = componentPathPrefix + "/volcano/vnpu-namespaces"
	volcanoVNPUAdmissionComponent  = componentPathPrefix + "/volcano/vnpu-admission"
	trainerComponent               = componentPathPrefix + "/trainer"
	devicePluginComponent          = componentPathPrefix + "/device-plugin"
	nodeDComponent                 = componentPathPrefix + "/noded"
	inferOperatorComponent         = componentPathPrefix + "/infer-operator"
	rdmaDevicePluginComponent      = componentPathPrefix + "/rdma-device-plugin"
	exporterComponent              = componentPathPrefix + "/npu-exporter"
	clusterdComponent              = componentPathPrefix + "/clusterd"
	rsControllerComponent          = componentPathPrefix + "/resilience-controller"
	mindIOACPComponent             = componentPathPrefix + "/mindio/mindioacp"
	mindIOTFTComponent             = componentPathPrefix + "/mindio/mindiotft"
	vnpuNamespaceComponent         = componentPathPrefix + "/vnpu/namespace"
	vnpuClientUpdateComponent      = componentPathPrefix + "/vnpu/client-update"
	vnpuDevicePluginComponent      = componentPathPrefix + "/vnpu/device-plugin"
	vnpuExporterComponent          = componentPathPrefix + "/vnpu/exporter"
	vnpuExporterSMComponent        = componentPathPrefix + "/vnpu/exporter-servicemonitor"
	draKubeletPluginComponent      = componentPathPrefix + "/dra/kubelet-plugin"
	draDeviceClassesComponent      = componentPathPrefix + "/dra/deviceclasses"
	draSoftVNPUComponent           = componentPathPrefix + "/dra/soft-vnpu"
	draVCANNRTInstallerComponent   = componentPathPrefix + "/dra/vcannrt-installer"
	draWebhookComponent            = componentPathPrefix + "/dra/webhook"
)

const (
	vnpuClientUpdateStatus      = componentPathPrefix + "/vnpu-client-update"
	vnpuDevicePluginStatus      = componentPathPrefix + "/vnpu-device-plugin"
	vnpuExporterStatus          = componentPathPrefix + "/vnpu-exporter"
	volcanoVNPUControllerStatus = componentPathPrefix + "/volcano-vnpu-controller"
	volcanoVNPUSchedulerStatus  = componentPathPrefix + "/volcano-vnpu-scheduler"
	volcanoVNPUAdmissionStatus  = componentPathPrefix + "/volcano-vnpu-admission"
	draKubeletPluginStatus      = componentPathPrefix + "/dra-kubelet-plugin"
	draVCANNRTInstallerStatus   = componentPathPrefix + "/dra-vcannrt-installer"
	draWebhookStatus            = componentPathPrefix + "/dra-webhook"
)

var (
	componentPaths = allComponentPaths()

	isComponentManaged = map[string]func(*NPUClusterPolicyReconciler) bool{
		inferOperatorComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.InferOperator.Managed
		},
		rdmaDevicePluginComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.RDMADevicePlugin.Managed
		},
		driverComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.Driver.Managed
		},
		ociRuntimeComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.OCIRuntime.Managed
		},
		trainerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.Trainer.Managed
		},
		devicePluginComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DevicePlugin.Managed
		},
		nodeDComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.NodeD.Managed
		},
		vccontrollerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VCController.Managed &&
				volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorMindCluster
		},
		vcschedulerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VCScheduler.Managed &&
				volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorMindCluster
		},
		volcanoVNPUCRDsComponent: func(r *NPUClusterPolicyReconciler) bool {
			return volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorVNPU
		},
		volcanoVNPUControllerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VCController.Managed &&
				volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorVNPU
		},
		volcanoVNPUSchedulerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VCScheduler.Managed &&
				volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorVNPU
		},
		volcanoVNPUNamespaceComponent: func(r *NPUClusterPolicyReconciler) bool {
			return volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorVNPU
		},
		volcanoVNPUAdmissionComponent: func(r *NPUClusterPolicyReconciler) bool {
			return volcanoFlavor(&r.instance.Spec) == v1.VolcanoFlavorVNPU && volcanoAdmissionEnabled(&r.instance.Spec)
		},
		clusterdComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.ClusterD.Managed
		},
		rsControllerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.RSController.Managed
		},
		exporterComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.Exporter.Managed
		},
		mindIOTFTComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.MindIOTFT.Managed
		},
		mindIOACPComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.MindIOACP.Managed
		},
		vnpuNamespaceComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VNPU.Managed
		},
		vnpuClientUpdateComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VNPU.Managed && r.instance.Spec.VNPU.ClientUpdate.Managed
		},
		vnpuDevicePluginComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VNPU.Managed && r.instance.Spec.VNPU.DevicePlugin.Managed
		},
		vnpuExporterComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VNPU.Managed && r.instance.Spec.VNPU.Exporter.Managed
		},
		vnpuExporterSMComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.VNPU.Managed &&
				r.instance.Spec.VNPU.Exporter.Managed &&
				r.instance.Spec.VNPU.Exporter.ServiceMonitor.Enabled
		},
		draKubeletPluginComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DRA.Managed && r.instance.Spec.DRA.KubeletPlugin.Managed
		},
		draDeviceClassesComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DRA.Managed && r.instance.Spec.DRA.DeviceClasses.Managed
		},
		draSoftVNPUComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DRA.Managed && r.instance.Spec.DRA.SoftVNPU.Managed
		},
		draVCANNRTInstallerComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DRA.Managed &&
				r.instance.Spec.DRA.SoftVNPU.Managed &&
				r.instance.Spec.DRA.SoftVNPU.VCANNRTInstaller.Managed
		},
		draWebhookComponent: func(r *NPUClusterPolicyReconciler) bool {
			return r.instance.Spec.DRA.Managed && r.instance.Spec.DRA.Webhook.Managed
		},
	}
)

func allComponentPaths() []string {
	paths := append([]string{}, baseComponentPaths()...)
	paths = append(paths,
		vccontrollerComponent,
		vcschedulerComponent,
		volcanoVNPUCRDsComponent,
		volcanoVNPUNamespaceComponent,
		volcanoVNPUAdmissionComponent,
		volcanoVNPUControllerComponent,
		volcanoVNPUSchedulerComponent,
	)
	return append(paths, addonComponentPaths()...)
}

func baseComponentPaths() []string {
	return []string{
		driverComponent,
		ociRuntimeComponent,
		devicePluginComponent,
		trainerComponent,
		nodeDComponent,
	}
}

func addonComponentPaths() []string {
	return []string{
		clusterdComponent,
		rsControllerComponent,
		exporterComponent,
		mindIOTFTComponent,
		mindIOACPComponent,
		vnpuNamespaceComponent,
		vnpuClientUpdateComponent,
		vnpuDevicePluginComponent,
		vnpuExporterComponent,
		vnpuExporterSMComponent,
		draSoftVNPUComponent,
		draVCANNRTInstallerComponent,
		draKubeletPluginComponent,
		draDeviceClassesComponent,
		draWebhookComponent,
		inferOperatorComponent,
		rdmaDevicePluginComponent,
	}
}

func componentStatusName(componentName string) string {
	switch componentName {
	case vnpuNamespaceComponent,
		volcanoVNPUCRDsComponent,
		volcanoVNPUNamespaceComponent,
		draDeviceClassesComponent,
		draSoftVNPUComponent:
		return ""
	case vnpuClientUpdateComponent:
		return vnpuClientUpdateStatus
	case vnpuDevicePluginComponent:
		return vnpuDevicePluginStatus
	case vnpuExporterComponent, vnpuExporterSMComponent:
		return vnpuExporterStatus
	case volcanoVNPUControllerComponent:
		return volcanoVNPUControllerStatus
	case volcanoVNPUSchedulerComponent:
		return volcanoVNPUSchedulerStatus
	case volcanoVNPUAdmissionComponent:
		return volcanoVNPUAdmissionStatus
	case draKubeletPluginComponent:
		return draKubeletPluginStatus
	case draVCANNRTInstallerComponent:
		return draVCANNRTInstallerStatus
	case draWebhookComponent:
		return draWebhookStatus
	default:
		return componentName
	}
}

func orderedComponentStatusNames(components []component) []string {
	seen := map[string]bool{}
	var names []string
	for _, component := range components {
		name := componentStatusName(component.name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		names = append(names, name)
	}
	return names
}

func syncComponentStatuses(
	status *v1.NPUClusterPolicyStatus,
	components []component,
	states map[string]v1.ComponentState,
) bool {
	oldStatuses := append([]v1.ComponentStatus(nil), status.ComponentStatuses...)
	existing := map[string]v1.ComponentStatus{}
	for _, componentStatus := range status.ComponentStatuses {
		existing[componentStatus.Name] = componentStatus
	}

	status.ComponentStatuses = status.ComponentStatuses[:0]
	for _, name := range orderedComponentStatusNames(components) {
		componentStatus, ok := existing[name]
		if !ok {
			componentStatus = v1.ComponentStatus{
				Name:  name,
				State: v1.ComponentState{Phase: v1.ComponentPending, Reason: reconcilingReason},
			}
		}
		if state, ok := states[name]; ok && componentStatus.State != state {
			componentStatus.PrevState = componentStatus.State
			componentStatus.State = state
		}
		status.ComponentStatuses = append(status.ComponentStatuses, componentStatus)
	}
	return !reflect.DeepEqual(oldStatuses, status.ComponentStatuses)
}

func mergeComponentStates(current, next v1.ComponentState) v1.ComponentState {
	if current.Phase == "" {
		return next
	}
	if next.Phase == "" {
		return current
	}
	if componentPhaseRank(next.Phase) > componentPhaseRank(current.Phase) {
		return next
	}
	return current
}

func componentPhaseRank(phase v1.ComponentPhase) int {
	switch phase {
	case v1.ComponentPending:
		return 5
	case v1.ComponentTerminated:
		return 4
	case v1.ComponentDeploying:
		return 3
	case v1.ComponentRunning:
		return 2
	case v1.ComponentUnmanaged:
		return 1
	default:
		return 0
	}
}

type component struct {
	name      string
	resources []componentResource
}

func (c *component) loadResource(
	ctx context.Context,
	manifest []byte,
	serializer *json.Serializer,
) error {
	logger := log.FromContext(ctx)
	obj, gvk, err := serializer.Decode(manifest, nil, nil)
	if err != nil {
		unstructuredObj := &unstructured.Unstructured{}
		jsonManifest, jsonErr := yaml.ToJSON(manifest)
		if jsonErr != nil {
			logger.V(1).Info("Failed to convert resource to JSON", "manifest", string(manifest))
			return fmt.Errorf("convert manifest to JSON: %w", jsonErr)
		}
		if unmarshalErr := unstructuredObj.UnmarshalJSON(jsonManifest); unmarshalErr != nil {
			logger.V(1).Info("Failed to load resource", "manifest", string(manifest))
			return fmt.Errorf("decode manifest: %w", err)
		}
		obj = unstructuredObj
		gvkValue := unstructuredObj.GroupVersionKind()
		gvk = &gvkValue
	}
	creator, ok := resourceCreators[*gvk]
	if !ok {
		return fmt.Errorf("unsupported resource GVK %s", gvk.String())
	}
	res := creator(obj.(client.Object))
	c.resources = append(c.resources, res)
	logger.V(1).Info("Resource loaded", "gvk", res.gvk())
	return nil
}

func (c *component) reconcile(ctx context.Context, r *NPUClusterPolicyReconciler) (v1.ComponentState, error) {
	logger := log.FromContext(ctx)

	statusName := componentStatusName(c.name)
	currentState := c.currentState(r, statusName)
	resources, err := c.resourcesForReconcile(r)
	if err != nil {
		return currentState, err
	}
	if deployingState, err := c.markDeploying(ctx, r, statusName, currentState, resources); err != nil {
		logger.Error(err, "Failed to update deploying state")
		return deployingState, err
	}
	finalState := v1.ComponentState{Phase: v1.ComponentRunning, Reason: reconciledReason}
	isManaged := isComponentManaged[c.name](r)
	if !isManaged {
		logger.V(1).Info("Component is unmanaged")
		finalState = v1.ComponentState{Phase: v1.ComponentUnmanaged, Reason: componentUnmanagedReason}
	}
	if c.name == draDeviceClassesComponent {
		if err := reconcileDRADeviceClassGarbage(ctx, r); err != nil {
			return v1.ComponentState{
				Phase:   v1.ComponentPending,
				Reason:  reconcileFailedReason,
				Message: err.Error(),
			}, err
		}
	}
	activeResourceRefs, err := r.activeManagedResourceRefs()
	if err != nil {
		return currentState, err
	}
	for _, resource := range resources {
		resourceManaged := c.isResourceManaged(r, resource)
		if !resourceManaged && activeResourceRefs[resource.ref(r.namespace)] {
			continue
		}
		state, err := resource.clone().reconcile(ctx, r, resourceManaged)
		if err != nil {
			if state == nil {
				state = &v1.ComponentState{
					Phase:   v1.ComponentPending,
					Reason:  reconcileFailedReason,
					Message: err.Error(),
				}
			}
			return *state, err
		}
		if state != nil {
			finalState = *state
		}
	}
	logger.Info("Component reconciled",
		"state", finalState.Phase, "reason", finalState.Reason, "message", finalState.Message)

	return finalState, nil
}

// isResourceManaged keeps resource reconciliation and ownership protection consistent.
func (c *component) isResourceManaged(r *NPUClusterPolicyReconciler, resource componentResource) bool {
	if !isComponentManaged[c.name](r) {
		return false
	}
	if c.name == exporterComponent && resource.ref(r.namespace).gvk == gvkServiceMonitor {
		return r.instance.Spec.Exporter.ServiceMonitor.Enabled
	}
	return true
}

func (r *NPUClusterPolicyReconciler) activeManagedResourceRefs() (map[componentResourceRef]bool, error) {
	refs := map[componentResourceRef]bool{}
	for i := range r.components {
		component := &r.components[i]
		if !isComponentManaged[component.name](r) {
			continue
		}
		resources, err := component.resourcesForReconcile(r)
		if err != nil {
			return nil, err
		}
		for _, resource := range resources {
			if component.isResourceManaged(r, resource) {
				refs[resource.ref(r.namespace)] = true
			}
		}
	}
	return refs, nil
}

func (c *component) currentState(r *NPUClusterPolicyReconciler, statusName string) v1.ComponentState {
	if statusName == "" {
		return v1.ComponentState{Phase: v1.ComponentPending}
	}
	if cs := r.instance.Status.GetComponentStatus(statusName); cs != nil {
		return cs.State
	}
	return v1.ComponentState{Phase: v1.ComponentPending}
}

func (c *component) markDeploying(
	ctx context.Context,
	r *NPUClusterPolicyReconciler,
	statusName string,
	currentState v1.ComponentState,
	resources []componentResource,
) (v1.ComponentState, error) {
	deployingState := v1.ComponentState{Phase: v1.ComponentDeploying, Reason: reconcilingReason}
	if statusName == "" || len(resources) == 0 ||
		currentState.Phase == v1.ComponentRunning ||
		currentState.Phase == v1.ComponentUnmanaged {
		return deployingState, nil
	}
	err := r.updateStatus(ctx, func(s *v1.NPUClusterPolicyStatus) bool {
		return syncComponentStatuses(s, r.components, map[string]v1.ComponentState{statusName: deployingState})
	})
	return deployingState, err
}

func (c *component) resourcesForReconcile(r *NPUClusterPolicyReconciler) ([]componentResource, error) {
	resources := append([]componentResource{}, c.resources...)
	if c.name != draDeviceClassesComponent {
		return resources, nil
	}
	objects, err := buildDRADeviceClassTemplateObjects(r.instance.Spec.DRA.DeviceClasses.Templates)
	if err != nil {
		return nil, err
	}
	for _, obj := range objects {
		resources = append(resources, resourceCreators[gvkDeviceClass](obj))
	}
	return resources, nil
}

func (r *NPUClusterPolicyReconciler) loadComponents(ctx context.Context) error {
	// Only check the number of components, since it's non-trivial to check if all resources of an arbitrary component
	// is loaded successfully
	desiredPaths := resolveComponentPaths(&r.instance.Spec)
	if !componentPathsLoaded(r.components, desiredPaths) {
		r.components = nil
		for _, path := range desiredPaths {
			ctx := log.IntoContext(ctx, log.FromContext(ctx, "component", path))
			if err := r.loadComponent(ctx, path); err != nil {
				return fmt.Errorf("load component: %w", err)
			}
		}
		if err := r.updateStatus(ctx, func(s *v1.NPUClusterPolicyStatus) (changed bool) {
			changed = s.Phase != v1.PolicyNotReady
			if changed {
				s.Phase = v1.PolicyNotReady
			}
			return syncComponentStatuses(s, r.components, nil) || changed
		}); err != nil {
			log.FromContext(ctx).V(1).Error(nil, err.Error())
		}
	}
	return nil
}

func resolveComponentPaths(spec *v1.NPUClusterPolicySpec) []string {
	paths := append([]string{}, baseComponentPaths()...)
	switch volcanoFlavor(spec) {
	case v1.VolcanoFlavorVNPU:
		paths = append(paths,
			vccontrollerComponent,
			vcschedulerComponent,
			volcanoVNPUNamespaceComponent,
			volcanoVNPUCRDsComponent,
			volcanoVNPUControllerComponent,
			volcanoVNPUSchedulerComponent,
			volcanoVNPUAdmissionComponent,
		)
	case v1.VolcanoFlavorMindCluster:
		paths = append(paths,
			volcanoVNPUAdmissionComponent,
			volcanoVNPUControllerComponent,
			volcanoVNPUSchedulerComponent,
			vccontrollerComponent,
			vcschedulerComponent,
		)
	case v1.VolcanoFlavorExternal:
		paths = append(paths,
			volcanoVNPUAdmissionComponent,
			vccontrollerComponent,
			vcschedulerComponent,
			volcanoVNPUCRDsComponent,
			volcanoVNPUControllerComponent,
			volcanoVNPUSchedulerComponent,
		)
	default:
	}
	return append(paths, addonComponentPaths()...)
}

func componentPathsLoaded(components []component, paths []string) bool {
	if len(components) != len(paths) {
		return false
	}
	for i := range paths {
		if components[i].name != paths[i] {
			return false
		}
	}
	return true
}

func (r *NPUClusterPolicyReconciler) loadComponent(ctx context.Context, path string) error {
	logger := log.FromContext(ctx)
	ctx = log.IntoContext(ctx, logger)

	manifests := readManifests(ctx, path)
	serializer := json.NewSerializerWithOptions(
		json.DefaultMetaFactory,
		r.Scheme,
		r.Scheme,
		json.SerializerOptions{Yaml: true, Pretty: false, Strict: false},
	)
	// Load component resources
	component := component{name: path}
	for _, manifest := range manifests {
		if err := component.loadResource(ctx, manifest, serializer); err != nil {
			return fmt.Errorf("load resource from %s: %w", path, err)
		}
	}
	r.components = append(r.components, component)

	return nil
}

func (r *NPUClusterPolicyReconciler) reconcileComponents(ctx context.Context) (result ctrl.Result, err error) {
	logger := log.FromContext(ctx)

	var reconcileErrors []error
	allReady := true
	statusStates := map[string]v1.ComponentState{}
	for _, component := range r.components {
		// Preserve managed component ordering after a failure, but always attempt
		// cleanup of disabled components so unrelated errors cannot strand them.
		if len(reconcileErrors) > 0 && isComponentManaged[component.name](r) {
			continue
		}
		logger := logger.WithValues("component", component.name)
		ctx := log.IntoContext(ctx, logger)
		state, componentErr := component.reconcile(ctx, r)
		if statusName := componentStatusName(component.name); statusName != "" {
			statusStates[statusName] = mergeComponentStates(statusStates[statusName], state)
		}
		if componentErr != nil {
			reconcileErrors = append(reconcileErrors, fmt.Errorf("reconcile component (%s): %w", component.name, componentErr))
		}
		allReady = allReady && (state.Phase == v1.ComponentRunning || state.Phase == v1.ComponentUnmanaged)
	}
	if statusErr := r.updateStatus(ctx, func(s *v1.NPUClusterPolicyStatus) bool {
		return syncComponentStatuses(s, r.components, statusStates)
	}); statusErr != nil {
		log.FromContext(ctx).V(1).Error(nil, statusErr.Error())
	}
	if err = errors.Join(reconcileErrors...); err != nil {
		ignoreError(r.setConditionsError(ctx, componentNotReadyReason, err.Error()))
		return
	}

	switch {
	case !allReady:
		msg := "at least one component is not ready"
		logger.V(-1).Info(fmt.Sprintf("NPUClusterPolicy not ready, %s", msg))
		ignoreError(r.setConditionsError(ctx, componentNotReadyReason, msg))
		result.RequeueAfter = notReadyRequeueInterval
	case !r.hasNFDLabels:
		ignoreError(r.setConditionsReady(ctx, noNFDLabelsReason, "no NFD labels found"))
		result.RequeueAfter = noNFDLabelRequeueInterval
	case !r.hasNPUNodes:
		err = r.setConditionsReady(ctx, noNPUNodesReason, "no NPU nodes with supported CRI runtime found")
	default:
		err = r.setConditionsReady(ctx, reconciledReason, "all components have been successfully reconciled")
	}
	// Infer CRDs are retained without owner references. Periodically reconcile
	// them and supporting RBAC even when no owned workload emits an event.
	if err == nil && result.RequeueAfter == 0 && componentMaintenanceEnabled(&r.instance.Spec) {
		result.RequeueAfter = 3 * time.Minute
	}

	return
}

func componentMaintenanceEnabled(spec *v1.NPUClusterPolicySpec) bool {
	return spec.InferOperator.Managed || spec.RDMADevicePlugin.Managed ||
		(spec.NodeD.Managed && spec.NodeD.ContainerSnapshot.Enabled)
}

func readManifests(ctx context.Context, dir string) [][]byte {
	logger := log.FromContext(ctx)
	logger.Info("Reading component manifest files")
	manifests := [][]byte{}
	if err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		logger := logger.V(1).WithValues()
		if err != nil {
			logger.Error(err, "Failed to stat file", "path", path)
			return nil
		}
		if !info.IsDir() {
			manifest, err := os.ReadFile(path)
			if err != nil {
				logger.Error(err, "Failed to read file", "path", path)
				return nil
			}
			manifests = append(manifests, manifest)
		}
		return nil
	}); err != nil {
		logger.Error(err, "Failed to list component manifest directory")
	}
	return manifests
}
