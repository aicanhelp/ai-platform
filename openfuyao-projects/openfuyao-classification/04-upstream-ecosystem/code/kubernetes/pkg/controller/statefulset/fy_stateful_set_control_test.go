/*
Copyright 2016 The Kubernetes Authors.
Copyright (c) 2025 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package statefulset

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/history"
	"k8s.io/kubernetes/pkg/features"
)

func setupControllerWithRecorder(client clientset.Interface) (*fakeObjectManager, *fakeStatefulSetStatusUpdater, StatefulSetControlInterface, *record.FakeRecorder) {
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	om := newFakeObjectManager(informerFactory)
	recorder := record.NewFakeRecorder(100)
	spc := NewStatefulPodControlFromManager(om, recorder)
	ssu := newFakeStatefulSetStatusUpdater(informerFactory.Apps().V1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc, ssu, history.NewFakeHistory(informerFactory.Apps().V1().ControllerRevisions()), recorder)

	// The informer is not started. The tests here manipulate the local cache (indexers) directly, and there is no waiting
	// for client state to sync. In fact, because the client is not updated during tests, informer updates will break tests
	// by unexpectedly deleting objects.
	//
	// TODO: It might be better to rewrite all these tests manipulate the client an explicitly sync to ensure consistent
	// state, or to create fake client that does not use a local cache.

	// The client is passed initial sets, so we have to put them in the local setsIndexer cache.
	if sets, err := client.AppsV1().StatefulSets("").List(context.TODO(), metav1.ListOptions{}); err != nil {
		panic(err)
	} else {
		for _, set := range sets.Items {
			if err := om.setsIndexer.Update(&set); err != nil {
				panic(err)
			}
		}
	}

	return om, ssu, ssc, recorder
}

func TestStatefulSetControlPVCResize(t *testing.T) {
	type testcase struct {
		name       string
		invariants func(set *apps.StatefulSet, om *fakeObjectManager) error
		initial    func() *apps.StatefulSet
		update     func(set *apps.StatefulSet) *apps.StatefulSet
		validate   func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string, updateError error) error
		features   map[featuregate.Feature]bool
	}

	testFn := func(test *testcase, t *testing.T) {
		set := test.initial()
		revert := enableFeaturesForTest(t, test.features)
		defer revert()
		client := fake.NewSimpleClientset(set)
		om, _, ssc, recorder := setupControllerWithRecorder(client)
		if err := scaleUpStatefulSetControl(set, ssc, om, test.invariants); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set, err := om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		oldSet := set.DeepCopy()
		set = test.update(set)
		updateErr := updateStatefulSetControl(set, ssc, om, assertUpdateInvariants)
		selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pods, err := om.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err = assertPVCResiszeUpdateInvariants(set, om); updateErr == nil && err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pvcs, err := getPVCsForSet(om, set, pods)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err := test.validate(set, oldSet, pods, pvcs, collectEvents(recorder.Events), updateErr); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
	}

	validatePVCsResizeEvents := func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, events []string, success bool) error {
		count := 0
		for _, pod := range pods {
			for _, ev := range events {
				ev = strings.ToLower(ev)
				if strings.Contains(ev, "success to resize") && strings.Contains(ev, pod.Name) && strings.Contains(ev, "success") {
					if success {
						count += 1
					} else {
						return fmt.Errorf("unexpected resize claim success: %s", ev)
					}
				}
				if strings.Contains(ev, "success to resize") && strings.Contains(ev, pod.Name) && strings.Contains(ev, "failed") {
					if success {
						return fmt.Errorf("unexpected resize claim error: %s", ev)
					} else {
						count += 1
					}
				}
			}
		}

		var want int
		if *set.Spec.Replicas > *oldSet.Spec.Replicas {
			// when we scale up the replica count, we don't resize the newly created PVC, so we compare with the old replica count
			want = int(*oldSet.Spec.Replicas)
		} else {
			want = int(*set.Spec.Replicas)
		}

		if count != want {
			return fmt.Errorf("want resize claim events count to be %d found %d", len(pods), count)
		}

		return nil
	}

	validatePVCsResized := func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string) error {
		for _, pvc := range pvcs {
			wantSize := *resource.NewQuantity(3, resource.BinarySI)
			foundSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if foundSize.String() != wantSize.String() {
				return fmt.Errorf("want pvc size to be %s found %s", wantSize.String(), foundSize.String())
			}
		}

		return validatePVCsResizeEvents(set, oldSet, pods, events, true)
	}

	tests := []testcase{
		{
			name:       "pod pvc resize",
			invariants: assertMonotonicInvariants,
			initial: func() *apps.StatefulSet {
				return newStatefulSet(3)
			},
			update: func(set *apps.StatefulSet) *apps.StatefulSet {
				set.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage] = *resource.NewQuantity(3, resource.BinarySI)
				return set
			},
			validate: func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string, updateError error) error {
				return validatePVCsResized(set, oldSet, pods, pvcs, events)
			},
			features: map[featuregate.Feature]bool{
				features.FYStatefulSetVolumeExpansion: true,
			},
		},
		{
			name:       "pod pvc resize and scale up",
			invariants: assertMonotonicInvariants,
			initial: func() *apps.StatefulSet {
				return newStatefulSet(3)
			},
			update: func(set *apps.StatefulSet) *apps.StatefulSet {
				set.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage] = *resource.NewQuantity(3, resource.BinarySI)
				*set.Spec.Replicas = 4
				return set
			},
			validate: func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string, updateError error) error {
				return validatePVCsResized(set, oldSet, pods, pvcs, events)
			},
			features: map[featuregate.Feature]bool{
				features.FYStatefulSetVolumeExpansion: true,
			},
		},
		{
			name:       "pod pvc resize and scale down",
			invariants: assertMonotonicInvariants,
			initial: func() *apps.StatefulSet {
				return newStatefulSet(3)
			},
			update: func(set *apps.StatefulSet) *apps.StatefulSet {
				set.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage] = *resource.NewQuantity(3, resource.BinarySI)
				*set.Spec.Replicas = 2
				return set
			},
			validate: func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string, updateError error) error {
				return validatePVCsResized(set, oldSet, pods, pvcs, events)
			},
			features: map[featuregate.Feature]bool{
				features.FYStatefulSetVolumeExpansion: true,
			},
		},
		{
			name:       "pod pvc resize and pod spec update",
			invariants: assertMonotonicInvariants,
			initial: func() *apps.StatefulSet {
				return newStatefulSet(3)
			},
			update: func(set *apps.StatefulSet) *apps.StatefulSet {
				set.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage] = *resource.NewQuantity(3, resource.BinarySI)
				set.Spec.Template.Spec.Containers[0].Image += "1"
				return set
			},
			validate: func(set *apps.StatefulSet, oldSet *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim, events []string, updateError error) error {
				return validatePVCsResized(set, oldSet, pods, pvcs, events)
			},
			features: map[featuregate.Feature]bool{
				features.FYStatefulSetVolumeExpansion: true,
			},
		},
	}
	for i := range tests {
		testFn(&tests[i], t)
	}
}

func TestStatefulSetControlPVCResizeRollback(t *testing.T) {
	type testcase struct {
		name             string
		invariants       func(set *apps.StatefulSet, om *fakeObjectManager) error
		initial          func() *apps.StatefulSet
		update           func(set *apps.StatefulSet) *apps.StatefulSet
		validateUpdate   func(set *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim) error
		validateRollback func(set *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim) error
		features         map[featuregate.Feature]bool
	}

	originalImage := newStatefulSet(3).Spec.Template.Spec.Containers[0].Image

	testFn := func(t *testing.T, test *testcase) {
		set := test.initial()
		revert := enableFeaturesForTest(t, test.features)
		defer revert()
		client := fake.NewSimpleClientset(set)
		om, _, ssc := setupController(client)
		if err := scaleUpStatefulSetControl(set, ssc, om, test.invariants); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set, err := om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set = test.update(set)
		if err := updateStatefulSetControl(set, ssc, om, assertUpdateInvariants); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pods, err := om.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pvcs, err := getPVCsForSet(om, set, pods)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err := test.validateUpdate(set, pods, pvcs); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		revisions, err := ssc.ListRevisions(set)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		history.SortControllerRevisions(revisions)
		set, err = ApplyRevision(set, revisions[0])
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err := updateStatefulSetControl(set, ssc, om, assertUpdateInvariants); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pods, err = om.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		pvcs, err = getPVCsForSet(om, set, pods)
		if err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
		if err := test.validateRollback(set, pods, pvcs); err != nil {
			t.Fatalf("%s: %s", test.name, err)
		}
	}

	// rollback shouldn't affect PVC resizes
	// i.e. if an update scales a PVC up, rollback shouldn't scale it back to its original size

	tests := []testcase{
		{
			name:       "monotonic image update and pvc resize",
			invariants: assertMonotonicInvariants,
			initial: func() *apps.StatefulSet {
				return newStatefulSet(3)
			},
			update: func(set *apps.StatefulSet) *apps.StatefulSet {
				set.Spec.Template.Spec.Containers[0].Image = "foo"
				set.Spec.VolumeClaimTemplates[0].Spec.Resources.Requests[v1.ResourceStorage] = *resource.NewQuantity(2, resource.BinarySI)
				return set
			},
			validateUpdate: func(set *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim) error {
				sort.Sort(ascendingOrdinal(pods))
				for i := range pods {
					if pods[i].Spec.Containers[0].Image != "foo" {
						return fmt.Errorf("want pod %s image foo found %s", pods[i].Name, pods[i].Spec.Containers[0].Image)
					}
				}
				for _, pvc := range pvcs {
					if !strings.Contains(pvc.Name, set.Spec.VolumeClaimTemplates[0].Name) {
						continue
					}

					wantSize := *resource.NewQuantity(2, resource.BinarySI)
					foundSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
					if foundSize.String() != wantSize.String() {
						return fmt.Errorf("want pvc size to be %s found %s", wantSize.String(), foundSize.String())
					}
				}
				return nil
			},
			validateRollback: func(set *apps.StatefulSet, pods []*v1.Pod, pvcs []*v1.PersistentVolumeClaim) error {
				sort.Sort(ascendingOrdinal(pods))
				for i := range pods {
					if pods[i].Spec.Containers[0].Image != originalImage {
						return fmt.Errorf("want pod %s image %s found %s", pods[i].Name, originalImage, pods[i].Spec.Containers[0].Image)
					}
				}
				for _, pvc := range pvcs {
					if !strings.Contains(pvc.Name, set.Spec.VolumeClaimTemplates[0].Name) {
						continue
					}

					// rollback shouldn't affect PVC resizes
					wantSize := *resource.NewQuantity(2, resource.BinarySI)
					foundSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
					if foundSize.String() != wantSize.String() {
						return fmt.Errorf("want pvc size to be %s found %s", wantSize.String(), foundSize.String())
					}
				}
				return nil
			},
			features: map[featuregate.Feature]bool{
				features.FYStatefulSetVolumeExpansion: true,
			},
		},
	}
	for i := range tests {
		testFn(t, &tests[i])
	}
}

func enableFeaturesForTest(t *testing.T, features map[featuregate.Feature]bool) func() {
	if features == nil {
		return func() {}
	}
	currentValues := make(map[featuregate.Feature]bool)
	for feature := range features {
		currentValues[feature] = utilfeature.DefaultFeatureGate.Enabled(feature)
	}
	for feature, value := range features {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, value)
	}
	return func() {
		for feature, value := range currentValues {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, value)
		}
	}
}

func assertPVCResiszeUpdateInvariants(set *apps.StatefulSet, om *fakeObjectManager) error {
	if err := assertUpdateInvariants(set, om); err != nil {
		return err
	}
	pods, err := getPodsForSet(om, set)
	if err != nil {
		return err
	}
	pvcs, err := getPVCsForSet(om, set, pods)
	if err != nil {
		return err
	}

	count := 0
	for _, pvc := range pvcs {
		belongsToSet := false
		for _, claimTpl := range set.Spec.VolumeClaimTemplates {
			if strings.HasPrefix(pvc.Name, fmt.Sprintf("%s-%s-", claimTpl.Name, set.Name)) {
				belongsToSet = true
				break
			}
		}
		if !belongsToSet {
			return fmt.Errorf("found PVC %s that doesn't belong to set %s", pvc.Name, set.Name)
		}
		count += 1
	}

	wantCount := int(*set.Spec.Replicas) * len(set.Spec.VolumeClaimTemplates)
	if count != wantCount {
		return fmt.Errorf("expected PVC count %d found %d", wantCount, count)
	}

	return nil
}

func getPodsForSet(om *fakeObjectManager, set *apps.StatefulSet) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	pods, err := om.podsLister.Pods(set.Namespace).List(selector)
	return pods, err
}

func getPVCsForSet(om *fakeObjectManager, set *apps.StatefulSet, pods []*v1.Pod) ([]*v1.PersistentVolumeClaim, error) {
	var pvcs []*v1.PersistentVolumeClaim
	for _, pvcTpl := range set.Spec.VolumeClaimTemplates {
		for _, pod := range pods {
			ordinal := getOrdinal(pod)
			pvcName := getPersistentVolumeClaimName(set, &pvcTpl, ordinal)
			pvc, err := om.claimsLister.PersistentVolumeClaims(set.Namespace).Get(pvcName)

			if err != nil {
				return nil, fmt.Errorf("failed to get PVC in namespace %s for claim name %s", set.Namespace, pvcTpl.Name)
			}
			pvcs = append(pvcs, pvc)
		}
	}
	return pvcs, nil
}

func (om *fakeObjectManager) PatchClaim(namespace, claimName string, data []byte) error {
	claim, err := om.claimsLister.PersistentVolumeClaims(namespace).Get(claimName)
	if err != nil {
		return fmt.Errorf("failed to find claim %s in namespace %s: %w", claimName, namespace, err)
	}

	claimBytes, err := json.Marshal(claim)
	if err != nil {
		return err
	}

	modified, err := strategicpatch.StrategicMergePatch(claimBytes, data, claim)
	if err != nil {
		return fmt.Errorf("failed to strategic merge patch claim %s in namespace %s: %w", claimName, namespace, err)
	}

	var modifiedClaim v1.PersistentVolumeClaim
	err = json.Unmarshal(modified, &modifiedClaim)
	if err != nil {
		return err
	}

	modifiedStorage := *modifiedClaim.Spec.Resources.Requests.Storage()
	currentStorage := *claim.Spec.Resources.Requests.Storage()

	if modifiedStorage.Cmp(currentStorage) < 0 {
		return fmt.Errorf("resized PVC size cannot be smaller that current size")
	}

	claim.Spec = modifiedClaim.Spec
	claim.Annotations = modifiedClaim.Annotations
	return om.claimsIndexer.Update(claim)
}
