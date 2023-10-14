/*

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

package xgboostjob

import (
	"testing"

	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	v1xgboost "github.com/kubeflow/xgboost-operator/pkg/apis/xgboostjob/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func NewXGBoostJobWithMaster(worker int) *v1xgboost.XGBoostJob {
	job := NewXGoostJob(worker)
	master := int32(1)
	masterReplicaSpec := &commonv1.ReplicaSpec{
		Replicas: &master,
		Template: NewXGBoostReplicaSpecTemplate(),
	}
	job.Spec.XGBReplicaSpecs[commonv1.ReplicaType(v1xgboost.XGBoostReplicaTypeMaster)] = masterReplicaSpec
	return job
}

func NewXGoostJob(worker int) *v1xgboost.XGBoostJob {

	job := &v1xgboost.XGBoostJob{
		TypeMeta: metav1.TypeMeta{
			Kind: v1xgboost.Kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-xgboostjob",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1xgboost.XGBoostJobSpec{
			XGBReplicaSpecs: make(map[commonv1.ReplicaType]*commonv1.ReplicaSpec),
		},
	}

	if worker > 0 {
		worker := int32(worker)
		workerReplicaSpec := &commonv1.ReplicaSpec{
			Replicas: &worker,
			Template: NewXGBoostReplicaSpecTemplate(),
		}
		job.Spec.XGBReplicaSpecs[commonv1.ReplicaType(v1xgboost.XGBoostReplicaTypeWorker)] = workerReplicaSpec
	}

	return job
}

func NewXGBoostReplicaSpecTemplate() v1.PodTemplateSpec {
	return v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				v1.Container{
					Name:  v1xgboost.DefaultContainerName,
					Image: "test-image-for-kubeflow-xgboost-operator:latest",
					Args:  []string{"Fake", "Fake"},
					Ports: []v1.ContainerPort{
						v1.ContainerPort{
							Name:          v1xgboost.DefaultContainerPortName,
							ContainerPort: v1xgboost.DefaultPort,
						},
					},
				},
			},
		},
	}
}

func TestClusterSpec(t *testing.T) {
	type tc struct {
		job                 *v1xgboost.XGBoostJob
		rt                  v1xgboost.XGBoostJobReplicaType
		index               string
		expectedClusterSpec map[string]string
	}
	testCase := []tc{
		tc{
			job:                 NewXGBoostJobWithMaster(0),
			rt:                  v1xgboost.XGBoostReplicaTypeMaster,
			index:               "0",
			expectedClusterSpec: map[string]string{"WORLD_SIZE": "1", "MASTER_PORT": "9999", "RANK": "0", "MASTER_ADDR": "test-xgboostjob-master-0"},
		},
		tc{
			job:                 NewXGBoostJobWithMaster(1),
			rt:                  v1xgboost.XGBoostReplicaTypeMaster,
			index:               "1",
			expectedClusterSpec: map[string]string{"WORLD_SIZE": "2", "MASTER_PORT": "9999", "RANK": "1", "MASTER_ADDR": "test-xgboostjob-master-0", "WORKER_PORT": "9999", "WORKER_ADDRS": "test-xgboostjob-worker-0"},
		},
		tc{
			job:                 NewXGBoostJobWithMaster(2),
			rt:                  v1xgboost.XGBoostReplicaTypeMaster,
			index:               "0",
			expectedClusterSpec: map[string]string{"WORLD_SIZE": "3", "MASTER_PORT": "9999", "RANK": "0", "MASTER_ADDR": "test-xgboostjob-master-0", "WORKER_PORT": "9999", "WORKER_ADDRS": "test-xgboostjob-worker-0,test-xgboostjob-worker-1"},
		},
		tc{
			job:                 NewXGBoostJobWithMaster(2),
			rt:                  v1xgboost.XGBoostReplicaTypeWorker,
			index:               "0",
			expectedClusterSpec: map[string]string{"WORLD_SIZE": "3", "MASTER_PORT": "9999", "RANK": "1", "MASTER_ADDR": "test-xgboostjob-master-0", "WORKER_PORT": "9999", "WORKER_ADDRS": "test-xgboostjob-worker-0,test-xgboostjob-worker-1"},
		},
		tc{
			job:                 NewXGBoostJobWithMaster(2),
			rt:                  v1xgboost.XGBoostReplicaTypeWorker,
			index:               "1",
			expectedClusterSpec: map[string]string{"WORLD_SIZE": "3", "MASTER_PORT": "9999", "RANK": "2", "MASTER_ADDR": "test-xgboostjob-master-0", "WORKER_PORT": "9999", "WORKER_ADDRS": "test-xgboostjob-worker-0,test-xgboostjob-worker-1"},
		},
	}
	for _, c := range testCase {
		demoTemplateSpec := c.job.Spec.XGBReplicaSpecs[commonv1.ReplicaType(c.rt)].Template
		if err := SetPodEnv(c.job, &demoTemplateSpec, string(c.rt), c.index); err != nil {
			t.Errorf("Failed to set cluster spec: %v", err)
		}
		actual := demoTemplateSpec.Spec.Containers[0].Env
		for _, env := range actual {
			if val, ok := c.expectedClusterSpec[env.Name]; ok {
				if val != env.Value {
					t.Errorf("For name %s Got %s. Expected %s ", env.Name, env.Value, c.expectedClusterSpec[env.Name])
				}
			}
		}
	}
}
