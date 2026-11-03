/*
 *
 *  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

package ray

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	rayconfv1 "github.com/ray-project/kuberay/ray-operator/pkg/client/applyconfiguration/ray/v1"
	rayversionedv1 "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned/typed/ray/v1"
	"github.com/stretchr/testify/mock"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes"
	rest "k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/utils/httputil"
)

type MockClientset struct {
	rayV1 *MockRayClient
}

func (c *MockClientset) RayV1() rayversionedv1.RayV1Interface {
	return c.rayV1
}

func (c *MockClientset) Discovery() discovery.DiscoveryInterface {
	return nil
}

type MockRayClient struct {
	restClient rest.Interface
	mock.Mock
}

func (c *MockRayClient) RESTClient() rest.Interface {
	return nil
}

func (m *MockRayClient) RayClusters(namespace string) rayversionedv1.RayClusterInterface {
	args := m.Called(namespace)
	return args.Get(0).(rayversionedv1.RayClusterInterface)
}

func (m *MockRayClient) RayJobs(namespace string) rayversionedv1.RayJobInterface {
	args := m.Called(namespace)
	return args.Get(0).(rayversionedv1.RayJobInterface)
}

func (m *MockRayClient) RayServices(namespace string) rayversionedv1.RayServiceInterface {
	args := m.Called(namespace)
	return args.Get(0).(rayversionedv1.RayServiceInterface)
}

type MockRayCluster struct {
	mock.Mock
}

func (m *MockRayCluster) Create(ctx context.Context,
	rayCluster *rayv1.RayCluster, opts metav1.CreateOptions) (*rayv1.RayCluster, error) {
	args := m.Called(ctx, rayCluster, opts)
	return args.Get(0).(*rayv1.RayCluster), args.Error(1)
}

func (m *MockRayCluster) Update(ctx context.Context,
	rayCluster *rayv1.RayCluster, opts metav1.UpdateOptions) (*rayv1.RayCluster, error) {
	args := m.Called(ctx, rayCluster, opts)
	return args.Get(0).(*rayv1.RayCluster), args.Error(1)
}

func (m *MockRayCluster) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	args := m.Called(ctx, name, opts)
	return args.Error(0)
}

func (m *MockRayCluster) Get(ctx context.Context,
	name string, opts metav1.GetOptions) (*rayv1.RayCluster, error) {
	args := m.Called(ctx, name, opts)
	return args.Get(0).(*rayv1.RayCluster), args.Error(1)
}

func (m *MockRayCluster) List(ctx context.Context, opts metav1.ListOptions) (*rayv1.RayClusterList, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(*rayv1.RayClusterList), args.Error(1)
}

func (m *MockRayCluster) Watch(ctx context.Context,
	opts metav1.ListOptions) (watch.Interface, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(watch.Interface), args.Error(1)
}

func (m *MockRayCluster) UpdateStatus(ctx context.Context,
	rayCluster *rayv1.RayCluster, opts metav1.UpdateOptions) (*rayv1.RayCluster, error) {
	updateStatusArgs := m.Called(ctx, rayCluster, opts)
	return updateStatusArgs.Get(0).(*rayv1.RayCluster), updateStatusArgs.Error(1)
}

func (m *MockRayCluster) Patch(ctx context.Context, name string,
	pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (*rayv1.RayCluster, error) {
	args := m.Called(ctx, name, pt, data, opts, subresources)
	return args.Get(0).(*rayv1.RayCluster), args.Error(1)
}

func (m *MockRayCluster) DeleteCollection(ctx context.Context,
	opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	args := m.Called(ctx, opts, listOpts)
	return args.Error(0)
}
func (m *MockRayCluster) Apply(ctx context.Context, rayCluster *rayconfv1.RayClusterApplyConfiguration,
	opts metav1.ApplyOptions) (*rayv1.RayCluster, error) {
	applyArgs := m.Called(ctx, rayCluster, opts)
	return applyArgs.Get(0).(*rayv1.RayCluster), applyArgs.Error(1)
}
func (m *MockRayCluster) ApplyStatus(ctx context.Context,
	rayCluster *rayconfv1.RayClusterApplyConfiguration, opts metav1.ApplyOptions) (*rayv1.RayCluster, error) {
	applyStatusArgs := m.Called(ctx, rayCluster, opts)
	return applyStatusArgs.Get(0).(*rayv1.RayCluster), applyStatusArgs.Error(1)
}

type MockRayJob struct {
	mock.Mock
}

func (m *MockRayJob) Create(ctx context.Context,
	rayCluster *rayv1.RayJob, opts metav1.CreateOptions) (*rayv1.RayJob, error) {
	args := m.Called(ctx, rayCluster, opts)
	return args.Get(0).(*rayv1.RayJob), args.Error(1)
}

func (m *MockRayJob) Update(ctx context.Context,
	rayCluster *rayv1.RayJob, opts metav1.UpdateOptions) (*rayv1.RayJob, error) {
	updateArgs := m.Called(ctx, rayCluster, opts)
	return updateArgs.Get(0).(*rayv1.RayJob), updateArgs.Error(1)
}

func (m *MockRayJob) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	args := m.Called(ctx, name, opts)
	return args.Error(0)
}

func (m *MockRayJob) Get(ctx context.Context, name string, opts metav1.GetOptions) (*rayv1.RayJob, error) {
	args := m.Called(ctx, name, opts)
	return args.Get(0).(*rayv1.RayJob), args.Error(1)
}

func (m *MockRayJob) List(ctx context.Context, opts metav1.ListOptions) (*rayv1.RayJobList, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(*rayv1.RayJobList), args.Error(1)
}

func (m *MockRayJob) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(watch.Interface), args.Error(1)
}

func (m *MockRayJob) UpdateStatus(ctx context.Context, rayCluster *rayv1.RayJob,
	opts metav1.UpdateOptions) (result *rayv1.RayJob, err error) {
	updateStatusArgs := m.Called(ctx, rayCluster, opts)
	return updateStatusArgs.Get(0).(*rayv1.RayJob), updateStatusArgs.Error(1)
}

func (m *MockRayJob) Patch(ctx context.Context, name string, pt types.PatchType,
	data []byte, opts metav1.PatchOptions, subresources ...string) (*rayv1.RayJob, error) {
	args := m.Called(ctx, name, pt, data, opts, subresources)
	return args.Get(0).(*rayv1.RayJob), args.Error(1)
}

func (m *MockRayJob) DeleteCollection(ctx context.Context,
	opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	args := m.Called(ctx, opts, listOpts)
	return args.Error(0)
}
func (m *MockRayJob) Apply(ctx context.Context,
	rayCluster *rayconfv1.RayJobApplyConfiguration, opts metav1.ApplyOptions) (*rayv1.RayJob, error) {
	applyArgs := m.Called(ctx, rayCluster, opts)
	return applyArgs.Get(0).(*rayv1.RayJob), applyArgs.Error(1)
}
func (m *MockRayJob) ApplyStatus(ctx context.Context,
	rayCluster *rayconfv1.RayJobApplyConfiguration, opts metav1.ApplyOptions) (*rayv1.RayJob, error) {
	applyStatusArgs := m.Called(ctx, rayCluster, opts)
	return applyStatusArgs.Get(0).(*rayv1.RayJob), applyStatusArgs.Error(1)
}

type MockRayService struct {
	mock.Mock
}

func (m *MockRayService) Create(ctx context.Context,
	rayCluster *rayv1.RayService, opts metav1.CreateOptions) (*rayv1.RayService, error) {
	args := m.Called(ctx, rayCluster, opts)
	return args.Get(0).(*rayv1.RayService), args.Error(1)
}

func (m *MockRayService) Update(ctx context.Context,
	rayCluster *rayv1.RayService, opts metav1.UpdateOptions) (*rayv1.RayService, error) {
	args := m.Called(ctx, rayCluster, opts)
	return args.Get(0).(*rayv1.RayService), args.Error(1)
}

func (m *MockRayService) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	args := m.Called(ctx, name, opts)
	return args.Error(0)
}

func (m *MockRayService) Get(ctx context.Context,
	name string, opts metav1.GetOptions) (*rayv1.RayService, error) {
	args := m.Called(ctx, name, opts)
	return args.Get(0).(*rayv1.RayService), args.Error(1)
}

func (m *MockRayService) List(ctx context.Context, opts metav1.ListOptions) (*rayv1.RayServiceList, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(*rayv1.RayServiceList), args.Error(1)
}

func (m *MockRayService) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	args := m.Called(ctx, opts)
	return args.Get(0).(watch.Interface), args.Error(1)
}

func (m *MockRayService) UpdateStatus(ctx context.Context,
	rayCluster *rayv1.RayService, opts metav1.UpdateOptions) (result *rayv1.RayService, err error) {
	updateStatusArgs := m.Called(ctx, rayCluster, opts)
	return updateStatusArgs.Get(0).(*rayv1.RayService), updateStatusArgs.Error(1)
}
func (m *MockRayService) Patch(ctx context.Context, name string,
	pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (*rayv1.RayService, error) {
	args := m.Called(ctx, name, pt, data, opts, subresources)
	return args.Get(0).(*rayv1.RayService), args.Error(1)
}

func (m *MockRayService) DeleteCollection(ctx context.Context,
	opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	args := m.Called(ctx, opts, listOpts)
	return args.Error(0)
}
func (m *MockRayService) Apply(ctx context.Context,
	rayCluster *rayconfv1.RayServiceApplyConfiguration, opts metav1.ApplyOptions) (*rayv1.RayService, error) {
	applyArgs := m.Called(ctx, rayCluster, opts)
	return applyArgs.Get(0).(*rayv1.RayService), applyArgs.Error(1)
}
func (m *MockRayService) ApplyStatus(ctx context.Context,
	rayCluster *rayconfv1.RayServiceApplyConfiguration, opts metav1.ApplyOptions) (*rayv1.RayService, error) {
	applyStatusArgs := m.Called(ctx, rayCluster, opts)
	return applyStatusArgs.Get(0).(*rayv1.RayService), applyStatusArgs.Error(1)
}

var (
	rayclusterTemp = `apiVersion: ray.io/v1
kind: RayCluster
metadata:
  labels:
    raycluster-createby: openfuyao.com
  name: %s
  namespace: vcjob
spec:
  autoscalerOptions:
    idleTimeoutSeconds: 120
    imagePullPolicy: IfNotPresent
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 500m
        memory: 512Mi
    securityContext: {}
    upscalingMode: Default
  enableInTreeAutoscaling: true
  headGroupSpec:
    rayStartParams:
      dashboard-host: 0.0.0.0
      dashboard-port: '8265'
      num-cpus: '0'
    serviceType: NodePort
    template:
      metadata: {}
      spec:
        serviceAccountName: test
        containers:
        - image: docker.io/library/ray:2.39.0
          lifecycle:
            preStop:
              exec:
                command:
                - /bin/sh
                - -c
                - ray stop
          name: ray-head
          ports:
          - containerPort: 6379
            name: gcs
            protocol: TCP
          - containerPort: 8265
            name: dashboard
            protocol: TCP
          - containerPort: 10001
            name: client
            protocol: TCP
          resources:
            limits:
              cpu: '1'
              memory: 2G
            requests:
              cpu: '1'
              memory: 2G
          volumeMounts:
          - mountPath: /tmp-ms
            name: host-pvc-vol
        volumes:
        - name: host-pvc-vol
          persistentVolumeClaim:
            claimName: host-512m-pvc
  rayVersion: 2.9.0
  workerGroupSpecs:
  - groupName: small-group
    maxReplicas: 8
    minReplicas: 0
    numOfHosts: 1
    rayStartParams: {}
    replicas: 0
    scaleStrategy: {}
    template:
      metadata: {}
      spec:
        containers:
        - image: docker.io/library/ray:2.39.0
          lifecycle:
            preStop:
              exec:
                command:
                - /bin/sh
                - -c
                - ray stop
          name: ray-worker
          resources:
            limits:
              cpu: '2'
              memory: 2G
            requests:
              cpu: '2'
              memory: 2G`

	rayJobTemp = `apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: %s
  namespace: vcjob
spec:
  entrypoint: python -c "print('Hello Kubeary RayJob!', end=' ')"
  rayClusterSpec:
    rayVersion: '2.41.0' 
    headGroupSpec:
      rayStartParams: {}
      template:
        spec:
          serviceAccountName: test
          containers:
            - name: ray-head
              image: docker.io/rayproject/ray:2.41.0
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
              resources:
                limits:
                  cpu: "1"
                requests:
                  cpu: "200m"
    workerGroupSpecs:
      - replicas: 0
        minReplicas: 0
        maxReplicas: 5
        groupName: small-group
        rayStartParams: {}
        #pod template
        template:
          spec:
            containers:
              - name: ray-worker 
                image: docker.io/rayproject/ray:2.41.0
                resources:
                  limits:
                    cpu: "1"
                  requests:
                    cpu: "200m"`

	rayServiceTemp = `apiVersion: ray.io/v1
kind: RayService
metadata:
  name: %s
  namespace: vcjob
spec:
  serveConfigV2: |
    applications:
      - name: fruit_app
        import_path: fruit.deployment_graph
        route_prefix: /fruit
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/78b4a5da38796123d9f9ffff59bab2792a043e95.zip"
        deployments:
          - name: MangoStand
            num_replicas: 2
            max_replicas_per_node: 1
            user_config:
              price: 3
            ray_actor_options:
              num_cpus: 0.1
          - name: OrangeStand
            num_replicas: 1
            user_config:
              price: 2
            ray_actor_options:
              num_cpus: 0.1
          - name: PearStand
            num_replicas: 1
            user_config:
              price: 1
            ray_actor_options:
              num_cpus: 0.1
          - name: FruitMarket
            num_replicas: 1
            ray_actor_options:
              num_cpus: 0.1
      - name: math_app
        import_path: conditional_dag.serve_dag
        route_prefix: /calc
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/78b4a5da38796123d9f9ffff59bab2792a043e95.zip"
        deployments:
          - name: Adder
            num_replicas: 1
            user_config:
              increment: 3
            ray_actor_options:
              num_cpus: 0.1
          - name: Multiplier
            num_replicas: 1
            user_config:
              factor: 5
            ray_actor_options:
              num_cpus: 0.1
          - name: Router
            num_replicas: 1
  rayClusterConfig:
    rayVersion: '2.41.0'
    headGroupSpec:
      rayStartParams: {}
      template:
        spec:
          serviceAccountName: test
          containers:
            - name: ray-head
              image: docker.io/rayproject/ray:2.41.0
              resources:
                limits:
                  cpu: 2
                  memory: 2Gi
                requests:
                  cpu: 2
                  memory: 2Gi
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
                - containerPort: 8000
                  name: serve
    workerGroupSpecs:
      - replicas: 1
        minReplicas: 1
        maxReplicas: 5
        groupName: small-group
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker 
                image: docker.io/rayproject/ray:2.41.0
                resources:
                  limits:
                    cpu: "1"
                    memory: "2Gi"
                  requests:
                    cpu: "500m"
                    memory: "2Gi"`
)

func buildMock() (*MockClientset, *MockRayCluster, *MockRayJob, *MockRayService) {
	// Set up Mock
	mockClientset := new(MockClientset)
	mockClient := new(MockRayClient)
	mockClientset.rayV1 = mockClient
	mockCluster := new(MockRayCluster)
	mockJob := new(MockRayJob)
	mockService := new(MockRayService)
	mockClient.On("RayClusters", mock.Anything).Return(mockCluster)
	mockClient.On("RayJobs", mock.Anything).Return(mockJob)
	mockClient.On("RayServices", mock.Anything).Return(mockService)
	return mockClientset, mockCluster, mockJob, mockService
}

func newRayClient(mock *MockClientset) *rayClient {
	db, _ := gorm.Open(sqlite.Open("file::memory:?cache=shared"), &gorm.Config{
		DryRun: false,
	})

	fakeKubeConfig := &rest.Config{}
	fakeClientset, err := kubernetes.NewForConfig(fakeKubeConfig)
	if err != nil {
		return nil
	}

	return &rayClient{
		dbConnection: db,
		clientset:    fakeClientset,
		rayClientset: mock,
	}
}

func getCrdSetupMock(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
	mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "raycluster",
			Namespace: "vcjob",
		},
		Spec: rayv1.RayClusterSpec{},
	}, nil)
	mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rayjob",
			Namespace: "vcjob",
		},
		Spec: rayv1.RayJobSpec{},
	}, nil)
	mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rayserivce",
			Namespace: "vcjob",
		},
		Spec: rayv1.RayServiceSpec{},
	}, nil)
}

func TestRayClientMethodInitOperation(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "Test MethodInitOperation",
		},
	}
	mockClientset, _, _, _ := buildMock()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			v.MethodInitOperation()
		})
	}
}

func TestRayClientMethodAllOverview(t *testing.T) {
	// Set up Mock
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{}, nil)
	mockJob.On("List", mock.Anything, metav1.ListOptions{}).Return(&rayv1.RayJobList{}, nil)
	mockService.On("List", mock.Anything, metav1.ListOptions{}).Return(&rayv1.RayServiceList{}, nil)
	tests := []struct {
		name  string
		want  *httputil.ResponseJson
		want1 int
	}{
		{
			name: "Test MethodAllOverview",
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{
					"rayclusters": 0,
					"rayservices": 0,
					"rayjobs":     0,
					"total":       0,
				},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			_, got1 := v.MethodAllOverview()
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodAllOverview() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodCreateCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	createSetupMock := func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
		mockCluster.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).Return(&rayv1.RayCluster{}, nil)
		mockJob.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).Return(&rayv1.RayJob{}, nil)
		mockService.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).Return(&rayv1.RayService{}, nil)
	}
	type args struct {
		crd      string
		data     string
		createBy string
		isStart  bool
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodCreateCrd_ForRayCluster",
			args: args{
				crd:      "raycluster",                              // crd type
				data:     fmt.Sprintf(rayclusterTemp, "raycluster"), // data
				createBy: "admin",                                   // create by
				isStart:  true,                                      // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},

		{
			name: "Test MethodCreateCrd_ForRayCluster_NoStart",
			args: args{
				crd:      "raycluster",                               // crd type
				data:     fmt.Sprintf(rayclusterTemp, "raycluster2"), // data
				createBy: "admin",                                    // create by
				isStart:  false,                                      // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},

		{
			name: "Test MethodCreateCrd_ForRayJob",
			args: args{
				crd:      "rayjob",                          // crd type
				data:     fmt.Sprintf(rayJobTemp, "rayjob"), // data
				createBy: "admin",                           // create by
				isStart:  true,                              // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},

		{
			name: "Test MethodCreateCrd_ForRayJob_NoStart",
			args: args{
				crd:      "rayjob",                           // crd type
				data:     fmt.Sprintf(rayJobTemp, "rayjob2"), // data
				createBy: "admin",                            // create by
				isStart:  false,                              // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},

		{
			name: "Test MethodCreateCrd_ForRayService",
			args: args{
				crd:      "rayservice",                              // crd type
				data:     fmt.Sprintf(rayServiceTemp, "rayservice"), // data
				createBy: "admin",                                   // create by
				isStart:  true,                                      // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},

		{
			name: "Test MethodCreateCrd_ForRayService_NoStart",
			args: args{
				crd:      "rayservice",                               // crd type
				data:     fmt.Sprintf(rayServiceTemp, "rayservice2"), // data
				createBy: "admin",                                    // create by
				isStart:  false,                                      // is start
			},
			setupMock: createSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			tt.setupMock(mockCluster, mockJob, mockService)
			got, got1 := v.MethodCreateCrd(tt.args.crd, tt.args.data, tt.args.createBy, tt.args.isStart)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rayClient.MethodCreateCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodCreateCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodListCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	listSetupMock := func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
		mockCluster.On("List", mock.Anything, metav1.ListOptions{}).Return(&rayv1.RayClusterList{
			Items: []rayv1.RayCluster{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "raycluster",
						Namespace: "vcjob",
					},
					Spec: rayv1.RayClusterSpec{},
				},
			},
		}, nil)
		mockJob.On("List", mock.Anything, metav1.ListOptions{}).Return(&rayv1.RayJobList{
			Items: []rayv1.RayJob{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "rayjob",
						Namespace: "vcjob",
					},
					Spec: rayv1.RayJobSpec{},
				},
			},
		}, nil)
		mockService.On("List", mock.Anything, metav1.ListOptions{}).Return(&rayv1.RayServiceList{
			Items: []rayv1.RayService{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "rayserivce",
						Namespace: "vcjob",
					},
					Spec: rayv1.RayServiceSpec{},
				},
			},
		}, nil)

		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "raycluster",
				Namespace: "vcjob",
			},
			Spec: rayv1.RayClusterSpec{},
		}, nil)
	}
	type args struct {
		crd string
		req *CrdListRequest
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodListCrd No Exist",
			args: args{
				crd: "raycluster",
				req: &CrdListRequest{
					Name: "raycluster-1234",
				},
			},
			setupMock: listSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{
					"total": 0,
					"items": []any{},
				},
			},
			want1: 200,
		},
		{
			name: "Test MethodListCrd For RayCluster",
			args: args{
				crd: "raycluster",
				req: &CrdListRequest{
					Name: "raycluster",
				},
			},
			setupMock: listSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{
					"total": 1,
					"items": []any{},
				},
			},
			want1: 200,
		},
		{
			name: "Test MethodListCrd For RayJob",
			args: args{
				crd: "rayjob",
				req: &CrdListRequest{
					Name: "rayjob",
				},
			},
			setupMock: listSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{
					"total": 1,
					"items": []any{},
				},
			},
			want1: 200,
		},
		{
			name: "Test MethodListCrd For RayService",
			args: args{
				crd: "rayservice",
				req: &CrdListRequest{
					Name: "rayservice",
				},
			},
			setupMock: listSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{
					"total": 1,
					"items": []any{},
				},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			tt.setupMock(mockCluster, mockJob, mockService)
			got, got1 := v.MethodListCrd(tt.args.crd, tt.args.req)
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodListCrd() got1 = %v, want %v", got1, tt.want1)
			}
			if got.Data == nil && tt.want.Data != nil {
				t.Errorf("rayClient.MethodListCrd() got = %v, want %v", got, tt.want)
			}
			if tt.want.Data == nil && got.Data != nil {
				t.Errorf("rayClient.MethodListCrd() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRayClientMethodGetCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	type args struct {
		crd  string
		ns   string
		name string
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodGetCrd No Exist",
			args: args{
				crd:  "raycluster",
				ns:   "default",
				name: "No-Exist-raycluster",
			},
			setupMock: getCrdSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.ResourceNotFound,
				Msg:  "Not Found",
				Data: []any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodGetCrd For RayCluster",
			args: args{
				crd:  "raycluster",
				ns:   "vcjob",
				name: "raycluster",
			},
			setupMock: getCrdSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: []any{"Some RayCluster Data"},
			},
			want1: 200,
		},
		{
			name: "Test MethodGetCrd For RayJob",
			args: args{
				crd:  "rayjob",
				ns:   "vcjob",
				name: "rayjob",
			},
			setupMock: getCrdSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: []any{"Some RayJob Data"},
			},
			want1: 200,
		},
		{
			name: "Test MethodGetCrd For RayService",
			args: args{
				crd:  "rayservice",
				ns:   "vcjob",
				name: "rayservice",
			},
			setupMock: getCrdSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: []any{"Some RayService Data"},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			tt.setupMock(mockCluster, mockJob, mockService)
			got, got1 := v.MethodGetCrd(tt.args.crd, tt.args.ns, tt.args.name)
			if tt.want.Data == nil && got.Data != nil {
				t.Errorf("rayClient.MethodGetCrd() got = %v, want %v", got, tt.want)
			}
			if tt.want.Data != nil && got.Data == nil {
				t.Errorf("rayClient.MethodGetCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodGetCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodUpdateCrd(t *testing.T) {
	mockClientset, _, _, _ := buildMock()
	type args struct {
		crd  string
		ns   string
		data string
		name string
	}
	tests := []struct {
		name  string
		args  args
		want  *httputil.ResponseJson
		want1 int
	}{
		{
			name: "Test MethodUpdateCrd",
			args: args{
				crd:  "raycluster",
				ns:   "default",
				data: "data",
				name: "raycluster",
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			got, got1 := v.MethodUpdateCrd(tt.args.crd, tt.args.ns, tt.args.data, tt.args.name)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rayClient.MethodUpdateCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodUpdateCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodStopCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	stopSetupMock := func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
		mockCluster.On("Delete", mock.Anything, "No-Exist-raycluster", mock.Anything).
			Return(errors.New("mock error"))
		mockCluster.On("Delete", mock.Anything, "raycluster", mock.Anything).Return(nil)
		mockJob.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockService.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
	}
	type args struct {
		crd  string
		ns   string
		name string
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodStopCrd No Exist",
			args: args{
				crd:  "raycluster",
				ns:   "default",
				name: "No-Exist-raycluster",
			},
			setupMock: stopSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 500,
		},
		{
			name: "Test MethodStopCrd For RayCluster",
			args: args{
				crd:  "raycluster",
				ns:   "vcjob",
				name: "raycluster",
			},
			setupMock: stopSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodStopCrd For RayJob",
			args: args{
				crd:  "rayjob",
				ns:   "vcjob",
				name: "rayjob",
			},
			setupMock: stopSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodStopCrd For RayService",
			args: args{
				crd:  "rayservice",
				ns:   "vcjob",
				name: "rayservice",
			},
			setupMock: stopSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			tt.setupMock(mockCluster, mockJob, mockService)
			_, got1 := v.MethodStopCrd(tt.args.crd, tt.args.ns, tt.args.name)
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodStopCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodRestartCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	restartSetupMock := func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return(&rayv1.RayCluster{}, errors.New("mock error"))
		mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return(&rayv1.RayJob{}, errors.New("mock error"))
		mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return(&rayv1.RayService{}, errors.New("mock error"))

		mockCluster.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).
			Return(&rayv1.RayCluster{}, nil)
		mockJob.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).
			Return(&rayv1.RayJob{}, nil)
		mockService.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).
			Return(&rayv1.RayService{}, nil)
	}
	type args struct {
		crd  string
		ns   string
		name string
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodRestartCrd No	Exist",
			args: args{
				crd:  "raycluster",
				ns:   "default",
				name: "No-Exist-raycluster",
			},
			setupMock: restartSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.ServerError,
				Msg:  "remote server busy",
				Data: nil,
			},
			want1: 500,
		},
		{
			name: "Test MethodRestartCrd For RayCluster",
			args: args{
				crd:  "raycluster",
				ns:   "vcjob",
				name: "raycluster",
			},
			setupMock: restartSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodRestartCrd For RayJob",
			args: args{
				crd:  "rayjob",
				ns:   "vcjob",
				name: "rayjob",
			},
			setupMock: restartSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodRestartCrd For RayService",
			args: args{
				crd:  "rayservice",
				ns:   "vcjob",
				name: "rayservice",
			},
			setupMock: restartSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			tt.setupMock(mockCluster, mockJob, mockService)
			got, got1 := v.MethodRestartCrd(tt.args.crd, tt.args.ns, tt.args.name)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rayClient.MethodRestartCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodRestartCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodDeleteCrd(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	deleteSetupMock := func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService) {
		mockCluster.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockJob.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockService.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{}, nil)
		mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayJob{}, nil)
		mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayService{}, nil)
	}
	type args struct {
		crd  string
		ns   string
		name string
	}
	tests := []struct {
		name      string
		args      args
		setupMock func(mockCluster *MockRayCluster, mockJob *MockRayJob, mockService *MockRayService)
		want      *httputil.ResponseJson
		want1     int
	}{
		{
			name: "Test MethodDeleteCrd No Exist",
			args: args{
				crd:  "raycluster",
				ns:   "default",
				name: "No-Exist-Raycluster",
			},
			setupMock: deleteSetupMock,
			want: httputil.GetResponseJson(
				constant.ClientError,
				"资源不存在!",
				map[string]any{}),
			want1: 200,
		},
		{
			name: "Test MethodDeleteCrd For RayCluster",
			args: args{
				crd:  "raycluster",
				ns:   "vcjob",
				name: "raycluster",
			},
			setupMock: deleteSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodDeleteCrd For RayJob",
			args: args{
				crd:  "rayjob",
				ns:   "vcjob",
				name: "rayjob",
			},
			setupMock: deleteSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
		{
			name: "Test MethodDeleteCrd For RayService",
			args: args{
				crd:  "rayservice",
				ns:   "vcjob",
				name: "rayservice",
			},
			setupMock: deleteSetupMock,
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: map[string]any{},
			},
			want1: 200,
		},
	}
	v := newRayClient(mockClientset)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupMock(mockCluster, mockJob, mockService)
			got, got1 := v.MethodDeleteCrd(tt.args.crd, tt.args.ns, tt.args.name)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rayClient.MethodDeleteCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodDeleteCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientMethodListMetaCrd(t *testing.T) {
	mockClientset, _, _, _ := buildMock()
	type args struct {
		crd string
	}
	tests := []struct {
		name  string
		args  args
		want  *httputil.ResponseJson
		want1 int
	}{
		{
			name: "Test MethodListMetaCrd",
			args: args{
				crd: "rayioss",
			},
			want: &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "bad request",
				Data: nil,
			},
			want1: 400,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			got, got1 := v.MethodListMetaCrd(tt.args.crd)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rayClient.MethodListMetaCrd() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("rayClient.MethodListMetaCrd() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestRayClientfetchPodLog(t *testing.T) {
	mockClientset, _, _, _ := buildMock()
	type args struct {
		ns            string
		crdName       string
		podName       string
		containerName string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test fetchPodLog",
			args: args{
				ns:            "default",
				crdName:       "raycluster",      // crd name
				podName:       "raycluster-1234", // pod name
				containerName: "raycluster",      // container name
			},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := newRayClient(mockClientset)
			if got := v.fetchPodLog(tt.args.ns, tt.args.crdName,
				tt.args.podName, tt.args.containerName); got != tt.want {
				t.Errorf("rayClient.fetchPodLog() = %v, want %v", got, tt.want)
			}
		})
	}
}
