/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package ray

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/mock"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	rest "k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/constant"
)

// extraDBSeq generates unique in-memory sqlite db names so that every extra
// test operates on its own isolated database.
var extraDBSeq int64

const (
	// runningDeleteRejectionMsg matches the message returned when deleting a
	// running resource is rejected.
	runningDeleteRejectionMsg = "不允许删除正在运行的资源!"
	// dashboardNotAvailable matches the placeholder returned when a dashboard
	// address cannot be resolved.
	dashboardNotAvailable = "N/A"
)

// newExtraDB opens an isolated in-memory sqlite database with the CrdInfo
// table created.
func newExtraDB(t *testing.T) *gorm.DB {
	t.Helper()
	seq := atomic.AddInt64(&extraDBSeq, 1)
	db, err := gorm.Open(sqlite.Open(fmt.Sprintf("file:ray_extra_%d?mode=memory&cache=shared", seq)), &gorm.Config{})
	if err != nil {
		t.Fatalf("failed to open isolated in-memory db: %v", err)
	}
	if err := db.AutoMigrate(&CrdInfo{}); err != nil {
		t.Fatalf("failed to migrate CrdInfo table: %v", err)
	}
	return db
}

// newExtraRayClient builds a rayClient with an isolated in-memory database.
// When clientset is nil a clientset backed by an empty rest.Config is used,
// which fails on any real API call (useful for error paths).
func newExtraRayClient(t *testing.T, mockClientset *MockClientset, clientset kubernetes.Interface) *rayClient {
	t.Helper()
	if clientset == nil {
		var err error
		clientset, err = kubernetes.NewForConfig(&rest.Config{})
		if err != nil {
			t.Fatalf("failed to build clientset: %v", err)
		}
	}
	return &rayClient{
		dbConnection: newExtraDB(t),
		clientset:    clientset,
		rayClientset: mockClientset,
	}
}

// closeDB closes the underlying sql connection pool of a gorm db.
func closeDB(t *testing.T, db *gorm.DB) {
	t.Helper()
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("failed to get sql db: %v", err)
	}
	if err := sqlDB.Close(); err != nil {
		t.Fatalf("failed to close sql db: %v", err)
	}
}

func TestValidateKubernetesResourceName(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{name: "valid name", input: "raycluster-1", wantErr: false},
		{name: "valid dns subdomain", input: "a.b-c.d", wantErr: false},
		{name: "single char", input: "a", wantErr: false},
		{name: "empty name", input: "", wantErr: true},
		{name: "too long name", input: strings.Repeat("a", 254), wantErr: true},
		{name: "upper case letters", input: "InvalidName", wantErr: true},
		{name: "underscore", input: "invalid_name", wantErr: true},
		{name: "leading hyphen", input: "-invalid", wantErr: true},
		{name: "trailing hyphen", input: "invalid-", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateKubernetesResourceName(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateKubernetesResourceName(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
		})
	}
}

func TestParseRayResource(t *testing.T) {
	t.Run("valid raycluster", func(t *testing.T) {
		resource, err := ParseRayResource(constant.RayCluster, fmt.Sprintf(rayclusterTemp, "parse-cluster"))
		if err != nil {
			t.Fatalf("ParseRayResource() error = %v, want nil", err)
		}
		if resource.GetMetadata().Name != "parse-cluster" {
			t.Errorf("ParseRayResource() name = %q, want parse-cluster", resource.GetMetadata().Name)
		}
		if resource.GetRayVersion() != "2.9.0" {
			t.Errorf("ParseRayResource() rayVersion = %q, want 2.9.0", resource.GetRayVersion())
		}
	})

	t.Run("valid rayservice", func(t *testing.T) {
		resource, err := ParseRayResource(constant.RayService, fmt.Sprintf(rayServiceTemp, "parse-service"))
		if err != nil {
			t.Fatalf("ParseRayResource() error = %v, want nil", err)
		}
		if resource.GetMetadata().Name != "parse-service" {
			t.Errorf("ParseRayResource() name = %q, want parse-service", resource.GetMetadata().Name)
		}
	})

	t.Run("valid rayjob", func(t *testing.T) {
		resource, err := ParseRayResource(constant.RayJob, fmt.Sprintf(rayJobTemp, "parse-job"))
		if err != nil {
			t.Fatalf("ParseRayResource() error = %v, want nil", err)
		}
		if resource.GetMetadata().Name != "parse-job" {
			t.Errorf("ParseRayResource() name = %q, want parse-job", resource.GetMetadata().Name)
		}
	})

	t.Run("invalid yaml for each type", func(t *testing.T) {
		for _, crdType := range []string{constant.RayCluster, constant.RayService, constant.RayJob} {
			if _, err := ParseRayResource(crdType, "\tinvalid: [yaml"); err == nil {
				t.Errorf("ParseRayResource(%q) error = nil, want error for invalid yaml", crdType)
			}
		}
	})

	t.Run("invalid name for each type", func(t *testing.T) {
		for _, crdType := range []string{constant.RayCluster, constant.RayService, constant.RayJob} {
			data := "apiVersion: ray.io/v1\nkind: " + crdType + "\nmetadata:\n  name: Invalid_Name\n"
			if _, err := ParseRayResource(crdType, data); err == nil {
				t.Errorf("ParseRayResource(%q) error = nil, want error for invalid name", crdType)
			}
		}
	})

	t.Run("unknown crd type", func(t *testing.T) {
		if _, err := ParseRayResource("unknown", "data"); err == nil {
			t.Errorf("ParseRayResource() error = nil, want error for unknown crd type")
		}
	})
}

func TestParseRayResourceAndSetDefaults(t *testing.T) {
	t.Run("default namespace is set when empty", func(t *testing.T) {
		data := "apiVersion: ray.io/v1\nkind: RayCluster\nmetadata:\n  name: defaults-cluster\n" +
			"spec:\n  rayVersion: '2.9.0'\n"
		resource, err := parseRayResourceAndSetDefaults(constant.RayCluster, data)
		if err != nil {
			t.Fatalf("parseRayResourceAndSetDefaults() error = %v, want nil", err)
		}
		if ns := resource.GetMetadata().Namespace; ns != constant.Default {
			t.Errorf("parseRayResourceAndSetDefaults() namespace = %q, want %q", ns, constant.Default)
		}
	})

	t.Run("existing namespace is kept", func(t *testing.T) {
		resource, err := parseRayResourceAndSetDefaults(constant.RayCluster, fmt.Sprintf(rayclusterTemp, "ns-cluster"))
		if err != nil {
			t.Fatalf("parseRayResourceAndSetDefaults() error = %v, want nil", err)
		}
		if ns := resource.GetMetadata().Namespace; ns != TestNamespace {
			t.Errorf("parseRayResourceAndSetDefaults() namespace = %q, want vcjob", ns)
		}
	})

	t.Run("parse error", func(t *testing.T) {
		if _, err := parseRayResourceAndSetDefaults("unknown", "data"); err == nil {
			t.Errorf("parseRayResourceAndSetDefaults() error = nil, want error for unknown type")
		}
	})
}

func TestGetDefaultNs(t *testing.T) {
	if got := getDefaultNs("", "default"); got != "default" {
		t.Errorf("getDefaultNs() = %q, want default", got)
	}
	if got := getDefaultNs(TestNamespace, "default"); got != TestNamespace {
		t.Errorf("getDefaultNs() = %q, want vcjob", got)
	}
}

func TestNewRayOperation(t *testing.T) {
	t.Run("invalid kubeconfig returns error", func(t *testing.T) {
		ctx := context.WithValue(context.Background(), constant.ContextDBKey, newExtraDB(t))
		if _, err := NewRayOperation(&rest.Config{Host: "://invalid"}, ctx); err == nil {
			t.Errorf("NewRayOperation() error = nil, want error for invalid kubeconfig")
		}
	})

	t.Run("valid kubeconfig returns operation", func(t *testing.T) {
		db := newExtraDB(t)
		ctx := context.WithValue(context.Background(), constant.ContextDBKey, db)
		operation, err := NewRayOperation(&rest.Config{Host: "http://127.0.0.1:18080"}, ctx)
		if err != nil {
			t.Fatalf("NewRayOperation() error = %v, want nil", err)
		}
		if operation == nil {
			t.Fatalf("NewRayOperation() = nil, want operation")
		}
	})
}

func TestConfigRayRegistersRoutes(t *testing.T) {
	db := newExtraDB(t)
	ctx := context.WithValue(context.Background(), constant.ContextDBKey, db)
	container := restful.NewContainer()

	if err := ConfigRay(container, &rest.Config{Host: "http://127.0.0.1:18080"}, ctx); err != nil {
		t.Fatalf("ConfigRay() error = %v, want nil", err)
	}
	services := container.RegisteredWebServices()
	if len(services) != 1 {
		t.Fatalf("ConfigRay() registered %d web services, want 1", len(services))
	}
	if len(services[0].Routes()) != 9 {
		t.Errorf("ConfigRay() registered %d routes, want 9", len(services[0].Routes()))
	}
}

func TestMethodAllOverviewErrors(t *testing.T) {
	t.Run("raycluster list error", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("List", mock.Anything, mock.Anything).
			Return(&rayv1.RayClusterList{}, errors.New("list error"))
		v := newExtraRayClient(t, mockClientset, nil)
		_, status := v.MethodAllOverview()
		if status != 500 {
			t.Errorf("MethodAllOverview() status = %d, want 500", status)
		}
	})

	t.Run("rayjob list error", func(t *testing.T) {
		mockClientset, mockCluster, mockJob, _ := buildMock()
		mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{}, nil)
		mockJob.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayJobList{}, errors.New("list error"))
		v := newExtraRayClient(t, mockClientset, nil)
		_, status := v.MethodAllOverview()
		if status != 500 {
			t.Errorf("MethodAllOverview() status = %d, want 500", status)
		}
	})

	t.Run("rayservice list error", func(t *testing.T) {
		mockClientset, mockCluster, mockJob, mockService := buildMock()
		mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{}, nil)
		mockJob.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayJobList{}, nil)
		mockService.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayServiceList{}, errors.New("list error"))
		v := newExtraRayClient(t, mockClientset, nil)
		_, status := v.MethodAllOverview()
		if status != 500 {
			t.Errorf("MethodAllOverview() status = %d, want 500", status)
		}
	})

	t.Run("total raycluster list error", func(t *testing.T) {
		mockClientset, mockCluster, mockJob, mockService := buildMock()
		mockCluster.On("List", mock.Anything, metav1.ListOptions{
			LabelSelector: "raycluster-createby=openfuyao.com",
		}).Return(&rayv1.RayClusterList{}, nil)
		mockCluster.On("List", mock.Anything, metav1.ListOptions{}).
			Return(&rayv1.RayClusterList{}, errors.New("list error"))
		mockJob.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayJobList{}, nil)
		mockService.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayServiceList{}, nil)
		v := newExtraRayClient(t, mockClientset, nil)
		_, status := v.MethodAllOverview()
		if status != 500 {
			t.Errorf("MethodAllOverview() status = %d, want 500", status)
		}
	})
}

func TestMethodAllOverviewSuccess(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{
		Items: []rayv1.RayCluster{{ObjectMeta: metav1.ObjectMeta{Name: "c1"}}},
	}, nil)
	mockJob.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayJobList{
		Items: []rayv1.RayJob{{ObjectMeta: metav1.ObjectMeta{Name: "j1"}}},
	}, nil)
	mockService.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayServiceList{}, nil)
	v := newExtraRayClient(t, mockClientset, nil)

	got, status := v.MethodAllOverview()
	if status != 200 {
		t.Fatalf("MethodAllOverview() status = %d, want 200", status)
	}
	data, ok := got.Data.(map[string]any)
	if !ok {
		t.Fatalf("MethodAllOverview() Data type = %T, want map[string]any", got.Data)
	}
	if data["total"] != 1 || data["rayjobs"] != 1 {
		t.Errorf("MethodAllOverview() Data = %v, want total=1 rayjobs=1", data)
	}
}

func TestMethodCreateCrdErrors(t *testing.T) {
	t.Run("invalid yaml returns 400", func(t *testing.T) {
		mockClientset, _, _, _ := buildMock()
		v := newExtraRayClient(t, mockClientset, nil)
		got, status := v.MethodCreateCrd(constant.RayCluster, "\tinvalid: [yaml", "admin", false)
		if status != 400 {
			t.Errorf("MethodCreateCrd() status = %d, want 400", status)
		}
		if got.Code != constant.ClientError {
			t.Errorf("MethodCreateCrd() code = %d, want %d", got.Code, constant.ClientError)
		}
	})

	t.Run("existing resource returns already exists", func(t *testing.T) {
		mockClientset, _, _, _ := buildMock()
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "raycluster", Namespace: TestNamespace, CrdType: constant.RayCluster,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodCreateCrd(constant.RayCluster,
			fmt.Sprintf(rayclusterTemp, "raycluster"), "admin", false)
		if status != 200 {
			t.Fatalf("MethodCreateCrd() status = %d, want 200", status)
		}
		if got.Msg != "Resource already exists!" {
			t.Errorf("MethodCreateCrd() msg = %q, want Resource already exists!", got.Msg)
		}
	})

	t.Run("k8s create failure returns 500 and records error", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Create", mock.Anything, mock.Anything, metav1.CreateOptions{}).
			Return(&rayv1.RayCluster{}, errors.New("create failed"))
		v := newExtraRayClient(t, mockClientset, nil)

		got, status := v.MethodCreateCrd(constant.RayCluster,
			fmt.Sprintf(rayclusterTemp, "failed-cluster"), "admin", true)
		if status != 500 {
			t.Fatalf("MethodCreateCrd() status = %d, want 500", status)
		}
		if got.Code != constant.ServerError {
			t.Errorf("MethodCreateCrd() code = %d, want %d", got.Code, constant.ServerError)
		}

		var record CrdInfo
		if err := v.dbConnection.Where("name = ?", "failed-cluster").First(&record).Error; err != nil {
			t.Fatalf("failed to load db record: %v", err)
		}
		if record.ErrorInfo == "" {
			t.Errorf("db record ErrorInfo should be set when k8s create fails")
		}
	})
}

func TestUpsertCrdInfoInTransaction(t *testing.T) {
	v := newExtraRayClient(t, nil, nil)

	rc := &CrdInfo{
		Name: "upsert-crd", Namespace: "default", CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "v1", RayVersion: "2.9.0",
	}
	if err := v.dbConnection.Transaction(func(tx *gorm.DB) error {
		return v.upsertCrdInfoInTransaction(tx, rc)
	}); err != nil {
		t.Fatalf("upsertCrdInfoInTransaction() create error = %v, want nil", err)
	}

	// upsert the same resource again: the existing record is updated, not duplicated
	rcUpdated := &CrdInfo{
		Name: "upsert-crd", Namespace: "default", CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "v2", RayVersion: "2.9.0",
	}
	if err := v.dbConnection.Transaction(func(tx *gorm.DB) error {
		return v.upsertCrdInfoInTransaction(tx, rcUpdated)
	}); err != nil {
		t.Fatalf("upsertCrdInfoInTransaction() update error = %v, want nil", err)
	}

	var count int64
	if err := v.dbConnection.Model(&CrdInfo{}).Where("name = ?", "upsert-crd").
		Count(&count).Error; err != nil {
		t.Fatalf("failed to count records: %v", err)
	}
	if count != 1 {
		t.Errorf("upsertCrdInfoInTransaction() should keep 1 record, got %d", count)
	}
	if rcUpdated.ID != rc.ID {
		t.Errorf("upsertCrdInfoInTransaction() ID = %d, want %d", rcUpdated.ID, rc.ID)
	}
}

func TestCheckResourceExistsInDB(t *testing.T) {
	v := newExtraRayClient(t, nil, nil)
	if err := v.dbConnection.Create(&CrdInfo{
		Name: "exists", Namespace: "default", CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
	}).Error; err != nil {
		t.Fatalf("failed to seed db record: %v", err)
	}

	exists, err := v.checkResourceExistsInDB("exists", constant.RayCluster, "default")
	if err != nil || !exists {
		t.Errorf("checkResourceExistsInDB() = %v, %v; want true, nil", exists, err)
	}

	exists, err = v.checkResourceExistsInDB("missing", constant.RayCluster, "default")
	if err != nil || exists {
		t.Errorf("checkResourceExistsInDB() = %v, %v; want false, nil", exists, err)
	}

	// a closed db surfaces a query error
	v.dbConnection = newExtraDB(t)
	closeDB(t, v.dbConnection)
	if _, err := v.checkResourceExistsInDB("any", constant.RayCluster, "default"); err == nil {
		t.Errorf("checkResourceExistsInDB() should return error when db is closed")
	}
}

func TestUpdateCrdStatusInDB(t *testing.T) {
	v := newExtraRayClient(t, nil, nil)
	rc := &CrdInfo{
		Name: "status-crd", Namespace: "default", CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0", Status: 1,
	}
	if err := v.updateCrdStatusInDB(rc); err != nil {
		t.Errorf("updateCrdStatusInDB() error = %v, want nil", err)
	}

	closedDB := newExtraDB(t)
	closeDB(t, closedDB)
	v.dbConnection = closedDB
	if err := v.updateCrdStatusInDB(rc); err == nil {
		t.Errorf("updateCrdStatusInDB() should return error when db is closed")
	}
}

func TestMethodDeleteCrdBranches(t *testing.T) {
	t.Run("running raycluster cannot be deleted", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
			ObjectMeta: metav1.ObjectMeta{Name: "running-cluster"},
			Status:     rayv1.RayClusterStatus{State: rayv1.Ready},
		}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		got, status := v.MethodDeleteCrd(constant.RayCluster, TestNamespace, "running-cluster")
		if status != 200 {
			t.Fatalf("MethodDeleteCrd() status = %d, want 200", status)
		}
		if got.Msg != runningDeleteRejectionMsg {
			t.Errorf("MethodDeleteCrd() msg = %q, want running rejection", got.Msg)
		}
	})

	t.Run("running rayservice cannot be deleted", func(t *testing.T) {
		mockClientset, _, _, mockService := buildMock()
		mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayService{
			ObjectMeta: metav1.ObjectMeta{Name: "running-service"},
			Status:     rayv1.RayServiceStatuses{ServiceStatus: rayv1.Running},
		}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		got, status := v.MethodDeleteCrd(constant.RayService, TestNamespace, "running-service")
		if status != 200 || got.Msg != runningDeleteRejectionMsg {
			t.Errorf("MethodDeleteCrd() = %q, %d; want running rejection", got.Msg, status)
		}
	})

	t.Run("running rayjob cannot be deleted", func(t *testing.T) {
		mockClientset, _, mockJob, _ := buildMock()
		mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayJob{
			ObjectMeta: metav1.ObjectMeta{Name: "running-job"},
			Status:     rayv1.RayJobStatus{JobDeploymentStatus: rayv1.JobDeploymentStatusRunning},
		}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		got, status := v.MethodDeleteCrd(constant.RayJob, TestNamespace, "running-job")
		if status != 200 || got.Msg != runningDeleteRejectionMsg {
			t.Errorf("MethodDeleteCrd() = %q, %d; want running rejection", got.Msg, status)
		}
	})

	t.Run("unknown crd returns 400", func(t *testing.T) {
		mockClientset, _, _, _ := buildMock()
		v := newExtraRayClient(t, mockClientset, nil)
		_, status := v.MethodDeleteCrd("unknown", TestNamespace, "any")
		if status != 400 {
			t.Errorf("MethodDeleteCrd() status = %d, want 400", status)
		}
	})

	t.Run("stopped resource is deleted from db", func(t *testing.T) {
		mockClientset, mockCluster, mockJob, mockService := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
			ObjectMeta: metav1.ObjectMeta{Name: "stopped-cluster"},
		}, nil)
		mockCluster.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockJob.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		mockService.On("Delete", mock.Anything, mock.Anything, mock.Anything).Return(nil)
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "stopped-cluster", Namespace: TestNamespace, CrdType: constant.RayCluster,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodDeleteCrd(constant.RayCluster, TestNamespace, "stopped-cluster")
		if status != 200 || got.Code != constant.Success {
			t.Errorf("MethodDeleteCrd() = %v, %d; want success", got, status)
		}

		var count int64
		if err := v.dbConnection.Model(&CrdInfo{}).Where("name = ?", "stopped-cluster").
			Count(&count).Error; err != nil {
			t.Fatalf("failed to count records: %v", err)
		}
		if count != 0 {
			t.Errorf("MethodDeleteCrd() should remove the db record, %d left", count)
		}
	})
}

func TestMethodStopCrdUnknown(t *testing.T) {
	mockClientset, _, _, _ := buildMock()
	v := newExtraRayClient(t, mockClientset, nil)
	_, status := v.MethodStopCrd("unknown", TestNamespace, "any")
	if status != 400 {
		t.Errorf("MethodStopCrd() status = %d, want 400", status)
	}
}

func TestMethodRestartCrdRunning(t *testing.T) {
	mockClientset, mockCluster, mockJob, mockService := buildMock()
	mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{Name: "restart-cluster"},
	}, nil)
	mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayJob{}, errors.New("no job"))
	mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).
		Return(&rayv1.RayService{}, errors.New("no service"))
	v := newExtraRayClient(t, mockClientset, nil)
	if err := v.dbConnection.Create(&CrdInfo{
		Name: "restart-cluster", Namespace: TestNamespace, CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
	}).Error; err != nil {
		t.Fatalf("failed to seed db record: %v", err)
	}

	got, status := v.MethodRestartCrd(constant.RayCluster, TestNamespace, "restart-cluster")
	if status != 200 {
		t.Fatalf("MethodRestartCrd() status = %d, want 200", status)
	}
	if got.Msg != "资源正在运行中!" {
		t.Errorf("MethodRestartCrd() msg = %q, want 资源正在运行中!", got.Msg)
	}
}

func TestMethodGetCrdFetchErrorBranches(t *testing.T) {
	t.Run("fetch error without recorded error info is suspended", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return(&rayv1.RayCluster{}, errors.New("get failed"))
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "suspended-cluster", Namespace: TestNamespace, CrdType: constant.RayCluster,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodGetCrd(constant.RayCluster, TestNamespace, "suspended-cluster")
		if status != 200 {
			t.Fatalf("MethodGetCrd() status = %d, want 200", status)
		}
		info, ok := got.Data.(*CrdInfo)
		if !ok {
			t.Fatalf("MethodGetCrd() Data type = %T, want *CrdInfo", got.Data)
		}
		if info.ClusterStatus != constant.Suspended {
			t.Errorf("MethodGetCrd() ClusterStatus = %q, want %q", info.ClusterStatus, constant.Suspended)
		}
		if info.TemplateStatus != constant.Template {
			t.Errorf("MethodGetCrd() TemplateStatus = %q, want %q", info.TemplateStatus, constant.Template)
		}
	})

	t.Run("fetch error with recorded error info is failed", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return(&rayv1.RayCluster{}, errors.New("get failed"))
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "failed-cluster", Namespace: TestNamespace, CrdType: constant.RayCluster,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
			ErrorInfo: "previous failure",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodGetCrd(constant.RayCluster, TestNamespace, "failed-cluster")
		if status != 200 {
			t.Fatalf("MethodGetCrd() status = %d, want 200", status)
		}
		info, ok := got.Data.(*CrdInfo)
		if !ok {
			t.Fatalf("MethodGetCrd() Data type = %T, want *CrdInfo", got.Data)
		}
		if info.ClusterStatus != constant.Failed {
			t.Errorf("MethodGetCrd() ClusterStatus = %q, want %q", info.ClusterStatus, constant.Failed)
		}
	})
}

func TestMethodGetCrdMissingRayCluster(t *testing.T) {
	t.Run("rayservice without raycluster returns missing", func(t *testing.T) {
		mockClientset, mockCluster, _, mockService := buildMock()
		mockService.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayService{
			ObjectMeta: metav1.ObjectMeta{Name: "no-cluster-service"},
			Status: rayv1.RayServiceStatuses{
				ActiveServiceStatus: rayv1.RayServiceStatus{
					RayClusterName: "missing-cluster",
				},
			},
		}, nil)
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return((*rayv1.RayCluster)(nil), errors.New("cluster missing"))
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "no-cluster-service", Namespace: TestNamespace, CrdType: constant.RayService,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.41.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodGetCrd(constant.RayService, TestNamespace, "no-cluster-service")
		if status != 200 {
			t.Fatalf("MethodGetCrd() status = %d, want 200", status)
		}
		if got.Code != constant.NoContent || got.Msg != "RayCluster is missing." {
			t.Errorf("MethodGetCrd() = %d, %q; want %d, RayCluster is missing.",
				got.Code, got.Msg, constant.NoContent)
		}
	})

	t.Run("rayjob without raycluster returns missing", func(t *testing.T) {
		mockClientset, mockCluster, mockJob, _ := buildMock()
		mockJob.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayJob{
			ObjectMeta: metav1.ObjectMeta{Name: "no-cluster-job"},
			Status:     rayv1.RayJobStatus{RayClusterName: "missing-cluster"},
		}, nil)
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return((*rayv1.RayCluster)(nil), errors.New("cluster missing"))
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "no-cluster-job", Namespace: TestNamespace, CrdType: constant.RayJob,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.41.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		got, status := v.MethodGetCrd(constant.RayJob, TestNamespace, "no-cluster-job")
		if status != 200 || got.Code != constant.NoContent {
			t.Errorf("MethodGetCrd() = %d, %d; want 200, %d", status, got.Code, constant.NoContent)
		}
	})
}

func TestDoGetCrdUnknown(t *testing.T) {
	mockClientset, _, _, _ := buildMock()
	v := newExtraRayClient(t, mockClientset, nil)
	if _, err := v.doGetCrd("unknown", TestNamespace, "any"); err == nil {
		t.Errorf("doGetCrd() error = nil, want error for unknown crd")
	}
}

func TestBuildMissingPodsResponse(t *testing.T) {
	v := newExtraRayClient(t, nil, nil)
	got := v.buildMissingPodsResponse()
	if got.Code != constant.NoContent || got.Msg != "Pod list is nil." {
		t.Errorf("buildMissingPodsResponse() = %v, want NoContent Pod list is nil.", got)
	}
}

func TestMethodListMetaCrdValidTypes(t *testing.T) {
	v := newExtraRayClient(t, nil, nil)
	if err := v.dbConnection.Create(&CrdInfo{
		Name: "meta-cluster", Namespace: "default", CrdType: constant.RayCluster,
		CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.9.0",
	}).Error; err != nil {
		t.Fatalf("failed to seed db record: %v", err)
	}

	tests := []struct {
		name              string
		crd               string
		wantRayVersionKey bool
	}{
		{name: "raycluster meta includes rayVersion", crd: constant.RayCluster, wantRayVersionKey: true},
		{name: "rayservice meta", crd: constant.RayService, wantRayVersionKey: false},
		{name: "rayjob meta", crd: constant.RayJob, wantRayVersionKey: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, status := v.MethodListMetaCrd(tt.crd)
			if status != 200 {
				t.Fatalf("MethodListMetaCrd() status = %d, want 200", status)
			}
			data, ok := got.Data.(map[string]any)
			if !ok {
				t.Fatalf("MethodListMetaCrd() Data type = %T, want map[string]any", got.Data)
			}
			if _, ok := data["createBy"]; !ok {
				t.Errorf("MethodListMetaCrd() Data should contain createBy, got %v", data)
			}
			if _, ok := data["status"]; !ok {
				t.Errorf("MethodListMetaCrd() Data should contain status, got %v", data)
			}
			_, hasVersion := data["rayVersion"]
			if hasVersion != tt.wantRayVersionKey {
				t.Errorf("MethodListMetaCrd() rayVersion presence = %v, want %v", hasVersion, tt.wantRayVersionKey)
			}
		})
	}
}

func TestMethodListCrdBranches(t *testing.T) {
	t.Run("empty db returns empty items", func(t *testing.T) {
		mockClientset, _, _, _ := buildMock()
		v := newExtraRayClient(t, mockClientset, nil)
		got, status := v.MethodListCrd(constant.RayCluster, &CrdListRequest{})
		if status != 200 {
			t.Fatalf("MethodListCrd() status = %d, want 200", status)
		}
		data, ok := got.Data.(map[string]any)
		if !ok {
			t.Fatalf("MethodListCrd() Data type = %T, want map", got.Data)
		}
		if data["total"] != 0 {
			t.Errorf("MethodListCrd() total = %v, want 0", data["total"])
		}
	})

	t.Run("populate error returns 500", func(t *testing.T) {
		mockClientset, _, mockJob, _ := buildMock()
		mockJob.On("List", mock.Anything, mock.Anything).Return((*rayv1.RayJobList)(nil), errors.New("list error"))
		v := newExtraRayClient(t, mockClientset, nil)
		if err := v.dbConnection.Create(&CrdInfo{
			Name: "job-1", Namespace: TestNamespace, CrdType: constant.RayJob,
			CreateBy: "admin", ConfigDetail: "data", RayVersion: "2.41.0",
		}).Error; err != nil {
			t.Fatalf("failed to seed db record: %v", err)
		}

		_, status := v.MethodListCrd(constant.RayJob, &CrdListRequest{})
		if status != 500 {
			t.Errorf("MethodListCrd() status = %d, want 500", status)
		}
	})

	t.Run("closed db returns 500", func(t *testing.T) {
		mockClientset, _, _, _ := buildMock()
		v := newExtraRayClient(t, mockClientset, nil)
		closeDB(t, v.dbConnection)
		_, status := v.MethodListCrd(constant.RayCluster, &CrdListRequest{})
		if status != 500 {
			t.Errorf("MethodListCrd() status = %d, want 500", status)
		}
	})

	t.Run("filters narrow down results", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{}, nil)
		v := newExtraRayClient(t, mockClientset, nil)
		records := []CrdInfo{
			{Name: "alpha", Namespace: "default", CrdType: constant.RayCluster,
				CreateBy: "admin", ConfigDetail: "d", RayVersion: "2.9.0"},
			{Name: "beta", Namespace: "default", CrdType: constant.RayCluster,
				CreateBy: "user", ConfigDetail: "d", RayVersion: "2.41.0"},
			{Name: "gamma", Namespace: "default", CrdType: constant.RayCluster,
				CreateBy: "admin", ConfigDetail: "d", RayVersion: "2.9.0"},
		}
		for i := range records {
			if err := v.dbConnection.Create(&records[i]).Error; err != nil {
				t.Fatalf("failed to seed db record: %v", err)
			}
		}

		assertTotal := func(t *testing.T, req *CrdListRequest, want int) {
			t.Helper()
			got, status := v.MethodListCrd(constant.RayCluster, req)
			if status != 200 {
				t.Fatalf("MethodListCrd() status = %d, want 200", status)
			}
			data, _ := got.Data.(map[string]any)
			if data["total"] != want {
				t.Errorf("MethodListCrd(%+v) total = %v, want %d", req, data["total"], want)
			}
		}

		assertTotal(t, &CrdListRequest{}, 3)
		assertTotal(t, &CrdListRequest{Name: "alp"}, 1)
		assertTotal(t, &CrdListRequest{RayVersion: "2.9.0"}, 2)
		assertTotal(t, &CrdListRequest{CreateBy: "user"}, 1)
	})
}

func TestEnrichWithRayClusterData(t *testing.T) {
	t.Run("nil cluster returns empty", func(t *testing.T) {
		v := newExtraRayClient(t, nil, nil)
		if got := v.enrichWithRayClusterData(TestNamespace, nil); got != "" {
			t.Errorf("enrichWithRayClusterData() = %q, want empty for nil cluster", got)
		}
	})

	t.Run("pod list error returns empty", func(t *testing.T) {
		// a clientset built from an empty rest.Config fails on any api call
		v := newExtraRayClient(t, nil, nil)
		cluster := &rayv1.RayCluster{
			ObjectMeta: metav1.ObjectMeta{Name: "list-error-cluster"},
			Spec: rayv1.RayClusterSpec{
				HeadGroupSpec: rayv1.HeadGroupSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "ray-head"}}},
					},
				},
			},
		}
		if got := v.enrichWithRayClusterData(TestNamespace, cluster); got != "" {
			t.Errorf("enrichWithRayClusterData() = %q, want empty for pod list error", got)
		}
	})

	t.Run("no pods returns empty", func(t *testing.T) {
		v := newExtraRayClient(t, nil, fake.NewSimpleClientset())
		cluster := &rayv1.RayCluster{ObjectMeta: metav1.ObjectMeta{Name: "no-pod-cluster"}}
		if got := v.enrichWithRayClusterData(TestNamespace, cluster); got != "" {
			t.Errorf("enrichWithRayClusterData() = %q, want empty when no pods exist", got)
		}
	})

	t.Run("head pod logs are fetched", func(t *testing.T) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "no-pod-cluster-head-xxx",
				Namespace: TestNamespace,
				Labels:    map[string]string{"ray.io/cluster": "no-pod-cluster"},
			},
		}
		v := newExtraRayClient(t, nil, fake.NewSimpleClientset(pod))
		cluster := &rayv1.RayCluster{
			ObjectMeta: metav1.ObjectMeta{Name: "no-pod-cluster"},
			Spec: rayv1.RayClusterSpec{
				HeadGroupSpec: rayv1.HeadGroupSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "ray-head"}}},
					},
				},
			},
		}
		if got := v.enrichWithRayClusterData(TestNamespace, cluster); got == "" {
			t.Errorf("enrichWithRayClusterData() should return head pod logs")
		}
	})
}

func TestFetchPodLog(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "log-pod", Namespace: TestNamespace},
	}
	v := newExtraRayClient(t, nil, fake.NewSimpleClientset(pod))

	got := v.fetchPodLog(TestNamespace, "log-crd", "log-pod", "ray-head")
	if got == "" {
		t.Errorf("fetchPodLog() should return logs from the fake clientset")
	}
}

func TestGetPodStatus(t *testing.T) {
	podRunning := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cluster-a-head",
			Namespace: TestNamespace,
			Labels: map[string]string{
				"ray.io/node-type": "head",
				"ray.io/cluster":   "cluster-a",
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			ContainerStatuses: []corev1.ContainerStatus{
				{State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
			},
		},
	}
	podNoClusterLabel := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "orphan-head",
			Namespace: TestNamespace,
			Labels:    map[string]string{"ray.io/node-type": "head"},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	v := newExtraRayClient(t, nil, fake.NewSimpleClientset(podRunning, podNoClusterLabel))

	raycls := []string{"cluster-a", "cluster-b"}
	stateMap := v.getPodStatus(TestNamespace, &raycls)

	if len(*stateMap) != 2 {
		t.Fatalf("getPodStatus() map size = %d, want 2", len(*stateMap))
	}
	aStatus, ok := (*stateMap)["cluster-a"]
	if !ok {
		t.Fatalf("getPodStatus() should contain cluster-a")
	}
	if aStatus.Phase != string(corev1.PodRunning) {
		t.Errorf("getPodStatus() cluster-a phase = %q, want Running", aStatus.Phase)
	}
	if aStatus.State == nil {
		t.Errorf("getPodStatus() cluster-a state should be set from container statuses")
	}
	bStatus, ok := (*stateMap)["cluster-b"]
	if !ok {
		t.Fatalf("getPodStatus() should contain cluster-b")
	}
	if bStatus.Phase != "Not Ready" || bStatus.State != nil {
		t.Errorf("getPodStatus() cluster-b = %+v, want Not Ready with nil state", bStatus)
	}
}

func TestPopulateInfosTypeAssertionBranches(t *testing.T) {
	t.Run("populateRayClusterInfos skips wrong type", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayClusterList{}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		keyMappings := map[string]interface{}{"x:y": "not-a-raycluster"}
		if err := v.populateRayClusterInfos(context.TODO(), keyMappings); err != nil {
			t.Fatalf("populateRayClusterInfos() error = %v, want nil", err)
		}
	})

	t.Run("populateRayServiceInfos skips wrong type", func(t *testing.T) {
		mockClientset, _, _, mockService := buildMock()
		mockService.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayServiceList{}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		keyMappings := map[string]interface{}{"x:y": "not-a-rayservice"}
		if err := v.populateRayServiceInfos(context.TODO(), keyMappings); err != nil {
			t.Fatalf("populateRayServiceInfos() error = %v, want nil", err)
		}
	})

	t.Run("populateRayJobInfos skips wrong type", func(t *testing.T) {
		mockClientset, _, mockJob, _ := buildMock()
		mockJob.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayJobList{}, nil)
		v := newExtraRayClient(t, mockClientset, nil)

		keyMappings := map[string]interface{}{"x:y": "not-a-rayjob"}
		if err := v.populateRayJobInfos(context.TODO(), keyMappings); err != nil {
			t.Fatalf("populateRayJobInfos() error = %v, want nil", err)
		}
	})
}

func TestPopulateRayServiceInfosUpdates(t *testing.T) {
	mockClientset, _, _, mockService := buildMock()
	mockService.On("List", mock.Anything, mock.Anything).Return(&rayv1.RayServiceList{
		Items: []rayv1.RayService{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "live-service", Namespace: TestNamespace},
				Status: rayv1.RayServiceStatuses{
					ServiceStatus: rayv1.Running,
					ActiveServiceStatus: rayv1.RayServiceStatus{
						RayClusterName: "live-service-cluster",
						RayClusterStatus: rayv1.RayClusterStatus{
							Head:      rayv1.HeadInfo{ServiceIP: "10.96.0.10"},
							Endpoints: map[string]string{"dashboard": "8265"},
						},
					},
				},
			},
		},
	}, nil)
	v := newExtraRayClient(t, mockClientset, nil)

	rs := &RayService{CrdInfo: &CrdInfo{
		Name: "live-service", Namespace: TestNamespace, CrdType: constant.RayService,
	}}
	keyMappings := map[string]interface{}{"live-service:vcjob": rs}
	if err := v.populateRayServiceInfos(context.TODO(), keyMappings); err != nil {
		t.Fatalf("populateRayServiceInfos() error = %v, want nil", err)
	}
	if rs.TemplateStatus != constant.Instance {
		t.Errorf("populateRayServiceInfos() TemplateStatus = %q, want %q", rs.TemplateStatus, constant.Instance)
	}
	if rs.ClusterStatus != string(rayv1.Running) {
		t.Errorf("populateRayServiceInfos() ClusterStatus = %q, want %q", rs.ClusterStatus, rayv1.Running)
	}
	if rs.DashboardAddress != "10.96.0.10:8265" {
		t.Errorf("populateRayServiceInfos() DashboardAddress = %q, want 10.96.0.10:8265", rs.DashboardAddress)
	}
}

func TestGetDashboardAddressForRayCluster(t *testing.T) {
	t.Run("empty cluster name or namespace returns N/A", func(t *testing.T) {
		v := newExtraRayClient(t, nil, nil)
		if got := v.getDashboardAddressForRayCluster(context.TODO(), "", "cluster"); got != dashboardNotAvailable {
			t.Errorf("getDashboardAddressForRayCluster() = %q, want N/A", got)
		}
		if got := v.getDashboardAddressForRayCluster(context.TODO(), TestNamespace, ""); got != dashboardNotAvailable {
			t.Errorf("getDashboardAddressForRayCluster() = %q, want N/A", got)
		}
	})

	t.Run("get error returns N/A", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).
			Return((*rayv1.RayCluster)(nil), errors.New("not found"))
		v := newExtraRayClient(t, mockClientset, nil)
		if got := v.getDashboardAddressForRayCluster(context.TODO(), TestNamespace, "cluster"); got != dashboardNotAvailable {
			t.Errorf("getDashboardAddressForRayCluster() = %q, want N/A", got)
		}
	})

	t.Run("cluster with dashboard endpoint returns address", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
			Status: rayv1.RayClusterStatus{
				Head:      rayv1.HeadInfo{PodIP: "10.244.0.5"},
				Endpoints: map[string]string{"dashboard": "8265"},
			},
		}, nil)
		v := newExtraRayClient(t, mockClientset, nil)
		if got := v.getDashboardAddressForRayCluster(context.TODO(), TestNamespace, "cluster"); got != "10.244.0.5:8265" {
			t.Errorf("getDashboardAddressForRayCluster() = %q, want 10.244.0.5:8265", got)
		}
	})

	t.Run("cluster without dashboard endpoint returns N/A", func(t *testing.T) {
		mockClientset, mockCluster, _, _ := buildMock()
		mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
			Status: rayv1.RayClusterStatus{Head: rayv1.HeadInfo{PodIP: "10.244.0.5"}},
		}, nil)
		v := newExtraRayClient(t, mockClientset, nil)
		if got := v.getDashboardAddressForRayCluster(context.TODO(), TestNamespace, "cluster"); got != dashboardNotAvailable {
			t.Errorf("getDashboardAddressForRayCluster() = %q, want N/A", got)
		}
	})
}

func TestNormalizeEntity(t *testing.T) {
	tests := []struct {
		name          string
		crdInfo       *CrdInfo
		wantStatus    string
		wantStatusZHCN string
	}{
		{
			name: "raycluster running",
			crdInfo: &CrdInfo{CrdType: constant.RayCluster, ClusterStatus: "ready",
				CreateAt: 1742354694, UpdateAt: 1742354694},
			wantStatus:     "ready",
			wantStatusZHCN: "运行中",
		},
		{
			name: "rayservice failed status is normalized",
			crdInfo: &CrdInfo{CrdType: constant.RayService,
				ClusterStatus: string(rayv1.FailedToGetServeDeploymentStatus)},
			wantStatus:     constant.Failed,
			wantStatusZHCN: "失败",
		},
		{
			name: "rayjob running",
			crdInfo: &CrdInfo{CrdType: constant.RayJob,
				ClusterStatus: string(rayv1.JobDeploymentStatusRunning)},
			wantStatus:     string(rayv1.JobDeploymentStatusRunning),
			wantStatusZHCN: "运行中",
		},
		{
			name:          "unknown crd type falls back to unknown label",
			crdInfo:       &CrdInfo{CrdType: "unknown", ClusterStatus: "whatever"},
			wantStatus:    "whatever",
			wantStatusZHCN: "未知",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			normalizeEntity(tt.crdInfo)
			if tt.crdInfo.ClusterStatus != tt.wantStatus {
				t.Errorf("normalizeEntity() ClusterStatus = %q, want %q", tt.crdInfo.ClusterStatus, tt.wantStatus)
			}
			if tt.crdInfo.ClusterStatusZHCN != tt.wantStatusZHCN {
				t.Errorf("normalizeEntity() ClusterStatusZHCN = %q, want %q",
					tt.crdInfo.ClusterStatusZHCN, tt.wantStatusZHCN)
			}
		})
	}

	t.Run("timestamps are formatted", func(t *testing.T) {
		info := &CrdInfo{CrdType: constant.RayCluster, ClusterStatus: "ready",
			CreateAt: 1742354694, UpdateAt: 1742354694}
		normalizeEntity(info)
		if info.CreateTime == "" || info.UpdateTime == "" {
			t.Errorf("normalizeEntity() should format CreateTime/UpdateTime, got %q/%q",
				info.CreateTime, info.UpdateTime)
		}
	})
}

func TestTransferRayClusterToEntity(t *testing.T) {
	r1 := &rayv1.RayCluster{
		Status: rayv1.RayClusterStatus{
			Head:      rayv1.HeadInfo{PodIP: "10.244.0.1"},
			Endpoints: map[string]string{"dashboard": "8265", "gcs": "6379"},
		},
	}
	r2 := &RayCluster{CrdInfo: &CrdInfo{}}
	transferRayClusterToEntity(r1, r2)
	if r2.ID != 0 || r2.DashboardPort != "8265" || r2.GcsPort != "6379" || r2.ClusterIP != "10.244.0.1" {
		t.Errorf("transferRayClusterToEntity() = %+v, want dashboard 8265, gcs 6379, ip 10.244.0.1", r2)
	}
}

func TestProcessRC(t *testing.T) {
	// processRC with a fake clientset: getPodStatus uses the fake pod list
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "proc-head",
			Namespace: TestNamespace,
			Labels: map[string]string{
				"ray.io/node-type": "head",
				"ray.io/cluster":   "proc-cluster",
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	v := newExtraRayClient(t, nil, fake.NewSimpleClientset(pod))

	crdInfo := &CrdInfo{Name: "proc-cluster", Namespace: TestNamespace, CrdType: constant.RayCluster}
	r := &rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{Name: "proc-cluster"},
		Status:     rayv1.RayClusterStatus{State: rayv1.Ready},
	}

	data, cluster := v.processRC(crdInfo, r)
	rc, ok := data.(*RayCluster)
	if !ok {
		t.Fatalf("processRC() data type = %T, want *RayCluster", data)
	}
	if rc.ClusterStatus != string(rayv1.Ready) {
		t.Errorf("processRC() ClusterStatus = %q, want %q", rc.ClusterStatus, rayv1.Ready)
	}
	if rc.RayHeadStatus.Phase != string(corev1.PodRunning) {
		t.Errorf("processRC() RayHeadStatus.Phase = %q, want %q", rc.RayHeadStatus.Phase, corev1.PodRunning)
	}
	if cluster != r {
		t.Errorf("processRC() should return the same ray cluster object")
	}
}

func TestProcessRS(t *testing.T) {
	mockClientset, mockCluster, _, _ := buildMock()
	mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return(&rayv1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{Name: "service-cluster"},
	}, nil)
	v := newExtraRayClient(t, mockClientset, nil)

	crdInfo := &CrdInfo{Name: "proc-service", Namespace: TestNamespace, CrdType: constant.RayService}
	r := &rayv1.RayService{
		ObjectMeta: metav1.ObjectMeta{Name: "proc-service"},
		Status: rayv1.RayServiceStatuses{
			ServiceStatus: rayv1.Running,
			ActiveServiceStatus: rayv1.RayServiceStatus{
				RayClusterName: "service-cluster",
				RayClusterStatus: rayv1.RayClusterStatus{
					Head:      rayv1.HeadInfo{ServiceIP: "10.96.0.20"},
					Endpoints: map[string]string{"dashboard": "8265"},
				},
			},
		},
	}

	data, cluster := v.processRS(TestNamespace, crdInfo, r)
	rs, ok := data.(*RayService)
	if !ok {
		t.Fatalf("processRS() data type = %T, want *RayService", data)
	}
	if rs.ClusterStatus != string(rayv1.Running) {
		t.Errorf("processRS() ClusterStatus = %q, want %q", rs.ClusterStatus, rayv1.Running)
	}
	if rs.DashboardAddress != "10.96.0.20:8265" {
		t.Errorf("processRS() DashboardAddress = %q, want 10.96.0.20:8265", rs.DashboardAddress)
	}
	if cluster == nil {
		t.Errorf("processRS() should return the backing ray cluster")
	}
}

func TestProcessRJ(t *testing.T) {
	mockClientset, mockCluster, _, _ := buildMock()
	mockCluster.On("Get", mock.Anything, mock.Anything, mock.Anything).Return((*rayv1.RayCluster)(nil),
		errors.New("no cluster"))
	v := newExtraRayClient(t, mockClientset, nil)

	crdInfo := &CrdInfo{Name: "proc-job", Namespace: TestNamespace, CrdType: constant.RayJob}
	r := &rayv1.RayJob{
		ObjectMeta: metav1.ObjectMeta{Name: "proc-job"},
		Status: rayv1.RayJobStatus{
			JobId:              "job-123",
			RayClusterName:     "job-cluster",
			JobDeploymentStatus: rayv1.JobDeploymentStatusRunning,
		},
	}

	data, cluster := v.processRJ(TestNamespace, crdInfo, r)
	rj, ok := data.(*RayJob)
	if !ok {
		t.Fatalf("processRJ() data type = %T, want *RayJob", data)
	}
	if rj.JobId != "job-123" || rj.RayClusterName != "job-cluster" {
		t.Errorf("processRJ() job info not copied: %+v", rj)
	}
	if rj.ClusterStatus != string(rayv1.JobDeploymentStatusRunning) {
		t.Errorf("processRJ() ClusterStatus = %q, want %q",
			rj.ClusterStatus, rayv1.JobDeploymentStatusRunning)
	}
	if cluster != nil {
		t.Errorf("processRJ() should return nil cluster when the ray cluster is missing")
	}
}

func TestBuildCrdInfoRecord(t *testing.T) {
	resource, err := ParseRayResource(constant.RayCluster, fmt.Sprintf(rayclusterTemp, "record-cluster"))
	if err != nil {
		t.Fatalf("ParseRayResource() error = %v, want nil", err)
	}
	record := buildCrdInfoRecord(constant.RayCluster, "config-data", "admin", resource)
	if record.Name != "record-cluster" || record.Namespace != TestNamespace {
		t.Errorf("buildCrdInfoRecord() name/namespace = %q/%q, want record-cluster/vcjob",
			record.Name, record.Namespace)
	}
	if record.CreateBy != "admin" || record.ConfigDetail != "config-data" {
		t.Errorf("buildCrdInfoRecord() createBy/configDetail = %q/%q, want admin/config-data",
			record.CreateBy, record.ConfigDetail)
	}
	if record.RayVersion != "2.9.0" || record.Status != 0 {
		t.Errorf("buildCrdInfoRecord() rayVersion/status = %q/%d, want 2.9.0/0", record.RayVersion, record.Status)
	}
}
