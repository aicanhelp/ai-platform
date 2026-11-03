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
	"io"
	"net/http"
	"regexp"
	"strings"

	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	rayversioned "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned"
	"gorm.io/gorm"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/yaml"

	"openfuyao-rayservice/pkg/constant"
	"openfuyao-rayservice/pkg/utils"
	"openfuyao-rayservice/pkg/utils/httputil"
	"openfuyao-rayservice/pkg/zlog"
)

type RayHeadStatus struct {
	State *corev1.ContainerState `json:"state"`
	Phase string                 `json:"phase"`
}

type CrdInfo struct {
	ID           uint   `gorm:"primarykey;autoIncrement" json:"-"`
	Name         string `gorm:"column:name;type:varchar(255);not null;uniqueIndex:idx_name_ns_crd" json:"name"`
	Namespace    string `gorm:"column:namespace;type:varchar(100);not null;uniqueIndex:idx_name_ns_crd" json:"namespace"`
	CreateBy     string `gorm:"column:create_by;type:varchar(255);not null;comment:creator name" json:"create_by"`
	ErrorInfo    string `gorm:"column:error_info;type:text;comment:error info mssage" json:"error_info"`
	ConfigDetail string `gorm:"column:config_detail;type:text;not null;comment:raycl detail config" json:"config_detail"`
	CreateAt     int64  `gorm:"column:create_at;autoCreateTime;comment:create time" json:"-"`
	UpdateAt     int64  `gorm:"column:update_at;autoUpdateTime;comment:update time" json:"-"`
	RayVersion   string `gorm:"column:ray_version;type:varchar(100);not null;comment:ray vesion" json:"ray_version"`
	Status       int    `gorm:"column:status;type:int(2);comment:status. {0: draft, 1: activate}" json:"-"`
	CrdType      string `gorm:"column:crd_type;type:varchar(100);not null;uniqueIndex:idx_name_ns_crd;comment:kuberay crd type" json:"-"`

	CreateTime        string `gorm:"-:all" json:"create_time"`
	UpdateTime        string `gorm:"-:all" json:"update_time"`
	ClusterStatus     string `gorm:"-:all" json:"cluster_status"`
	RayHeadLogs       string `gorm:"-:all" json:"ray_head_logs"`
	TemplateStatus    string `gorm:"-:all" json:"template_status"`
	ClusterStatusZHCN string `gorm:"-:all" json:"cluster_status_zhcn"`
}

type RayCluster struct {
	*CrdInfo
	GcsPort       string        `gorm:"-:all" json:"gcs_port"`
	ClusterIP     string        `gorm:"-:all" json:"cluster_ip"`
	DashboardPort string        `gorm:"-:all" json:"dashboard_port"`
	RayHeadStatus RayHeadStatus `gorm:"-:all" json:"ray_head_status"`
}

type RayService struct {
	*CrdInfo
	DashboardAddress string `gorm:"-:all" json:"dashboard_address"`
}

type RayJob struct {
	*CrdInfo
	JobId               string `gorm:"-:all" json:"job_id"`
	RayClusterName      string `gorm:"-:all" json:"raycluster_name"`
	DashboardAddress    string `gorm:"-:all" json:"dashboard_address"`
	JobDeploymentStatus string `gorm:"-:all" json:"job_deployment_status"`
}

func (CrdInfo) TableName() string {
	return "kuberay_crd"
}

func (v rayClient) MethodInitOperation() {
	table := CrdInfo{}
	zlog.Infof("init table %s ...", table.TableName())

	err := v.dbConnection.Table(table.TableName()).Migrator().AutoMigrate(&table)
	if err != nil {
		zlog.FormatFatal("Create Table: `kuberay_crd` error, %v: \n", err)
	}

}

func (v rayClient) MethodAllOverview() (*httputil.ResponseJson, int) {

	rayclusters, err := v.rayClientset.RayV1().RayClusters("").List(context.TODO(), metav1.ListOptions{
		LabelSelector: fmt.Sprintf("%s=%s", constant.RayClusterCreateBy, constant.ConsoleServiceDefaultOrgName),
	})
	if err != nil {
		zlog.Errorf("Get system RayClusters[Fuyao] failed, %v", err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}

	rayjobs, err := v.rayClientset.RayV1().RayJobs("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("Get system rayjobs failed, %v", err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}

	rayservices, err := v.rayClientset.RayV1().RayServices("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("Get system rayservices failed, %v", err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}

	total, err := v.rayClientset.RayV1().RayClusters("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("Get system RayClusters[Total] failed, %v", err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{
			"rayclusters": len(rayclusters.Items),
			"rayservices": len(rayservices.Items),
			"rayjobs":     len(rayjobs.Items),
			"total":       len(total.Items),
		},
	}, http.StatusOK

}

func (v rayClient) MethodCreateCrd(crd, data, createBy string, isStart bool) (*httputil.ResponseJson, int) {
	return v.doCreateCrd(crd, data, createBy, isStart, false)
}

// ResourceRay test
type ResourceRay interface {
	// 获取资源的元数据（包含 Name, Namespace, Labels 等）
	GetMetadata() *metav1.ObjectMeta

	// 设置资源的命名空间
	SetNamespace(ns string)

	// 获取资源的 Ray 版本
	GetRayVersion() string

	// 在创建前进行准备工作，例如添加必要的 Label
	PrepareForCreation(createBy string)

	// 获取该资源对应的 Kubernetes runtime.Object，用于 client-go 调用
	GetK8sObject() runtime.Object

	// 获取创建该资源所需的具体函数，这里使用函数变量更灵活
	// 也可以在接口中直接定义 Create(ctx, clientset) error 方法
	GetCreatorFunc() func(ctx context.Context, clientset rayversioned.Interface,
		obj runtime.Object, opts metav1.CreateOptions) (runtime.Object, error)
}

// ClusterWrapper test
type ClusterWrapper struct {
	*rayv1.RayCluster
}

// ServiceWrapper test
type ServiceWrapper struct {
	*rayv1.RayService
}

// JobWrapper test
type JobWrapper struct {
	*rayv1.RayJob
}

// GetMetadata test
func (w *ClusterWrapper) GetMetadata() *metav1.ObjectMeta {
	return &w.ObjectMeta
}

// SetNamespace test
func (w *ClusterWrapper) SetNamespace(ns string) {
	w.Namespace = ns
}

// GetRayVersion test
func (w *ClusterWrapper) GetRayVersion() string {
	return w.Spec.RayVersion
}

// PrepareForCreation test
func (cw *ClusterWrapper) PrepareForCreation(createBy string) {
	if cw.Labels == nil {
		cw.Labels = make(map[string]string)
	}
	cw.Labels[constant.KuberayCrdCreateBy] = createBy
	cw.Labels[constant.RayClusterCreateBy] = createBy

	headPod := &cw.Spec.HeadGroupSpec.Template
	if headPod.Labels == nil {
		headPod.Labels = make(map[string]string)
	}
	headPod.Labels[constant.KuberayCrdCreateBy] = createBy
}

// GetK8sObject test
func (w *ClusterWrapper) GetK8sObject() runtime.Object {
	return w.RayCluster
}

// GetCreatorFunc test
func (w *ClusterWrapper) GetCreatorFunc() func(ctx context.Context, clientset rayversioned.Interface,
	obj runtime.Object, opts metav1.CreateOptions) (runtime.Object, error) {
	return func(ctx context.Context, clientset rayversioned.Interface, obj runtime.Object,
		opts metav1.CreateOptions) (runtime.Object, error) {
		return clientset.RayV1().RayClusters(w.Namespace).Create(ctx, w.RayCluster, opts)
	}
}

// GetMetadata test
func (w *ServiceWrapper) GetMetadata() *metav1.ObjectMeta { return &w.ObjectMeta }

// SetNamespace test
func (w *ServiceWrapper) SetNamespace(ns string) { w.Namespace = ns }

// GetRayVersion test
func (w *ServiceWrapper) GetRayVersion() string { return w.Spec.RayClusterSpec.RayVersion }

// PrepareForCreation test
func (w *ServiceWrapper) PrepareForCreation(createBy string) {
	if w.Labels == nil {
		w.Labels = make(map[string]string)
	}
	w.Labels[constant.KuberayCrdCreateBy] = createBy
	headPod := &w.Spec.RayClusterSpec.HeadGroupSpec.Template
	if headPod.Labels == nil {
		headPod.Labels = make(map[string]string)
	}
	headPod.Labels[constant.KuberayCrdCreateBy] = createBy
}

// GetK8sObject test
func (w *ServiceWrapper) GetK8sObject() runtime.Object { return w.RayService }

// GetCreatorFunc test
func (w *ServiceWrapper) GetCreatorFunc() func(ctx context.Context, clientset rayversioned.Interface,
	obj runtime.Object, opts metav1.CreateOptions) (runtime.Object, error) {
	return func(ctx context.Context, clientset rayversioned.Interface, obj runtime.Object,
		opts metav1.CreateOptions) (runtime.Object, error) {
		return clientset.RayV1().RayServices(w.Namespace).Create(ctx, w.RayService, opts)
	}
}

// GetMetadata test
func (w *JobWrapper) GetMetadata() *metav1.ObjectMeta { return &w.ObjectMeta }

// SetNamespace test
func (w *JobWrapper) SetNamespace(ns string) { w.Namespace = ns }

// GetRayVersion test
func (w *JobWrapper) GetRayVersion() string { return w.Spec.RayClusterSpec.RayVersion }

// PrepareForCreation test
func (jw *JobWrapper) PrepareForCreation(createBy string) {
	if jw.Labels == nil {
		jw.Labels = make(map[string]string)
	}
	jw.Labels[constant.KuberayCrdCreateBy] = createBy
	headPod := &jw.Spec.RayClusterSpec.HeadGroupSpec.Template
	if headPod.Labels == nil {
		headPod.Labels = make(map[string]string)
	}
	headPod.Labels[constant.KuberayCrdCreateBy] = createBy
}

// GetK8sObject test
func (w *JobWrapper) GetK8sObject() runtime.Object { return w.RayJob }

// GetCreatorFunc test
func (w *JobWrapper) GetCreatorFunc() func(ctx context.Context, clientset rayversioned.Interface,
	obj runtime.Object, opts metav1.CreateOptions) (runtime.Object, error) {
	return func(ctx context.Context, clientset rayversioned.Interface, obj runtime.Object,
		opts metav1.CreateOptions) (runtime.Object, error) {
		return clientset.RayV1().RayJobs(w.Namespace).Create(ctx, w.RayJob, opts)
	}
}

func validateKubernetesResourceName(name string) error {
	const maxLen = 253
	// 1. 长度校验
	if len(name) == 0 {
		return fmt.Errorf("resource name cannot be empty")
	}
	if len(name) > maxLen {
		return fmt.Errorf("resource name '%s' exceeds the maximum length of 253 characters", name)
	}

	// 2. 正则校验
	// RFC 1123 DNS子域名格式:
	// - 最多253个字符
	// - 包含小写字母、数字、连字符(-)和点(.)
	// - 必须以字母或数字开头和结尾
	nameRegex := `^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$`
	re := regexp.MustCompile(nameRegex)
	if !re.MatchString(name) {
		return fmt.Errorf("resource name '%s' is invalid", name)
	}

	return nil
}

// ParseRayResource test
func ParseRayResource(crdType, data string) (ResourceRay, error) {
	switch crdType {
	case constant.RayCluster:
		instance := rayv1.RayCluster{}
		if err := yaml.Unmarshal([]byte(data), &instance); err != nil {
			return nil, fmt.Errorf("failed to unmarshal RayCluster: %w", err)
		}
		name := instance.Name
		if validationErr := validateKubernetesResourceName(name); validationErr != nil {
			return nil, fmt.Errorf("invalid RayCluster name: %w", validationErr)
		}
		return &ClusterWrapper{RayCluster: &instance}, nil
	case constant.RayService:
		instance := rayv1.RayService{}
		if err := yaml.Unmarshal([]byte(data), &instance); err != nil {
			return nil, fmt.Errorf("failed to unmarshal RayService: %w", err)
		}
		name := instance.Name
		if validationErr := validateKubernetesResourceName(name); validationErr != nil {
			return nil, fmt.Errorf("invalid RayService name: %w", validationErr)
		}
		return &ServiceWrapper{RayService: &instance}, nil
	case constant.RayJob:
		instance := rayv1.RayJob{}
		if err := yaml.Unmarshal([]byte(data), &instance); err != nil {
			return nil, fmt.Errorf("failed to unmarshal RayJob: %w", err)
		}
		name := instance.Name
		if validationErr := validateKubernetesResourceName(name); validationErr != nil {
			return nil, fmt.Errorf("invalid RayJob name: %w", validationErr)
		}
		return &JobWrapper{RayJob: &instance}, nil
	default:
		return nil, fmt.Errorf("unknown crd type: %s", crdType)
	}
}

// parseRayResourceAndSetDefaults 解析资源并设置默认命名空间
func parseRayResourceAndSetDefaults(crdType, data string) (ResourceRay, error) {
	resource, err := ParseRayResource(crdType, data)
	if err != nil {
		return nil, fmt.Errorf("parse user resource failed: %w", err)
	}

	if resource == nil {
		return nil, fmt.Errorf("internal error: ParseRayResource returned a nil resource")
	}

	meta := resource.GetMetadata()
	resource.SetNamespace(getDefaultNs(meta.Namespace, constant.Default))

	return resource, nil
}

// buildCrdInfoRecord 根据资源和输入构建数据库记录
func buildCrdInfoRecord(crdType, data, createBy string, resource ResourceRay) CrdInfo {
	meta := resource.GetMetadata()
	return CrdInfo{
		CreateBy:     createBy,
		ErrorInfo:    "",
		ConfigDetail: data,
		CrdType:      crdType,
		Status:       0, // Initial status
		Name:         meta.Name,
		Namespace:    meta.Namespace,
		RayVersion:   resource.GetRayVersion(),
	}
}

// checkResourceExistsInDB 检查数据库中是否已存在指定的资源
func (v *rayClient) checkResourceExistsInDB(name, crdType, namespace string) (bool, error) {
	var existingCrd CrdInfo
	err := v.dbConnection.Where("name = ? AND crd_type = ? AND namespace = ?",
		name, crdType, namespace).First(&existingCrd).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return false, nil
		}
		return false, fmt.Errorf("database query failed: %w", err)
	}
	return true, nil
}

// upsertCrdInfoInTransaction 在事务中创建或更新 CrdInfo 记录
func (v *rayClient) upsertCrdInfoInTransaction(tx *gorm.DB, rc *CrdInfo) error {
	var crdInfoFromDB CrdInfo
	if err := tx.Where("name = ? AND crd_type = ? AND namespace = ?",
		rc.Name, rc.CrdType, rc.Namespace).First(&crdInfoFromDB).Error; err != nil {
		// Not found, create it
		if err := tx.Create(rc).Error; err != nil {
			return fmt.Errorf("create crd info failed: %w", err)
		}
	} else {
		// Found, update it
		// GORM's Updates doesn't update zero-value fields, so we need to handle it carefully or assign all fields.
		// Here we update the record ID and save the whole struct to ensure all fields are updated.
		rc.ID = crdInfoFromDB.ID // Keep the primary key
		if err := tx.Save(rc).Error; err != nil {
			return fmt.Errorf("update crd info failed: %w", err)
		}
	}
	return nil
}

// createKubernetesResource 在 K8s 中创建资源
func (v *rayClient) createKubernetesResource(crdType string, resource ResourceRay, rc *CrdInfo) error {
	resource.PrepareForCreation(constant.ConsoleServiceDefaultOrgName)
	creator := resource.GetCreatorFunc()
	_, err := creator(context.TODO(), v.rayClientset, resource.GetK8sObject(), metav1.CreateOptions{})
	if err != nil {
		// Log the error and update the record's ErrorInfo in the caller
		return fmt.Errorf("create %s %s in k8s failed: %w", crdType, rc.Name, err)
	}
	return nil
}

// updateCrdStatusInDB 更新数据库中 CRD 的状态
func (v *rayClient) updateCrdStatusInDB(rc *CrdInfo) error {
	if err := v.dbConnection.Save(rc).Error; err != nil {
		return fmt.Errorf("failed to update crd status in db: %w", err)
	}
	return nil
}

// doCreateCrd 创建或更新 CRD，并可选择性地在 K8s 中启动它。
// 此函数作为核心流程的编排器。
func (v *rayClient) doCreateCrd(crdType, data, createBy string, isStart,
	fromRestart bool) (*httputil.ResponseJson, int) {

	// 1. 解析资源并设置默认值
	resource, err := parseRayResourceAndSetDefaults(crdType, data)
	if err != nil {
		zlog.Errorf("%v", err)
		return httputil.GetClientFailureResponseJsonWithData(err.Error()), http.StatusBadRequest
	}

	// 2. 构建数据库记录对象
	rc := buildCrdInfoRecord(crdType, data, createBy, resource)

	// 3. 检查资源是否已存在（仅在非重启时）
	if !fromRestart {
		exists, err := v.checkResourceExistsInDB(rc.Name, rc.CrdType, rc.Namespace)
		if err != nil {
			zlog.Errorf("%v", err)
			return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
		}
		if exists {
			return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "Resource already exists!",
				Data: []any{},
			}, http.StatusOK
		}
	}

	// 4. 通过事务持久化 CrdInfo 记录
	if err := v.dbConnection.Transaction(func(tx *gorm.DB) error {
		return v.upsertCrdInfoInTransaction(tx, &rc)
	}); err != nil {
		zlog.Errorf("%v", err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}

	// 5. 如果需要，则在 K8s 中创建资源
	if isStart {
		if err := v.createKubernetesResource(crdType, resource, &rc); err != nil {
			zlog.Errorf("%v", err)
			rc.ErrorInfo = err.Error()
			// Update DB with the error, but don't fail the response if the save fails to avoid hiding the primary error.
			_ = v.updateCrdStatusInDB(&rc)
			return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
		}

		// 6. K8s 创建成功，更新状态
		rc.Status = 1
		if err := v.updateCrdStatusInDB(&rc); err != nil {
			zlog.Errorf("%v", err)
		}
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{},
	}, http.StatusOK
}

func (v rayClient) MethodUpdateCrd(crd, ns, data, name string) (*httputil.ResponseJson, int) {
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{},
	}, http.StatusOK
}

func (v rayClient) MethodDeleteCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	var resourceName string
	var isRunning bool = false
	switch crd {
	case constant.RayCluster:
		raycluster, _ := v.rayClientset.RayV1().RayClusters(ns).Get(context.TODO(), name, metav1.GetOptions{})
		resourceName = raycluster.Name
		isRunning = raycluster.Status.State == rayv1.Ready
	case constant.RayService:
		rayservice, _ := v.rayClientset.RayV1().RayServices(ns).Get(context.TODO(), name, metav1.GetOptions{})
		resourceName = rayservice.Name
		isRunning = rayservice.Status.ServiceStatus == rayv1.Running
	case constant.RayJob:
		rayjob, _ := v.rayClientset.RayV1().RayJobs(ns).Get(context.TODO(), name, metav1.GetOptions{})
		resourceName = rayjob.Name
		isRunning = rayjob.Status.JobDeploymentStatus == rayv1.JobDeploymentStatusRunning
	default:
		zlog.Errorf("Unknown Resource Found ..")
		return httputil.GetDefaultClientFailureResponseJson(), http.StatusBadRequest
	}
	if resourceName != "" {
		if isRunning {
			return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "不允许删除正在运行的资源!",
				Data: map[string]any{},
			}, http.StatusOK
		}
		// other delete k8s resource
		v.MethodStopCrd(crd, ns, name)
	}
	delRes := v.dbConnection.Model(&CrdInfo{}).Where("name = ?", name).
		Where("crd_type = ?", crd).
		Where("namespace = ?", ns).Delete(&CrdInfo{})
	if delRes.RowsAffected == 0 {
		zlog.Errorf("Delete crd failed, crd: %s, ns: %s, name: %s", crd, ns, name)
		return httputil.GetResponseJson(
			constant.ClientError,
			"资源不存在!",
			map[string]any{}), http.StatusOK
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{},
	}, http.StatusOK
}

func (v rayClient) MethodStopCrd(crd, ns, name string) (*httputil.ResponseJson, int) {

	var err error
	switch crd {
	case constant.RayCluster:
		err = v.rayClientset.RayV1().RayClusters(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
	case constant.RayService:
		err = v.rayClientset.RayV1().RayServices(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
	case constant.RayJob:
		err = v.rayClientset.RayV1().RayJobs(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
	default:
		zlog.Errorf("Unknown Resource Found ..")
		return httputil.GetDefaultClientFailureResponseJson(), http.StatusBadRequest
	}
	if err != nil {
		zlog.Errorf("Stop Crd failed, type: %s, name: %s,  error: %v", crd, name, err)
		return httputil.GetServerFailureResponseJsonWithData(err.Error()), http.StatusInternalServerError
	}
	v.dbConnection.Model(&CrdInfo{}).
		Where("name = ?", name).
		Where("crd_type = ?", crd).
		Where("namespace = ?", ns).
		Update("error_info", "")
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{},
	}, http.StatusOK
}

func (v rayClient) MethodRestartCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	crdInfo := CrdInfo{}
	v.dbConnection.Where("name = ?", name).
		Where("namespace = ?", ns).
		Where("crd_type = ?", crd).
		First(&crdInfo)
	if crdInfo.Name == "" {
		zlog.Errorf("Restart crd failed, name: %s", name)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	_, err := v.doGetCrd(crd, ns, name)
	if err == nil { // Already exists resource
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "资源正在运行中!",
			Data: []any{},
		}, http.StatusOK
	}
	return v.doCreateCrd(crd, crdInfo.ConfigDetail, crdInfo.CreateBy, true, true)
}

func (v rayClient) buildMissingRayClusterResponse() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: constant.NoContent,
		Msg:  "RayCluster is missing.",
		Data: nil,
	}
}

func (v rayClient) buildMissingCrdInfoNameResponse() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: constant.ResourceNotFound,
		Msg:  "Not Found",
		Data: []any{},
	}
}

func (v rayClient) buildMissingPodsResponse() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: constant.NoContent,
		Msg:  "Pod list is nil.",
		Data: nil,
	}
}

func (v rayClient) buildSuccessResponse(data interface{}) *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: data,
	}
}

func (v rayClient) processRC(c *CrdInfo, r *rayv1.RayCluster) (interface{}, *rayv1.RayCluster) {
	rayCluster := RayCluster{
		CrdInfo: c,
	}
	transferRayClusterToEntity(r, &rayCluster)

	stateMap := v.getPodStatus(rayCluster.Namespace, &[]string{rayCluster.Name})
	rayCluster.RayHeadStatus = *(*stateMap)[rayCluster.Name]
	rayCluster.ClusterStatus = string(r.Status.State)

	return &rayCluster, r
}

func (v rayClient) processRS(ns string, c *CrdInfo, r *rayv1.RayService) (interface{}, *rayv1.RayCluster) {
	rayService := RayService{
		CrdInfo: c,
	}

	rayV1RayCluster, _ := v.rayClientset.RayV1().RayClusters(ns).
		Get(context.TODO(), r.Status.ActiveServiceStatus.RayClusterName, metav1.GetOptions{})

	if rayV1RayCluster != nil {
		rayService.DashboardAddress = fmt.Sprintf("%s:%s",
			r.Status.ActiveServiceStatus.RayClusterStatus.Head.ServiceIP,
			r.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["dashboard"])
	}

	rayService.ClusterStatus = string(r.Status.ServiceStatus)
	return &rayService, rayV1RayCluster
}

func (v rayClient) processRJ(ns string, c *CrdInfo, r *rayv1.RayJob) (interface{}, *rayv1.RayCluster) {
	rayJob := RayJob{
		CrdInfo: c,
	}

	rayV1RayCluster, _ := v.rayClientset.RayV1().RayClusters(ns).
		Get(context.TODO(), r.Status.RayClusterName, metav1.GetOptions{})

	if rayV1RayCluster != nil {
		rayJob.DashboardAddress = fmt.Sprintf("%s:%s",
			rayV1RayCluster.Status.Head.PodIP,
			rayV1RayCluster.Status.Endpoints["dashboard"])
	}

	rayJob.JobId = r.Status.JobId
	rayJob.RayClusterName = r.Status.RayClusterName
	rayJob.JobDeploymentStatus = string(r.Status.JobDeploymentStatus)
	rayJob.ClusterStatus = string(r.Status.JobDeploymentStatus)

	return &rayJob, rayV1RayCluster
}

func (v rayClient) MethodGetCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	// 步骤 1: 从数据库获取缓存的 CrdInfo
	crdInfo, found := v.getCrdInfoFromDB(crd, ns, name)
	if !found {
		return v.buildMissingCrdInfoNameResponse(), http.StatusOK
	}

	// 步骤 2: 从 Kubernetes API 获取实时 CRD 资源
	crdResource, err := v.doGetCrd(crd, ns, name)
	if err != nil {
		// 步骤 2a: 如果获取失败，处理错误并更新状态
		v.handleCrdFetchError(err, &crdInfo)
		// 直接返回更新后的数据库信息，不再进行后续的 Pod/Log 获取
		normalizeEntity(&crdInfo)
		return v.buildSuccessResponse(&crdInfo), http.StatusOK
	}

	// 步骤 3: 处理获取到的 CRD 资源，并提取核心的 RayCluster 信息
	data, rayV1RayCluster := v.processCrdResource(crd, ns, &crdInfo, crdResource)

	// 步骤 4: 使用 RayCluster 信息进一步丰富 CrdInfo（如获取 Pod 日志）
	if rayV1RayCluster != nil {
		crdInfo.RayHeadLogs = v.enrichWithRayClusterData(ns, rayV1RayCluster)
	} else {
		// 如果处理后的 RayCluster 为空，说明存在问题，直接返回
		// 例如，RayService 可能还未创建 RayCluster
		return v.buildMissingRayClusterResponse(), http.StatusOK
	}

	// 步骤 5: 规范化数据并构建成功响应
	normalizeEntity(&crdInfo)
	return v.buildSuccessResponse(data), http.StatusOK
}

func (v rayClient) getCrdInfoFromDB(crd, ns, name string) (CrdInfo, bool) {
	var crdInfo CrdInfo
	v.dbConnection.Where("name = ?", name).
		Where("crd_type = ?", crd).
		Where("namespace = ?", ns).First(&crdInfo)

	if crdInfo.Name == "" {
		return crdInfo, false
	}
	return crdInfo, true
}

// handleCrdFetchError 处理从 Kubernetes 获取 CRD 失败的情况。
// 它会根据数据库中已有的错误信息来更新 CrdInfo 的状态。
func (v rayClient) handleCrdFetchError(err error, crdInfo *CrdInfo) {
	zlog.Errorf("Get crd failed, name: %s,  error: %v", crdInfo.Name, err)
	if crdInfo.ErrorInfo != "" {
		crdInfo.ClusterStatus = constant.Failed
	} else {
		crdInfo.ClusterStatus = constant.Suspended
	}
	crdInfo.TemplateStatus = constant.Template
}

// processCrdResource 根据不同的 CRD 类型，调用相应的处理逻辑。
// 返回处理后的数据和从资源中提取的 RayCluster 对象。
func (v rayClient) processCrdResource(crd, ns string, crdInfo *CrdInfo, crdResource interface{}) (interface{}, *rayv1.RayCluster) {
	crdInfo.TemplateStatus = constant.Instance

	var data interface{} = &crdInfo
	var rayV1RayCluster *rayv1.RayCluster

	switch crd {
	case constant.RayCluster:
		data, rayV1RayCluster = v.processRC(crdInfo, crdResource.(*rayv1.RayCluster))
	case constant.RayService:
		data, rayV1RayCluster = v.processRS(ns, crdInfo, crdResource.(*rayv1.RayService))
	case constant.RayJob:
		data, rayV1RayCluster = v.processRJ(ns, crdInfo, crdResource.(*rayv1.RayJob))
	default:
		zlog.Errorf("Unknown Resource Found, type: %s", crd)
	}

	return data, rayV1RayCluster
}

// enrichWithRayClusterData 获取与 RayCluster 关联的 Pod，并获取 Head Pod 的日志。
// 返回获取到的日志字符串。
func (v rayClient) enrichWithRayClusterData(ns string, rayV1RayCluster *rayv1.RayCluster) string {
	if rayV1RayCluster == nil {
		zlog.Error("enrichWithRayClusterData called with nil rayV1RayCluster")
		return ""
	}

	// 获取 RayCluster 对应的 Pod 列表
	podList, err := v.clientset.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{
		LabelSelector: fmt.Sprintf("ray.io/cluster=%s", rayV1RayCluster.Name),
	})

	if err != nil {
		zlog.Errorf("Get rayCluster Pod failed for cluster %s, error: %v", rayV1RayCluster.Name, err)
		return "" // 获取失败，返回空字符串
	}

	if podList == nil || len(podList.Items) == 0 {
		zlog.Warnf("No pods found for RayCluster: %s", rayV1RayCluster.Name)
		return "" // 找不到Pod，返回空字符串
	}

	// 获取 Head Pod 的日志
	// 注意：这里假设第一个 Pod 是 Head Pod，这是一个较强的假设。
	// 更严谨的做法是根据 "ray.io/node-type: head" 标签来筛选。
	headPod := podList.Items[0]
	containerName := rayV1RayCluster.Spec.HeadGroupSpec.Template.Spec.Containers[0].Name
	logs := v.fetchPodLog(headPod.Namespace, rayV1RayCluster.Name, headPod.Name, containerName)

	return logs
}

func (v rayClient) doGetCrd(crd, ns, name string) (interface{}, error) {
	var crdResource interface{} = rayv1.RayCluster{}
	var err error
	switch crd {
	case constant.RayCluster:
		crdResource, err = v.rayClientset.RayV1().RayClusters(ns).Get(context.TODO(), name, metav1.GetOptions{})
	case constant.RayService:
		crdResource, err = v.rayClientset.RayV1().RayServices(ns).Get(context.TODO(), name, metav1.GetOptions{})
	case constant.RayJob:
		crdResource, err = v.rayClientset.RayV1().RayJobs(ns).Get(context.TODO(), name, metav1.GetOptions{})
	default:
		zlog.Errorf("Unknown Resource Found ..")
		return nil, errors.New("unknown resource found")
	}
	return crdResource, err
}

func (v rayClient) MethodListMetaCrd(crd string) (*httputil.ResponseJson, int) {
	data := make(map[string]any)
	var createBy []string
	v.dbConnection.Model(&CrdInfo{}).Distinct().Pluck("create_by", &createBy)
	data["createBy"] = createBy

	switch crd {
	case constant.RayCluster:
		data["status"] = constant.GetServiceStatusMap()
		var rayVersion []string
		v.dbConnection.Model(&CrdInfo{}).Distinct().Pluck("ray_version", &rayVersion)
		data["rayVersion"] = rayVersion
	case constant.RayService:
		data["status"] = constant.GetServiceStatusMap()
	case constant.RayJob:
		data["status"] = constant.GetJobDeploymentStatusMap()
	default:
		zlog.Errorf("Unknown Resource Found ..")
		return httputil.GetDefaultClientFailureResponseJson(), http.StatusBadRequest
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: data,
	}, http.StatusOK
}

func (v *rayClient) populateRayClusterInfos(ctx context.Context, keyMappings map[string]interface{}) error {
	rayClusters, err := v.rayClientset.RayV1().RayClusters("").List(ctx, metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("list rayCluster failed")
	}

	// Keep track of live clusters, needed for pod status fetching
	aliveNamespacedRaycls := make(map[string][]string)
	utils.MapInPlace(rayClusters.Items, func(item *rayv1.RayCluster) {
		if _, ok := aliveNamespacedRaycls[item.Namespace]; !ok {
			aliveNamespacedRaycls[item.Namespace] = []string{}
		}
		aliveNamespacedRaycls[item.Namespace] = append(aliveNamespacedRaycls[item.Namespace], item.Name)

		key := item.Name + ":" + item.Namespace
		if v, ok := keyMappings[key]; ok {
			rc, assertOk := v.(*RayCluster)
			if !assertOk {
				// This should not happen if keyMappings is set up correctly
				zlog.Error(fmt.Errorf("type assertion failed"), "Type assertion failed", "key",
					key, "expected", "*RayCluster", "got", fmt.Sprintf("%T", v))
				return
			}
			rc.TemplateStatus = constant.Instance
			rc.ClusterStatus = string(item.Status.State)
			rc.DashboardPort = item.Status.Endpoints["dashboard"]
			rc.GcsPort = item.Status.Endpoints["gcs"]
			rc.ClusterIP = item.Status.Head.PodIP
		}
	})

	// Fetch pod status for all live clusters
	for ns, raycls := range aliveNamespacedRaycls {
		stateMap := v.getPodStatus(ns, &raycls)
		utils.MapInPlace(raycls, func(rayclusterName *string) {
			key := *rayclusterName + ":" + ns
			v, ok := keyMappings[key]
			if !ok {
				return
			}
			rc, assertOk := v.(*RayCluster)
			if !assertOk {
				zlog.Error(fmt.Errorf("type assertion failed"), "Type assertion failed", "key", key,
					"expected", "*RayCluster", "got", fmt.Sprintf("%T", v))
				return
			}
			if headStatus, ok := (*stateMap)[*rayclusterName]; ok {
				rc.RayHeadStatus = *headStatus
			}
		})
	}

	return nil
}

// populateRayServiceInfos fetches live RayService data from Kubernetes
// and updates the provided keyMappings with the latest status.
func (v *rayClient) populateRayServiceInfos(ctx context.Context, keyMappings map[string]interface{}) error {
	rayServices, err := v.rayClientset.RayV1().RayServices("").List(ctx, metav1.ListOptions{})
	if err != nil {
		zlog.Errorf("list rayServices failed")
	}

	utils.MapInPlace(rayServices.Items, func(item *rayv1.RayService) {
		key := item.Name + ":" + item.Namespace
		if v, ok := keyMappings[key]; ok {
			rs, assertOk := v.(*RayService)
			if !assertOk {
				zlog.Error(fmt.Errorf("type assertion failed"), "Type assertion failed", "key",
					key, "expected", "*RayService", "got", fmt.Sprintf("%T", v))
				return
			}
			rs.TemplateStatus = constant.Instance
			rs.ClusterStatus = string(item.Status.ServiceStatus)

			headInfo := item.Status.ActiveServiceStatus.RayClusterStatus.Head
			if headInfo != (rayv1.HeadInfo{}) {
				rs.DashboardAddress = fmt.Sprintf("%s:%s",
					headInfo.ServiceIP,
					item.Status.ActiveServiceStatus.RayClusterStatus.Endpoints["dashboard"])
			}
			if _, ok := constant.GetServiceFailedMap()[rayv1.ServiceStatus(rs.ClusterStatus)]; ok {
				rs.ClusterStatus = constant.Failed
			}
		}
	})
	return nil
}

// populateRayJobInfos fetches live RayJob data from Kubernetes
// and updates the provided keyMappings with the latest status.
func (v *rayClient) getDashboardAddressForRayCluster(ctx context.Context, namespace, rayClusterName string) string {
	if rayClusterName == "" || namespace == "" {
		return "N/A"
	}

	rayV1RayCluster, err := v.rayClientset.RayV1().RayClusters(namespace).Get(ctx, rayClusterName, metav1.GetOptions{})
	if err != nil {
		zlog.Error(err, "Failed to get RayCluster for RayJob", "RayCluster", rayClusterName, "Namespace", namespace)
		return "N/A"
	}

	headInfo := rayV1RayCluster.Status.Head
	if headInfo.PodIP != "" && rayV1RayCluster.Status.Endpoints != nil {
		if dashboardPort, ok := rayV1RayCluster.Status.Endpoints["dashboard"]; ok && dashboardPort != "" {
			return fmt.Sprintf("%s:%s", headInfo.PodIP, dashboardPort)
		}
	}

	return "N/A"
}

func (v *rayClient) updateRayJobFromK8s(ctx context.Context, k8sRayJob *rayv1.RayJob, internalRj *RayJob) {
	internalRj.TemplateStatus = constant.Instance
	internalRj.ClusterStatus = string(k8sRayJob.Status.JobDeploymentStatus)
	internalRj.JobId = k8sRayJob.Status.JobId
	internalRj.RayClusterName = k8sRayJob.Status.RayClusterName
	internalRj.JobDeploymentStatus = string(k8sRayJob.Status.JobDeploymentStatus)

	internalRj.DashboardAddress = v.getDashboardAddressForRayCluster(
		ctx,
		internalRj.Namespace,
		k8sRayJob.Status.RayClusterName,
	)
}

// populateRayJobInfos 主函数，负责协调和调用其他辅助函数。
func (v *rayClient) populateRayJobInfos(ctx context.Context, keyMappings map[string]interface{}) error {
	rayJobs, err := v.rayClientset.RayV1().RayJobs("").List(ctx, metav1.ListOptions{})
	if err != nil {
		zlog.Error(err, "Failed to list RayJobs from all namespaces")
		return err
	}

	for _, item := range rayJobs.Items {
		key := item.Name + ":" + item.Namespace
		if elem, ok := keyMappings[key]; ok {
			internalRj, assertOk := elem.(*RayJob)
			if !assertOk {
				zlog.Error(fmt.Errorf("type assertion failed"), "Type assertion failed for key", "key", key,
					"expected", "*RayJob", "got", fmt.Sprintf("%T", elem))
				continue
			}
			v.updateRayJobFromK8s(ctx, &item, internalRj)
		}
	}

	return nil
}

// MethodListCrd is the main function, now refactored to use the helper functions.
func (v *rayClient) MethodListCrd(crd string, req *CrdListRequest) (*httputil.ResponseJson, int) {
	// 1. 从数据库获取基础的 CRD 信息
	crdInfos, err := v.fetchCrdInfosFromDB(crd, req)
	if err != nil {
		zlog.Error(err, "Failed to fetch crd infos from database")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	// 如果没有数据，直接返回空列表
	if len(crdInfos) == 0 {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  "success",
			Data: map[string]any{"items": []interface{}{}, "total": 0},
		}, http.StatusOK
	}

	// 2. 创建初始的对象映射，并设置基础状态
	keyMappings := v.createInitialCrdMappings(crdInfos, crd)

	// 3. 从 K8s 等外部源填充实时信息
	if err := v.populateCrdDetails(context.TODO(), crd, keyMappings); err != nil {
		zlog.Error(err, "Failed to populate crd details")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	// 4. 根据请求筛选并构建最终的响应数据
	data := v.buildFinalResponseData(crdInfos, keyMappings, req.Status)

	// 5. 构建并返回成功的响应
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: map[string]any{
			"items": data,
			"total": len(data),
		},
	}, http.StatusOK
}

func (v *rayClient) fetchCrdInfosFromDB(crd string, req *CrdListRequest) ([]CrdInfo, error) {
	var crdInfos []CrdInfo
	query := v.dbConnection.Order("update_at desc").Where("crd_type = ?", crd)

	if req.Name != "" {
		query = query.Where("name like ?", "%"+req.Name+"%")
	}
	if req.RayVersion != "" {
		query = query.Where("ray_version = ?", req.RayVersion)
	}
	if req.CreateBy != "" {
		query = query.Where("create_by = ?", req.CreateBy)
	}

	// 返回查询结果和可能的错误
	return crdInfos, query.Find(&crdInfos).Error
}

// createInitialCrdMappings 遍历原始数据，创建具体类型的对象映射，并设置其初始状态。
func (v *rayClient) createInitialCrdMappings(crdInfos []CrdInfo, crd string) map[string]interface{} {
	keyMappings := make(map[string]interface{})
	for i := range crdInfos {
		item := &crdInfos[i] // 获取指针，以便修改

		// 根据类型创建具体实例并存入 map
		switch crd {
		case constant.RayCluster:
			keyMappings[item.Name+":"+item.Namespace] = &RayCluster{CrdInfo: item}
		case constant.RayService:
			keyMappings[item.Name+":"+item.Namespace] = &RayService{CrdInfo: item}
		case constant.RayJob:
			keyMappings[item.Name+":"+item.Namespace] = &RayJob{CrdInfo: item}
		default:
			zlog.Error(fmt.Errorf("unknown resource type"), "Unknown Resource Found", "crd", crd)
			continue
		}

		// 设置初始状态
		if item.ErrorInfo != "" {
			item.ClusterStatus = constant.Failed
		} else {
			item.ClusterStatus = constant.Suspended
		}
		item.TemplateStatus = constant.Template
		item.ConfigDetail = ""
	}
	return keyMappings
}

// populateCrdDetails 封装了根据 crd 类型调用相应填充方法的逻辑。
func (v *rayClient) populateCrdDetails(ctx context.Context, crd string, keyMappings map[string]interface{}) error {
	switch crd {
	case constant.RayCluster:
		return v.populateRayClusterInfos(ctx, keyMappings)
	case constant.RayService:
		return v.populateRayServiceInfos(ctx, keyMappings)
	case constant.RayJob:
		return v.populateRayJobInfos(ctx, keyMappings)
	default:
		// 这个错误通常在 createInitialCrdMappings 中已记录，这里返回错误以统一处理
		return fmt.Errorf("cannot populate details for unknown crd type: %s", crd)
	}
}

// buildFinalResponseData 负责筛选数据并规范化，生成最终返回的列表。
func (v *rayClient) buildFinalResponseData(crdInfos []CrdInfo, keyMappings map[string]interface{},
	statusFilter string) []interface{} {
	var data []interface{}
	for i := range crdInfos {
		r := &crdInfos[i]

		// 根据状态进行筛选
		if statusFilter == "" || strings.EqualFold(statusFilter, r.ClusterStatus) {
			// 规范化实体，这可能会修改 r 指向的内容，从而影响 keyMappings 中的对象
			normalizeEntity(r)
			data = append(data, keyMappings[r.Name+":"+r.Namespace])
		}
	}
	return data
}

func (v rayClient) fetchPodLog(ns, crdName, podName, containerName string) string {
	// 设置日志选项
	logOptions := &corev1.PodLogOptions{
		Container: containerName,    // 容器名称
		Follow:    false,            // 是否持续跟踪日志（类似 `kubectl logs -f`）
		TailLines: &[]int64{100}[0], // 获取最后 100 行日志
	}

	// 请求日志流
	req := v.clientset.CoreV1().Pods(ns).GetLogs(podName, logOptions) // podList.Items[0].Name
	podLogs, err := req.Stream(context.TODO())
	if err != nil {
		zlog.Errorf("Get rayCluster head-pod logs failed, name: %s,  error: %v", crdName, err)
		return ""
	}
	defer podLogs.Close()

	// 读取日志内容
	logBytes, err := io.ReadAll(podLogs)
	if err != nil {
		zlog.Errorf("Read rayCluster head-pod logs failed, name: %s,  error: %v", crdName, err)
		return ""
	}
	return string(logBytes)
}

func (v rayClient) getPodStatus(ns string, raycls *[]string) *map[string]*RayHeadStatus {

	statusMap := make(map[string]*RayHeadStatus)

	for _, v := range *raycls {
		statusMap[v] = &RayHeadStatus{
			State: nil,
			Phase: "Not Ready",
		}
	}

	pods, err := v.clientset.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labels.SelectorFromSet(map[string]string{
			"ray.io/node-type": "head",
		}).String(),
	})

	if err != nil {
		zlog.Errorf("List head-pod error %v", err)
		return &statusMap
	}

	for _, item := range pods.Items {
		var podClusterName string
		var ok bool
		if podClusterName, ok = item.Labels["ray.io/cluster"]; !ok {
			continue
		}
		if rayHeadStatus, ok := statusMap[podClusterName]; ok {
			rayHeadStatus.Phase = string(item.Status.Phase)
			cs := len(item.Status.ContainerStatuses)
			if cs > 0 {
				rayHeadStatus.State = &item.Status.ContainerStatuses[cs-1].State
			}
		}
	}

	return &statusMap
}

func transferRayClusterToEntity(r1 *rayv1.RayCluster, r2 *RayCluster) {
	r2.ID = 0
	r2.DashboardPort = r1.Status.Endpoints["dashboard"]
	r2.GcsPort = r1.Status.Endpoints["gcs"]
	r2.ClusterIP = r1.Status.Head.PodIP
}

func normalizeEntity(r *CrdInfo) {
	if r.CreateAt != 0 {
		r.CreateTime = utils.UnixFmtTime(r.CreateAt)
	}
	if r.UpdateAt != 0 {
		r.UpdateTime = utils.UnixFmtTime(r.UpdateAt)
	}
	switch r.CrdType {
	case constant.RayCluster:
		if v, ok := constant.GetClusterStatusMap()[rayv1.ClusterState(r.ClusterStatus)]; ok {
			r.ClusterStatusZHCN = v
		}
	case constant.RayService:
		if v, ok := constant.GetServiceStatusMap()[rayv1.ServiceStatus(r.ClusterStatus)]; ok {
			r.ClusterStatusZHCN = v
		}
		if _, ok := constant.GetServiceFailedMap()[rayv1.ServiceStatus(r.ClusterStatus)]; ok {
			r.ClusterStatus = constant.Failed
		}
	case constant.RayJob:
		if v, ok := constant.GetJobDeploymentStatusMap()[rayv1.JobDeploymentStatus(r.ClusterStatus)]; ok {
			r.ClusterStatusZHCN = v
		}
	default:
		zlog.Errorf("Unknown Resource Found ..")
	}

	if v, ok := constant.GetGlobalStatusMap()[r.ClusterStatus]; ok {
		r.ClusterStatusZHCN = v
	}

	if r.ClusterStatusZHCN == "" {
		r.ClusterStatusZHCN = "未知"
	}
}

func getDefaultNs(ns, defaultNs string) string {
	if ns == "" {
		return defaultNs
	}
	return ns
}
