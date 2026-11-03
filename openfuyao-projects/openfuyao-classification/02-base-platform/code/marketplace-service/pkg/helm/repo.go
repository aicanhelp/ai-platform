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

package helm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"helm.sh/helm/v3/pkg/repo"
	v1 "k8s.io/api/core/v1"
	k8sErrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/utils/k8sutil"
	"marketplace-service/pkg/utils/util"
	"marketplace-service/pkg/zlog"
)

// map key
const (
	modificationTimestamp = "modificationTimestamp"
	syncInProgressMsg     = "in_progress"
	syncCompleteMsg       = "complete"
	syncFailedMsg         = "failed"
)

var (
	repoCRDGVR = schema.GroupVersionResource{Group: constant.CRDRepoGroup, Version: constant.CRDRepoVersion,
		Resource: constant.CRDRepoResource}
	repoAsyncTaskMap = sync.Map{}
)

const (
	repoCRSecretBasicAuth = "basic-auth"
	repoCRSecretTLS       = "tls"
	repoCRConfigMapCA     = "ca"

	mapKeyUsername = "username"
	mapKeyPassword = "password"
	mapKeyTLSCrt   = "tls.crt"
	mapKeyTLSKey   = "tls.key"
	mapKeyCAKey    = "ca"

	requestLastTime = 10
)

func (c *helmClient) isRepoModificationAllowed(repoName string) (*httputil.ResponseJson, int, error) {
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil || config == nil {
		zlog.Errorf("error getting marketplace-service config: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting marketplace-service config"}, http.StatusInternalServerError, err
	}
	// 统一转为小写，避免大小写绕过
	lowerRepoName := strings.ToLower(repoName)
	lowerOfficialName := strings.ToLower(config.OfficialHarborDisplayName)
	lowerLocalName := strings.ToLower(config.LocalHarborDisplayName)
	if lowerRepoName == lowerOfficialName || lowerRepoName == lowerLocalName {
		const msg = "repo not allowed for modification"
		return &httputil.ResponseJson{
				Code: http.StatusBadRequest, // 补充：错误码修正为400（原500不合理）
				Msg:  msg}, http.StatusBadRequest,
			errors.New(msg)
	}
	return nil, 0, nil
}

func (c *helmClient) UpdateRepo(repoEntry *helm.SafeRepoEntry) (*httputil.ResponseJson, int) {
	if responseJson, status, err := c.isRepoModificationAllowed(repoEntry.Name); err != nil {
		return responseJson, status
	}
	repository, err := c.getCustomRepoByName(repoEntry.Name)
	if err != nil {
		if k8sErrors.IsNotFound(err) {
			zlog.Errorf("update repository：%s not exist", repoEntry.Name)
			return &httputil.ResponseJson{
					Code: constant.ClientError,
					Msg:  fmt.Sprintf("creation failed, repository：%s doesn't exist.", repoEntry.Name)},
				http.StatusBadRequest
		}
		zlog.Errorf("update repository %s failed in check existence ,%v", repoEntry.Name, err)
		return httputil.GetDefaultServerFailureResponseJson(),
			http.StatusInternalServerError
	}
	index, err := LoadRepoIndex(safeRepoEntryToRepoEntry(repoEntry))
	if err != nil {
		zlog.Errorf("load repository index failed in update, %v", err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "unable to connect to the remote server, please try another url",
			},
			http.StatusBadRequest
	}

	// create corresponding secret and configmap
	responseJson, status := c.updateRepoSecretAndConfigmap(repoEntry, repository)
	if status == http.StatusOK {
		cachedData.SetChartCache(repoEntry.Name, index)
	}
	return responseJson, status
}

/*
CreateRepo create repo based on user input
repoEntry contains user custom name, and corresponding CRD's name is created in lower case of that name
in order to fulfill k8s metadata requirement
so keep in mind in lowercase conversion and use of strings.toLower()
*/
func (c *helmClient) CreateRepo(repoEntry *helm.SafeRepoEntry) (*httputil.ResponseJson, int) {
	// check & validate
	if valid, err := k8sutil.ResourceMetadataRegexValid(strings.ToLower(repoEntry.Name)); !valid || err != nil {
		zlog.Errorf("create repository name check failed, name %s , %v", repoEntry.Name, err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg: "must consist of lower case alphanumeric characters, '-' or '.'," +
					" and must start and end with an alphanumeric character"},
			http.StatusBadRequest
	}
	listCustomRepo, err := c.listCustomRepo()
	if err != nil {
		zlog.Errorf("create repo %s failed in list existed repo cr ,%v", repoEntry.Name, err)
		return httputil.GetDefaultServerFailureResponseJson(),
			http.StatusInternalServerError
	}
	for _, customRepo := range listCustomRepo {
		if customRepo.Spec.URL == repoEntry.URL || customRepo.Spec.DisplayName == repoEntry.Name {
			zlog.Errorf("creation failed, repo：%s already exist.", repoEntry.Name)
			return &httputil.ResponseJson{
					Code: constant.ClientError,
					Msg:  "creation failed, name or url already exist."},
				http.StatusBadRequest
		}
	}

	index, err := LoadRepoIndex(safeRepoEntryToRepoEntry(repoEntry))
	if err != nil {
		zlog.Errorf("load repo index failed, %v", err)
		return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  "unable to find index.yaml, please provide correct ChartMuseum project url",
			},
			http.StatusBadRequest
	}
	// create corresponding secret and configmap
	responseJson, status := c.createRepoSecretAndConfigmap(repoEntry)
	if status == http.StatusCreated {
		cachedData.SetChartCache(repoEntry.Name, index)
	}

	return responseJson, status
}

func (c *helmClient) updateRepoSecretAndConfigmap(repoEntry *helm.SafeRepoEntry,
	repository *helm.HelmChartRepository) (*httputil.ResponseJson, int) {
	_, err := c.updateRepoCRBasicAuth(repository, repoEntry.Name, repoEntry.Username, repoEntry.Password)
	if err != nil {
		zlog.Errorf("update auth secret failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	_, err = c.updateRepoCRTLS(repository, repoEntry.Name, repoEntry.KeyFile, repoEntry.CertFile)
	if err != nil {
		zlog.Errorf("update tls secret failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	_, err = c.updateRepoCRCA(repository, repoEntry.Name, repoEntry.CAFile)
	if err != nil {
		zlog.Errorf("update ca config map failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	updatedRepoCR, err := c.updateRepoCR(repository, repoEntry)
	if err != nil {
		zlog.Errorf("update repo cr failed %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "repository updated",
		Data: updatedRepoCR,
	}, http.StatusOK
}

func (c *helmClient) createRepoSecretAndConfigmap(repoEntry *helm.SafeRepoEntry) (*httputil.ResponseJson, int) {
	authRef, err := c.createRepoCRBasicAuth(repoEntry.Name, repoEntry.Username, repoEntry.Password)
	if err != nil {
		zlog.Errorf("create auth secret failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	tlsRef, err := c.createRepoCRTLS(repoEntry.Name, repoEntry.KeyFile, repoEntry.CertFile)
	if err != nil {
		zlog.Errorf("create tls secret failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	caRef, err := c.createRepoCRCA(repoEntry.Name, repoEntry.CAFile)
	if err != nil {
		zlog.Errorf("create ca config map failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	createdRepoCR, err := c.createRepoCR(repoEntry, authRef, tlsRef, caRef)
	if err != nil {
		zlog.Errorf("create repo cr failed %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "repository created",
		Data: createdRepoCR,
	}, http.StatusCreated
}

// safeRepoEntryToRepoEntry convert from custom SafeRepoEntry struct to helm repo.Entry struct
func safeRepoEntryToRepoEntry(safeRepoEntry *helm.SafeRepoEntry) *repo.Entry {
	return &repo.Entry{
		Name:                  safeRepoEntry.Name,
		URL:                   safeRepoEntry.URL,
		Username:              string(safeRepoEntry.Username),
		Password:              string(safeRepoEntry.Password),
		CertFile:              safeRepoEntry.CertFile,
		KeyFile:               safeRepoEntry.KeyFile,
		CAFile:                safeRepoEntry.CAFile,
		InsecureSkipTLSverify: safeRepoEntry.InsecureSkipTLSVerify,
		PassCredentialsAll:    safeRepoEntry.PassCredentialsAll,
	}
}

func (c *helmClient) DeleteRepo(repoName string) (*httputil.ResponseJson, int) {
	if responseJson, status, err := c.isRepoModificationAllowed(repoName); err != nil {
		return responseJson, status
	}
	repoName = strings.ToLower(repoName)
	exists, err := k8sutil.CrExists(c.dynamicClient, repoName, "", repoCRDGVR)
	if err != nil {
		zlog.Warnf("delete repo %s failed in check existence", repoName)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	if !exists {
		zlog.Warnf("deletion failed, repo：%s doesn't exists.", repoName)
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("deletion failed, repo：%s doesn't exists.", repoName),
		}, http.StatusBadRequest
	}
	err = c.deleteRepoCR(repoName)
	if err != nil {
		zlog.Errorf("delete repo cr failed %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	cachedData.DeleteChartCache(repoName)
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  fmt.Sprintf("%s deleted", repoName),
		Data: nil,
	}, http.StatusOK
}

func (c *helmClient) GetRepo(repoName string) (*httputil.ResponseJson, int) {
	if repoName == "" {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide repo name for search",
			Data: nil,
		}, http.StatusBadRequest
	}
	customRepo, err := c.getCustomRepoByName(repoName)
	if err != nil {
		if k8sErrors.IsNotFound(err) {
			return &httputil.ResponseJson{
				Code: constant.ResourceNotFound,
				Msg:  fmt.Sprintf("repository %s not found", repoName),
			}, http.StatusNotFound
		}
		zlog.Errorf("get custom helm repo %s failed, %v", repoName, err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: customRepo,
	}, http.StatusOK
}

func (c *helmClient) ListRepo(query *param.Query, repo string) (*httputil.ResponseJson, int) {
	repoList := make([]*helm.RepoResponse, 0)
	customRepoList, err := c.listCustomRepo()
	if err != nil {
		zlog.Errorf("list custom helm repo failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	for i := range customRepoList {
		if repo == "" || strings.Contains(customRepoList[i].Spec.DisplayName, strings.ToLower(repo)) {
			repoList = append(repoList, &helm.RepoResponse{
				Name: customRepoList[i].Spec.DisplayName,
				URL:  customRepoList[i].Spec.URL,
			})
		}
	}
	sortRepoResponse(query, repoList)
	length := len(repoList)
	start, end, totalPages := query.Pagination.GetPaginationResult(length)

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: &helm.ListResponse{
			TotalItems:  length,
			Items:       ConvertToListInterface(repoList[start:end]),
			CurrentPage: query.Pagination.CurrentPage,
			TotalPages:  totalPages,
		},
	}, http.StatusOK
}

func sortRepoResponse(query *param.Query, repoList []*helm.RepoResponse) {
	switch query.SortBy {
	case param.Name:
		sortRepoByName(repoList, query.Ascending)
	default:
		sortRepoByName(repoList, true)
	}
}

// Custom sort function
func sortRepoByName(repoList []*helm.RepoResponse, ascending bool) {
	if ascending {
		sort.Slice(repoList, func(i, j int) bool {
			return customCompare(repoList[i].Name, repoList[j].Name, ascendingOrder)
		})
	} else {
		sort.Slice(repoList, func(i, j int) bool {
			return customCompare(repoList[i].Name, repoList[j].Name, descendingOrder)
		})
	}
}

func (c *helmClient) listCustomRepo() ([]helm.HelmChartRepository, error) {
	crList, err := c.dynamicClient.Resource(repoCRDGVR).List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var customRepoList []helm.HelmChartRepository
	for _, unstructuredCR := range crList.Items {
		var customRepo helm.HelmChartRepository
		err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredCR.Object, &customRepo)
		if err != nil {
			zlog.Debugf("error converting to HelmChartRepository: %s", unstructuredCR.GetName())
			return nil, err
		} else {
			customRepoList = append(customRepoList, customRepo)
		}
	}
	return customRepoList, nil
}

func (c *helmClient) getCustomRepoByName(repoName string) (*helm.HelmChartRepository, error) {
	metaName := strings.ToLower(repoName)
	cr, err := c.dynamicClient.Resource(repoCRDGVR).Get(context.Background(), metaName, metav1.GetOptions{})
	if err != nil {
		if k8sErrors.IsNotFound(err) {
			zlog.Warnf("can't find repo cr based on name %s", metaName)
			return nil, err
		}
		return nil, err
	}
	var customRepo *helm.HelmChartRepository
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(cr.Object, &customRepo)
	if err != nil {
		zlog.Debugf("error converting repo cr to HelmChartRepository: %s", cr.GetName())
		return nil, err
	}
	return customRepo, nil
}

func (c *helmClient) repoCRtoRepoEntry(repoCR *helm.HelmChartRepository) (*repo.Entry, error) {
	repoEntry := &repo.Entry{
		Name:                  repoCR.Spec.DisplayName,
		URL:                   repoCR.Spec.URL,
		Username:              "",
		Password:              "",
		CertFile:              "",
		KeyFile:               "",
		CAFile:                "",
		InsecureSkipTLSverify: repoCR.Spec.InsecureSkipTLSVerify,
		PassCredentialsAll:    repoCR.Spec.PassCredentialsAll,
	}

	if repoCR.Spec.BasicAuth.Name != "" {
		basicAuth, err := k8sutil.GetSecret(c.clientset, repoCR.Spec.BasicAuth.Name,
			constant.MarketplaceServiceDefaultNamespace)
		if err != nil || basicAuth == nil {
			return nil, err
		}
		repoEntry.Username = string(basicAuth.Data[mapKeyUsername])
		repoEntry.Password = string(basicAuth.Data[mapKeyPassword])
	}
	if repoCR.Spec.TLS.Name != "" {
		tls, err := k8sutil.GetSecret(c.clientset, repoCR.Spec.TLS.Name, constant.MarketplaceServiceDefaultNamespace)
		if err != nil || tls == nil {
			return nil, err
		}
		repoEntry.CertFile = string(tls.Data[mapKeyTLSCrt])
		repoEntry.KeyFile = string(tls.Data[mapKeyTLSKey])
	}
	if repoCR.Spec.CA.Name != "" {
		ca, err := k8sutil.GetConfigMap(c.clientset, repoCR.Spec.CA.Name, constant.MarketplaceServiceDefaultNamespace)
		if err != nil && ca == nil {
			return nil, err
		}
		repoEntry.CAFile = ca.Data[mapKeyCAKey]
	}
	return repoEntry, nil
}

type errRepo struct {
	Name string
	Err  error
}

// SyncAllRepos sync all repository index file, save all chart cache
func (c *helmClient) SyncAllRepos() (*httputil.ResponseJson, int) {
	errRepoList := make(chan errRepo, 10)
	repoList, err := c.listCustomRepo()
	if err != nil {
		zlog.Errorf("syncAll repos failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	var wg sync.WaitGroup
	for _, repository := range repoList {
		tmpRepository := repository
		wg.Add(1)
		go func(repoCopy *helm.HelmChartRepository) {
			defer wg.Done()
			c.asyncUpdateRepository(repoCopy, errRepoList)
		}(&tmpRepository)
	}
	go func() {
		wg.Wait()
		close(errRepoList)
		zlog.Infof("sync all repository complete")
		for errRepo := range errRepoList {
			zlog.Errorf("repo %s sync failed: %v", errRepo.Name, errRepo.Err)
		}
	}()
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "synchronizing repositories",
	}, http.StatusAccepted
}

func (c *helmClient) asyncUpdateRepository(repository *helm.HelmChartRepository, errRepoList chan<- errRepo) {
	zlog.Infof("syncing %s", repository.Spec.DisplayName)
	repoEntry, err := c.repoCRtoRepoEntry(repository)
	if err != nil {
		zlog.Errorf("error convert repo cr %s to repo entry", repository.Name)
		errRepoList <- errRepo{
			Name: repository.Name,
			Err:  err,
		}
		return
	}
	err = cachedData.UpdateChartCache(repoEntry)
	if err != nil {
		zlog.Errorf("error update repo %s chart cache", repository.Name)
		errRepoList <- errRepo{
			Name: repository.Name,
			Err:  err,
		}
		return
	}
	err = c.patchRepoModificationTime(repository.Name)
	if err != nil {
		zlog.Errorf("error updating modification time for repository %s", repository.Spec.DisplayName)
		errRepoList <- errRepo{
			Name: repository.Name,
			Err:  err,
		}
	}
}

// SyncRepo sync repository index file with repo name, save chart cache
func (c *helmClient) SyncRepo(repoDisplayName string) (*httputil.ResponseJson, int) {
	if repoDisplayName == "" {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide repo name for sync",
			Data: nil,
		}, http.StatusBadRequest
	}
	repository, err := c.getCustomRepoByName(repoDisplayName)
	if err != nil {
		if k8sErrors.IsNotFound(err) {
			zlog.Warnf("repository %s not found", repoDisplayName)
			return &httputil.ResponseJson{
				Code: constant.ClientError,
				Msg:  fmt.Sprintf(" repository %s not found", repoDisplayName),
			}, http.StatusBadRequest
		}
		zlog.Errorf("error get repository %s, %v", repoDisplayName, err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	status, _ := repoAsyncTaskMap.Load(repoDisplayName)
	switch status {
	case syncInProgressMsg:
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf(" repository %s syncronizing in progress", repoDisplayName),
		}, http.StatusBadRequest
	case syncFailedMsg, syncCompleteMsg, nil:
		repoAsyncTaskMap.Store(repoDisplayName, syncInProgressMsg)
	default:
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	go func() {
		errRepoList := make(chan errRepo, 1)
		c.asyncUpdateRepository(repository, errRepoList)
		close(errRepoList)
		zlog.Infof("sync repo %s complete", repoDisplayName)
		repoAsyncTaskMap.Store(repoDisplayName, syncCompleteMsg)
		for errRepo := range errRepoList {
			zlog.Errorf("repo %s sync failed: %v", errRepo.Name, errRepo.Err)
			repoAsyncTaskMap.Store(repoDisplayName, syncFailedMsg)
		}
	}()
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  fmt.Sprintf("syncronizing repository %s", repoDisplayName),
	}, http.StatusAccepted
}

func (c *helmClient) GetRepoSyncStatus(repoName string) (*httputil.ResponseJson, int) {
	status, exist := repoAsyncTaskMap.Load(repoName)

	if !exist {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("%s update task not found", repoName),
		}, http.StatusBadRequest
	} else {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  status.(string),
		}, http.StatusOK
	}
}

func ResetSyncStatusForTest() {
	repoAsyncTaskMap = sync.Map{}
}

func (c *helmClient) patchRepoModificationTime(name string) error {
	utcTime := time.Now().UTC().Format(time.RFC3339)
	annotationKey := fmt.Sprintf("%s/%s", constant.MarketplaceServiceDefaultOrgName, modificationTimestamp)
	patchData, err := json.Marshal(map[string]interface{}{
		"metadata": map[string]interface{}{
			"annotations": map[string]string{
				annotationKey: utcTime,
			},
		},
	})
	if err != nil {
		return err
	}
	_, err = c.dynamicClient.Resource(repoCRDGVR).Namespace("").Patch(context.TODO(), name,
		types.MergePatchType, patchData, metav1.PatchOptions{})
	if err != nil {
		return err
	}
	return nil
}

func (c *helmClient) updateRepoCR(customHelmRepository *helm.HelmChartRepository,
	repoEntry *helm.SafeRepoEntry) (*unstructured.Unstructured, error) {
	customHelmRepository.Spec.URL = repoEntry.URL
	repoUnstructured, err := k8sutil.StructToUnstructured(customHelmRepository)
	if err != nil {
		zlog.Errorf("convert repo cr to unstructured error, %v", err)
		return nil, err
	}
	repoUnstructured, err = c.dynamicClient.Resource(repoCRDGVR).Update(context.TODO(), repoUnstructured,
		metav1.UpdateOptions{})
	if err != nil {
		return nil, err
	}
	zlog.Infof("repo CR: %s updated", customHelmRepository.Name)
	return repoUnstructured, nil
}

func (c *helmClient) createRepoCR(repoEntry *helm.SafeRepoEntry, authRef *v1.Secret,
	tlsRef *v1.Secret, caRef *v1.ConfigMap) (*unstructured.Unstructured, error) {
	// construct struct and pass it to dynamic client for cr creation
	customHelmRepository := &helm.HelmChartRepository{
		TypeMeta:   getRepoCRTypeMeta(),
		ObjectMeta: getRepoCRObjectMeta(strings.ToLower(repoEntry.Name)),
		Spec: helm.HelmChartRepositorySpec{
			DisplayName:           repoEntry.Name,
			URL:                   repoEntry.URL,
			InsecureSkipTLSVerify: repoEntry.InsecureSkipTLSVerify,
			PassCredentialsAll:    repoEntry.PassCredentialsAll,
		},
	}
	if authRef != nil {
		customHelmRepository.Spec.BasicAuth.Name = authRef.Name
	}
	if tlsRef != nil {
		customHelmRepository.Spec.TLS.Name = tlsRef.Name
	}
	if caRef != nil {
		customHelmRepository.Spec.CA.Name = caRef.Name
	}
	repoUnstructured, err := k8sutil.StructToUnstructured(customHelmRepository)
	if err != nil {
		zlog.Errorf("convert repo cr to unstructured error, %v", err)
		return nil, err
	}
	repoUnstructured, err = c.dynamicClient.Resource(repoCRDGVR).Create(context.TODO(),
		repoUnstructured, metav1.CreateOptions{})

	if err != nil {
		return nil, err
	}
	zlog.Infof("repo CR: %s created", customHelmRepository.Name)
	return repoUnstructured, nil
}

func (c *helmClient) deleteRepoCR(repoCRName string) error {
	// request last at most 10 seconds
	ctx, cancel := context.WithTimeout(context.Background(), requestLastTime*time.Second)
	defer cancel()

	chartRepository, err := c.getCustomRepoByName(repoCRName)
	if err != nil {
		return err
	}
	if chartRepository.Spec.BasicAuth.Name != "" {
		err = k8sutil.DeleteSecret(c.clientset, chartRepository.Spec.BasicAuth.Name,
			constant.MarketplaceServiceDefaultNamespace)
		if err != nil {
			zlog.Errorf("delete repo cr basic auth failed %v", err)
		}
	}
	if chartRepository.Spec.TLS.Name != "" {
		err = k8sutil.DeleteSecret(c.clientset, chartRepository.Spec.TLS.Name,
			constant.MarketplaceServiceDefaultNamespace)
		if err != nil {
			zlog.Errorf("delete repo cr tls failed %v", err)
		}
	}
	if chartRepository.Spec.CA.Name != "" {
		err = k8sutil.DeleteConfigMap(c.clientset, chartRepository.Spec.CA.Name,
			constant.MarketplaceServiceDefaultNamespace)
		if err != nil {
			zlog.Errorf("delete repo cr ca failed %v", err)
		}
	}

	err = c.dynamicClient.Resource(repoCRDGVR).Delete(ctx, repoCRName, metav1.DeleteOptions{})
	if err != nil {
		return err
	}
	zlog.Infof("repo cr: %s deleted", repoCRName)
	return nil
}

func (c *helmClient) createRepoCRBasicAuth(repoName string, username, password []byte) (*v1.Secret, error) {
	if len(username) == 0 || len(password) == 0 {
		zlog.Infof("skip creating repo auth %s, username or pasword is empty", repoName)
		return nil, nil
	}
	secretName := repoName + repoCRSecretBasicAuth
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: constant.MarketplaceServiceDefaultNamespace,
		},
		Data: map[string][]byte{
			mapKeyUsername: username,
			mapKeyPassword: password,
		},
	}
	var err error
	secret, err = k8sutil.CreateSecret(c.clientset, secret)
	if err != nil {
		if k8sErrors.IsAlreadyExists(err) {
			return k8sutil.UpdateSecret(c.clientset, secret)
		}
		return nil, err
	}
	for _, value := range secret.Data {
		util.ClearByte(value)
	}
	return secret, nil
}

func (c *helmClient) updateRepoCRBasicAuth(repo *helm.HelmChartRepository, repoName string,
	username, password []byte) (*v1.Secret, error) {

	if len(repo.Spec.BasicAuth.Name) == 0 && len(username) == 0 {
		zlog.Infof("skip updating repo auth %s, username or pasword is empty", repoName)
		return nil, nil
	} else if len(repo.Spec.BasicAuth.Name) == 0 && len(username) != 0 {
		return c.createRepoCRBasicAuth(repoName, username, password)
	} else if len(repo.Spec.BasicAuth.Name) != 0 && len(username) == 0 {
		return nil, k8sutil.DeleteSecret(c.clientset, repo.Spec.BasicAuth.Name,
			constant.MarketplaceServiceDefaultNamespace)
	} else if len(repo.Spec.BasicAuth.Name) != 0 && len(username) != 0 {
		secret := &v1.Secret{}
		secret.Data = map[string][]byte{
			mapKeyUsername: username,
			mapKeyPassword: password,
		}
		return c.updateAndClearSecretData(secret)
	} else {
		zlog.Errorf("something goes wrong in update repo cr basic auth, repo cr %v, username %s ", repo, username)
		return nil, fmt.Errorf("something goes wrong in update repo cr basic auth, repo cr %v,"+
			" username %s ", repo, username)
	}
}

func (c *helmClient) updateAndClearSecretData(secret *v1.Secret) (*v1.Secret, error) {
	_, err := k8sutil.UpdateSecret(c.clientset, secret)
	if err != nil {
		return nil, err
	}
	for _, value := range secret.Data {
		util.ClearByte(value)
	}
	return secret, nil
}

func (c *helmClient) updateRepoCRTLS(repo *helm.HelmChartRepository, repoName string,
	certFile, keyFile string) (*v1.Secret, error) {
	// make sure tls secret certFile and keyFile is always valid
	// either two of them are valid, or no-exist
	if len(repo.Spec.TLS.Name) == 0 && (len(certFile) == 0 && len(keyFile) == 0) {
		zlog.Infof("skip updating repo tls %s, username or pasword is empty", repoName)
		return nil, nil
	} else if len(repo.Spec.TLS.Name) == 0 && (len(certFile) != 0 || len(keyFile) != 0) {
		return c.createRepoCRTLS(repoName, certFile, keyFile)
	} else if len(repo.Spec.TLS.Name) != 0 && (len(certFile) == 0 && len(keyFile) == 0) {
		return nil, k8sutil.DeleteSecret(c.clientset, repo.Spec.TLS.Name,
			constant.MarketplaceServiceDefaultNamespace)
	} else if len(repo.Spec.TLS.Name) != 0 && (len(certFile) != 0 || len(keyFile) != 0) {
		secret := &v1.Secret{}
		secret.StringData = map[string]string{
			mapKeyTLSCrt: certFile,
			mapKeyTLSKey: keyFile,
		}
		return k8sutil.UpdateSecret(c.clientset, secret)
	} else {
		zlog.Errorf("something goes wrong in update repo cr tls, repo cr %v,certFile %s ,keyFile %s",
			repo, certFile, keyFile)
		return nil, fmt.Errorf("something goes wrong in update repo cr tls, repo cr %v,certFile %s"+
			" ,keyFile %s", repo, certFile, keyFile)
	}
}

func (c *helmClient) createRepoCRTLS(repoName, certFile, keyFile string) (*v1.Secret, error) {
	if len(certFile) == 0 || len(keyFile) == 0 {
		zlog.Infof("create repo TLS %s, certFile or keyFile is empty", repoName)
		return nil, nil
	}
	secretName := repoName + repoCRSecretTLS
	namespace := constant.MarketplaceServiceDefaultNamespace
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
		StringData: map[string]string{
			mapKeyTLSCrt: certFile,
			mapKeyTLSKey: keyFile,
		},
	}

	_, err := k8sutil.CreateSecret(c.clientset, secret)
	if err != nil {
		if k8sErrors.IsAlreadyExists(err) {
			return k8sutil.UpdateSecret(c.clientset, secret)
		}
		return nil, err
	}
	return secret, nil
}

func (c *helmClient) updateRepoCRCA(repo *helm.HelmChartRepository, repoName, caFile string) (*v1.ConfigMap, error) {
	if len(repo.Spec.CA.Name) == 0 && len(caFile) == 0 {
		zlog.Infof("skip updating repo ca %s, username or pasword is empty", repoName)
		return nil, nil
	} else if len(repo.Spec.CA.Name) == 0 && len(caFile) != 0 {
		return c.createRepoCRCA(repoName, caFile)
	} else if len(repo.Spec.CA.Name) != 0 && len(caFile) == 0 {
		return nil, k8sutil.DeleteConfigMap(c.clientset, repo.Spec.CA.Name,
			constant.MarketplaceServiceDefaultNamespace)
	} else if len(repo.Spec.CA.Name) != 0 && len(caFile) != 0 {
		configmap := &v1.ConfigMap{}
		configmap.Data = map[string]string{
			mapKeyCAKey: caFile,
		}
		return k8sutil.UpdateConfigMap(c.clientset, configmap)
	} else {
		zlog.Errorf("something goes wrong in update repo cr ca, repo cr %v, caFile %s ", repo, caFile)
		return nil, fmt.Errorf("something goes wrong in update repo cr ca, repo cr %v,"+
			" caFile %s ", repo, caFile)
	}
}

func (c *helmClient) createRepoCRCA(repoName, caFile string) (*v1.ConfigMap, error) {
	if len(caFile) == 0 {
		zlog.Infof("create repo CA %s, caFile is empty", repoName)
		return nil, nil
	}

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      repoName + repoCRConfigMapCA,
			Namespace: constant.MarketplaceServiceDefaultNamespace,
		},
		Data: map[string]string{
			mapKeyCAKey: caFile,
		},
	}
	_, err := k8sutil.CreateConfigMap(c.clientset, configMap)
	if err != nil {
		if k8sErrors.IsAlreadyExists(err) {
			return k8sutil.UpdateConfigMap(c.clientset, configMap)
		}
		return nil, err
	}
	return configMap, nil
}

func getRepoCRObjectMeta(repoName string) metav1.ObjectMeta {
	objectMeta := metav1.ObjectMeta{
		Name: repoName,
		Annotations: map[string]string{
			fmt.Sprintf("%s/%s", constant.MarketplaceServiceDefaultOrgName,
				modificationTimestamp): time.Now().UTC().Format(time.RFC3339),
		},
	}
	return objectMeta
}

func getRepoCRTypeMeta() metav1.TypeMeta {
	typeMeta := metav1.TypeMeta{
		Kind:       constant.CRDRepoKind,
		APIVersion: fmt.Sprintf("%s/%s", constant.CRDRepoGroup, constant.CRDRepoVersion),
	}
	return typeMeta
}
