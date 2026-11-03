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
	"bytes"
	"encoding/json"
	goErrors "errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/repo"
	"k8s.io/apimachinery/pkg/api/errors"

	"marketplace-service/pkg/constant"
	marketplaceErrors "marketplace-service/pkg/errors"
	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/utils/k8sutil"
	"marketplace-service/pkg/utils/util"
	"marketplace-service/pkg/zlog"
)

var (
	officialHarborBearerToken = ""
	splitTokenPartNum         = 2
	ascendingOrder            = 1
	descendingOrder           = -1
)

const (
	valuesFileName             = "values.yaml"
	chartFileName              = "chart.yaml"
	templateDirPrefix          = "templates/"
	harborDefaultFormDataField = "chart"

	localHarborSecret       = "local-harbor-core"
	localHarborSecretMapKey = "HARBOR_ADMIN_PASSWORD"
	localHarborUser         = "admin"

	fileTypeDetail   = "detail"
	fileTypeTemplate = "template"
	fileTypeAll      = ""

	token           = "token"
	authorization   = "Authorization"
	wwwAuthenticate = "Www-Authenticate"
	realmParam      = "realm"
	serviceParam    = "service"
	scopeParam      = "scope"

	maxRetries = 3

	// chart tgz max file size 2MB
	maxFileSize = 2 * 1024 * 1024
	// read 64 kb size buffer every time
	bufferSize   = 64 * 1024
	megabyteSize = 1024 * 1024

	orderZero  = 0
	orderOne   = 1
	orderTwo   = 2
	orderThree = 3
)

func validUploadChart(formFile multipart.File,
	fileHeader *multipart.FileHeader) *httputil.ResponseJson {
	valid, err := util.CheckFileSize(formFile, bufferSize, maxFileSize)
	if err != nil || !valid {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("file excced %.2f megabytes", float64(maxFileSize)/(megabyteSize)),
		}
	}

	_, err = formFile.Seek(0, io.SeekStart)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("file cannot be sought"),
		}
	}

	valid, err = util.ValidateTarGzFile(formFile, fileHeader, maxFileSize)
	if err != nil || !valid {
		return &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  fmt.Sprintf("danger file"),
		}
	}
	return nil
}

func (c *helmClient) uploadChartCheck(formFile multipart.File,
	fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int) {
	response := validUploadChart(formFile, fileHeader)
	if response != nil {
		return response, http.StatusBadRequest
	}

	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil || config == nil {
		zlog.Errorf("error getting marketplace-service config: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting marketplace-service config"}, http.StatusInternalServerError
	}
	numbers, err := c.countChartNumbers()
	if err != nil {
		zlog.Errorf("error counting chart numbers: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error counting chart numbers"}, http.StatusInternalServerError
	}
	chartLimit, err := strconv.ParseInt(config.ChartLimit, constant.BaseTen, constant.SixtyFourBits)
	if err != nil {
		zlog.Errorf("error parsing chart limit: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error parsing chart limit"}, http.StatusInternalServerError
	}
	if numbers > chartLimit {
		zlog.Errorf("exceeding chart upload limit %v", chartLimit)
		return &httputil.ResponseJson{
			Code: constant.ExceedChartUploadLimit,
			Msg:  fmt.Sprintf("reach chart upload limit %v", chartLimit)}, http.StatusBadRequest
	}
	return nil, constant.Success
}

// UploadChart upload chart into built-in repository
func (c *helmClient) UploadChart(formFile multipart.File,
	fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int) {

	checkingResponse, httpStatus := c.uploadChartCheck(formFile, fileHeader)
	if checkingResponse != nil {
		return checkingResponse, httpStatus
	}
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil {
		zlog.Errorf("error getting marketplace service config: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting marketplace-service config"}, http.StatusInternalServerError
	}
	request, response := c.buildUploadRequest(config, formFile, fileHeader)
	if response != nil {
		return response, http.StatusInternalServerError
	}
	password, exists, err := c.getLocalHarborPassword()
	if err != nil || !exists {
		zlog.Errorf("error getting local harbor password: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting local harbor password"}, http.StatusInternalServerError
	}
	request.SetBasicAuth(localHarborUser, string(password))
	util.ClearByte(password)

	_, repoEntry, err := c.getRepoEntryAndChartVersions("local")
	if err != nil {
		zlog.Errorf("error getting repo entry & chartVersions %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting repo entry & chartVersions"}, http.StatusInternalServerError
	}
	resp, body, err := getHttpRequestResponse(request, repoEntry)
	if err != nil {
		zlog.Errorf("error sending HTTP request: %v", err)
		return httputil.GetResponseJson(constant.ServerError,
			"error reaching registry", nil), http.StatusInternalServerError
	}
	zlog.Infof("status %v, status code %v, body %v", resp.Status, resp.StatusCode, string(body))

	if util.IsHTTPClientError(resp.StatusCode) {
		return getUploadErrorGeneralResponse(), http.StatusBadRequest
	}
	if resp.StatusCode != http.StatusCreated {
		zlog.Errorf(fmt.Sprintf("failed to upload file, status %s, status code %v, %v", resp.Status, resp.StatusCode,
			string(body)))
		return httputil.GetResponseJson(constant.ServerError, "failed to upload file", nil),
			http.StatusInternalServerError
	}
	_, _ = c.SyncRepo(config.LocalHarborDisplayName)
	return httputil.GetResponseJson(constant.FileCreated, "upload success", nil), http.StatusCreated
}

func getBuiltInHarborChartURI(builtInHarborHost, builtInHarborProject string) string {
	zlog.Infof("%s/api%s/charts", builtInHarborHost, builtInHarborProject)
	return fmt.Sprintf("%s/api%s/charts", builtInHarborHost, builtInHarborProject)
}

func (c *helmClient) buildUploadRequest(config *helm.MarketplaceServiceConfig, formFile multipart.File,
	fileHeader *multipart.FileHeader) (*http.Request, *httputil.ResponseJson) {
	_, err := (formFile).Seek(0, io.SeekStart)
	if err != nil {
		zlog.Errorf("reset file pointer to zero failed: %v", err)
		return nil, httputil.GetDefaultServerFailureResponseJson()
	}
	request, err := http.NewRequest(http.MethodPost,
		getBuiltInHarborChartURI(config.LocalHarborHost, config.LocalHarborProject), nil)
	if err != nil {
		zlog.Errorf("error creating post request: %v", err)
		return nil, httputil.GetDefaultServerFailureResponseJson()
	}

	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)
	defer writer.Close()
	part, err := writer.CreateFormFile(harborDefaultFormDataField, fileHeader.Filename)
	if err != nil {
		zlog.Errorf("error creating form file, %v", err)
		return nil, getUploadErrorGeneralResponse()
	}
	_, err = io.Copy(part, formFile)
	if err != nil {
		zlog.Errorf("error copy from file to writer part, %v", err)
		return nil, getUploadErrorGeneralResponse()
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.Body = io.NopCloser(&requestBody)
	return request, nil
}

func (c *helmClient) getLocalHarborPassword() ([]byte, bool, error) {
	secret, err := k8sutil.GetSecret(c.clientset, localHarborSecret, constant.MarketplaceServiceDefaultNamespace)
	if err != nil {
		return nil, false, err
	}
	password, exists := secret.Data[localHarborSecretMapKey]
	return password, exists, nil
}

func (c *helmClient) DeleteChart(chartName string) (*httputil.ResponseJson, int) {
	return c.DeleteChartVersion(chartName, "")
}

func (c *helmClient) buildDeleteRequest(config *helm.MarketplaceServiceConfig, chartName,
	version string) (*http.Request,
	*httputil.ResponseJson, int) {
	if chartName == "" {
		return nil, &httputil.ResponseJson{
			Code: constant.ClientError,
			Msg:  "please provide chart name for delete",
			Data: nil,
		}, http.StatusBadRequest
	}
	var deleteURL string
	if version != "" {
		deleteURL = fmt.Sprintf("%s/%s/%s",
			getBuiltInHarborChartURI(config.LocalHarborHost, config.LocalHarborProject), chartName, version)
	} else {
		deleteURL = fmt.Sprintf("%s/%s",
			getBuiltInHarborChartURI(config.LocalHarborHost, config.LocalHarborProject), chartName)
	}

	request, err := http.NewRequest(http.MethodDelete, deleteURL, nil)
	if err != nil {
		zlog.Errorf("error creating delete request: %v", err)
		return nil, httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	password, exists, err := c.getLocalHarborPassword()
	if err != nil || !exists {
		zlog.Errorf("error getting local harbor password: %v", err)
		return nil, getUploadErrorGeneralResponse(), http.StatusInternalServerError
	}
	request.SetBasicAuth(localHarborUser, string(password))
	util.ClearByte(password)
	return request, nil, http.StatusOK
}

// DeleteChartVersion delete chart all versions in built-in repository
func (c *helmClient) DeleteChartVersion(chartName, version string) (*httputil.ResponseJson, int) {
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil || config == nil {
		zlog.Errorf("error getting marketplace-service config: %v", err)
		return httputil.GetResponseJson(constant.ServerError, "error getting marketplace-service config", nil),
			http.StatusInternalServerError
	}
	req, response, status := c.buildDeleteRequest(config, chartName, version)
	if response != nil {
		return response, status
	}
	_, repoEntry, err := c.getRepoEntryAndChartVersions("local")
	if err != nil {
		zlog.Errorf("error getting repo entry & chartVersions %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting repo entry & chartVersions"}, http.StatusInternalServerError
	}
	resp, body, err := getHttpRequestResponse(req, repoEntry)
	if err != nil {
		zlog.Errorf("error sending HTTP request: %v", err)
		return httputil.GetResponseJson(constant.ServerError,
			"error sending HTTP request", nil), http.StatusInternalServerError
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusNotFound {
			zlog.Infof(string(body))
			return httputil.GetResponseJson(constant.ClientError,
					fmt.Sprintf("failed to delete %s %s, no such file", chartName, version), nil),
				http.StatusInternalServerError
		}
		zlog.Errorf(fmt.Sprintf("failed to delete file, %v", string(body)))
		return httputil.GetResponseJson(constant.ClientError, "failed to delete file", nil),
			http.StatusInternalServerError
	}
	_, _ = c.SyncRepo(config.LocalHarborDisplayName)
	return httputil.GetResponseJson(constant.Success, "delete success", nil), http.StatusOK
}

func getHttpRequestResponse(req *http.Request, repoEntry *repo.Entry) (*http.Response, []byte, error) {
	client, err := httputil.GetHarborHttpClientByRepo(repoEntry)
	if err != nil {
		zlog.Errorf("error create http client %v", err)
		return nil, nil, err
	}
	if client == nil {
		return nil, nil, goErrors.New("error create http client")
	}

	resp, err := client.Do(req)
	if err != nil {
		zlog.Errorf("error sending HTTP request: %v", err)
		return nil, nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		zlog.Errorf("error reading response body: %v", err)
		return nil, nil, err
	}
	return resp, body, nil
}

func getUploadErrorGeneralResponse() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: constant.ClientError,
		Msg:  "please check the file you upload, only .tgz and .tar.gz are permitted",
		Data: nil,
	}
}

// GetLatestCharts list charts from repositories
// return all latest charts from repositories listed in repos
// return all latest charts if repos is empty
func (c *helmClient) GetLatestCharts(searchParam *helm.ChartSearchParam) (*httputil.ResponseJson, int) {
	chartList, err := c.getLatestChartsByRepos(searchParam.Repositories)
	if err != nil {
		zlog.Errorf("list chart failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil {
		zlog.Errorf("get marketplace service config failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	if !util.ContainsAll(config.MarketplaceScenes, searchParam.Scene) {
		zlog.Warnf("check if scene is defined by marketplace service, marketplace config %v,"+
			" search param %v", config.MarketplaceScenes, searchParam.Scene)
		return httputil.GetDefaultClientFailureResponseJson(), http.StatusBadRequest
	}
	filteredList := filterChartVersionSlice(searchParam, chartList)
	sortChartVersionSlice(searchParam.Query, filteredList)

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: pageResult(ConvertToListInterface(filteredList), searchParam.Query.Pagination),
	}, http.StatusOK
}

func (c *helmClient) getChartWithOfficialTag(tag string,
	config *helm.MarketplaceServiceConfig) (*helm.ChartVersionResponseWithTag,
	error) {
	var officialTagsResponse *helm.OfficialTagsResponse
	for i := 0; i < maxRetries; i++ {
		officialTagsResponse = c.handleTagRequest(tag, config)
		if officialTagsResponse != nil {
			break
		}
	}
	if officialTagsResponse == nil {
		return nil, &marketplaceErrors.HttpResponseNotOKError{
			Message: "official tags request no response with max retries",
		}
	}
	taggedCharts := make([]string, len(officialTagsResponse.TaggedCharts))
	for index, base64String := range officialTagsResponse.TaggedCharts {
		resultString, err := decodeURLSafeBase64String(base64String)
		if err != nil {
			return nil, err
		}
		taggedCharts[index] = resultString
	}
	chartWithTag, err := c.filterOfficialChartWithTag(taggedCharts, config.OfficialHarborDisplayName)
	if err != nil {
		zlog.Errorf("filter official chart with tag %s , error %v", tag, err)
		return nil, err
	}
	return &helm.ChartVersionResponseWithTag{
		Tag:                   tag,
		ChartVersionResponses: chartWithTag,
	}, nil
}

func (c *helmClient) handleTagRequest(tag string, config *helm.MarketplaceServiceConfig) *helm.OfficialTagsResponse {
	officialTagsResponse, err := c.getOfficialTagWithToken(config.OfficialHarborHost+config.OfficialHarborTagsURL, tag)
	if err != nil {
		zlog.Errorf("request tag respond error %v", err)
		var expiredError *marketplaceErrors.TokenExpiredError
		if goErrors.As(err, &expiredError) {
			if expiredError == nil {
				return nil
			}
			err = c.updateOfficialHarborBearerToken(expiredError.Response)
			if err != nil {
				zlog.Errorf("update Official Harbor Bearer Token failed, %v", err)
			}
		}
	}
	return officialTagsResponse
}

func (c *helmClient) GetChartsWithOfficialTags(tags []string, query *param.Query) (*httputil.ResponseJson, int) {
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil {
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	result := make([]*helm.ChartVersionResponseWithTag, 0)
	for _, tag := range tags {
		chartVersionResponseWithTag, err := c.getChartWithOfficialTag(tag, config)
		if err != nil {
			zlog.Errorln(err)
		} else {
			result = append(result, chartVersionResponseWithTag)
		}
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: result,
	}, http.StatusOK
}

func (c *helmClient) filterOfficialChartWithTag(taggedCharts []string,
	officialHarborDisplayName string) ([]*helm.ChartVersionResponse, error) {
	resultChartList := make([]*helm.ChartVersionResponse, 0)
	repository, err := c.getCustomRepoByName(officialHarborDisplayName)
	if err != nil {
		return resultChartList, err
	}
	c.asyncUpdateRepository(repository, nil)
	chartList, err := c.getLatestChartsByRepos([]string{officialHarborDisplayName})
	if err != nil {
		zlog.Errorf("list chart failed, %v", err)
		return nil, err
	}
	for _, taggedChart := range taggedCharts {
		for _, chartInfo := range chartList {
			if chartInfo.Metadata.Name == taggedChart {
				resultChartList = append(resultChartList, chartInfo)
			}
		}
	}
	return resultChartList, nil
}

func decodeURLSafeBase64String(input string) (string, error) {
	input = strings.ReplaceAll(input, "-", "+")
	input = strings.ReplaceAll(input, "_", "/")
	input = strings.ReplaceAll(input, ".", "=")
	return util.DecodeBase64String(input)
}
func (c *helmClient) updateOfficialHarborBearerToken(tagRequestResponse *http.Response) error {
	authenticate := tagRequestResponse.Header[wwwAuthenticate]
	if len(authenticate) < 1 {
		return &marketplaceErrors.FieldNotFoundError{Message: "field authenticate not found"}
	}
	realm, service, scope := getBearTokenHeadParameter(authenticate[0])
	OfficialHarborBearerTokenRequestURL := getOfficialHarborBearerTokenRequestURL(realm, service, scope)
	request, err := http.NewRequest(http.MethodGet, OfficialHarborBearerTokenRequestURL, nil)
	if err != nil {
		return err
	}
	_, repoEntry, err := c.getRepoEntryAndChartVersions("openfuyao")
	if err != nil {
		zlog.Errorf("error getting repo entry & chartVersions %v", err)
		return err
	}
	response, body, err := getHttpRequestResponse(request, repoEntry)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return &marketplaceErrors.HttpResponseNotOKError{Message: string(body)}
	}

	if !json.Valid(body) {
		return &marketplaceErrors.InvalidJsonHttpBodyError{Message: string(body)}
	}
	var data map[string]interface{}
	err = json.Unmarshal(body, &data)
	if err != nil {
		return err
	}
	tokenField, ok := data[token]
	if !ok {
		return getTokenNotFoundError(body)
	}
	officialHarborBearerToken, ok = tokenField.(string)
	if !ok {
		return getTokenNotFoundError(body)
	}
	return nil
}

func getTokenNotFoundError(body []byte) error {
	return &marketplaceErrors.FieldNotFoundError{
		Message: fmt.Sprintf("failed to find field %s in response %s", token, string(body)),
		Field:   token,
	}
}

func getBearTokenHeadParameter(input string) (string, string, string) {
	input = strings.TrimPrefix(input, "Bearer ")
	parts := strings.Split(input, ",")
	var realmString, serviceString, scopeString string

	for _, part := range parts {
		kv := strings.SplitN(part, "=", splitTokenPartNum)
		if len(kv) == splitTokenPartNum {
			key := strings.TrimSpace(kv[0])
			value := strings.Trim(kv[1], `"`)
			switch key {
			case realmParam:
				realmString = value
			case serviceParam:
				serviceString = value
			case scopeParam:
				scopeString = value
			default:
			}
		}
	}
	return realmString, serviceString, scopeString
}

func (c *helmClient) getOfficialTagWithToken(harborHost, tagToBeListed string) (*helm.OfficialTagsResponse, error) {
	officialHarborTagRequestURL := getOfficialHarborTagsRequestURL(harborHost, tagToBeListed)
	request, err := http.NewRequest(http.MethodGet, officialHarborTagRequestURL, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set(authorization, officialHarborBearerToken)
	_, repoEntry, err := c.getRepoEntryAndChartVersions("openfuyao")
	if err != nil {
		zlog.Errorf("error getting repo entry & chartVersions %v", err)
		return nil, &marketplaceErrors.HttpResponseNotOKError{Message: "error getting repo entry & chartVersions"}
	}
	response, body, err := getHttpRequestResponse(request, repoEntry)
	if err != nil {
		return nil, err
	}
	if response.StatusCode == http.StatusUnauthorized {
		return nil, &marketplaceErrors.TokenExpiredError{
			Message:  string(body),
			Response: response,
		}
	} else if response.StatusCode != http.StatusOK {
		return nil, &marketplaceErrors.HttpResponseNotOKError{Message: string(body)}
	}

	var tagResponse helm.OfficialTagsResponse
	err = json.Unmarshal(body, &tagResponse)
	if err != nil {
		return nil, err
	}
	return &tagResponse, nil
}

func getOfficialHarborTagsRequestURL(harborHost, tagToBeListed string) string {
	return fmt.Sprintf("%s/%s%s", harborHost, tagToBeListed, "/tags/list")
}

func getOfficialHarborBearerTokenRequestURL(realm, service, scope string) string {
	return fmt.Sprintf("%s?service=%s&scope=%s", realm, service, scope)
}

func filterChartVersionSlice(searchParam *helm.ChartSearchParam,
	chartList []*helm.ChartVersionResponse) []*helm.ChartVersionResponse {
	// Helper function to filter chartList based on a condition
	filter := func(chartList []*helm.ChartVersionResponse,
		condition func(*helm.ChartVersionResponse) bool) []*helm.ChartVersionResponse {
		tmpChartList := make([]*helm.ChartVersionResponse, 0)
		for _, chartInfo := range chartList {
			zlog.Infof("check chart %s, version: %s, containing keywords: %v", chartInfo.Metadata.Name,
				chartInfo.Metadata.Version, chartInfo.Metadata.Keywords)
			if condition(chartInfo) {
				tmpChartList = append(tmpChartList, chartInfo)
			}
		}
		return tmpChartList
	}

	// Filter by Chart name
	if searchParam.Chart != "" {
		chartList = filter(chartList, func(chartInfo *helm.ChartVersionResponse) bool {
			return strings.Contains(strings.ToLower(chartInfo.Metadata.Name), strings.ToLower(searchParam.Chart))
		})
	}

	if len(searchParam.Types) > 0 {
		chartList = filter(chartList, func(chartInfo *helm.ChartVersionResponse) bool {
			for _, typeString := range searchParam.Types {
				zlog.Infof("check appType: %s", typeString)
				if typeString == "application" {
					if util.NotContains(chartInfo.Metadata.Keywords, constant.FuyaoExtensionKeyword) {
						return true
					}
				} else if typeString != "" {
					if util.Contains(chartInfo.Metadata.Keywords, typeString) {
						return true
					}
				} else {
					continue
				}
			}
			return false
		})
	}

	// Filter by Scene
	if len(searchParam.Scene) > 0 {
		chartList = filter(chartList, func(chartInfo *helm.ChartVersionResponse) bool {
			return util.ContainsOne(chartInfo.Metadata.Keywords, searchParam.Scene)
		})
	}
	return chartList
}

func sortChartVersionSlice(query *param.Query, chartList []*helm.ChartVersionResponse) {
	switch query.SortBy {
	case param.Time:
		sortChartByTime(chartList, query.Ascending)
	case param.Name:
		sortChartByName(chartList, query.Ascending)
	default:
		sortChartByName(chartList, true)
	}
}

// Custom sort function
func sortChartByTime(chartList []*helm.ChartVersionResponse, ascending bool) {
	if ascending {
		sort.Slice(chartList, func(i, j int) bool {
			return chartList[i].Metadata.Created.Before(chartList[j].Metadata.Created)
		})
	} else {
		sort.Slice(chartList, func(i, j int) bool {
			return chartList[i].Metadata.Created.After(chartList[j].Metadata.Created)
		})
	}
}

// Custom sort function
func sortChartByName(chartList []*helm.ChartVersionResponse, ascending bool) {
	if ascending {
		sort.Slice(chartList, func(i, j int) bool {
			return customCompare(chartList[i].Metadata.Name, chartList[j].Metadata.Name, ascendingOrder)
		})
	} else {
		sort.Slice(chartList, func(i, j int) bool {
			return customCompare(chartList[i].Metadata.Name, chartList[j].Metadata.Name, descendingOrder)
		})
	}
}

func customCompare(a, b string, sortOrder int) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		ra, rb := rune(a[i]), rune(b[i])

		pa, pb := getPriority(ra), getPriority(rb)

		if pa != pb {
			return (pa < pb) == (sortOrder == 1)
		}
		// 如果优先级相同，按字符本身比较
		if ra != rb {
			if unicode.ToLower(ra) == unicode.ToLower(rb) {
				return (ra < rb) == (sortOrder == 1)
			}
			return (unicode.ToLower(ra) < unicode.ToLower(rb)) == (sortOrder == 1)
		}
	}

	// 如果遍历到一方结束，则按长度比较
	return (len(a) < len(b)) == (sortOrder == 1)
}

func getPriority(r rune) int {
	switch {
	case unicode.IsSymbol(r) || unicode.IsPunct(r):
		return orderZero
	case unicode.IsLetter(r):
		return orderOne
	case unicode.IsDigit(r):
		return orderTwo
	default:
		return orderThree
	}
}

func pageResult(result []any, pagination *param.Pagination) *helm.ListResponse {
	total := len(result)
	if pagination == nil {
		pagination = param.GetNoPagination()
	}
	start, end, totalPages := pagination.GetPaginationResult(total)
	return &helm.ListResponse{
		TotalItems:  total,
		Items:       result[start:end],
		CurrentPage: pagination.CurrentPage,
		TotalPages:  totalPages,
	}
}

func (c *helmClient) getLatestChartsByRepos(repositories []string) ([]*helm.ChartVersionResponse, error) {
	chartList := make([]*helm.ChartVersionResponse, 0)
	repoCRList, err := c.listCustomRepo()
	if err != nil {
		zlog.Errorf("error list repositories %v", err)
		return nil, err
	}
	if len(repositories) == 0 {
		chartList = c.getLatestChartsFromAllRepositories(repoCRList)
	} else {
		selectedRepositories := filterSelectedRepositories(repositories, repoCRList)
		for _, selectedRepository := range selectedRepositories {
			charts := c.getLatestChartsFromRepository(selectedRepository)
			chartList = append(chartList, charts...)
		}
	}
	return chartList, nil
}

func filterSelectedRepositories(repositories []string,
	repoCRList []helm.HelmChartRepository) []helm.HelmChartRepository {
	result := make([]helm.HelmChartRepository, 0)
	for _, repoName := range repositories {
		for _, repoCR := range repoCRList {
			if repoName == repoCR.Spec.DisplayName {
				result = append(result, repoCR)
			}
		}
	}
	return result
}

func (c *helmClient) getLatestChartsFromAllRepositories(
	repoList []helm.HelmChartRepository) []*helm.ChartVersionResponse {

	result := make([]*helm.ChartVersionResponse, 0)
	for _, repository := range repoList {
		charts := c.getLatestChartsFromRepository(repository)
		result = append(result, charts...)
	}
	return result
}

func (c *helmClient) getLatestChartsFromRepository(repository helm.HelmChartRepository) []*helm.ChartVersionResponse {
	result := make([]*helm.ChartVersionResponse, 0)
	chartList, exist := cachedData.GetChartCacheByRepo(repository.Spec.DisplayName)
	if !exist {
		zlog.Errorf("repo cr %s exist, but cache is missing", repository.Spec.DisplayName)
	}
	zlog.Debugf("getLatestChartsByRepos repository is: %s", repository.Spec.DisplayName)
	// go through every chart, get the latest version order by time
	// the latest version contains all the keywords of this chart
	// repo.ChartVersions has too many fields, only return fields we are interested
	for _, chartVersions := range chartList {
		sort.Slice(chartVersions, func(i, j int) bool {
			return chartVersions[i].Created.After(chartVersions[j].Created)
		})

		availIdx := 0
		if availIdx == chartVersions.Len() {
			continue
		}
		chartVersions[availIdx].Keywords = ExtractUniqueKeywords(chartVersions)

		latestChart := &helm.ChartVersionResponse{
			Metadata: chartVersions[availIdx],
			Repo:     repository.Spec.DisplayName,
			RepoUrl:  repository.Spec.URL,
		}
		result = append(result, latestChart)
	}
	return result
}

// GetChartVersions  get all chart versions from repository in repoName
func (c *helmClient) GetChartVersions(repoName, chartName string) (*httputil.ResponseJson, int) {
	return c.GetChartVersion(repoName, chartName, "")
}

func needShow(entry *repo.ChartVersion, version string) bool {
	if version != "" && entry.Version != version {
		return false
	}
	return true
}

// GetChartVersion  get chart with specified version from repository in repoName
func (c *helmClient) GetChartVersion(repoName, chartName, version string) (*httputil.ResponseJson, int) {
	repository, err := c.getCustomRepoByName(repoName)
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.Errorf("repository %s not found", repoName)
			return &httputil.ResponseJson{
				Code: constant.ResourceNotFound,
				Msg:  "repository not found",
			}, http.StatusNotFound
		}
		zlog.Errorf("error get repository %s, %v", repoName, err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}
	indexEntries, exist := cachedData.GetChartCacheByRepo(repository.Spec.DisplayName)

	if !exist {
		zlog.Errorf("repo cr %s exist, but cache is missing", repository.Spec.DisplayName)
		return &httputil.ResponseJson{
			Code: constant.ResourceNotFound,
			Msg:  "please sync your repository",
		}, http.StatusNotFound
	}
	chartList := make([]helm.ChartVersionResponse, 0)
	for _, entry := range indexEntries[chartName] {
		if needShow(entry, version) {
			chartInfo := helm.ChartVersionResponse{
				Metadata: entry, Repo: repository.Spec.DisplayName, RepoUrl: repository.Spec.URL,
			}
			chartList = append(chartList, chartInfo)
		}
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: chartList,
	}, http.StatusOK
}

func (c *helmClient) getRepoEntryAndChartVersions(repoName string) (map[string]repo.ChartVersions,
	*repo.Entry, error) {
	indexEntries, exist := cachedData.GetChartCacheByRepo(repoName)
	if !exist {
		return nil, nil, fmt.Errorf("%s is no longer in the helm repository, please add repository first", repoName)
	}
	repository, err := c.getCustomRepoByName(repoName)
	if err != nil {
		if errors.IsNotFound(err) {
			zlog.Errorf("repository %s not found", repoName)
			return nil, nil, err
		}
		return nil, nil, err
	}
	repoEntry, err := c.repoCRtoRepoEntry(repository)
	if err != nil {
		return nil, nil, err
	}
	return indexEntries, repoEntry, nil
}

// GetChartBytesByVersion get chart from repoName repository with chartName and chartVersion
func (c *helmClient) GetChartBytesByVersion(repoName, chartName, chartVersion string) (*bytes.Buffer, error) {
	chartVersions, repoEntry, err := c.getRepoEntryAndChartVersions(repoName)
	if err != nil {
		zlog.Errorf("error get repo entry & chartVersions %v", err)
		return nil, err
	}
	for _, entry := range chartVersions[chartName] {
		if entry.Version == chartVersion {
			chartInfo, err := LoadChartBytes(entry.URLs[0], repoEntry)
			if err != nil {
				return nil, err
			} else {
				return chartInfo, nil
			}
		}
	}
	return nil, fmt.Errorf("no chart information found: %s-%s-%s", repoName, chartName, chartVersion)
}

// GetChartByVersion get chart from repoName repository with chartName and chartVersion
func (c *helmClient) getChartByVersion(repoName, chartName, chartVersion string) (*chart.Chart, error) {
	chartVersions, repoEntry, err := c.getRepoEntryAndChartVersions(repoName)
	if err != nil {
		zlog.Errorf("error get repo entry & chartVersions %v", err)
		return nil, err
	}
	for _, entry := range chartVersions[chartName] {
		if entry.Version == chartVersion {
			chartInfo, err := LoadChart(entry.URLs[0], repoEntry)
			if err != nil {
				return nil, err
			} else {
				return chartInfo, nil
			}
		}
	}
	return nil, fmt.Errorf("no chart information found: %s-%s-%s", repoName, chartName, chartVersion)
}

func (c *helmClient) GetChartFiles(repoName, chartName, version, fileType string) (*httputil.ResponseJson, int) {
	restfulResponse := httputil.GetDefaultSuccessResponseJson()
	chartByVersion, err := c.getChartByVersion(repoName, chartName, version)
	if err != nil {
		restfulResponse.Code = constant.ServerError
		restfulResponse.Msg =
			fmt.Sprintf("not found chart %s with version %s in repo %s, please check input or sync repository",
				chartName, version, repoName)
		return restfulResponse, http.StatusInternalServerError
	}

	switch fileType {
	case fileTypeDetail:
		restfulResponse.Data = c.getChartDetail(chartByVersion)
	case fileTypeTemplate:
		restfulResponse.Data = c.getChartTemplates(chartByVersion)
	case fileTypeAll:
		files := make(map[string]string)
		for _, file := range chartByVersion.Raw {
			files[file.Name] = string(file.Data)
		}
		restfulResponse.Data = files
	default:
		restfulResponse.Code = constant.ClientError
		restfulResponse.Msg = "invalid file type"
		return restfulResponse, http.StatusBadRequest
	}
	return restfulResponse, http.StatusOK
}

var readmes = map[string]bool{
	"readme":     true,
	"readme.txt": true,
	"readme.md":  true,
}

func (c *helmClient) getChartDetail(chart *chart.Chart) *helm.ChartDetailResponse {
	var detail = &helm.ChartDetailResponse{}
	for _, file := range chart.Raw {
		if readmes[strings.ToLower(file.Name)] {
			detail.Readme = string(file.Data)
		} else if strings.ToLower(file.Name) == valuesFileName {
			detail.Values = string(file.Data)
		} else if strings.ToLower(file.Name) == chartFileName {
			detail.Chart = string(file.Data)
		} else {
			zlog.Debugf("%s is not part of chart detail file list", file.Name)
		}
	}
	return detail
}

func (c *helmClient) getChartTemplates(chart *chart.Chart) []helm.ChartTemplateResponse {
	templates := make([]helm.ChartTemplateResponse, 0)
	for _, file := range chart.Raw {
		if strings.HasPrefix(file.Name, templateDirPrefix) {
			template := helm.ChartTemplateResponse{
				Name: file.Name,
				Data: string(file.Data),
			}
			templates = append(templates, template)
		}
	}
	return templates
}

// Chart chart response from ChartMuseum
type Chart struct {
	Name          string `json:"name"`
	TotalVersions int    `json:"total_versions"`
}

// TotalResponse response data json for api: CountCharts
type TotalResponse struct {
	Total int64 `json:"total"`
	Limit int64 `json:"limit"`
}

func (c *helmClient) CountCharts() (*httputil.ResponseJson, int) {
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil || config == nil {
		zlog.Errorf("error getting marketplace-service config: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error getting marketplace-service config"}, http.StatusInternalServerError
	}
	chartLimit, err := strconv.ParseInt(config.ChartLimit, constant.BaseTen, constant.SixtyFourBits)
	if err != nil {
		zlog.Errorf("error parsing chart limit: %v", err)
		return &httputil.ResponseJson{
			Code: http.StatusInternalServerError,
			Msg:  "error parsing chart limit"}, http.StatusInternalServerError
	}
	numbers, err := c.countChartNumbers()
	if err != nil {
		zlog.Error(err)
		return httputil.GetResponseJson(constant.ServerError, "failed to count files", nil),
			http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: TotalResponse{
			Total: numbers,
			Limit: chartLimit,
		},
	}, http.StatusOK
}

func (c *helmClient) countChartNumbers() (int64, error) {
	config, err := util.GetMarketplaceServiceConfig(c.clientset)
	if err != nil {
		return 0, goErrors.New(fmt.Sprintf("error getting marketplace service config: %v", err))
	}
	countURL := getBuiltInHarborChartURI(config.LocalHarborHost, config.LocalHarborProject)

	request, err := http.NewRequest(http.MethodGet, countURL, nil)
	if err != nil {
		return 0, goErrors.New(fmt.Sprintf("error creating get request: %v", err))
	}
	password, exists, err := c.getLocalHarborPassword()
	if err != nil || !exists {
		return 0, goErrors.New(fmt.Sprintf("error getting local harbor password: %v", err))
	}
	request.SetBasicAuth(localHarborUser, string(password))
	util.ClearByte(password)
	_, repoEntry, err := c.getRepoEntryAndChartVersions("local")
	if err != nil {
		zlog.Errorf("error getting repo entry & chartVersions %v", err)
		return 0, err
	}
	resp, body, err := getHttpRequestResponse(request, repoEntry)
	if err != nil {
		return 0, goErrors.New(fmt.Sprintf("error reaching registry: %v", err))
	}
	if resp.StatusCode != http.StatusOK {
		return 0, goErrors.New(fmt.Sprintf("failed to count files, %v", string(body)))
	}

	// Parse the JSON response
	var charts []Chart
	if err := json.Unmarshal(body, &charts); err != nil {
		zlog.Errorf(fmt.Sprintf("failed to count files, %v", string(body)))
	}

	var totalCharts int64 = 0
	for _, chartVersions := range charts {
		totalCharts += int64(chartVersions.TotalVersions)
	}
	zlog.Infof("Total number of chart packages: %d\n", totalCharts)
	return totalCharts, err
}
