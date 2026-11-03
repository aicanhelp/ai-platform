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
package marketplace

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"marketplace-service/pkg/constant"
)

// ==================== 创建仓库测试用例 ====================

var _ = Describe("应用市场-添加仓库", Label("CreateRepo"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		if tc != nil {
			tc.teardown()
		}
	})

	// 【应用市场-添加仓库-001】无需认证
	It("无需认证：添加仓库成功（无需用户验证）", func() {

		repoName := fmt.Sprintf("test-repo-api-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)
	})

	// 【应用市场-添加仓库-002】需要认证
	It("需要认证：添加仓库成功（需要用户名/密码认证）", func() {
		tc.mockHelmRepo.AddValidUser("testuser", "testpassword")

		repoName := fmt.Sprintf("test-repo-auth-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			Username:              []byte("testuser"),
			Password:              []byte("testpassword"),
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)
	})

	// 【应用市场-添加仓库-003】缺少认证凭据
	It("缺少认证凭据：添加需要认证的仓库但不提供凭据应该失败", func() {
		tc.mockHelmRepo.AddValidUser("testuser", "testpassword")

		repoName := fmt.Sprintf("test-repo-no-auth-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		req, err := tc.createRepoRequest(reqBody)
		Expect(err).NotTo(HaveOccurred())

		resp, respBody, err := tc.sendRequest(req)
		Expect(err).NotTo(HaveOccurred())
		verifyErrorResponse(resp, respBody, http.StatusBadRequest, "unable to find index.yaml")
	})

	// 【应用市场-添加仓库-004】index.yaml不存在/不可解析
	It("index.yaml 不存在：应该返回 HTTP 400", func() {
		tc.mockHelmRepo.SetReturn404(true)

		repoName := fmt.Sprintf("test-repo-404-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		req, err := tc.createRepoRequest(reqBody)
		Expect(err).NotTo(HaveOccurred())

		resp, respBody, err := tc.sendRequest(req)
		Expect(err).NotTo(HaveOccurred())
		verifyErrorResponse(resp, respBody, http.StatusBadRequest, "unable to find index.yaml")
	})

	It("index.yaml 不可解析：应该返回 HTTP 400", func() {
		// 显式声明本用例需要的 mock 行为
		tc.mockHelmRepo.SetReturn404(false)
		tc.mockHelmRepo.SetReturnInvalidYAML(true)

		repoName := fmt.Sprintf("test-repo-invalid-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		req, err := tc.createRepoRequest(reqBody)
		Expect(err).NotTo(HaveOccurred())

		resp, respBody, err := tc.sendRequest(req)
		Expect(err).NotTo(HaveOccurred())
		verifyErrorResponse(resp, respBody, http.StatusBadRequest, "unable to find index.yaml")
	})

	// 【应用市场-添加仓库-005】TLS认证
	It("TLS 认证：添加仓库成功", func() {

		repoName := fmt.Sprintf("test-repo-tls-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			CertFile:              "-----BEGIN CERTIFICATE-----\nMOCK_CERT_CONTENT\n-----END CERTIFICATE-----",
			KeyFile:               "-----BEGIN PRIVATE KEY-----\nMOCK_KEY_CONTENT\n-----END PRIVATE KEY-----",
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		Expect(repoResp.Data.Spec.TLS.Name).To(Equal(strings.ToLower(repoName) + "tls"))
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)
	})

	// 【应用市场-添加仓库-006】CA认证
	It("CA 认证：添加仓库成功", func() {

		repoName := fmt.Sprintf("test-repo-ca-%d", time.Now().Unix())
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			CAFile:                "-----BEGIN CERTIFICATE-----\nMOCK_CA_CERT_CONTENT\n-----END CERTIFICATE-----",
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		Expect(repoResp.Data.Spec.CA.Name).To(Equal(strings.ToLower(repoName) + "ca"))
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)
	})
})

// ==================== 删除仓库测试用例 ====================

var _ = Describe("应用市场-删除仓库", Label("DeleteRepo"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})

	// 【应用市场-删除仓库-001】登录
	It("登录用户删除远端仓库应该成功", func() {
		repoName := fmt.Sprintf("remote-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		deleteResp, deleteRespBody, err := tc.deleteRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())

		verifyDeleteSuccessResponse(deleteResp, deleteRespBody, repoNameLower)

		deleteResp2, deleteRespBody2, err := tc.deleteRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		expectedMsg := fmt.Sprintf("deletion failed, repo：%s doesn't exists.", repoNameLower)
		verifyErrorResponse(deleteResp2, deleteRespBody2, http.StatusBadRequest, expectedMsg)
	})

	// 【应用市场-删除仓库-002】repo不存在
	It("当repo不存在时，应该返回400错误", func() {
		repoName := fmt.Sprintf("non-existent-repo-%d", time.Now().Unix())

		deleteResp, deleteRespBody, err := tc.deleteRepo(repoName)
		Expect(err).NotTo(HaveOccurred())

		expectedMsg := fmt.Sprintf("deletion failed, repo：%s doesn't exists.", strings.ToLower(repoName))
		verifyErrorResponse(deleteResp, deleteRespBody, http.StatusBadRequest, expectedMsg)
	})

	// 【应用市场-删除仓库-003】删除仓库并清理缓存
	It("当所有操作成功时，应该返回200并清理cache", func() {
		repoName := fmt.Sprintf("test-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		repoResp, resp, err := tc.createRepo(RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		})
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusCreated))
		Expect(repoResp.Code).To(Equal(200))

		deleteResp, deleteRespBody, err := tc.deleteRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())

		verifyDeleteSuccessResponse(deleteResp, deleteRespBody, repoNameLower)

		deleteResp2, deleteRespBody2, err := tc.deleteRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		expectedMsg := fmt.Sprintf("deletion failed, repo：%s doesn't exists.", repoNameLower)
		verifyErrorResponse(deleteResp2, deleteRespBody2, http.StatusBadRequest, expectedMsg)
	})

	// 【应用市场-删除仓库-004】不允许删除内置仓库
	It("删除内置仓库 'openFuyao' 和 'local' 应该失败", func() {
		builtinRepos := []string{"openFuyao", "local", "OpenFuyao", "LOCAL"}

		for _, repoName := range builtinRepos {
			By(fmt.Sprintf("尝试删除内置仓库: %s", repoName), func() {
				deleteResp, deleteRespBody, err := tc.deleteRepo(repoName)
				Expect(err).NotTo(HaveOccurred())
				verifyErrorResponse(deleteResp, deleteRespBody, http.StatusBadRequest, "repo not allowed for modification")
			})
		}
	})
})

// ==================== 同步仓库测试用例 ====================
var _ = Describe("应用市场-同步仓库", Label("SyncRepo"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})
	// 【应用市场-同步仓库-001】同步远端仓库
	It("登录用户同步远端仓库应该成功", func() {
		repoName := fmt.Sprintf("remote-repo-sync-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		syncResp, syncRespBody, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		verifySyncSuccessResponse(syncResp, syncRespBody, repoNameLower)
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))

		err = tc.waitForSyncComplete(repoNameLower, 30*time.Second)
		Expect(err).NotTo(HaveOccurred(), "同步应该成功完成")

		statusResp, statusRespBody, err := tc.getRepoSyncStatus(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(statusResp.StatusCode).To(Equal(200))
		var statusRespData SyncStatusResponse
		err = json.Unmarshal(statusRespBody, &statusRespData)
		Expect(err).NotTo(HaveOccurred())
		Expect(statusRespData.Code).To(Equal(200))
		Expect(statusRespData.Msg).To(Equal("complete"), "同步状态应该为complete")
	})

	// 【应用市场-同步仓库-002】repo不存在
	It("同步不存在的仓库应该返回400错误", func() {
		nonExistentRepo := fmt.Sprintf("non-existent-repo-%d", time.Now().Unix())
		nonExistentRepoLower := strings.ToLower(nonExistentRepo)

		// 触发同步（直接对不存在的仓库发起同步请求）
		syncResp, syncRespBody, err := tc.syncRepo(nonExistentRepoLower)
		Expect(err).NotTo(HaveOccurred())

		Expect(syncResp.StatusCode).To(Equal(http.StatusBadRequest))

		var errorResp struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		err = json.Unmarshal(syncRespBody, &errorResp)
		Expect(err).NotTo(HaveOccurred())
		Expect(errorResp.Code).To(Equal(400))
		Expect(errorResp.Msg).To(ContainSubstring("repository"))
		Expect(errorResp.Msg).To(ContainSubstring("not found"))
	})

	// 【应用市场-同步仓库-003】首次同步触发成功
	It("首次触发同步应该成功，状态从in_progress变为complete", func() {
		repoName := fmt.Sprintf("first-sync-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		// 首次触发同步
		syncResp, syncRespBody, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		verifySyncSuccessResponse(syncResp, syncRespBody, repoNameLower)
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))

		// 等待同步完成
		err = tc.waitForSyncComplete(repoNameLower, 30*time.Second)
		Expect(err).NotTo(HaveOccurred(), "同步应该成功完成")

		// 验证同步状态为complete
		statusResp, statusRespBody, err := tc.getRepoSyncStatus(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(statusResp.StatusCode).To(Equal(200))
		var statusRespData SyncStatusResponse
		err = json.Unmarshal(statusRespBody, &statusRespData)
		Expect(err).NotTo(HaveOccurred())
		Expect(statusRespData.Code).To(Equal(200))
		Expect(statusRespData.Msg).To(Equal("complete"), "同步状态应该为complete")
	})

	// 【应用市场-同步仓库-004】同步进行中重复触发
	It("在同步进行中再次触发同步应该返回400错误", func() {
		repoName := fmt.Sprintf("duplicate-sync-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		// 为了稳定复现“同步进行中”状态：让 mock HelmRepo 的 /index.yaml 响应变慢一点
		tc.mockHelmRepo.SetResponseDelay(300 * time.Millisecond)
		defer tc.mockHelmRepo.SetResponseDelay(0)

		// 第一次触发同步
		syncResp1, syncRespBody1, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		verifySyncSuccessResponse(syncResp1, syncRespBody1, repoNameLower)
		Expect(syncResp1.StatusCode).To(Equal(http.StatusAccepted))

		// 立即再次触发同步（在同步完成前）
		time.Sleep(50 * time.Millisecond)
		syncResp2, syncRespBody2, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())

		Expect(syncResp2.StatusCode).To(Equal(http.StatusBadRequest))

		var errorResp struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		err = json.Unmarshal(syncRespBody2, &errorResp)
		Expect(err).NotTo(HaveOccurred())
		Expect(errorResp.Code).To(Equal(400))
		Expect(errorResp.Msg).To(ContainSubstring("syncronizing in progress"))
		Expect(errorResp.Msg).To(ContainSubstring(repoNameLower))
		Expect(tc.waitForSyncComplete(repoNameLower, 30*time.Second)).To(Succeed())
	})

	// 【应用市场-同步仓库-005】同步成功后的再次触发
	It("同步成功后再次触发同步", func() {
		repoName := fmt.Sprintf("retrigger-sync-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		repoResp, resp, err := tc.createRepo(RepoRequest{
			Name: repoName,
			URL:  tc.repoURL,
		})
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		By("第一次触发同步", func() {
			syncResp1, syncRespBody1, err := tc.syncRepo(repoNameLower)
			Expect(err).NotTo(HaveOccurred())
			verifySyncSuccessResponse(syncResp1, syncRespBody1, repoNameLower)
			Expect(syncResp1.StatusCode).To(Equal(http.StatusAccepted))
		})

		By("等待第一次同步完成", func() {
			err = tc.waitForSyncComplete(repoNameLower, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
		})

		By("确认第一次同步状态为 complete", func() {
			statusResp, statusRespBody, err := tc.getRepoSyncStatus(repoNameLower)
			Expect(err).NotTo(HaveOccurred())
			Expect(statusResp.StatusCode).To(Equal(http.StatusOK))
			var statusData SyncStatusResponse
			Expect(json.Unmarshal(statusRespBody, &statusData)).To(Succeed())
			Expect(statusData.Msg).To(Equal("complete"))
		})

		By("第二次触发同步：complete 状态仍允许触发", func() {
			syncResp2, syncRespBody2, err := tc.syncRepo(repoNameLower)
			Expect(err).NotTo(HaveOccurred())
			verifySyncSuccessResponse(syncResp2, syncRespBody2, repoNameLower)
			Expect(syncResp2.StatusCode).To(Equal(http.StatusAccepted))
		})

		By("等待第二次同步完成并确认状态为 complete", func() {
			err = tc.waitForSyncComplete(repoNameLower, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())

			_, statusRespBody2, err := tc.getRepoSyncStatus(repoNameLower)
			Expect(err).NotTo(HaveOccurred())
			var statusData2 SyncStatusResponse
			Expect(json.Unmarshal(statusRespBody2, &statusData2)).To(Succeed())
			Expect(statusData2.Msg).To(Equal("complete"))
		})
	})
})

// ==================== 列出仓库测试用例（GET /helm-repos） ====================

// 【应用市场-仓库列表-001】不带过滤参数，默认返回全部
var _ = Describe("应用市场-仓库列表", Label("ListRepo"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})

	// 【应用市场-仓库列表-001listCustomRepo 失败
	It("listCustomRepo 失败应返回500", func() {
		mdc, ok := tc.dynamicClient.(*MockDynamicClient)
		Expect(ok).To(BeTrue(), "dynamicClient should be *MockDynamicClient, got %T", tc.dynamicClient)
		mdc.SetListFail(true)
		defer mdc.SetListFail(false)

		resp, body, err := tc.listRepos("")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ServerError))
	})

	// 【应用市场-仓库列表-002】不带过滤参数，默认返回全部
	It("不带过滤参数，默认返回全部", func() {
		// 创建 2 个仓库
		repoNames := []string{
			fmt.Sprintf("list-all-a-%d", time.Now().Unix()),
			fmt.Sprintf("list-all-b-%d", time.Now().Unix()),
		}
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for _, name := range repoNames {
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)
			Expect(tc.seedRepoCR(name, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data struct {
				TotalItems  int `json:"totalItems"`
				CurrentPage int `json:"currentPage"`
				TotalPages  int `json:"totalPage"`
				Items       []struct {
					Name string `json:"name"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
		Expect(out.Data.TotalItems).To(BeNumerically(">=", 2))
		Expect(out.Data.CurrentPage).To(Equal(1))
		Expect(out.Data.TotalPages).To(BeNumerically(">=", 1))
		Expect(out.Data.Items).ToNot(BeEmpty())
	})

	// 【应用市场-仓库列表-003】带过滤参数 repo（子串匹配）
	It("带过滤参数 repo（子串匹配：repo 参数会被 ToLower，但 DisplayName 不会 ToLower）", func() {
		repoNameHit := fmt.Sprintf("my-repo-abc-%d", time.Now().Unix())
		repoNameMiss := fmt.Sprintf("other-repo-%d", time.Now().Unix())
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for _, name := range []string{repoNameHit, repoNameMiss} {
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)

			Expect(tc.seedRepoCR(name, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("repo=abc")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Name string `json:"name"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.Items).ToNot(BeEmpty())
		for _, it := range out.Data.Items {
			Expect(it.Name).To(ContainSubstring("abc"))
		}
	})

	// 【应用市场-仓库列表-004】排序（按 Name 升序/降序）
	It("按 Name 升序排序", func() {
		// 按规则准备：符号 < 字母 < 数字 < 其他
		input := []string{"repo-0", "arepo", "2repo", "repo-zh"}
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for _, name := range input {
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)

			repoName := name + fmt.Sprintf("-%d", time.Now().UnixNano())
			Expect(tc.seedRepoCR(repoName, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("sortBy=name&ascending=true")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Name string `json:"name"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(len(out.Data.Items)).To(BeNumerically(">=", 4))

		// 只检查前四个是否满足大类顺序
		first4 := out.Data.Items[:4]
		ranks := make([]int, 0, 4)
		for _, it := range first4 {
			ranks = append(ranks, firstRuneRank(it.Name))
		}
		Expect(ranks[0]).To(BeNumerically("<=", ranks[1]))
		Expect(ranks[1]).To(BeNumerically("<=", ranks[2]))
		Expect(ranks[2]).To(BeNumerically("<=", ranks[3]))
	})

	It("按 Name 降序排序", func() {
		input := []string{"repo-0", "arepo", "2repo", "repo-zh"}
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for _, name := range input {
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)

			repoName := name + fmt.Sprintf("-%d", time.Now().UnixNano())
			Expect(tc.seedRepoCR(repoName, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("sortBy=name&ascending=false")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Name string `json:"name"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(len(out.Data.Items)).To(BeNumerically(">=", 4))

		first4 := out.Data.Items[:4]
		ranks := make([]int, 0, 4)
		for _, it := range first4 {
			ranks = append(ranks, firstRuneRank(it.Name))
		}
		Expect(ranks[0]).To(BeNumerically(">=", ranks[1]))
		Expect(ranks[1]).To(BeNumerically(">=", ranks[2]))
		Expect(ranks[2]).To(BeNumerically(">=", ranks[3]))
	})

	// 【应用市场-仓库列表-005】分页功能
	It("指定 page 和 limit 时返回对应页的数据", func() {
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for i := 0; i < 5; i++ {
			name := fmt.Sprintf("page-repo-%d-%d", i, time.Now().UnixNano())
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)

			Expect(tc.seedRepoCR(name, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("page=2&limit=2&sortBy=name&ascending=true")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				TotalItems  int           `json:"totalItems"`
				CurrentPage int           `json:"currentPage"`
				TotalPages  int           `json:"totalPage"`
				Items       []interface{} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.TotalPages).To(Equal(3))
		Expect(out.Data.CurrentPage).To(Equal(2))
		Expect(len(out.Data.Items)).To(Equal(2))
	})

	It("当 page 超出范围时，重置到第一页并返回第一页数据", func() {
		var stops []func()
		defer func() {
			for _, stop := range stops {
				stop()
			}
		}()

		for i := 0; i < 3; i++ {
			name := fmt.Sprintf("oob-repo-%d-%d", i, time.Now().UnixNano())
			url, stop := startNewMockHelmRepo()
			stops = append(stops, stop)

			Expect(tc.seedRepoCR(name, url)).To(Succeed())
		}

		resp, body, err := tc.listRepos("page=3&limit=2&sortBy=name&ascending=true")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data struct {
				TotalItems  int           `json:"totalItems"`
				CurrentPage int           `json:"currentPage"`
				TotalPages  int           `json:"totalPage"`
				Items       []interface{} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
		Expect(out.Data.TotalItems).To(Equal(3))
		Expect(out.Data.TotalPages).To(Equal(2))
		Expect(out.Data.CurrentPage).To(Equal(3))
		Expect(len(out.Data.Items)).To(Equal(2))
	})

	It("当 totalItems 为 0 时，返回空 items 和 totalPages=1", func() {
		resp, body, err := tc.listRepos("page=1&limit=10")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				TotalItems  int           `json:"totalItems"`
				CurrentPage int           `json:"currentPage"`
				TotalPages  int           `json:"totalPage"`
				Items       []interface{} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.TotalItems).To(Equal(0))
		Expect(out.Data.TotalPages).To(Equal(1))
		Expect(out.Data.CurrentPage).To(Equal(1))
		Expect(out.Data.Items).To(BeEmpty())
	})

	// 【应用市场-仓库列表-006】详情信息展示
	It("同步仓库后获取指定Chart版本信息应返回200并包含metadata", func() {
		repoName := fmt.Sprintf("chart-detail-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}
		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		syncResp, _, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))
		err = tc.waitForSyncComplete(repoNameLower, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())

		chartName := "test-chart"
		version := "1.0.0"
		chartResp, chartRespBody, err := tc.getChartVersion(repoNameLower, chartName, version)
		Expect(err).NotTo(HaveOccurred())
		Expect(chartResp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data []struct {
				Metadata struct {
					Name    string `json:"name"`
					Version string `json:"version"`
				} `json:"metadata"`
				Repo    string `json:"repo"`
				RepoUrl string `json:"repoUrl"`
			} `json:"data"`
		}
		err = json.Unmarshal(chartRespBody, &out)
		Expect(err).NotTo(HaveOccurred())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
		Expect(out.Data).NotTo(BeEmpty())
		Expect(out.Data[0].Metadata.Name).To(Equal(chartName))
		Expect(out.Data[0].Metadata.Version).To(Equal(version))
		Expect(out.Data[0].Repo).To(Equal(repoNameLower))
		Expect(out.Data[0].RepoUrl).To(Equal(tc.repoURL))
	})
})

// ==================== 获取仓库详情测试用例 ====================

var _ = Describe("应用市场-获取仓库", Label("GetRepo"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})

	// 【应用市场-获取仓库-001】repoName 为空
	It("repoName 为空时路由会命中仓库列表接口并返回200", func() {
		req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet,
			fmt.Sprintf("%s%s/helm-repos/", tc.apiURL, apiBasePath), nil)
		Expect(err).NotTo(HaveOccurred())
		req.Header.Set("Authorization", "Bearer valid-token-12345")

		resp, body, err := tc.sendRequest(req)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data struct {
				Items []interface{} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
	})

	// 【应用市场-获取仓库-002】repo 不存在
	It("获取不存在的仓库应该返回404错误", func() {
		nonExistentRepo := fmt.Sprintf("non-existent-repo-%d", time.Now().Unix())
		nonExistentRepoLower := strings.ToLower(nonExistentRepo)

		resp, respBody, err := tc.getRepo(nonExistentRepoLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusNotFound))
		var errorResp struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		err = json.Unmarshal(respBody, &errorResp)
		Expect(err).NotTo(HaveOccurred())
		Expect(errorResp.Code).To(Equal(404))
		Expect(errorResp.Msg).To(ContainSubstring("repository"))
		Expect(errorResp.Msg).To(ContainSubstring("not found"))
		Expect(errorResp.Msg).To(ContainSubstring(nonExistentRepoLower))
	})

	// 【应用市场-获取仓库-003】正常获取
	It("获取存在的仓库详情应该返回200并包含完整信息", func() {
		repoName := fmt.Sprintf("test-repo-get-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)
		reqBody := RepoRequest{
			Name:                  repoName,
			URL:                   tc.repoURL,
			InsecureSkipTLSVerify: false,
			PassCredentialsAll:    false,
		}

		repoResp, resp, err := tc.createRepo(reqBody)
		Expect(err).NotTo(HaveOccurred())
		verifySuccessResponse(resp, repoResp, repoName, tc.repoURL)
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)
		getResp, getRespBody, err := tc.getRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(getResp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data struct {
				APIVersion string `json:"apiVersion"`
				Kind       string `json:"kind"`
				Metadata   struct {
					Name              string            `json:"name"`
					CreationTimestamp string            `json:"creationTimestamp"`
					Annotations       map[string]string `json:"annotations"`
				} `json:"metadata"`
				Spec struct {
					DisplayName string `json:"displayName"`
					URL         string `json:"url"`
					Description string `json:"description"`
				} `json:"spec"`
			} `json:"data"`
		}
		err = json.Unmarshal(getRespBody, &out)
		Expect(err).NotTo(HaveOccurred())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
		Expect(out.Data.APIVersion).To(Equal("console.openfuyao.com/v1beta1"))
		Expect(out.Data.Kind).To(Equal("HelmChartRepository"))
		Expect(out.Data.Metadata.Name).To(Equal(repoNameLower))
		Expect(out.Data.Spec.DisplayName).To(Equal(repoName))
		Expect(out.Data.Spec.URL).To(Equal(tc.repoURL))
		Expect(out.Data.Metadata.Annotations).NotTo(BeNil())
	})
})

// ==================== Chart 查询测试用例 ====================

var _ = Describe("应用市场-查询chart版本", Label("GetChartVersion"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})

	// 【应用市场-查询chart版本-001】repo 不存在
	It("repo 不存在时应返回404：repository not found", func() {
		repoName := "openfuyao"
		chartName := "aiaio-installer"
		version := "0.14.0"

		resp, body, err := tc.getChartVersion(repoName, chartName, version)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusNotFound))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusNotFound))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("repository not found"))
	})

	// 【应用市场-查询chart版本-002】cache 缺失（需要 sync）
	It("未同步仓库时应返回404：please sync your repository", func() {
		repoName := fmt.Sprintf("repo-cache-miss-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		url, stop := startNewMockHelmRepo()
		defer stop()

		Expect(tc.seedRepoCR(repoName, url)).To(Succeed())

		// 查询任意 chart 版本（因为 cache 不存在，应该返回 404）
		chartName := "test-chart"
		version := "1.0.0"
		getResp, getBody, err := tc.getChartVersion(repoNameLower, chartName, version)
		Expect(err).NotTo(HaveOccurred())
		Expect(getResp.StatusCode).To(Equal(http.StatusNotFound))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(getBody, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusNotFound))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("please sync"))
	})

	// 【应用市场-查询chart版本-003】指定版本过滤
	It("同步后 GetChartVersion 应返回指定版本", func() {
		repoName := fmt.Sprintf("chart-ver-repo-%d", time.Now().Unix())
		repoNameLower := strings.ToLower(repoName)

		// 1) 创建 repo
		repoResp, resp, err := tc.createRepo(RepoRequest{Name: repoName, URL: tc.repoURL})
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusCreated))
		tc.createdRepos = append(tc.createdRepos, repoResp.Data.Metadata.Name)

		// 2) 触发 sync，使 mock 写入 chartCache
		syncResp, _, err := tc.syncRepo(repoNameLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))
		Expect(tc.waitForSyncComplete(repoNameLower, 30*time.Second)).To(Succeed())

		chartName := "test-chart"
		version := "1.0.0"

		// 3) GetChartVersion：过滤指定版本
		verResp, verBody, err := tc.getChartVersion(repoNameLower, chartName, version)
		Expect(err).NotTo(HaveOccurred())
		Expect(verResp.StatusCode).To(Equal(http.StatusOK))

		var outOne struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data []struct {
				Metadata struct {
					Name    string `json:"name"`
					Version string `json:"version"`
				} `json:"metadata"`
				Repo string `json:"repo"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(verBody, &outOne)).To(Succeed())
		Expect(outOne.Code).To(Equal(200))
		Expect(outOne.Msg).To(Equal("success"))
		Expect(outOne.Data).NotTo(BeEmpty())
		Expect(outOne.Data[0].Metadata.Name).To(Equal(chartName))
		Expect(outOne.Data[0].Metadata.Version).To(Equal(version))
	})
})

// ==================== 获取最新 Chart 列表测试用例 ====================

var _ = Describe("应用市场-获取最新Chart列表", Label("GetLatestCharts"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
	})

	AfterEach(func() {
		tc.teardown()
	})

	seedReposAndCharts := func() (string, string, func()) {
		repoA := fmt.Sprintf("openFuyao-%d", time.Now().UnixNano())
		repoB := fmt.Sprintf("local-%d", time.Now().UnixNano())

		// 为 repoA 创建独立的 Mock Helm Repository
		mockHelmRepoA := NewMockHelmRepositoryServer(0)
		Expect(mockHelmRepoA.Start()).To(Succeed())

		// 为 repoB 创建独立的 Mock Helm Repository
		mockHelmRepoB := NewMockHelmRepositoryServer(0)
		Expect(mockHelmRepoB.Start()).To(Succeed())

		// 返回 cleanup 函数，用于在测试用例结束时清理
		cleanup := func() {
			_ = mockHelmRepoA.Stop()
			_ = mockHelmRepoB.Stop()
		}

		// 构造 repoA 的 index.yaml
		now := time.Now()
		indexYAMLA := fmt.Sprintf(`apiVersion: v1
entries:
  "-a":
  - version: 1.0.0
    name: "-a"
    description: Chart with symbol prefix
    keywords:
    - ai
    urls:
    - http://localhost/-a-1.0.0.tgz
    created: "%s"
  "A":
  - version: 1.0.0
    name: "A"
    description: Chart with uppercase letter
    keywords:
    - database
    urls:
    - http://localhost/A-1.0.0.tgz
    created: "%s"
  "a":
  - version: 1.0.0
    name: "a"
    description: Chart with lowercase letter
    keywords:
    - monitor
    urls:
    - http://localhost/a-1.0.0.tgz
    created: "%s"
  "1":
  - version: 1.0.0
    name: "1"
    description: Chart with digit prefix
    keywords:
    - middleware
    urls:
    - http://localhost/1-1.0.0.tgz
    created: "%s"
  "_":
  - version: 1.0.0
    name: "_"
    description: Chart with underscore
    keywords:
    - openfuyao-extension
    urls:
    - http://localhost/_-1.0.0.tgz
    created: "%s"
  "中文":
  - version: 1.0.0
    name: "中文"
    description: Chart with Chinese characters
    keywords:
    - artificial-intelligence
    urls:
    - http://localhost/中文-1.0.0.tgz
    created: "%s"
  "nginx":
  - version: 1.0.0
    name: "nginx"
    description: Nginx chart v1
    keywords:
    - middleware
    - ai
    urls:
    - http://localhost/nginx-1.0.0.tgz
    created: "%s"
  - version: 2.0.0
    name: "nginx"
    description: Nginx chart v2
    keywords:
    - middleware
    - database
    urls:
    - http://localhost/nginx-2.0.0.tgz
    created: "%s"
generated: "%s"
`, now.Add(-10*time.Minute).Format(time.RFC3339),
			now.Add(-9*time.Minute).Format(time.RFC3339),
			now.Add(-8*time.Minute).Format(time.RFC3339),
			now.Add(-7*time.Minute).Format(time.RFC3339),
			now.Add(-6*time.Minute).Format(time.RFC3339),
			now.Add(-5*time.Minute).Format(time.RFC3339),
			now.Add(-4*time.Minute).Format(time.RFC3339),
			now.Add(-1*time.Minute).Format(time.RFC3339),
			now.Format(time.RFC3339))

		// 构造 repoB 的 index.yaml
		indexYAMLB := fmt.Sprintf(`apiVersion: v1
entries:
  "ai-app":
  - version: 0.1.0
    name: "ai-app"
    description: AI application chart
    keywords:
    - ai
    urls:
    - http://localhost/ai-app-0.1.0.tgz
    created: "%s"
generated: "%s"
`, now.Add(-2*time.Minute).Format(time.RFC3339),
			now.Format(time.RFC3339))

		// 设置 index.yaml
		mockHelmRepoA.SetIndexYAML(indexYAMLA)
		mockHelmRepoB.SetIndexYAML(indexYAMLB)

		// 创建 repoA
		repoRespA, respA, err := tc.createRepo(RepoRequest{Name: repoA, URL: mockHelmRepoA.GetURL()})
		Expect(err).NotTo(HaveOccurred())
		Expect(respA.StatusCode).To(Equal(http.StatusCreated))
		tc.createdRepos = append(tc.createdRepos, repoRespA.Data.Metadata.Name)

		// 创建 repoB
		repoRespB, respB, err := tc.createRepo(RepoRequest{Name: repoB, URL: mockHelmRepoB.GetURL()})
		Expect(err).NotTo(HaveOccurred())
		Expect(respB.StatusCode).To(Equal(http.StatusCreated))
		tc.createdRepos = append(tc.createdRepos, repoRespB.Data.Metadata.Name)

		// 同步 repoA（触发真实的 chart 缓存解析）
		repoALower := strings.ToLower(repoA)
		syncRespA, _, err := tc.syncRepo(repoALower)
		Expect(err).NotTo(HaveOccurred())
		Expect(syncRespA.StatusCode).To(Equal(http.StatusAccepted))
		Expect(tc.waitForSyncComplete(repoALower, 30*time.Second)).To(Succeed())

		// 同步 repoB
		repoBLower := strings.ToLower(repoB)
		syncRespB, _, err := tc.syncRepo(repoBLower)
		Expect(err).NotTo(HaveOccurred())
		Expect(syncRespB.StatusCode).To(Equal(http.StatusAccepted))
		Expect(tc.waitForSyncComplete(repoBLower, 30*time.Second)).To(Succeed())

		return repoA, repoB, cleanup
	}

	// 【应用市场-获取列表-001】repos为空
	It("repos 为空时应返回分页结构并合并所有仓库", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
			Data struct {
				TotalItems  int `json:"totalItems"`
				CurrentPage int `json:"currentPage"`
				TotalPages  int `json:"totalPage"`
				Items       []struct {
					Metadata struct {
						Name string `json:"name"`
					} `json:"metadata"`
					Repo string `json:"repo"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(200))
		Expect(out.Msg).To(Equal("success"))
		Expect(out.Data.TotalItems).To(BeNumerically(">", 0))
		Expect(out.Data.CurrentPage).To(Equal(1))
		Expect(out.Data.Items).NotTo(BeEmpty())
	})

	// 【应用市场-获取列表-002】指定repos子集
	It("指定 repo 子集时只返回命中仓库的数据", func() {
		repoA, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts(fmt.Sprintf("page=1&limit=50&repo=%s", repoA))
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Repo string `json:"repo"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		for _, it := range out.Data.Items {
			Expect(it.Repo).To(Equal(repoA))
		}
	})

	// 【应用市场-获取列表-003】scene 校验失败
	It("scene 不在 MarketplaceScenes 中时应返回400", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()

		resp, _, err := tc.getLatestCharts("page=1&limit=50&scene=not-exist")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
	})

	// 【应用市场-获取列表-004】chart 名称模糊匹配
	It("chart 名称模糊匹配应生效（不区分大小写）", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50&chart=AI")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Name string `json:"name"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		for _, it := range out.Data.Items {
			Expect(strings.ToLower(it.Metadata.Name)).To(ContainSubstring("ai"))
		}
	})

	// 【应用市场-获取列表-005】Types=application
	It("appType=application 应过滤掉包含 fuyao-extension 的 chart", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50&appType=application")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Keywords []string `json:"keywords"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		for _, it := range out.Data.Items {
			Expect(it.Metadata.Keywords).NotTo(ContainElement("fuyao-extension"))
		}
	})

	// 【应用市场-获取列表-006】Types=middleware
	It("appType=middleware 应只返回 keywords 含 middleware 的 chart", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50&appType=middleware")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Keywords []string `json:"keywords"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.Items).NotTo(BeEmpty())
		for _, it := range out.Data.Items {
			Expect(it.Metadata.Keywords).To(ContainElement("middleware"))
		}
	})

	// 【应用市场-获取列表-007】Scene 过滤
	It("scene 过滤应返回 keywords 命中任一 scene 的 chart", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()

		resp, body, err := tc.getLatestCharts("page=1&limit=50&scene=artificial-intelligence&scene=monitor")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Keywords []string `json:"keywords"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		for _, it := range out.Data.Items {
			Expect(it.Metadata.Keywords).To(SatisfyAny(
				ContainElement("artificial-intelligence"),
				ContainElement("monitor"),
			))
		}
	})

	// 【应用市场-获取列表-008】按时间排序（sortType=time 默认 desc）
	It("sortType=time 应按 created 时间倒序", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50&sortType=time&order=desc")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Created time.Time `json:"created"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		if len(out.Data.Items) >= 2 {
			for i := 1; i < len(out.Data.Items); i++ {
				Expect(out.Data.Items[i-1].Metadata.Created.After(out.Data.Items[i].Metadata.Created) ||
					out.Data.Items[i-1].Metadata.Created.Equal(out.Data.Items[i].Metadata.Created)).To(BeTrue())
			}
		}
	})

	// 【应用市场-获取列表-009】按名称排序（customCompare 规则）
	It("默认按名称排序应符合优先级：符号<字母<数字<其它", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=1&limit=50")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				Items []struct {
					Metadata struct {
						Name string `json:"name"`
					} `json:"metadata"`
				} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.Items).NotTo(BeEmpty())

		// 只验证相邻元素的 rank 不逆序
		for i := 1; i < len(out.Data.Items); i++ {
			a := out.Data.Items[i-1].Metadata.Name
			b := out.Data.Items[i].Metadata.Name
			Expect(compareChartNameMock(a, b)).To(BeNumerically("<=", 0))
		}
	})

	// 【应用市场-获取列表-010】分页：超界页重置到第一页
	It("分页超界时应重置到第一页并返回第一页数据", func() {
		_, _, cleanup := seedReposAndCharts()
		defer cleanup()
		resp, body, err := tc.getLatestCharts("page=999&limit=10")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Data struct {
				TotalItems  int           `json:"totalItems"`
				CurrentPage int           `json:"currentPage"`
				Items       []interface{} `json:"items"`
			} `json:"data"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Data.CurrentPage).To(Equal(999))
		Expect(len(out.Data.Items)).To(BeNumerically("<=", 10))
		if out.Data.TotalItems > 0 {
			Expect(len(out.Data.Items)).To(BeNumerically(">", 0))
		}
	})
})

// ==================== Upload Chart 测试用例 ====================
var _ = Describe("应用市场-上传Chart", Label("UploadChart"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
		// 确保 local repo 存在并已同步
		Expect(tc.seedRepoCR("local", tc.mockHarbor.GetURL())).To(Succeed())
		resp, _, err := tc.syncRepo("local")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusAccepted))
		Expect(tc.waitForSyncComplete("local", 30*time.Second)).To(Succeed())
	})

	// 创建一个自定义 k8s mock 配置的 testContext，并完成 local repo sync。
	createUploadTC := func(opts ...mockK8sOption) *testContext {
		t := setupTestContext(nil, opts...)
		Expect(t.seedRepoCR("local", t.mockHarbor.GetURL())).To(Succeed())
		resp, _, err := t.syncRepo("local")
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusAccepted))
		Expect(t.waitForSyncComplete("local", 30*time.Second)).To(Succeed())
		return t
	}

	AfterEach(func() {
		if tc != nil && len(tc.createdRepos) > 0 {
			cleanupRepos(tc.ctx, tc.apiURL, tc.httpClient, tc.createdRepos)
		}
		if tc != nil && tc.server != nil {
			tc.server.Close()
		}
	})

	// 【应用市场-上传Chart-001】登录上传成功
	It("登录上传成功", func() {
		chartContent, err := createTestChartTgz("test-upload-chart", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := tc.uploadChart(chartContent, "test-upload-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusCreated))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.FileCreated))
		Expect(out.Msg).To(ContainSubstring("upload success"))
		Expect(tc.mockHarbor.UploadCount()).To(BeNumerically(">=", 1))
	})

	// 【应用市场-上传Chart-003】超 2MB
	It("超 2MB 应返回错误", func() {
		chartContent, err := createLargeChartTgz("test-large-chart", "1.0.0")
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := tc.uploadChart(chartContent, "test-large-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
		Expect(strings.ToLower(out.Msg)).To(Or(ContainSubstring("exceed"), ContainSubstring("danger")))
	})

	// 【应用市场-上传Chart-004】非法包/危险条目
	It("非法包应返回错误", func() {
		invalidContent := createInvalidChartTgz()

		resp, body, err := tc.uploadChart(invalidContent, "invalid-chart.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
		Expect(strings.ToLower(out.Msg)).To(Or(ContainSubstring("danger"), ContainSubstring("invalid")))
	})

	// 【应用市场-上传Chart-005】Harbor 返回 4xx
	It("Harbor 返回 4xx 应返回错误", func() {
		tc.mockHarbor.SetResponseCode(http.StatusBadRequest)

		chartContent, err := createTestChartTgz("test-4xx-chart", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := tc.uploadChart(chartContent, "test-4xx-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusInternalServerError))
		Expect(out.Msg).To(ContainSubstring("error counting chart numbers"))
	})

	// 【应用市场-上传Chart-006】Harbor 返回 500/非201
	It("Harbor 返回 500 应返回错误", func() {
		tc.mockHarbor.SetResponseCode(http.StatusInternalServerError)

		chartContent, err := createTestChartTgz("test-500-chart", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := tc.uploadChart(chartContent, "test-500-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ServerError))
	})

	// 【应用市场-上传Chart-007】获取 marketplace-service config 失败
	It("获取 marketplace-service config 失败应返回500", func() {
		t := createUploadTC(withSkipConfigMap())
		defer t.teardown()

		chartContent, err := createTestChartTgz("test-a7", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := t.uploadChart(chartContent, "test-a7-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusInternalServerError))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("error getting marketplace-service config"))
		Expect(t.mockHarbor.UploadCount()).To(Equal(0))
	})

	// 【应用市场-上传Chart-008】ChartLimit 配置不可解析
	It("ChartLimit 配置不可解析应返回500", func() {
		t := createUploadTC(withChartLimit("abc"))
		defer t.teardown()

		chartContent, err := createTestChartTgz("test-a9", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := t.uploadChart(chartContent, "test-a9-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusInternalServerError))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("error parsing chart limit"))
		Expect(t.mockHarbor.UploadCount()).To(Equal(0))
	})

	// 【应用市场-上传Chart-009】超出上传数量限制
	It("超出上传数量限制应返回400", func() {
		t := createUploadTC(withChartLimit("0"))
		defer t.teardown()

		// 先上传一个 chart（在 chart-limit 校验之前，UploadChart 会先 countChartNumbers，numbers=0 时允许上传）
		seed, err := createTestChartTgz("seed-chart", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())
		seedResp, _, err := t.uploadChart(seed, "seed-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(seedResp.StatusCode).To(Equal(http.StatusCreated))

		// 再上传第二个 chart：此时 countChartNumbers() 返回 1，且 chart-limit=0，应该触发超限返回 400
		chartContent, err := createTestChartTgz("test-a10", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := t.uploadChart(chartContent, "test-a10-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ExceedChartUploadLimit))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("reach chart upload limit"))
		// 第二次请求不应命中 Harbor 上传，因此 uploadCount 仍为 1
		Expect(t.mockHarbor.UploadCount()).To(Equal(1))
	})

	// 【应用市场-上传Chart-010】获取 Harbor 密码失败
	It("获取 Harbor 密码失败应返回500", func() {
		t := createUploadTC(withSkipSecret())
		defer t.teardown()

		chartContent, err := createTestChartTgz("test-a11", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())

		resp, body, err := t.uploadChart(chartContent, "test-a11-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusInternalServerError))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("error counting chart numbers"))
		Expect(t.mockHarbor.UploadCount()).To(Equal(0))
	})
})

// ==================== Delete Chart 测试用例 ====================
var _ = Describe("应用市场-删除Chart", Label("DeleteChart"), func() {
	var tc *testContext

	BeforeEach(func() {
		tc = setupTestContext(nil)
		// 确保 local repo 存在并已同步
		Expect(tc.seedRepoCR("local", tc.mockHarbor.GetURL())).To(Succeed())
		syncResp, _, err := tc.syncRepo("local")
		Expect(err).NotTo(HaveOccurred())
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))
		Expect(tc.waitForSyncComplete("local", 30*time.Second)).To(Succeed())

		// 预先上传一个 chart 用于删除测试
		chartContent, err := createTestChartTgz("test-delete-chart", "1.0.0", 0)
		Expect(err).NotTo(HaveOccurred())
		uploadResp, _, err := tc.uploadChart(chartContent, "test-delete-chart-1.0.0.tgz", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(uploadResp.StatusCode).To(Equal(http.StatusCreated))
		Expect(tc.waitForSyncComplete("local", 30*time.Second)).To(Succeed())
	})

	// 创建一个自定义 k8s mock 配置的 testContext，并完成 local repo sync。
	createDeleteTC := func(opts ...mockK8sOption) *testContext {
		t := setupTestContext(nil, opts...)
		Expect(t.seedRepoCR("local", t.mockHarbor.GetURL())).To(Succeed())
		syncResp, _, err := t.syncRepo("local")
		Expect(err).NotTo(HaveOccurred())
		Expect(syncResp.StatusCode).To(Equal(http.StatusAccepted))
		Expect(t.waitForSyncComplete("local", 30*time.Second)).To(Succeed())
		return t
	}

	AfterEach(func() {
		if tc != nil && len(tc.createdRepos) > 0 {
			cleanupRepos(tc.ctx, tc.apiURL, tc.httpClient, tc.createdRepos)
		}
		if tc != nil && tc.server != nil {
			tc.server.Close()
		}
	})

	// 【应用市场-删除Chart-001】登录删除 chart 成功
	It("登录删除 chart 成功", func() {
		resp, body, err := tc.deleteChart("test-delete-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.Success))
		Expect(out.Msg).To(ContainSubstring("delete success"))

		Expect(tc.mockHarbor.DeleteCount()).To(BeNumerically(">=", 1))
	})

	// 【应用市场-删除Chart-003】登录删除指定版本成功
	It("登录删除指定版本成功", func() {
		resp, body, err := tc.deleteChart("test-delete-chart", "1.0.0", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.Success))
		Expect(out.Msg).To(ContainSubstring("delete success"))
	})

	// 【应用市场-删除Chart-004】chartName 为空
	It("chartName 为空应返回错误", func() {
		resp, body, err := tc.deleteChart("", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("chart name"))
	})

	// 【应用市场-删除Chart-005】Harbor 404 → 当前返回 500
	It("Harbor 404 应返回 500（缺陷固化）", func() {
		tc.mockHarbor.SetResponseCode(http.StatusNotFound)

		resp, body, err := tc.deleteChart("non-existent-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
	})

	// 【应用市场-删除Chart-006】Harbor 500/403
	It("Harbor 500 应返回错误", func() {
		tc.mockHarbor.SetResponseCode(http.StatusInternalServerError)

		resp, body, err := tc.deleteChart("test-delete-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
	})

	It("Harbor 403 应返回错误", func() {
		tc.mockHarbor.SetResponseCode(http.StatusForbidden)

		resp, body, err := tc.deleteChart("test-delete-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
	})

	// 【应用市场-删除Chart-008】获取 marketplace-service config 失败
	It("获取 marketplace-service config 失败应返回500", func() {
		t := createDeleteTC(withSkipConfigMap())
		defer t.teardown()

		resp, body, err := t.deleteChart("test-delete-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(http.StatusInternalServerError))
		Expect(strings.ToLower(out.Msg)).To(ContainSubstring("error getting marketplace-service config"))
		Expect(t.mockHarbor.DeleteCount()).To(Equal(0))
	})

	// 【应用市场-删除Chart-009】获取 Harbor 密码失败
	It("获取 Harbor 密码失败应返回500", func() {
		t := createDeleteTC(withSkipSecret())
		defer t.teardown()

		resp, body, err := t.deleteChart("test-delete-chart", "", true)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		var out struct {
			Code int    `json:"code"`
			Msg  string `json:"msg"`
		}
		Expect(json.Unmarshal(body, &out)).To(Succeed())
		Expect(out.Code).To(Equal(constant.ClientError))
		Expect(strings.ToLower(out.Msg)).To(Or(
			ContainSubstring("please check the file you upload"),
			ContainSubstring("error getting local harbor password"),
		))
		Expect(t.mockHarbor.DeleteCount()).To(Equal(0))
	})
})
