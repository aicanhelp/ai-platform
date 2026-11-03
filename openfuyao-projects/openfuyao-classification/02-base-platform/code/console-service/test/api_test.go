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

package test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"console-service/pkg/auth"
	"console-service/pkg/utils/authutil"
)

var _ = Describe("登录", func() {

	It("未提供sessionID", func() {
		// cookie中未包含sessionID
		resp, err := testHttpClient.Get(serverAddr + "/rest/auth/login")
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 检查响应码302，重定向到 /oauth2/oauth/authorize
		Expect(resp.StatusCode).To(Equal(302))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/oauth2/oauth/authorize"))

		// 检查query中包含state
		state := u.Query().Get("state")
		Expect(state).NotTo(BeEmpty())

		// 检查响应 Set-Cookie 中包含 state，值与 query 相同
		found := false
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "state" {
				Expect(cookie.Value).To(Equal(state))
				found = true
				break
			}
		}
		Expect(found).To(BeTrue(), "响应未设置 state cookie")
	})

	It("提供不存在的sessionID", func() {
		// cookie中包含不存在的sessionID
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/login", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "non-existing-sessionID"})
		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 检查响应码302，重定向到 /oauth2/oauth/authorize
		Expect(resp.StatusCode).To(Equal(302))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/oauth2/oauth/authorize"))

		// 检查query中包含state
		state := u.Query().Get("state")
		Expect(state).NotTo(BeEmpty())

		// 检查响应 Set-Cookie 中包含 state，值与 query 相同
		found := false
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "state" {
				Expect(cookie.Value).To(Equal(state))
				found = true
				break
			}
		}
		Expect(found).To(BeTrue(), "响应未设置 state cookie")

		// 检查响应中删除原有sessionID cookie
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "sessionID" {
				Expect(cookie.Value).To(BeEmpty())
				break
			}
		}
	})

	It("提供过期的sessionID", func() {
		// 创建临时的过期 session
		err := auth.StoreSession(k8sClient, expiredSession, false)
		Expect(err).NotTo(HaveOccurred())

		// cookie中包含过期的sessionID
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/login", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: expiredSession.GetSessionID()})
		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 检查响应码302，重定向到 /oauth2/oauth/authorize
		Expect(resp.StatusCode).To(Equal(302))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/oauth2/oauth/authorize"))

		// 检查query中包含state
		state := u.Query().Get("state")
		Expect(state).NotTo(BeEmpty())

		// 检查响应 Set-Cookie 中包含 state，值与 query 相同
		found := false
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "state" {
				Expect(cookie.Value).To(Equal(state))
				found = true
				break
			}
		}
		Expect(found).To(BeTrue(), "响应未设置 state cookie")

		// 响应中删除原有sessionID cookie
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "sessionID" {
				Expect(cookie.Value).To(BeEmpty())
				break
			}
		}
	})

	It("提供有效的sessionID", func() {
		// cookie中包含有效的sessionID
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/login", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "12345678"})
		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 响应码302，重定向到 /
		Expect(resp.StatusCode).To(Equal(302))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/"))
	})
})

var _ = Describe("登录回调", func() {

	It("错误提示", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?error=access_denied&error_description=Invalid+credentials", nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码200
		Expect(resp.StatusCode).To(Equal(200))

		// 读取响应内容
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		// 预期返回包含报错信息的HTML文档
		Expect(string(body)).To(ContainSubstring("access_denied"))
		Expect(string(body)).To(ContainSubstring("Invalid credentials"))
		Expect(string(body)).To(ContainSubstring("<html"))
		Expect(string(body)).To(ContainSubstring("</html>"))
	})

	It("缺少状态码参数", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?code=validcode", nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码401
		Expect(resp.StatusCode).To(Equal(http.StatusUnauthorized))
	})

	It("缺少验证码参数", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?state=validstate", nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码401
		Expect(resp.StatusCode).To(Equal(http.StatusUnauthorized))
	})

	It("无效的状态码", func() {
		// 发送请求 /rest/auth/callback，query中的state与cookie中的state不同
		// 先设置cookie中的state
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?code=validcode&state=wrongstate", nil)
		Expect(err).NotTo(HaveOccurred())
		// cookie 中的 state 与 query 中的不同
		req.AddCookie(&http.Cookie{Name: "state", Value: "correctstate"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码303
		Expect(resp.StatusCode).To(Equal(http.StatusSeeOther))
	})

	It("无效的验证码", func() {
		// 设置正确的 state cookie
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?code=invalidcode&state=correctstate", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "state", Value: "correctstate"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码500（因为OAuth服务器返回错误）
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
	})

	It("有效的状态码与验证码", func() {
		// 统计原始 session 数量
		secrets, err := k8sClient.CoreV1().Secrets("session-secret").List(context.Background(), metav1.ListOptions{})
		originalCount := len(secrets.Items)

		// 设置正确的 state cookie
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/callback?code=validcode&state=correctstate", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "state", Value: "correctstate"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 期望响应码302，重定向到 /
		Expect(resp.StatusCode).To(Equal(http.StatusFound))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/"))

		// 检查响应中删除原有state cookie
		foundState := false
		foundSessionID := false
		foundWsID := false
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "state" && cookie.Value == "" {
				foundState = true
			}
			if cookie.Name == "sessionID" && cookie.Value != "" {
				foundSessionID = true
			}
			if cookie.Name == "wsID" && cookie.Value != "" {
				foundWsID = true
			}
		}
		Expect(foundState).To(BeTrue(), "响应未包含删除 state cookie")
		Expect(foundSessionID).To(BeTrue(), "响应未包含 sessionID cookie")
		Expect(foundWsID).To(BeTrue(), "响应未包含 wsID cookie")

		// 检查 session-secret 命名空间是否新增了 secret
		// 这里我们通过检查是否有新创建的 session 来验证
		secrets, err = k8sClient.CoreV1().Secrets("session-secret").List(context.Background(), metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(len(secrets.Items)).To(BeNumerically(">", originalCount), "session-secret 命名空间应包含至少一个 secret")
	})

})

var _ = Describe("登出", func() {

	It("未提供sessionID", func() {
		// 不携带 sessionID cookie，直接调用登出接口
		req, err := http.NewRequest("POST", serverAddr+"/rest/auth/logout", nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期返回 204
		Expect(resp.StatusCode).To(Equal(http.StatusNoContent))
	})

	It("有效的sessionID", func() {
		// 创建临时的有效 session
		err := auth.StoreSession(k8sClient, &auth.SessionStore{
			"AccessToken":   []byte("xxxxxxxx"),
			"RefreshToken":  []byte("xxxxxxxx"),
			"SessionID":     []byte("xxxxxxxx"),
			"AccessExpiry":  []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
			"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
		}, false)
		Expect(err).NotTo(HaveOccurred())

		// 携带有效的 sessionID 调用登出接口
		req, err := http.NewRequest("POST", serverAddr+"/rest/auth/logout", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "xxxxxxxx"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期返回 204
		Expect(resp.StatusCode).To(Equal(http.StatusNoContent))

		// 预期响应中通过 Set-Cookie 清空 sessionID
		found := false
		for _, c := range resp.Cookies() {
			if c.Name == "sessionID" {
				Expect(c.Value).To(BeEmpty())
				found = true
				break
			}
		}
		Expect(found).To(BeTrue(), "响应未包含清除 sessionID 的 Set-Cookie")

		// 对应的 session 应被删除
		_, err = auth.GetSession(k8sClient, "xxxxxxxx")
		Expect(err).To(HaveOccurred())
	})

})

var _ = Describe("获取当前用户", func() {

	It("未提供sessionID", func() {
		// 不携带任何认证头和 sessionID，直接调用接口
		resp, err := testHttpClient.Get(serverAddr + "/rest/auth/user")
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 401，错误消息为 cookie not found
		Expect(resp.StatusCode).To(Equal(http.StatusUnauthorized))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32       `json:"code"`
			Msg  string      `json:"msg"`
			Data interface{} `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		Expect(res.Msg).To(Equal("cookie not found"))
	})

	It("无效的sessionID（过期）", func() {
		// 创建临时的过期 session
		err := auth.StoreSession(k8sClient, expiredSession, false)
		Expect(err).NotTo(HaveOccurred())

		// 使用过期的 session
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/user", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: expiredSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 401，错误消息为 cannot get token from sessionID
		Expect(resp.StatusCode).To(Equal(http.StatusUnauthorized))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32       `json:"code"`
			Msg  string      `json:"msg"`
			Data interface{} `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		Expect(res.Msg).To(Equal("cannot get token from sessionID"))
	})

	It("有效的sessionID，无效的accessCode", func() {
		// 使用有效 sessionID（BeforeSuite 中写入的 validSession，ID 为 12345678）
		// 其 accessToken 为无效格式，期望解析失败
		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/user", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "12345678"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 500，错误消息为 cannot extract user from token
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32       `json:"code"`
			Msg  string      `json:"msg"`
			Data interface{} `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		Expect(res.Msg).To(Equal("cannot extract user from token"))
	})

	It("有效的sessionID，有效的accessCode", func() {
		// 为本用例单独创建一个带有效 JWT accessToken 的 session
		token := &auth.AccessRefreshToken{
			AccessToken:        authutil.GenerateToken("user"),
			AccessTokenExpiry:  time.Now().Add(2 * time.Hour),
			RefreshToken:       "dummy-refresh-token",
			RefreshTokenExpiry: time.Now().Add(2 * time.Hour),
		}
		session, err := auth.NewStoreSession(token)
		Expect(err).NotTo(HaveOccurred())
		Expect(auth.StoreSession(k8sClient, session, false)).To(Succeed())

		req, err := http.NewRequest("GET", serverAddr+"/rest/auth/user", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: session.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 200，msg 为 get current user succeed，data 为 user
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32  `json:"code"`
			Msg  string `json:"msg"`
			Data string `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		Expect(res.Msg).To(Equal("get current user succeed"))
		Expect(res.Data).To(Equal("user"))

		// 删除临时 session
		Expect(auth.DeleteSession(k8sClient, session.GetSessionID())).To(Succeed())
	})

})

var _ = Describe("时间偏移", func() {

	It("时间偏移", func() {
		// 使用有效 sessionID 访问时间偏移接口
		req, err := http.NewRequest("GET", serverAddr+"/rest/console/v1beta1/time-offset", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "12345678"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 200
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32             `json:"code"`
			Msg  string            `json:"msg"`
			Data map[string]string `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())

		// currentServerTime 为非空时间戳，offset 为 UTC±hh:mm
		Expect(res.Data["currentServerTime"]).NotTo(BeEmpty())
		Expect(res.Data["offset"]).To(Or(HavePrefix("UTC+"), HavePrefix("UTC-")))
	})

})

var _ = Describe("登录状态websocket", func() {

	It("Upgrade失败，未提供wsID", func() {
		// 使用有效 sessionID 访问登录状态 websocket 接口
		req, err := http.NewRequest("GET", serverAddr+"/ws/auth/login-status", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "12345678"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 400，错误消息为 websocket connection request failed, fail to get cookieNameWsID from cookie
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32       `json:"code"`
			Msg  string      `json:"msg"`
			Data interface{} `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		Expect(res.Msg).To(Equal("websocket connection request failed, fail to get cookieNameWsID from cookie"))
	})

	It("Upgrade失败，未提供Upgrade请求头", func() {
		req, err := http.NewRequest("GET", serverAddr+"/ws/auth/login-status", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "12345678"})
		req.AddCookie(&http.Cookie{Name: "wsID", Value: "12345678"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 预期 400
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var res struct {
			Code int32       `json:"code"`
			Msg  string      `json:"msg"`
			Data interface{} `json:"data"`
		}
		Expect(json.Unmarshal(body, &res)).To(Succeed())
		// 请求体不带 Connection: Upgrade 会导致 ws 升级失败
		Expect(res.Msg).To(ContainSubstring("the client is not using the websocket protocol: 'upgrade' token"))
	})

	It("Upgrade成功 - 验证心跳包并登出", func() {
		dialer := websocket.Dialer{}
		wsURL := strings.Replace(serverAddr, "http", "ws", 1) + "/ws/auth/login-status"
		header := http.Header{}
		header.Set("Cookie", "sessionID=12345678; wsID=12345678")

		conn, resp, err := dialer.Dial(wsURL, header)
		Expect(err).NotTo(HaveOccurred())
		defer conn.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusSwitchingProtocols))

		conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		// 读取心跳包
		messageType, message, err := conn.ReadMessage()
		Expect(err).NotTo(HaveOccurred())
		Expect(messageType).To(Equal(websocket.TextMessage))
		var heartbeatMsg struct {
			LoginStatus string `json:"loginStatus"`
		}
		Expect(json.Unmarshal(message, &heartbeatMsg)).To(Succeed())
		Expect(heartbeatMsg.LoginStatus).To(Equal("true"))

		// 发送无效一般请求，触发服务端WS清理
		req, err := http.NewRequest("GET", serverAddr+"/rest/alert/v1/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "wsID", Value: "12345678"})

		_, err = testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())

		// 读取心跳包
		messageType, message, err = conn.ReadMessage()
		Expect(err).NotTo(HaveOccurred())
		Expect(messageType).To(Equal(websocket.TextMessage))
		Expect(json.Unmarshal(message, &heartbeatMsg)).To(Succeed())
		Expect(heartbeatMsg.LoginStatus).To(Equal("false"))
	})
})

var _ = Describe("OAuth登录重定向", func() {

	BeforeEach(func() {
		// Ensure session-secret namespace exists
		_, err := k8sClient.CoreV1().Namespaces().Get(context.Background(), "session-secret", metav1.GetOptions{})
		if err != nil {
			_, err = k8sClient.CoreV1().Namespaces().Create(context.Background(), &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "session-secret"},
			}, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("登录成功后重定向到原始页面", func() {
		// Step 1: 访问需要认证的页面，触发重定向到登录页，并检查是否设置了 Referrer Cookie
		targetURL := "/container_platform/workload/pods"
		req, err := http.NewRequest("GET", serverAddr+targetURL, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		// 检查重定向到登录页
		Expect(resp.StatusCode).To(Equal(http.StatusSeeOther))
		location := resp.Header.Get("Location")
		Expect(location).To(ContainSubstring("/rest/auth/login"))

		// 检查是否设置了 Referrer Cookie
		var referrerCookie *http.Cookie
		for _, cookie := range resp.Cookies() {
			if cookie.Name == "X-OpenFuyao-Referrer" {
				referrerCookie = cookie
				break
			}
		}
		Expect(referrerCookie).NotTo(BeNil(), "应该设置 X-OpenFuyao-Referrer Cookie")
		Expect(referrerCookie.Value).To(ContainSubstring(targetURL), "Referrer Cookie 应包含原始 URL")

		// Step 2: 模拟登录流程 - 获取 login state
		loginReq, err := http.NewRequest("GET", serverAddr+"/rest/auth/login", nil)
		Expect(err).NotTo(HaveOccurred())

		loginResp, err := testHttpClient.Do(loginReq)
		Expect(err).NotTo(HaveOccurred())
		defer loginResp.Body.Close()

		Expect(loginResp.StatusCode).To(Equal(http.StatusFound))

		// 获取 state cookie 和 query 参数
		var stateCookie *http.Cookie
		var stateValue string
		for _, cookie := range loginResp.Cookies() {
			if cookie.Name == "state" {
				stateCookie = cookie
				stateValue = cookie.Value
				break
			}
		}
		Expect(stateCookie).NotTo(BeNil(), "登录请求应设置 state Cookie")

		// Step 3: 模拟 OAuth callback - 使用有效的 code 和 state
		callbackURL := serverAddr + "/rest/auth/callback?code=validcode&state=" + stateValue
		callbackReq, err := http.NewRequest("GET", callbackURL, nil)
		Expect(err).NotTo(HaveOccurred())

		// 添加 state cookie 和 referrer cookie
		callbackReq.AddCookie(stateCookie)
		callbackReq.AddCookie(referrerCookie)

		callbackResp, err := testHttpClient.Do(callbackReq)
		Expect(err).NotTo(HaveOccurred())
		defer callbackResp.Body.Close()

		// Step 4: 验证重定向到原始页面
		Expect(callbackResp.StatusCode).To(Equal(http.StatusFound))
		redirectLocation := callbackResp.Header.Get("Location")

		// 验证重定向到原始目标页面（包含原始路径）
		u, err := url.Parse(redirectLocation)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal(targetURL), "登录成功后应重定向到原始页面")

		// 验证 Referrer Cookie 已被清除
		for _, cookie := range callbackResp.Cookies() {
			if cookie.Name == "X-OpenFuyao-Referrer" {
				Expect(cookie.Value).To(BeEmpty(), "Referrer Cookie 应被清除")
				break
			}
		}
	})

	It("无 Referrer Cookie 时重定向到根路径", func() {
		// 直接访问登录页，无 Referrer Cookie
		loginReq, err := http.NewRequest("GET", serverAddr+"/rest/auth/login", nil)
		Expect(err).NotTo(HaveOccurred())

		loginResp, err := testHttpClient.Do(loginReq)
		Expect(err).NotTo(HaveOccurred())
		defer loginResp.Body.Close()

		Expect(loginResp.StatusCode).To(Equal(http.StatusFound))

		// 获取 state cookie
		var stateCookie *http.Cookie
		var stateValue string
		for _, cookie := range loginResp.Cookies() {
			if cookie.Name == "state" {
				stateCookie = cookie
				stateValue = cookie.Value
				break
			}
		}
		Expect(stateCookie).NotTo(BeNil())

		// 模拟 OAuth callback - 不带 Referrer Cookie
		callbackURL := serverAddr + "/rest/auth/callback?code=validcode&state=" + stateValue
		callbackReq, err := http.NewRequest("GET", callbackURL, nil)
		Expect(err).NotTo(HaveOccurred())
		callbackReq.AddCookie(stateCookie)

		callbackResp, err := testHttpClient.Do(callbackReq)
		Expect(err).NotTo(HaveOccurred())
		defer callbackResp.Body.Close()

		// 验证重定向到根路径 "/"
		Expect(callbackResp.StatusCode).To(Equal(http.StatusFound))
		redirectLocation := callbackResp.Header.Get("Location")
		u, err := url.Parse(redirectLocation)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/"), "无 Referrer Cookie 时应重定向到根路径")
	})
})
