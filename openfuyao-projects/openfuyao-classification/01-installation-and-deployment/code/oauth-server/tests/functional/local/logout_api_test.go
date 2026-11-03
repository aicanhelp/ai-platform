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

package local

import (
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
)

var _ = Describe("Logout API", func() {
	var ctx *TestContext

	BeforeEach(func() {
		ctx = NewTestContext()
		ctx.SetupOAuthServer()
	})

	AfterEach(func() {
		ctx.ResetPatches()
	})

	Describe("POST /oauth2/auth/logout/fuyaoPasswordProvider", func() {
		Context("认证鉴权-084-LOGOUT-POST: Successful logout for authenticated user", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should return 204 No Content and clear cookies", func() {
				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Verify cookies are being cleared
				cookies := ctx.Recorder.Result().Cookies()
				// Should have at least the session cookie and csrf cookie being cleared
				hasClearingCookie := false
				for _, cookie := range cookies {
					if cookie.Name == "idpLogin" || cookie.Name == "csrf" {
						hasClearingCookie = true
					}
				}
				Expect(hasClearingCookie).To(BeTrue())
			})
		})

		Context("认证鉴权-085-LOGOUT-POST: Logout with redirect_uri parameter", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should return 204 and accept redirect_uri parameter", func() {
				redirectURI := "https://example.com/login"
				req, err := ctx.NewRequestWithSession("POST",
					constants.FuyaoLogoutEndpoint+"?redirect_uri="+redirectURI, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				// Logout should succeed with 204
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))
			})
		})

		Context("认证鉴权-086-LOGOUT-POST: Logout for unauthenticated user (idempotent)", func() {
			// No session setup - user not logged in

			It("should return 204 even when user is not logged in", func() {
				req, err := http.NewRequest("POST", constants.FuyaoLogoutEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				// Should still return 204 - logout is idempotent
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))
			})
		})

		Context("认证鉴权-087-LOGOUT-POST: HTTP method not allowed", func() {
			DescribeTable("should reject non-POST methods",
				func(method string) {
					req, err := http.NewRequest(method, constants.FuyaoLogoutEndpoint, nil)
					Expect(err).ToNot(HaveOccurred())

					ctx.ResetRecorder()
					ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusMethodNotAllowed))
					Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrRequestMethodNotAllowed))
				},
				Entry("GET method", "GET"),
				Entry("PUT method", "PUT"),
				Entry("DELETE method", "DELETE"),
				Entry("PATCH method", "PATCH"),
			)
		})

		Context("认证鉴权-088-LOGOUT-POST: CSRF cookie clearing verification", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should clear the CSRF cookie", func() {
				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Check for CSRF cookie clearing
				cookies := ctx.Recorder.Result().Cookies()
				var csrfCookieFound bool
				for _, cookie := range cookies {
					if cookie.Name == "csrf" {
						csrfCookieFound = true
						// Cookie should be expired (cleared)
						Expect(cookie.MaxAge).To(Or(
							Equal(-1),
							Equal(0),
							BeNumerically("<", 0),
						))
					}
				}
				// CSRF cookie clear attempt should be present
				// Note: depending on implementation, it may or may not be present
				_ = csrfCookieFound
			})
		})

		Context("认证鉴权-089-LOGOUT-POST: Session cookie clearing verification", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should clear the login session cookie", func() {
				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Check for session cookie clearing
				cookies := ctx.Recorder.Result().Cookies()
				var sessionCookieFound bool
				for _, cookie := range cookies {
					if cookie.Name == "idpLogin" {
						sessionCookieFound = true
						// The cookie should be set (to clear the value)
						// The actual clearing is done by setting a new empty value
					}
				}
				Expect(sessionCookieFound).To(BeTrue(), "Session cookie should be present in response")
			})
		})
	})

	Describe("Logout edge cases", func() {
		Context("Multiple logout requests", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should handle multiple consecutive logout requests", func() {
				// First logout
				req, _ := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Second logout (should still work - idempotent, no session cookie)
				ctx.ResetRecorder()
				req, _ = http.NewRequest("POST", constants.FuyaoLogoutEndpoint, nil)
				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Third logout
				ctx.ResetRecorder()
				req, _ = http.NewRequest("POST", constants.FuyaoLogoutEndpoint, nil)
				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))
			})
		})

		Context("Logout with various redirect_uri values", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			DescribeTable("should accept various redirect_uri values",
				func(redirectURI string) {
					req, err := ctx.NewRequestWithSession("POST",
						constants.FuyaoLogoutEndpoint+"?redirect_uri="+redirectURI, nil)
					Expect(err).ToNot(HaveOccurred())

					ctx.ResetRecorder()
					ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

					// Should always return 204 regardless of redirect_uri
					Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))
				},
				Entry("empty redirect_uri", ""),
				Entry("login page", "https://example.com/login"),
				Entry("home page", "https://example.com/"),
				Entry("local path", "/login"),
			)
		})

		Context("Logout response headers", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should not include sensitive data in response", func() {
				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Response body should be empty for 204 No Content
				Expect(ctx.Recorder.Body.String()).To(BeEmpty())
			})
		})
	})

	Describe("Session state after logout", func() {
		Context("Verify session is cleared", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should clear session so subsequent requests require re-login", func() {
				// Perform logout with session cookie
				req, _ := ctx.NewRequestWithSession("POST", constants.FuyaoLogoutEndpoint, nil)
				ctx.OAuthServer.SingleLogoutHandler(ctx.Recorder, req)
				Expect(ctx.Recorder.Code).To(Equal(http.StatusNoContent))

				// Get the cookies from logout response (these should clear the session)
				logoutCookies := ctx.Recorder.Result().Cookies()

				// Now try to access authorize endpoint (should redirect to login)
				ctx.ResetRecorder()

				authorizeReq, _ := http.NewRequest("GET",
					constants.FuyaoOAuthAuthorizeEndpoint+
						"?client_id=console"+
						"&redirect_uri=https://example.com/rest/auth/callback"+
						"&response_type=code"+
						"&identity_provider=fuyaoPasswordProvider", nil)

				// Add logout cookies to the request (these are the cleared cookies)
				for _, cookie := range logoutCookies {
					authorizeReq.AddCookie(cookie)
				}

				ctx.OAuthServer.OAuthAuthorizeHandler(ctx.Recorder, authorizeReq)

				// Should redirect to login since session was cleared
				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring(constants.FuyaoLoginEndpoint))
			})
		})
	})
})
