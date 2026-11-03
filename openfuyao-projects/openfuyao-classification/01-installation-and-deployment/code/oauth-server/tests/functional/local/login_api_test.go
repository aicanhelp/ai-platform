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
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
)

var _ = Describe("Login API", func() {
	var ctx *TestContext

	BeforeEach(func() {
		ctx = NewTestContext()
		ctx.SetupLoginHandler(false)
	})

	AfterEach(func() {
		ctx.ResetPatches()
	})

	Describe("GET /oauth2/auth/login/fuyaoPasswordProvider", func() {
		Context("认证鉴权-001-LOGIN-GET: Display login form with valid then parameter", func() {
			It("should return 200 and display login form HTML", func() {
				req, err := http.NewRequest("GET",
					constants.FuyaoLoginEndpoint+"?then="+url.QueryEscape(ctx.ValidThenURL), nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusOK))
				body := ctx.Recorder.Body.String()
				Expect(body).To(ContainSubstring("login"))
				Expect(body).To(ContainSubstring("username"))
				Expect(body).To(ContainSubstring("password"))
			})
		})

		Context("认证鉴权-002-LOGIN-GET: Invalid then parameter", func() {
			It("should redirect to root when then parameter is invalid", func() {
				req, err := http.NewRequest("GET",
					constants.FuyaoLoginEndpoint+"?then=invalid-url", nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))
			})
		})

		Context("认证鉴权-003-LOGIN-GET: Empty then parameter", func() {
			It("should redirect to root when then parameter is empty", func() {
				req, err := http.NewRequest("GET", constants.FuyaoLoginEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))
			})
		})

		Context("认证鉴权-004-LOGIN-GET: Display login form with error message", func() {
			It("should display error message in the form", func() {
				errorMsg := "test error message"
				req, err := http.NewRequest("GET",
					constants.FuyaoLoginEndpoint+"?then="+url.QueryEscape(ctx.ValidThenURL)+
						"&error="+url.QueryEscape(errorMsg), nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusOK))
				body := ctx.Recorder.Body.String()
				Expect(body).To(ContainSubstring(errorMsg))
			})
		})
	})

	Describe("POST /oauth2/auth/login/fuyaoPasswordProvider", func() {
		Context("认证鉴权-005-LOGIN-POST: Successful login", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should redirect to then URL and set session cookie", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte(ctx.TestUser.Password),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL))

				// Verify session cookie is set
				cookies := ctx.Recorder.Result().Cookies()
				Expect(cookies).To(HaveLen(1))
				Expect(cookies[0].Name).To(Equal("idpLogin"))
			})
		})

		Context("认证鉴权-006-LOGIN-POST: Empty username", func() {
			It("should redirect to login page with error message", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: "",
					Password: []byte(ctx.TestUser.Password),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrUsernameOrPasswordMissing)))
			})
		})

		Context("认证鉴权-007-LOGIN-POST: Empty password", func() {
			It("should redirect to login page with error message", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte(""),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrUsernameOrPasswordMissing)))
			})
		})

		Context("认证鉴权-008-LOGIN-POST: Non-existent username", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should redirect to login page with authentication error", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: "nonexistentuser",
					Password: []byte(ctx.TestUser.Password),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrPasswordAuthenticationFailed)))
			})
		})

		Context("认证鉴权-009-LOGIN-POST: Wrong password", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should redirect to login page with password error and remaining attempts", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte("wrongpassword"),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				// Should contain remaining attempts count
				Expect(location).To(ContainSubstring(url.QueryEscape("用户名或密码错误，再输错")))
			})
		})

		Context("认证鉴权-010-LOGIN-POST: Username case sensitivity error", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			DescribeTable("should fail for case-mismatched usernames",
				func(username string) {
					reqBody := fuyaopassword.LoginRequest{
						Username: username,
						Password: []byte(ctx.TestUser.Password),
						Then:     ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					location := ctx.Recorder.Header().Get("Location")
					Expect(location).To(ContainSubstring("error="))
				},
				Entry("Admin (capital A)", "Admin"),
				Entry("ADMIN (all caps)", "ADMIN"),
				Entry("aDmIn (mixed case)", "aDmIn"),
			)
		})

		Context("认证鉴权-011-LOGIN-POST: Invalid request body format", func() {
			It("should return 400 Bad Request for non-JSON body", func() {
				// Send plain text instead of JSON
				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint,
					bytes.NewReader([]byte("invalid-json-body")))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest))
			})
		})

		Context("认证鉴权-012-LOGIN-POST: Invalid then parameter", func() {
			It("should redirect to login page with invalid then error", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte(ctx.TestUser.Password),
					Then:     "invalid-then-url",
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrInvalidThen)))
			})
		})

		Context("认证鉴权-014-LOGIN-POST: HTTP method not allowed", func() {
			DescribeTable("should reject non-GET/POST methods",
				func(method string) {
					req, err := http.NewRequest(method, constants.FuyaoLoginEndpoint, nil)
					Expect(err).ToNot(HaveOccurred())

					ctx.ResetRecorder()
					ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusMethodNotAllowed))
				},
				Entry("PUT method", "PUT"),
				Entry("DELETE method", "DELETE"),
				Entry("PATCH method", "PATCH"),
			)
		})

		Context("认证鉴权-015-LOGIN-POST: First login user", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.FirstLoginUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should login successfully and redirect (will be handled by authorize endpoint)", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.FirstLoginUser.Username,
					Password: []byte(ctx.FirstLoginUser.Password),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				// First login user should still redirect to then URL after successful login
				// The password confirmation is handled at /oauth/authorize endpoint
				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL))

				// Verify session cookie is set with first-login flag
				cookies := ctx.Recorder.Result().Cookies()
				Expect(cookies).To(HaveLen(1))
				Expect(cookies[0].Name).To(Equal("idpLogin"))
			})
		})

		Context("认证鉴权-016-LOGIN-POST: Already logged in user", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
				ctx.SetupSession(CreateLoggedInSession(ctx.TestUser.Username, false))
			})

			It("should redirect directly to then URL without re-authentication", func() {
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte(ctx.TestUser.Password),
					Then:     ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL))
			})
		})

		Context("认证鉴权-017-018-LOGIN-POST: Username edge cases", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			DescribeTable("should fail for malformed usernames",
				func(username string, description string) {
					reqBody := fuyaopassword.LoginRequest{
						Username: username,
						Password: []byte(ctx.TestUser.Password),
						Then:     ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					location := ctx.Recorder.Header().Get("Location")
					Expect(location).To(ContainSubstring("error="))
					Expect(location).To(ContainSubstring(url.QueryEscape("用户名或密码错误")),
						"Failed for: %s (%s)", username, description)
				},
				// LOGIN-POST-013: Username spelling error
				Entry("username spelling error - missing letter", "admi", "missing last letter"),
				Entry("username spelling error - extra letter", "adminn", "extra letter"),
				Entry("username spelling error - typo", "adimn", "transposed letters"),

				// LOGIN-POST-014: Username with special characters
				Entry("username with special chars @#$%", "admin@#$%", "special characters appended"),
				Entry("username with special chars prefix", "@admin", "special character prefix"),
				Entry("username with spaces", "admin user", "contains space"),
			)
		})
	})

	Describe("User lockout scenarios", func() {
		Context("认证鉴权-013-LOGIN-POST: User locked after multiple failed attempts", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should lock user after configured number of failed attempts", func() {
				// Simulate multiple failed login attempts
				for i := 0; i < TestFailTimes; i++ {
					reqBody := fuyaopassword.LoginRequest{
						Username: ctx.TestUser.Username,
						Password: []byte("wrongpassword"),
						Then:     ctx.ValidThenURL,
					}
					body, _ := json.Marshal(reqBody)

					req, _ := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
					req.Header.Set("Content-Type", "application/json")

					ctx.Recorder = httptest.NewRecorder()
					ctx.LoginHandler.LoginHandler(ctx.Recorder, req)
				}

				// Next attempt should show locked message
				reqBody := fuyaopassword.LoginRequest{
					Username: ctx.TestUser.Username,
					Password: []byte("wrongpassword"),
					Then:     ctx.ValidThenURL,
				}
				body, _ := json.Marshal(reqBody)

				req, _ := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")

				ctx.Recorder = httptest.NewRecorder()
				ctx.LoginHandler.LoginHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				// Should contain locked message
				errorMsg := ExtractErrorFromLocation(location)
				Expect(errorMsg).To(Or(
					ContainSubstring("锁定"),
					ContainSubstring("locked"),
				))
			})
		})
	})
})
