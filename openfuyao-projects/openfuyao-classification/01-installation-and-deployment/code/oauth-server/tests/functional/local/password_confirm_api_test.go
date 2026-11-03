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
	"net/url"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
)

var _ = Describe("Password Confirm API", func() {
	var ctx *TestContext

	BeforeEach(func() {
		ctx = NewTestContext()
		ctx.SetupLoginHandler(true)
	})

	AfterEach(func() {
		ctx.ResetPatches()
	})

	Describe("GET /oauth2/auth/password/confirm/fuyaoPasswordProvider", func() {
		Context("认证鉴权-019-CONFIRM-GET: Display password confirm form with valid session", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.FirstLoginUser)
				ctx.SetupUserInfoMock(mockUser)
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should return 200 and display password confirmation form", func() {
				req, err := ctx.NewRequestWithSession("GET",
					constants.FuyaoPasswordConfirmEndpoint+"?then="+url.QueryEscape(ctx.ValidThenURL), nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusOK))
				// Form should be displayed
				body := ctx.Recorder.Body.String()
				Expect(body).ToNot(BeEmpty())
			})
		})

		Context("认证鉴权-020-CONFIRM-GET: Unauthenticated user access", func() {
			// No session setup - user not logged in

			It("should redirect to root when user is not logged in", func() {
				req, err := http.NewRequest("GET",
					constants.FuyaoPasswordConfirmEndpoint+"?then="+url.QueryEscape(ctx.ValidThenURL), nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))
			})
		})

		Context("认证鉴权-021-CONFIRM-GET: Invalid then parameter", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should redirect to root when then parameter is invalid", func() {
				req, err := ctx.NewRequestWithSession("GET",
					constants.FuyaoPasswordConfirmEndpoint+"?then=invalid-url", nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))
			})
		})

		Context("认证鉴权-022-CONFIRM-GET: Display form with error message", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should display error message in the form", func() {
				errorMsg := "password complexity error"
				req, err := ctx.NewRequestWithSession("GET",
					constants.FuyaoPasswordConfirmEndpoint+"?then="+url.QueryEscape(ctx.ValidThenURL)+
						"&error="+url.QueryEscape(errorMsg), nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusOK))
				body := ctx.Recorder.Body.String()
				Expect(body).To(ContainSubstring(errorMsg))
			})
		})
	})

	Describe("POST /oauth2/auth/password/confirm/fuyaoPasswordProvider", func() {
		Context("认证鉴权-023-CONFIRM-POST: Successful password confirmation", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.FirstLoginUser)
				ctx.SetupUserInfoMock(mockUser)
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should confirm password and redirect to then URL", func() {
				reqBody := fuyaopassword.PasswordConfirmRequest{
					NewPassword: []byte("NewPass123!"),
					Then:        ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL))

				// Verify session is updated
				cookies := ctx.Recorder.Result().Cookies()
				Expect(cookies).To(HaveLen(1))
				Expect(cookies[0].Name).To(Equal("idpLogin"))
			})
		})

		Context("认证鉴权-024-CONFIRM-POST: Empty new password", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should redirect with missing password error", func() {
				reqBody := fuyaopassword.PasswordConfirmRequest{
					NewPassword: []byte(""),
					Then:        ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrUsernameOrPasswordMissing)))
			})
		})

		Context("Password complexity validation", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.FirstLoginUser)
				ctx.SetupUserInfoMock(mockUser)
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			DescribeTable("认证鉴权-025-029-CONFIRM-POST: Should reject invalid passwords",
				func(password string, description string) {
					reqBody := fuyaopassword.PasswordConfirmRequest{
						NewPassword: []byte(password),
						Then:        ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					location := ctx.Recorder.Header().Get("Location")
					errorMsg := ExtractErrorFromLocation(location)
					Expect(errorMsg).To(ContainSubstring(fuyaoerrors.ErrStrPasswordTooWeak),
						"Failed for: %s (%s)", password, description)
				},
				// CONFIRM-POST-003: Password too short
				Entry("7 chars", "pass12!", "length 7 - too short"),
				Entry("6 chars", "pa12!x", "length 6 - too short"),

				// CONFIRM-POST-004: Password too long
				Entry("33 chars", "passwordwith123and@specialchar33!", "length 33 - too long"),
				Entry("40 chars", "password123456789012345678901234567890!A", "length 40 - too long"),

				// CONFIRM-POST-005: Missing letters
				Entry("no letters 1", "12345678!", "only numbers and special char"),
				Entry("no letters 2", "!@#$%^&*12", "special chars and numbers"),

				// CONFIRM-POST-006: Missing numbers
				Entry("no numbers 1", "password!", "only letters and special char"),
				Entry("no numbers 2", "Password!@", "letters and special char"),

				// CONFIRM-POST-007: Missing special characters
				Entry("no special 1", "password123", "only letters and numbers"),
				Entry("no special 2", "Password12", "letters and numbers"),
			)

			DescribeTable("认证鉴权-032-033-CONFIRM-POST: Should accept valid passwords at boundaries",
				func(password string, description string) {
					reqBody := fuyaopassword.PasswordConfirmRequest{
						NewPassword: []byte(password),
						Then:        ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL),
						"Failed for: %s (%s)", password, description)
				},
				// CONFIRM-POST-010: Password length 8 (minimum boundary)
				Entry("8 chars valid", "pass12!A", "length 8 - minimum valid"),
				Entry("8 chars valid 2", "Pa$$w0rd", "length 8 - minimum valid with special"),

				// CONFIRM-POST-011: Password length 32 (maximum boundary)
				Entry("32 chars valid", "passwordwith123and@specialchar3", "length 32 - maximum valid"),
			)
		})

		Context("认证鉴权-030-CONFIRM-POST: Unauthenticated user submission", func() {
			// No session setup - user not logged in

			It("should redirect to root when user is not logged in", func() {
				reqBody := fuyaopassword.PasswordConfirmRequest{
					NewPassword: []byte("NewPass123!"),
					Then:        ctx.ValidThenURL,
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := http.NewRequest("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))
			})
		})

		Context("认证鉴权-031-CONFIRM-POST: Invalid then parameter in POST", func() {
			BeforeEach(func() {
				ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
			})

			It("should redirect with invalid then error", func() {
				reqBody := fuyaopassword.PasswordConfirmRequest{
					NewPassword: []byte("NewPass123!"),
					Then:        "invalid-then-url",
				}
				body, err := json.Marshal(reqBody)
				Expect(err).ToNot(HaveOccurred())

				req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				location := ctx.Recorder.Header().Get("Location")
				Expect(location).To(ContainSubstring("error="))
				Expect(location).To(ContainSubstring(url.QueryEscape(fuyaoerrors.ErrStrInvalidThen)))
			})
		})
	})

	Describe("DELETE /oauth2/auth/password/confirm/fuyaoPasswordProvider", func() {
		Context("认证鉴权-037-CONFIRM-DELETE: Cancel password confirmation", func() {
			It("should clear session and redirect to root", func() {
				req, err := http.NewRequest("DELETE", constants.FuyaoPasswordConfirmEndpoint, nil)
				Expect(err).ToNot(HaveOccurred())

				ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
				Expect(ctx.Recorder.Header().Get("Location")).To(Equal("/"))

				// Verify session cookie is set (to clear the session)
				cookies := ctx.Recorder.Result().Cookies()
				Expect(cookies).To(HaveLen(1))
				Expect(cookies[0].Name).To(Equal("idpLogin"))
			})
		})
	})

	Describe("Password complexity comprehensive tests", func() {
		BeforeEach(func() {
			mockUser := CreateMockUserInfo(ctx.FirstLoginUser)
			ctx.SetupUserInfoMock(mockUser)
			ctx.SetupSession(CreateLoggedInSession(ctx.FirstLoginUser.Username, true))
		})

		Context("认证鉴权-035-CONFIRM-POST: Valid passwords with all complexity requirements", func() {
			DescribeTable("should accept passwords meeting all requirements",
				func(password string) {
					reqBody := fuyaopassword.PasswordConfirmRequest{
						NewPassword: []byte(password),
						Then:        ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					Expect(ctx.Recorder.Header().Get("Location")).To(Equal(ctx.ValidThenURL))
				},
				// Basic valid combinations
				Entry("lowercase + number + special", "password123!"),
				Entry("uppercase + number + special", "PASSWORD123!"),
				Entry("mixed case + number + special", "PassWord123!"),
				Entry("multiple special chars", "pass123!@#"),
				Entry("long valid password", "MySecurePassword123!"),

				// Additional valid combinations from idpfuyaopassword
				Entry("mixed case + digits + multiple special", "Pa123!@#Abc"),
				Entry("hello world pattern", "HelloWorld123!"),
				Entry("minimum length 8 with all requirements", "a1!bcdef"),
				Entry("near maximum length with all requirements", "z1!ABCDEFGHIJKLMNOPQRSTUVWX"),
			)
		})

		Context("认证鉴权-034-036-CONFIRM-POST: Invalid passwords edge cases", func() {
			DescribeTable("should reject passwords missing requirements",
				func(password string, reason string) {
					reqBody := fuyaopassword.PasswordConfirmRequest{
						NewPassword: []byte(password),
						Then:        ctx.ValidThenURL,
					}
					body, err := json.Marshal(reqBody)
					Expect(err).ToNot(HaveOccurred())

					req, err := ctx.NewRequestWithSession("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(body))
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Content-Type", "application/json")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordConfirmHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusFound))
					location := ctx.Recorder.Header().Get("Location")
					Expect(location).To(ContainSubstring("error="), "Should have error for: %s", reason)
				},
				// Basic invalid cases
				Entry("only letters", "passwordonly", "missing numbers and special chars"),
				Entry("only numbers", "12345678", "missing letters and special chars"),
				Entry("only special", "!@#$%^&*", "missing letters and numbers"),
				Entry("very short", "Pa1!", "too short"),

				// CONFIRM-POST-012: Whitespace edge cases
				Entry("whitespace only - spaces", "        ", "only spaces"),
				Entry("whitespace only - 3 spaces", "   ", "only 3 spaces"),

				// Additional invalid cases from idpfuyaopassword
				Entry("letters only - no digits no special", "password", "missing digits and special"),
				Entry("digits only - no letters no special", "12345678", "missing letters and special"),
				Entry("special only - no letters no digits", "!@#$%^&*", "missing letters and digits"),
				Entry("letters + digits - no special", "pass1234", "lowercase + digits, no special"),
				Entry("letters + digits - no special (uppercase)", "PASS1234", "uppercase + digits, no special"),
				Entry("letters + special - no digits", "pass!@#$", "letters + special, no digits"),
				Entry("digits + special - no letters", "1234!@#$", "digits + special, no letters"),
				Entry("too short with all types", "Pass", "too short even with mixed types"),
				Entry("way too long", "Pass123456789012345678901234567890123", "exceeds 32 chars"),
			)
		})
	})
})
