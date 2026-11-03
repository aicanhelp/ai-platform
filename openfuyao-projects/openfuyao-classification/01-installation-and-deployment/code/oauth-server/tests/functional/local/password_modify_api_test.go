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
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
)

var _ = Describe("Password Modify API", func() {
	var ctx *TestContext

	BeforeEach(func() {
		ctx = NewTestContext()
		ctx.SetupLoginHandler(true) // Enable TokenReview mock
	})

	AfterEach(func() {
		ctx.ResetPatches()
	})

	// Helper function to create password reset request
	createPasswordResetRequest := func(username, oldPassword, newPassword string) *http.Request {
		reqBody := fuyaopassword.PasswordResetRequest{
			Username:         username,
			OriginalPassword: []byte(oldPassword),
			NewPassword:      []byte(newPassword),
		}
		body, _ := json.Marshal(reqBody)

		req, _ := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+TestValidToken)
		return req
	}

	Describe("POST /oauth2/auth/password/modify/fuyaoPasswordProvider", func() {
		Context("认证鉴权-038-MODIFY-POST: Successful password modification", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should return 200 with success message", func() {
				req := createPasswordResetRequest(ctx.TestUser.Username, ctx.TestUser.Password, "NewPass123!")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusOK))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring("Password Reset OK"))
			})
		})

		Context("认证鉴权-039-MODIFY-POST: Missing Authorization header", func() {
			It("should return 401 Unauthorized", func() {
				reqBody := fuyaopassword.PasswordResetRequest{
					Username:         ctx.TestUser.Username,
					OriginalPassword: []byte(ctx.TestUser.Password),
					NewPassword:      []byte("NewPass123!"),
				}
				body, _ := json.Marshal(reqBody)

				req, _ := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				// No Authorization header

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrNotLogin))
			})
		})

		Context("认证鉴权-040-MODIFY-POST: Invalid Bearer Token", func() {
			It("should return 401 for invalid token", func() {
				reqBody := fuyaopassword.PasswordResetRequest{
					Username:         ctx.TestUser.Username,
					OriginalPassword: []byte(ctx.TestUser.Password),
					NewPassword:      []byte("NewPass123!"),
				}
				body, _ := json.Marshal(reqBody)

				req, _ := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				// Use an invalid token format (not "Bearer ...")
				req.Header.Set("Authorization", "InvalidToken")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrNotLogin))
			})
		})

		Context("认证鉴权-041-MODIFY-POST: Expired Token", func() {
			It("should return 401 for expired token", func() {
				// Create a fresh context with expired token reactor
				expiredCtx := NewTestContext()
				expiredCtx.SetupLoginHandlerWithExpiredToken()

				reqBody := fuyaopassword.PasswordResetRequest{
					Username:         expiredCtx.TestUser.Username,
					OriginalPassword: []byte(expiredCtx.TestUser.Password),
					NewPassword:      []byte("NewPass123!"),
				}
				body, _ := json.Marshal(reqBody)

				req, _ := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Authorization", "Bearer expired-token")

				expiredCtx.LoginHandler.PasswordResetHandler(expiredCtx.Recorder, req)

				Expect(expiredCtx.Recorder.Code).To(Equal(http.StatusUnauthorized))
			})
		})

		Context("认证鉴权-042-MODIFY-POST: Wrong original password", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should return 401 with authentication error", func() {
				req := createPasswordResetRequest(ctx.TestUser.Username, "wrongpassword", "NewPass123!")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
				body := ctx.Recorder.Body.String()
				Expect(body).To(ContainSubstring("用户名或密码错误"))
			})
		})

		Context("认证鉴权-043-MODIFY-POST: Original password case sensitivity error", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			DescribeTable("should fail for case-mismatched original passwords",
				func(wrongPassword string) {
					req := createPasswordResetRequest(ctx.TestUser.Username, wrongPassword, "NewPass123!")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
					body := ctx.Recorder.Body.String()
					Expect(body).To(ContainSubstring("用户名或密码错误"))
				},
				Entry("lowercase soup", "soup4@LL"), // Original is Soup4@LL
				Entry("uppercase SOUP", "SOUP4@LL"), // Original is Soup4@LL
				Entry("mixed case", "sOuP4@ll"),     // Original is Soup4@LL
			)
		})

		Context("认证鉴权-054-057-MODIFY-POST: Original password edge cases", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			DescribeTable("should fail for various original password errors",
				func(wrongPassword string, description string) {
					req := createPasswordResetRequest(ctx.TestUser.Username, wrongPassword, "NewPass123!")

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized),
						"Failed for: %s (%s)", wrongPassword, description)
					body := ctx.Recorder.Body.String()
					Expect(body).To(ContainSubstring("用户名或密码错误"))
				},
				// MODIFY-POST-017: Original password missing special character
				Entry("missing special char", "Soup4LL", "missing @ symbol"),
				Entry("wrong special char", "Soup4#LL", "wrong special character"),

				// MODIFY-POST-018: Original password too short
				Entry("too short - missing one char", "Sop4@LL", "missing 'u'"),
				Entry("too short - missing two chars", "So4@LL", "missing 'ou'"),

				// MODIFY-POST-019: Original password too long
				Entry("too long - extra chars", "Soup4@LLLLLLLLLLLLLLLLLLLL", "many extra L's"),
				Entry("too long - doubled", "Soup4@LLSoup4@LL", "password doubled"),

				// MODIFY-POST-020: Original password wrong digit
				Entry("wrong digit - 5 instead of 4", "Soup5@LL", "digit 5 instead of 4"),
				Entry("wrong digit - 0 instead of 4", "Soup0@LL", "digit 0 instead of 4"),
				Entry("extra digit", "Soup44@LL", "extra digit"),
			)
		})

		Context("认证鉴权-044-MODIFY-POST: Empty username", func() {
			It("should return 400 Bad Request", func() {
				req := createPasswordResetRequest("", ctx.TestUser.Password, "NewPass123!")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrUsernameOrPasswordMissing))
			})
		})

		Context("认证鉴权-045-MODIFY-POST: Empty original password", func() {
			It("should return 400 Bad Request", func() {
				req := createPasswordResetRequest(ctx.TestUser.Username, "", "NewPass123!")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrUsernameOrPasswordMissing))
			})
		})

		Context("认证鉴权-046-MODIFY-POST: Empty new password", func() {
			It("should return 400 Bad Request", func() {
				req := createPasswordResetRequest(ctx.TestUser.Username, ctx.TestUser.Password, "")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrUsernameOrPasswordMissing))
			})
		})

		Context("Password complexity validation for new password", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			DescribeTable("认证鉴权-047-049-MODIFY-POST: Should reject invalid new passwords",
				func(newPassword string, description string) {
					req := createPasswordResetRequest(ctx.TestUser.Username, ctx.TestUser.Password, newPassword)

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest),
						"Failed for: %s (%s)", newPassword, description)
					Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrPasswordTooWeak))
				},
				// MODIFY-POST-010: New password too short
				Entry("7 chars", "Pass12!", "length 7 - too short"),

				// MODIFY-POST-011: New password too long
				Entry("33 chars", "passwordwith123and@specialchar33!", "length 33 - too long"),

				// MODIFY-POST-012: Missing complexity
				Entry("no special", "password123", "missing special char"),
				Entry("no number", "password!@", "missing number"),
				Entry("no letter", "12345678!", "missing letter"),
			)

			DescribeTable("Should accept valid new passwords",
				func(newPassword string, description string) {
					req := createPasswordResetRequest(ctx.TestUser.Username, ctx.TestUser.Password, newPassword)

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusOK),
						"Failed for: %s (%s)", newPassword, description)
					Expect(ctx.Recorder.Body.String()).To(ContainSubstring("Password Reset OK"))
				},
				Entry("8 chars valid", "pass12!A", "minimum length valid"),
				Entry("32 chars valid", "passwordwith123and@specialchar3", "maximum length valid"),
				Entry("mixed case", "NewPass123!", "typical valid password"),
			)
		})

		Context("认证鉴权-050-MODIFY-POST: HTTP method not allowed", func() {
			DescribeTable("should reject non-POST methods",
				func(method string) {
					req, err := http.NewRequest(method, constants.FuyaoPasswordModifyEndpoint, nil)
					Expect(err).ToNot(HaveOccurred())
					req.Header.Set("Authorization", "Bearer "+TestValidToken)

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					Expect(ctx.Recorder.Code).To(Equal(http.StatusMethodNotAllowed))
					Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrRequestMethodNotAllowed))
				},
				Entry("GET method", "GET"),
				Entry("PUT method", "PUT"),
				Entry("DELETE method", "DELETE"),
				Entry("PATCH method", "PATCH"),
			)
		})

		Context("认证鉴权-051-MODIFY-POST: Invalid request body format", func() {
			It("should return 400 for invalid JSON", func() {
				req, err := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint,
					strings.NewReader("invalid json body"))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Authorization", "Bearer "+TestValidToken)

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusBadRequest))
				Expect(ctx.Recorder.Body.String()).To(ContainSubstring(fuyaoerrors.ErrStrFailToUnmarshalData))
			})
		})

		Context("认证鉴权-052-MODIFY-POST: Non-existent user", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should return 401 for non-existent user", func() {
				req := createPasswordResetRequest("nonexistentuser", ctx.TestUser.Password, "NewPass123!")

				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
			})
		})

		Context("认证鉴权-053-MODIFY-POST: User lockout after multiple failed attempts", func() {
			BeforeEach(func() {
				mockUser := CreateMockUserInfo(ctx.TestUser)
				ctx.SetupUserInfoMock(mockUser)
			})

			It("should lock user after multiple failed password attempts", func() {
				// Simulate multiple failed attempts
				for i := 0; i < TestFailTimes; i++ {
					req := createPasswordResetRequest(ctx.TestUser.Username, "wrongpassword", "NewPass123!")

					ctx.Recorder = httptest.NewRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)
				}

				// Next attempt should show reduced remaining attempts or locked status
				req := createPasswordResetRequest(ctx.TestUser.Username, "wrongpassword", "NewPass123!")

				ctx.Recorder = httptest.NewRecorder()
				ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

				// Should indicate lockout or reduced attempts
				body := ctx.Recorder.Body.String()
				Expect(body).To(Or(
					ContainSubstring("锁定"),
					ContainSubstring("locked"),
					ContainSubstring("输错"),
				))
			})
		})
	})

	Describe("Authorization header validation", func() {
		Context("Various Authorization header formats", func() {
			DescribeTable("should handle different authorization header formats",
				func(authHeader string, shouldFail bool) {
					reqBody := fuyaopassword.PasswordResetRequest{
						Username:         ctx.TestUser.Username,
						OriginalPassword: []byte(ctx.TestUser.Password),
						NewPassword:      []byte("NewPass123!"),
					}
					body, _ := json.Marshal(reqBody)

					req, _ := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewReader(body))
					req.Header.Set("Content-Type", "application/json")
					if authHeader != "" {
						req.Header.Set("Authorization", authHeader)
					}

					ctx.ResetRecorder()
					ctx.LoginHandler.PasswordResetHandler(ctx.Recorder, req)

					if shouldFail {
						Expect(ctx.Recorder.Code).To(Equal(http.StatusUnauthorized))
					}
				},
				Entry("empty header", "", true),
				Entry("missing Bearer prefix", TestValidToken, true),
				Entry("lowercase bearer", "bearer "+TestValidToken, true),
				Entry("valid Bearer format", "Bearer "+TestValidToken, false),
			)
		})
	})
})
