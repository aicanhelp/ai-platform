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

package fuyao

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/authentication/v1"
)

// TestJWTValidatorValidate test the validation logic for JWT
func TestJWTValidatorValidate(t *testing.T) {
	type fields struct {
		signingKey []byte
	}
	type args struct {
		req v1.TokenReview
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *v1.TokenReview
		want1  bool
	}{
		{
			"wrong access token",
			fields{signingKey: []byte("i_am_the_secrets")},
			args{v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsImtpZCI6ImFjY2Vzc190b2tlbl9zaWduX2tle" +
					"SIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTc0NjAwMDM1OCwic3ViIjoiYWRtaW4if" +
					"Q.JbvABvQ0E-hKN5tKCMwSOAFzUcry_vQIpEWCZy4JidXXMO5PCoGVqeuhP9abcdar-3nXl39J2GTn5KFw7V1IzA"},
			}},
			&v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsImtpZCI6ImFjY2Vzc190b2tlbl9zaWduX2tle" +
					"SIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTc0NjAwMDM1OCwic3ViIjoiYWRtaW4if" +
					"Q.JbvABvQ0E-hKN5tKCMwSOAFzUcry_vQIpEWCZy4JidXXMO5PCoGVqeuhP9abcdar-3nXl39J2GTn5KFw7V1IzA"},
				Status: v1.TokenReviewStatus{
					Authenticated: false,
					User:          v1.UserInfo{},
					Error:         "signature is invalid",
				},
			},
			false,
		},
		{
			"access token expired",
			fields{signingKey: []byte("i_am_the_secrets")},
			args{v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsImtpZCI6ImFjY2Vzc190b2tlbl9zaWduX2t" +
					"leSIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTcxNDQ1NTM3MSwic3ViIjoiYWRtaW4ifQ." +
					"oMYX7t1UNUox6u5aQHK0jCYwFcqkbIYhB6JoHLYj1Iq_6CKwwnBbUaRLzlMzSrqoRlEVnaHxrWmA5_JyirhmWw"},
			}},
			&v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsImtpZCI6ImFjY2Vzc190b2tlbl9zaWduX2t" +
					"leSIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTcxNDQ1NTM3MSwic3ViIjoiYWRtaW4ifQ." +
					"oMYX7t1UNUox6u5aQHK0jCYwFcqkbIYhB6JoHLYj1Iq_6CKwwnBbUaRLzlMzSrqoRlEVnaHxrWmA5_JyirhmWw"},
				Status: v1.TokenReviewStatus{
					Authenticated: false,
					User:          v1.UserInfo{},
					Error:         "fail to validate access token through JWT",
				},
			},
			false,
		},
		{
			"successfully validate",
			fields{signingKey: []byte("a-valid-string-secret-that-is-at-least-512-bits-long-which-is-very-long")},
			args{v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV" +
					"4cCI6MTkyNDkyMDAwMCwic3ViIjoiYWRtaW4ifQ.49rrmIRdFXfhigt70993BThscDv6fZECMQYb6g1c9JPVpXYL7gh" +
					"KJ_OCOp8eKsisZBlgsm0jAXpo0ffJzfuaYQ"},
			}},
			&v1.TokenReview{
				Spec: v1.TokenReviewSpec{Token: "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsIm" +
					"V4cCI6MTkyNDkyMDAwMCwic3ViIjoiYWRtaW4ifQ.49rrmIRdFXfhigt70993BThscDv6fZECMQYb6g1c9JPVpXYL7ghKJ_O" +
					"COp8eKsisZBlgsm0jAXpo0ffJzfuaYQ"},
				Status: v1.TokenReviewStatus{
					Authenticated: true,
					User: v1.UserInfo{
						Username: "admin",
						UID:      "",
						Groups:   nil,
						Extra:    nil,
					},
					Audiences: nil,
					Error:     "",
				},
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &JWTValidator{
				signingKey: tt.fields.signingKey,
			}
			got, got1 := v.Validate(tt.args.req)
			if !reflect.DeepEqual(got, tt.want) {
				// the access token expiration time is returned by jwt.ParseWithClaims (dynamic)
				if tt.name == "access token expired" {
					if !strings.HasPrefix(got.Status.Error, "token is expired by ") {
						t.Errorf("Validate() got = %v, want %v", got, tt.want)
					}
				} else {
					t.Errorf("Validate() got = %v, want %v", got, tt.want)
				}
			}
			if got1 != tt.want1 {
				t.Errorf("Validate() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
