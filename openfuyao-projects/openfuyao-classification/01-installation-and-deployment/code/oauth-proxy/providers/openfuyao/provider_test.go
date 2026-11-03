/*
 * Copyright OAuth2_proxy Authors.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://opensource.org/licenses/MIT
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

// Package openfuyao defines the openfuyao provider configs
package openfuyao

import (
	"context"
	"net/http"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

type mockAuthRequestHandler struct {
}

type mockAuthorizer struct {
}

// if you're seeing cert expiration fuyaoerrors on 'Nov  3 11:57:34 2119 GMT', I am sorry
const longLivedCACert = `
-----BEGIN CERTIFICATE-----
MIIFjjCCA3agAwIBAgIUYICrP1shKbhgEbQsmHdf64W7hGwwDQYJKoZIhvcNAQEN
BQAwTzELMAkGA1UEBhMCQ1oxEDAOBgNVBAgMB01vcmF2aWExHDAaBgNVBAoME015
IFByaXZhdGUgT3JnIEx0ZC4xEDAOBgNVBAMMB1Rlc3QgQ0EwIBcNMTkxMDA4MTE1
NzMzWhgPMjExOTExMDMxMTU3MzNaME8xCzAJBgNVBAYTAkNaMRAwDgYDVQQIDAdN
b3JhdmlhMRwwGgYDVQQKDBNNeSBQcml2YXRlIE9yZyBMdGQuMRAwDgYDVQQDDAdU
ZXN0IENBMIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAqGrcAHxo2Iiu
jNABMdasP0lHRiV3m6DGmDFGEWI9A5s4hSL+2Nh9Hnu1bmCqmm88EB8wQBxgte08
hhxtamFHhqTvsr2zvZIinPI+ntgHuKWH2fKVNmHUA0/DfA51yPppRZXws2J2OhwG
VBfmztV6StSWP5HuCbujGnuMG37+CEiOqqR8nfvwtXhebEYCEGcRJmPQLWZuhohh
7Ie/M6auSQS29Xnezy/6To1V7kMuBwKq+ywTftfNiWRTRRAtx5+cd5EeZf8svO5z
WSYWQK+OzyjqCTwYDmm5WhHid112jsjhNMHVM8mL9za4E7zgZBYBRSkKiM5UVWTs
Lb6kO3FkIlQzqt9eSYzZfcQxUfuSOKviubtNghGI2TmoElcbgIIZ0zVBxa5k4DMY
Hr36B+PggXPbzF+pxAMpmR0qYKth6mGW6SJZTXdjwEbFSRE+zrpcttCGJgQsseTl
hV2BCyVq8aDvmMKh63sGAkalK1TmqNRplFuohSFW523Ilm2I93EF0/L4pRQ7+KZ3
8+tFvrv1XswX0wWMNnsrUVIkvmsX2olZgvlN/taqovgTvC0zcO7EopDDveXMMLRY
C3wPP222sJ5wOGpT+m8HmddNaVWuW/9MzOgAEr4kuFlQUcvGdP/Z3IUgp8cVrjM7
g6wVyVvguWE0a2q8xLw6Y3CKp5bLHh8CAwEAAaNgMF4wHQYDVR0OBBYEFNb1bu9A
OeRUWyN15uG/aIBtIgyTMB8GA1UdIwQYMBaAFNb1bu9AOeRUWyN15uG/aIBtIgyT
MA8GA1UdEwEB/wQFMAMBAf8wCwYDVR0PBAQDAgEGMA0GCSqGSIb3DQEBDQUAA4IC
AQACR2hSEMqlkFZ7RX/csZgpMt4E5z0TJZ7Uny+yKV/ibIFcy7sfU2bXnZX63Sdl
do89DkVTqI4T48byvF8KQ+pHr4ow5nvA2rigmQEySrSBT9GseZm9XIFy/Sb4vUml
dXYcmeJYNVgGAOspwrFg8mJ8a+afkBArSJyNLIemv+P2Bb4fChUhpoVt3XngJJJZ
5SxvF9g++0ZaDEse80wHCaHlgeh48Yo0SczNHv5lJ5uQzNIjxBEad/4P02Uj7wXf
J8TX3NK15P+Iwvf+UY8odtjIsLMd2KltaJ7P4MqTAS+b7Xb9i0CZgEtnCG3Fup8b
xM5S9S55qLUNUQtolNs2jxSnMGOciG3G/sdcl/qbiQZchvKvYZp8Q8NnavBIcRkQ
mZ0P2BPrg6rfofaNvOpTz+NeaWDFfQzC7+2QnfiiIOL8le7b4lOjmLyCfZaNW8WN
PlYMGYA460xdn/IWPJcLCdt3rNw+CKZCw4pxZvUWqzRnCrNkM4zA7JgLn7M9Vx1l
3q4sUFMZuUjWIxACwk9u/U4sc2rLYelwHhg/2j0hUoqbDhyHRYUVruptwRSebE2U
KvcuxUCTIws0kHzgUX6qT6gDFKDl9A+EgIcusosjUNIjLUsgUPs6THNvQadMEEV7
w9aR8p+EwE+/BERIzwURZmyINWafvMjVGNHCKC1w7AhFEA==
-----END CERTIFICATE-----
`

func TestParseSubjectAccessReviews(t *testing.T) {

	tests := []struct {
		sar            string
		expectedResult []string
	}{
		{
			sar: `{"foo":"bar"}`,
			expectedResult: []string{
				`{"foo":"bar","scopes":[]}`,
			},
		},
		{
			sar: `[{"foo":"bar"}, {"baz":"bad"}]`,
			expectedResult: []string{
				`{"foo":"bar","scopes":[]}`,
				`{"baz":"bad","scopes":[]}`,
			},
		},
	}

	for _, test := range tests {
		result, err := parseSubjectAccessReviews(test.sar)
		if err != nil {
			t.Fatalf("unexpected error %s", err.Error())
		}
		if !reflect.DeepEqual(result, test.expectedResult) {
			t.Fatalf("expected %v, got %v", test.expectedResult, result)
		}
	}
}

func (mock *mockAuthRequestHandler) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	return &authenticator.Response{User: &user.DefaultInfo{Name: "username", UID: "uid"}}, true, nil
}

func (mock *mockAuthorizer) Authorize(
	ctx context.Context,
	record authorizer.Attributes,
) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "", nil
}

func TestPassOAuthToken(t *testing.T) {
	req, _ := http.NewRequest("GET", "/someurl", nil)
	req.Header.Set("Authorization", "Bearer this-is-the-token")
	p := &FuyaoProvider{}
	p.paths = recordsByPath{
		pathRecord{
			path:   "/someurl",
			record: authorizer.AttributesRecord{},
		},
	}
	p.authenticator = &mockAuthRequestHandler{}
	p.authorizer = &mockAuthorizer{}

	session, err := p.ValidateRequest(req)
	if err != nil {
		t.Fatalf("failed to validate request %s", err.Error())
	}
	if session == nil {
		t.Fatal("failed to validate request, no session received")
	}
	if g, e := session.AccessToken, "this-is-the-token"; g != e {
		t.Errorf("access token not set in session to expected value: %v", session)
	}
}

func TestDontPassBasicAuthentication(t *testing.T) {
	req, _ := http.NewRequest("GET", "/someurl", nil)
	req.Header.Set("Authorization", "Basic dXNlcm5hbWU6cGFzc3dvcmQK")
	p := &FuyaoProvider{}
	p.authenticator = &mockAuthRequestHandler{}
	p.authorizer = &mockAuthorizer{}
	p.paths = recordsByPath{
		pathRecord{
			path:   "/someurl",
			record: authorizer.AttributesRecord{},
		},
	}

	session, err := p.ValidateRequest(req)
	if err != nil {
		t.Fatalf("failed to validate request %s", err.Error())
	}
	if session == nil {
		t.Fatal("failed to validate request, no session received")
	}
	if g, e := session.AccessToken, ""; g != e {
		t.Errorf("access token should be empty string for basic authentication: %v", session)
	}
}
