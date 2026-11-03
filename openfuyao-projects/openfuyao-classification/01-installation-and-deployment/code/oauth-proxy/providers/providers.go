package providers

import (
	"errors"
	"net/http"

	"openfuyao/oauth-proxy/cookie"
)

type Provider interface {
	Data() *ProviderData
	GetEmailAddress(*SessionState) (string, error)
	GetUserName(accessToken string) (string, error)
	Redeem(string, string, string) (*SessionState, error)
	ValidateGroup(string) bool
	ValidateSessionState(*SessionState) bool
	GetLoginURL(redirectURI, state string) string
	RefreshSessionIfNeeded(*SessionState) (bool, error)
	SessionFromCookie(string, *cookie.Cipher) (*SessionState, error)
	CookieForSession(*SessionState, *cookie.Cipher) (string, error)

	ReviewUser(name, host string) error
	ValidateRequest(*http.Request) (*SessionState, error)
	ValidateMultiClusterRequest(*http.Request) (*SessionState, error)
	ValidateSessionAuth(*SessionState, *http.Request) (bool, error)
	DeleteSession(string) error

	SetRedeemURL(req *http.Request, scheme string)
	SetLoginURL(req *http.Request, scheme string)
}

// ErrPermissionDenied may be returned from Redeem() to indicate the user is not allowed to login.
var ErrPermissionDenied = errors.New("permission denied")
