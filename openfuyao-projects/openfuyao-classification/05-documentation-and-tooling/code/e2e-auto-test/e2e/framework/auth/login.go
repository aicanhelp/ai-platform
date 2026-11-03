package auth

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"strings"
	"time"

	"golang.org/x/net/html"
)

// NewInsecureClient 返回带 cookie jar、跳过 TLS 校验、不自动跟随重定向的 HTTP 客户端。
func NewInsecureClient() (*http.Client, error) {
	jar, err := cookiejar.New(nil)
	if err != nil {
		return nil, err
	}
	return &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
		Timeout: 30 * time.Second,
		Jar:     jar,
	}, nil
}

// BaseURLFromEnv 解析控制台入口。优先 TEST_FUYAOURL，否则 https://TEST_NODE2_IP:31616。
func BaseURLFromEnv() string {
	if u := strings.TrimRight(strings.TrimSpace(os.Getenv("TEST_FUYAOURL")), "/"); u != "" {
		return u
	}
	if ip := strings.TrimSpace(os.Getenv("TEST_NODE2_IP")); ip != "" {
		return "https://" + ip + ":31616"
	}
	return ""
}

// Login 走 OAuth 密码登录链：/rest/auth/login → authorize → fuyaoPasswordProvider → callback → /。
// client 须带 cookie jar 且 CheckRedirect 为 ErrUseLastResponse（见 NewInsecureClient）。
func Login(client *http.Client, baseURL, username, password string) error {
	baseURL = strings.TrimRight(baseURL, "/")
	resp, err := client.Get(baseURL + "/rest/auth/login")
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusFound, "GET /rest/auth/login"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	loc := resp.Header.Get("Location")
	_ = resp.Body.Close()
	if err := checkPrefix(loc, "/oauth2/oauth/authorize", "login"); err != nil {
		return err
	}

	resp, err = follow(client, baseURL, loc)
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusFound, "GET authorize"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	loc = resp.Header.Get("Location")
	_ = resp.Body.Close()
	if err := checkPrefix(loc, "/oauth2/auth/login/fuyaoPasswordProvider", "authorize"); err != nil {
		return err
	}

	resp, err = follow(client, baseURL, loc)
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusOK, "GET password provider"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	body, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return err
	}
	csrfToken, thenValue, err := extractLoginFormData(string(body))
	if err != nil {
		return err
	}

	passwordInts := make([]int, 0, len(password))
	for _, char := range password {
		passwordInts = append(passwordInts, int(char))
	}
	jsonData, err := json.Marshal(map[string]interface{}{
		"username": username,
		"password": passwordInts,
		"then":     thenValue,
	})
	if err != nil {
		return err
	}
	loginURL, err := url.Parse(baseURL + "/oauth2/auth/login/fuyaoPasswordProvider")
	if err != nil {
		return err
	}
	loginReq := &http.Request{
		Method: "POST",
		URL:    loginURL,
		Header: http.Header{
			"Content-Type": {"application/json"},
			"X-CSRF-Token": {csrfToken},
			"Referer":      {loc},
		},
		Body: io.NopCloser(bytes.NewReader(jsonData)),
	}
	resp, err = client.Do(loginReq)
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusFound, "POST password login"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	loc = resp.Header.Get("Location")
	_ = resp.Body.Close()
	if err := checkPrefix(loc, "/oauth2/oauth/authorize", "post login"); err != nil {
		return err
	}

	resp, err = follow(client, baseURL, loc)
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusFound, "GET authorize after login"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	loc = resp.Header.Get("Location")
	_ = resp.Body.Close()
	if err := checkPrefix(loc, "/rest/auth/callback", "callback"); err != nil {
		return err
	}

	resp, err = follow(client, baseURL, loc)
	if err != nil {
		return err
	}
	if err := checkStatus(resp, http.StatusFound, "GET callback"); err != nil {
		_ = resp.Body.Close()
		return err
	}
	loc = resp.Header.Get("Location")
	_ = resp.Body.Close()
	if loc != "/" {
		return fmt.Errorf("callback location %q want /", loc)
	}

	resp, err = follow(client, baseURL, loc)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if err := checkStatus(resp, http.StatusOK, "GET /"); err != nil {
		return err
	}
	home, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if !strings.Contains(string(home), `<div id="root"></div>`) {
		return fmt.Errorf("home page missing root element")
	}
	return nil
}

// Logout 调用 /rest/auth/logout，期望 204。
func Logout(client *http.Client, baseURL string) error {
	if client == nil {
		return nil
	}
	resp, err := client.Post(strings.TrimRight(baseURL, "/")+"/rest/auth/logout", "application/json", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("POST /rest/auth/logout: status %d want %d", resp.StatusCode, http.StatusNoContent)
	}
	return nil
}

func extractLoginFormData(htmlContent string) (csrfToken, thenValue string, err error) {
	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return "", "", err
	}
	var findInputs func(*html.Node)
	findInputs = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "input" {
			var name, value string
			for _, attr := range n.Attr {
				switch attr.Key {
				case "name":
					name = attr.Val
				case "value":
					value = attr.Val
				}
			}
			switch name {
			case "gorilla.csrf.Token":
				csrfToken = value
			case "then":
				thenValue = value
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			findInputs(c)
		}
	}
	findInputs(doc)
	if csrfToken == "" {
		return "", "", fmt.Errorf("CSRF token not found in HTML")
	}
	if thenValue == "" {
		return "", "", fmt.Errorf("then parameter not found in HTML")
	}
	return csrfToken, thenValue, nil
}

func follow(client *http.Client, base, loc string) (*http.Response, error) {
	if strings.HasPrefix(loc, "http://") || strings.HasPrefix(loc, "https://") {
		return client.Get(loc)
	}
	return client.Get(base + loc)
}

func checkStatus(resp *http.Response, want int, step string) error {
	if resp.StatusCode != want {
		return fmt.Errorf("%s: status %d want %d loc=%s", step, resp.StatusCode, want, resp.Header.Get("Location"))
	}
	return nil
}

func checkPrefix(loc, prefix, step string) error {
	if !strings.HasPrefix(loc, prefix) {
		return fmt.Errorf("%s: location %q want prefix %q", step, loc, prefix)
	}
	return nil
}
