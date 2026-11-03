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
 *
 * NEW COPYRIGHT AND LICENSE
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 * functions modified:
 * type SessionState struct
 * func (s *SessionState) EncodeSessionInfo()
 * func DecodeSessionState
 * func (s *SessionState) EncryptedString
 */

package providers

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"openfuyao/oauth-proxy/cookie"
)

type SessionState struct {
	AccessToken  string
	ExpiresOn    time.Time
	RefreshToken string
	Email        string
	User         string
	SessionID    string
}

func (s *SessionState) IsExpired() bool {
	if !s.ExpiresOn.IsZero() && s.ExpiresOn.Before(time.Now()) {
		return true
	}
	return false
}

func (s *SessionState) String() string {
	o := fmt.Sprintf("Session{%s", s.userOrEmail())
	if s.AccessToken != "" {
		o += " token:true"
	}
	if !s.ExpiresOn.IsZero() {
		o += fmt.Sprintf(" expires:%s", s.ExpiresOn)
	}
	if s.RefreshToken != "" {
		o += " refresh_token:true"
	}
	return o + "}"
}

func (s *SessionState) EncodeSessionState(c *cookie.Cipher) (string, error) {
	if c == nil || s.AccessToken == "" {
		return s.EncodeSessionInfo(), nil
	}
	return s.EncryptedString(c)
}

func (s *SessionState) userOrEmail() string {
	u := s.User
	if s.Email != "" {
		u = s.Email
	}
	return u
}

// EncodeSessionInfo encodes the sessionInfo
func (s *SessionState) EncodeSessionInfo() string {
	return fmt.Sprintf("%s|%d", s.SessionID, s.ExpiresOn.Unix())
}

func (s *SessionState) EncryptedString(c *cookie.Cipher) (string, error) {
	var err error
	if c == nil {
		panic("error. missing cipher")
	}
	a := s.AccessToken
	if a != "" {
		a, err = c.Encrypt(a)
		if err != nil {
			return "", err
		}
	}
	r := s.RefreshToken
	if r != "" {
		r, err = c.Encrypt(r)
		if err != nil {
			return "", err
		}
	}
	sessID := s.SessionID
	if sessID != "" {
		sessID, err = c.Encrypt(sessID)
		if err != nil {
			return "", err
		}
	}
	return fmt.Sprintf("%s|%s|%d|%s|%s", s.userOrEmail(), a, s.ExpiresOn.Unix(), r, sessID), nil
}

// DecodeSessionState decodes the session state
func DecodeSessionState(v string, c *cookie.Cipher) (*SessionState, error) {
	var s = &SessionState{}
	var err error
	chunks := strings.Split(v, "|")
	const MinChunk = 1
	const MidChunk = 2
	const LargeChunk = 5

	// len(chunks) == 1 means only sessionID is returned
	if len(chunks) == MinChunk {
		return &SessionState{SessionID: v}, nil
	}

	// len(chunks) == 2 means only sessionID + expiresOn is returned
	if len(chunks) == MidChunk {
		ts, err := strconv.Atoi(chunks[1])
		if err != nil {
			return nil, fmt.Errorf("invalid expires_on timestamp %q: %v", chunks[1], err)
		}
		s.ExpiresOn = time.Unix(int64(ts), 0)
		s.SessionID = chunks[0]
		return s, nil
	}

	if len(chunks) != LargeChunk {
		err = fmt.Errorf("invalid number of fields (got %d expected 4)", len(chunks))
		return nil, err
	}

	if c != nil && chunks[1] != "" {
		s.AccessToken, err = c.Decrypt(chunks[1])
		if err != nil {
			return nil, err
		}
	}
	if c != nil && chunks[3] != "" {
		s.RefreshToken, err = c.Decrypt(chunks[3])
		if err != nil {
			return nil, err
		}
	}
	if c != nil && chunks[4] != "" {
		s.SessionID, err = c.Decrypt(chunks[4])
		if err != nil {
			return nil, err
		}
	}
	if u := chunks[0]; strings.Contains(u, "@") {
		s.Email = u
		s.User = strings.Split(u, "@")[0]
	} else {
		s.User = u
	}
	ts, _ := strconv.Atoi(chunks[2])
	s.ExpiresOn = time.Unix(int64(ts), 0)
	return s, nil
}
