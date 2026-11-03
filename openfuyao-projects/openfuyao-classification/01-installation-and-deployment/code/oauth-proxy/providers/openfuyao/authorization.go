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
 * function modified:
 * func (s *DelegatingAuthorizationOptions) newSubjectAccessReview()
 */

// Package openfuyao defines the openfuyao provider configs
package openfuyao

// this file is partially copied from k8s.io/apiserver/pkg/server/options
import (
	"flag"
	"time"

	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/kubernetes/typed/authorization/v1"

	"openfuyao/oauth-proxy/constants"
)

// DelegatingAuthorizationOptions provides an easy way for composing API servers to delegate their authorization to
// the root kube API server
type DelegatingAuthorizationOptions struct {
	// RemoteKubeConfigFile is the file to use to connect to a "normal" kube API server which hosts the
	// SubjectAccessReview.authorization.k8s.io endpoint for checking tokens.
	RemoteKubeConfigFile string

	// AllowCacheTTL is the length of time that a successful authorization response will be cached
	AllowCacheTTL time.Duration

	// DenyCacheTTL is the length of time that an unsuccessful authorization response will be cached.
	// You generally want more responsive, "deny, try again" flows.
	DenyCacheTTL time.Duration
}

// NewDelegatingAuthorizationOptions inits DelegatingAuthorizationOptions
func NewDelegatingAuthorizationOptions() *DelegatingAuthorizationOptions {
	return &DelegatingAuthorizationOptions{
		// very low for responsiveness, but high enough to handle storms
		AllowCacheTTL: constants.AuthzAllowCacheTTL * time.Second,
		DenyCacheTTL:  constants.AuthzDenyCacheTTL * time.Second,
	}
}

// Validate validates authzOption
func (s *DelegatingAuthorizationOptions) Validate() []error {
	var allErrors []error
	return allErrors
}

// AddFlags add necessary configs for openFuyao authz client
func (s *DelegatingAuthorizationOptions) AddFlags(fs *flag.FlagSet) {
	fs.StringVar(&s.RemoteKubeConfigFile, "authorization-kubeconfig", s.RemoteKubeConfigFile, ""+
		"kubeconfig file pointing at the 'core' kubernetes server with enough rights to create "+
		" subjectaccessreviews.authorization.k8s.io.")

	fs.DurationVar(&s.AllowCacheTTL, "authorization-webhook-cache-authorized-ttl",
		s.AllowCacheTTL,
		"The duration to cache 'authorized' responses from the webhook authorizer.")

	fs.DurationVar(&s.DenyCacheTTL,
		"authorization-webhook-cache-unauthorized-ttl", s.DenyCacheTTL,
		"The duration to cache 'unauthorized' responses from the webhook authorizer.")
}

// ToAuthorizationConfig generate the authz webhook config
func (s *DelegatingAuthorizationOptions) ToAuthorizationConfig() (authorizerfactory.DelegatingAuthorizerConfig, error) {
	sarClient, err := s.newSubjectAccessReview()
	if err != nil {
		return authorizerfactory.DelegatingAuthorizerConfig{}, err
	}

	ret := authorizerfactory.DelegatingAuthorizerConfig{
		SubjectAccessReviewClient: sarClient,
		AllowCacheTTL:             s.AllowCacheTTL,
		DenyCacheTTL:              s.DenyCacheTTL,
		WebhookRetryBackoff:       options.DefaultAuthWebhookRetryBackoff(),
	}
	return ret, nil
}

func (s *DelegatingAuthorizationOptions) newSubjectAccessReview() (v1.AuthorizationV1Interface, error) {
	config, err := GetClientConfig(s.RemoteKubeConfigFile)
	if err != nil {
		return nil, err
	}

	config.Wrap(auditIDRountripper)

	client, err := v1.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return client, nil
}
