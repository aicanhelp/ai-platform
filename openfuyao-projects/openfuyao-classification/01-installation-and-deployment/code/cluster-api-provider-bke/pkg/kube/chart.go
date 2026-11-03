/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package kube

import (
	"context"
	"fmt"
	"io/ioutil"
	stdlog "log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/cli"
	"helm.sh/helm/v3/pkg/registry"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const releaseNotFound = "release: not found"

func (c *Client) installChartAddon(addon *confv1beta1.Product, addonOperate bkeaddon.AddonOperate,
	bkeClusterNS string, cfg bkeinit.BkeConfig, localClient client.Client) error {
	c.Log.Infof("starting %s chart addon %s", addonOperate, addon.Name)

	var err error
	switch addonOperate {
	case bkeaddon.CreateAddon:
		err = c.handleCreateChartOperation(addon, cfg, bkeClusterNS, localClient)
	case bkeaddon.UpdateAddon:
		err = c.handleUpgradeChartOperation(addon, cfg, bkeClusterNS, localClient)
	case bkeaddon.UpgradeAddon:
		err = c.handleUpgradeChartOperation(addon, cfg, bkeClusterNS, localClient)
	case bkeaddon.RemoveAddon:
		err = c.handleRemoveChartOperation(addon)
	default:
		c.Log.Warnf("Unknown operation type: %s", addonOperate)
		err = fmt.Errorf("unknown operation type: %s", addonOperate)
	}

	if addonOperate == bkeaddon.RemoveAddon && err != nil {
		return nil
	}
	c.Log.Infof("completed %s chart addon %s, error: %v", addonOperate, addon.Name, err)

	return err
}

func (c *Client) getDataFromCMByKey(name, namespace, key string, localClient client.Client) (string, error) {
	valuesCM := &corev1.ConfigMap{}
	if err := localClient.Get(context.TODO(), client.ObjectKey{Namespace: namespace, Name: name}, valuesCM); err != nil {
		return "", fmt.Errorf("failed to get ConfigMap %s/%s: %w", namespace, name, err)
	}

	data, ok := valuesCM.Data[key]
	if !ok {
		c.Log.Warnf("key %s not found in ConfigMap %s/%s", key, namespace, name)
		return "", nil
	}
	return data, nil
}

func (c *Client) getDataFromSecretByKeys(name, namespace string, keys []string, localClient client.Client) (map[string][]byte, error) {
	secret := &corev1.Secret{}
	if err := localClient.Get(context.TODO(), client.ObjectKey{Namespace: namespace, Name: name}, secret); err != nil {
		c.Log.Error(err, "failed to get secret ", name, " in namespace ", namespace)
	}

	data := map[string][]byte{}
	for _, key := range keys {
		value, ok := secret.Data[key]
		if !ok {
			data[key] = []byte("")
			c.Log.Warn("key ", key, " not found in secret ", name, " in namespace ", namespace)
			continue
		}
		data[key] = value
	}
	return data, nil
}

func (c *Client) getChartTimeout(addon *confv1beta1.Product) time.Duration {
	if addon.Timeout > 0 {
		return time.Duration(addon.Timeout) * time.Minute
	}
	return bkeinit.DefaultAddonTimeout
}

func (c *Client) initActionConfig(namespace string) (*action.Configuration, error) {
	restClientGetter := NewRESTClientConfig(namespace, c.RestConfig)
	actionConfig := new(action.Configuration)
	if err := actionConfig.Init(restClientGetter, namespace, "secret", c.Log.Infof); err != nil {
		c.Log.Error(err, "failed to init action config")
		return nil, err
	}
	return actionConfig, nil
}

func (c *Client) releaseExists(actionConfig *action.Configuration, releaseName, ns string) (bool, error) {
	helmStatus := action.NewStatus(actionConfig)
	_, err := helmStatus.Run(releaseName)
	if err != nil {
		// release not found is expected behavior and should not report as error
		if err.Error() != releaseNotFound {
			c.Log.Errorf("namespace: %s, name: %s, run command failed, error: %v", ns, releaseName, err)
			return true, err
		}
		return false, nil
	}
	c.Log.Infof("namespace: %s, name: %s, run command success", ns, releaseName)
	return true, nil
}

func (c *Client) handleRemoveChartOperation(addon *confv1beta1.Product) error {
	c.Log.Info("starting remove chart addon ", addon.Name)

	actionConfig, err := c.initActionConfig(addon.Namespace)
	if err != nil {
		return err
	}
	releaseName := addon.ReleaseName
	if releaseName == "" {
		releaseName = addon.Name
	}

	uninstall := action.NewUninstall(actionConfig)
	uninstall.Wait = addon.Block
	uninstall.Timeout = c.getChartTimeout(addon)
	_, err = uninstall.Run(releaseName)
	if err != nil {
		c.Log.Error(err, "failed to uninstall chart ", addon.Name)
		return err
	}
	c.Log.Info("uninstall chart ", addon.Name, " success")
	return nil
}

func (c *Client) handleUpgradeChartOperation(addon *confv1beta1.Product, cfg bkeinit.BkeConfig, bkeClusterNS string, localClient client.Client) error {
	c.Log.Info("starting upgrade chart addon ", addon.Name)
	actionConfig, err := c.initActionConfig(addon.Namespace)
	if err != nil {
		return err
	}
	releaseName := addon.ReleaseName
	if releaseName == "" {
		releaseName = addon.Name
	}

	upgrade := action.NewUpgrade(actionConfig)
	upgrade.Namespace = addon.Namespace
	upgrade.Timeout = c.getChartTimeout(addon)
	upgrade.Wait = addon.Block
	upgrade.WaitForJobs = true

	values, err := c.getChartValues(addon, bkeClusterNS, localClient)
	if err != nil {
		return err
	}
	chartFile, err := c.fetchChartPackage(addon, cfg, bkeClusterNS, localClient)
	if err != nil {
		c.Log.Error(err, "failed to fetch chart package ", addon.Name)
		return err
	}

	_, err = upgrade.Run(releaseName, chartFile, values)
	if err != nil {
		c.Log.Error(err, "failed to upgrade chart ", addon.Name)
		return err
	}
	c.Log.Info("upgrade chart ", addon.Name, " success")
	return nil
}

func (c *Client) handleCreateChartOperation(addon *confv1beta1.Product, cfg bkeinit.BkeConfig, bkeClusterNS string, localClient client.Client) error {
	c.Log.Info("starting install chart addon ", addon.Name)
	actionConfig, err := c.initActionConfig(addon.Namespace)
	if err != nil {
		return err
	}
	releaseName := addon.ReleaseName
	if releaseName == "" {
		releaseName = addon.Name
	}

	// 检查是否安装过该release
	exists, err := c.releaseExists(actionConfig, releaseName, addon.Namespace)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("release %s already exists in namespace %s", releaseName, addon.Namespace)
	}

	values, err := c.getChartValues(addon, bkeClusterNS, localClient)
	if err != nil {
		return err
	}
	chartFile, err := c.fetchChartPackage(addon, cfg, bkeClusterNS, localClient)
	if err != nil {
		c.Log.Error(err, "failed to fetch chart package ", addon.Name)
		return err
	}

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: addon.Namespace}}
	if _, err := c.ClientSet.CoreV1().Namespaces().Create(c.Ctx, ns, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("failed to create remote ns %s : %w", addon.Namespace, err)
	}

	install := action.NewInstall(actionConfig)
	install.ReleaseName = releaseName
	install.Namespace = addon.Namespace
	install.Timeout = c.getChartTimeout(addon)
	install.Wait = addon.Block
	install.WaitForJobs = true

	_, err = install.Run(chartFile, values)
	if err != nil {
		c.Log.Error(err, "failed to install chart ", addon.Name)
		return err
	}
	c.Log.Info("install chart ", addon.Name, " success")
	return nil
}

func (c *Client) getChartValues(addon *confv1beta1.Product, ns string, localClient client.Client) (chartutil.Values, error) {
	if addon.ValuesConfigMapRef == nil {
		c.Log.Info("no set values.yaml of chart addon ", addon.Name)
		return chartutil.Values{}, nil
	}

	name := addon.ValuesConfigMapRef.Name
	namespace := addon.ValuesConfigMapRef.Namespace
	if namespace == "" {
		namespace = ns
	}
	valuesYamlKey := addon.ValuesConfigMapRef.ValuesKey
	if valuesYamlKey == "" {
		valuesYamlKey = constant.ValuesYamlKey
	}

	valuesYaml, err := c.getDataFromCMByKey(name, namespace, valuesYamlKey, localClient)
	if err != nil {
		return chartutil.Values{}, err
	}

	values, err := chartutil.ReadValues([]byte(valuesYaml))
	if err != nil {
		c.Log.Error(err, "failed to parse values.yaml", addon.Name)
		return chartutil.Values{}, err
	}
	return values, nil
}

func (c *Client) getChartRepoTLSCerts(cfg bkeinit.BkeConfig, ns string, localClient client.Client) (*AuthConfig, error) {
	authConfig := NewAuthConfig()
	if cfg.Cluster.ChartRepo.TlsSecretRef == nil {
		c.Log.Info("no set tls secret of chart repo")
		return authConfig, nil
	}

	name := cfg.Cluster.ChartRepo.TlsSecretRef.Name
	namespace := cfg.Cluster.ChartRepo.TlsSecretRef.Namespace
	if namespace == "" {
		namespace = ns
	}
	caKey := cfg.Cluster.ChartRepo.TlsSecretRef.CaKey
	if caKey == "" {
		caKey = constant.CaKey
	}
	certKey := cfg.Cluster.ChartRepo.TlsSecretRef.CertKey
	if certKey == "" {
		certKey = constant.CertKey
	}
	keyKey := cfg.Cluster.ChartRepo.TlsSecretRef.KeyKey
	if keyKey == "" {
		keyKey = constant.KeyKey
	}

	data, err := c.getDataFromSecretByKeys(name, namespace, []string{caKey, certKey, keyKey}, localClient)
	if err != nil {
		return authConfig, err
	}
	authConfig = authConfig.SetCertFile(data[certKey]).SetKeyFile(data[keyKey]).SetCaFile(data[caKey])
	c.Log.Info("completed get tls info of chart repo")
	return authConfig, nil
}

func (c *Client) getChartRepoLoginInfo(cfg bkeinit.BkeConfig, ns string, localClient client.Client) ([]byte, []byte, error) {
	if cfg.Cluster.ChartRepo.AuthSecretRef == nil {
		c.Log.Info("no set auth secret of chart repo")
		return []byte(""), []byte(""), nil
	}

	name := cfg.Cluster.ChartRepo.AuthSecretRef.Name
	namespace := cfg.Cluster.ChartRepo.AuthSecretRef.Namespace
	if namespace == "" {
		namespace = ns
	}
	usernameKey := cfg.Cluster.ChartRepo.AuthSecretRef.UsernameKey
	if usernameKey == "" {
		usernameKey = constant.UsernameKey
	}
	passwordKey := cfg.Cluster.ChartRepo.AuthSecretRef.PasswordKey
	if passwordKey == "" {
		passwordKey = constant.PasswordKey
	}

	data, err := c.getDataFromSecretByKeys(name, namespace, []string{usernameKey, passwordKey}, localClient)
	if err != nil {
		return []byte(""), []byte(""), err
	}
	c.Log.Info("completed get login info of chart repo")
	return data[usernameKey], data[passwordKey], nil
}

func (c *Client) fetchChartPackage(addon *confv1beta1.Product, cfg bkeinit.BkeConfig, ns string, localClient client.Client) (*chart.Chart, error) {
	chartRepo, err := cfg.ResolveReachableChartRepo()
	if err != nil {
		return nil, err
	}

	authConfig, err := c.getChartRepoTLSCerts(cfg, ns, localClient)
	if err != nil {
		return nil, err
	}

	username, password, err := c.getChartRepoLoginInfo(cfg, ns, localClient)
	if err != nil {
		return nil, err
	}

	authConfig = authConfig.SetUsername(username).SetPassword(password).
		SetInsecureSkipTLSVerify(cfg.Cluster.ChartRepo.InsecureSkipTLSVerify)
	c.Log.Infof("starting fetch chart package %s from %s version is %s", addon.Name, chartRepo, addon.Version)
	return FetchChartUniversal(chartRepo, addon.Name, addon.Version, authConfig, c.Log)
}

// AuthConfig 封装认证和 TLS 配置
type AuthConfig struct {
	Username              []byte
	Password              []byte
	CaFile                []byte
	CertFile              []byte
	KeyFile               []byte
	CaFilePath            string
	CertFilePath          string
	KeyFilePath           string
	InsecureSkipTLSVerify bool
}

// NewAuthConfig is new AuthConfig
func NewAuthConfig() *AuthConfig {
	return &AuthConfig{
		Username:              []byte(""),
		Password:              []byte(""),
		CaFile:                []byte(""),
		CertFile:              []byte(""),
		KeyFile:               []byte(""),
		InsecureSkipTLSVerify: false,
	}
}

func (a *AuthConfig) SetCaFilePath(caFilePath string) *AuthConfig {
	a.CaFilePath = caFilePath
	return a
}

func (a *AuthConfig) SetCertFilePath(certFilePath string) *AuthConfig {
	a.CertFilePath = certFilePath
	return a
}

func (a *AuthConfig) SetKeyFilePath(keyFilePath string) *AuthConfig {
	a.KeyFilePath = keyFilePath
	return a
}

func (a *AuthConfig) SetCaFile(caFile []byte) *AuthConfig {
	a.CaFile = caFile
	return a
}

func (a *AuthConfig) SetCertFile(CertFile []byte) *AuthConfig {
	a.CertFile = CertFile
	return a
}

func (a *AuthConfig) SetKeyFile(keyFile []byte) *AuthConfig {
	a.KeyFile = keyFile
	return a
}

func (a *AuthConfig) SetUsername(username []byte) *AuthConfig {
	a.Username = username
	return a
}

func (a *AuthConfig) SetPassword(password []byte) *AuthConfig {
	a.Password = password
	return a
}

func (a *AuthConfig) SetInsecureSkipTLSVerify(insecureSkipTLSVerify bool) *AuthConfig {
	a.InsecureSkipTLSVerify = insecureSkipTLSVerify
	return a
}

// Cleanup 清理认证和TLS配置
func (a *AuthConfig) Cleanup() {
	a.CertFile = []byte("")
	a.KeyFile = []byte("")
	a.CaFile = []byte("")
	a.Username = []byte("")
	a.Password = []byte("")
}

// FetchChartUniversal 拉取chart包，支持oci和传统格式
func FetchChartUniversal(
	chartRepo, chartName, version string,
	auth *AuthConfig, logger *log.Logger,
) (*chart.Chart, error) {
	defer auth.Cleanup()

	// 1. 拉取oci格式chart
	if isOCIReference(chartRepo) {
		logger.Info("fetching chart from oci registry")
		return FetchChartOCI(chartRepo, chartName, version, auth, logger)
	}

	// 2. 拉取传统格式chart
	logger.Info("fetching chart from http registry")
	chartFile, err := FetchChartTraditional(chartRepo, chartName, version, auth, logger)
	if err == nil {
		return chartFile, nil
	}
	logger.Warnf("failed to fetch chart from http registry err is %v, try oci registry", err)

	// 3. 如果传统方式失败，尝试 OCI 方式
	ociURL := convertToOCIURL(chartRepo)
	return FetchChartOCI(ociURL, chartName, version, auth, logger)
}

func isOCIReference(url string) bool {
	return strings.HasPrefix(url, "oci://")
}

func convertToOCIURL(repoURL string) string {
	// 示例转换：https://192.168.100.202:30043/chartrepo/library
	// 转换为：oci://192.168.100.202:30043/library

	// 移除协议
	cleanURL := strings.TrimPrefix(repoURL, "https://")
	cleanURL = strings.TrimPrefix(cleanURL, "http://")

	// 移除 /chartrepo/ 前缀
	cleanURL = strings.Replace(cleanURL, "/chartrepo/", "/", 1)

	return "oci://" + cleanURL
}

var helmDriver string = os.Getenv("HELM_DRIVER")

func getCertTmpPath(auth *AuthConfig) (*AuthConfig, string, error) {
	tmpDir, err := os.MkdirTemp("", "helm-certs-*")
	if err != nil {
		return auth, "", fmt.Errorf("failed to create temp dir: %w", err)
	}

	certFiles := map[string][]byte{"ca-*.crt": auth.CaFile, "cert-*.crt": auth.CertFile, "key-*.key": auth.KeyFile}
	var certFilePath []string
	for pattern, cert := range certFiles {
		if len(cert) == 0 {
			certFilePath = append(certFilePath, "")
			continue
		}
		f, err := ioutil.TempFile(tmpDir, pattern)
		if err != nil {
			return auth, tmpDir, fmt.Errorf("failed to create temp cert file: %w", err)
		}
		if _, err := f.Write(cert); err != nil {
			return auth, tmpDir, fmt.Errorf("failed to write temp cret file: %w", err)
		}
		if err := f.Close(); err != nil {
			return auth, tmpDir, fmt.Errorf("failed to close temp cret file: %w", err)
		}
		certFilePath = append(certFilePath, f.Name())
	}

	auth = auth.SetCaFilePath(certFilePath[0])
	// ssl 校验证书要同时存在
	if len(certFilePath) >= len(certFiles) && (certFilePath[1] == "" || certFilePath[2] == "") {
		auth = auth.SetCertFilePath("").SetKeyFilePath("")
	} else {
		auth = auth.SetCertFilePath(certFilePath[1]).SetKeyFilePath(certFilePath[2])
	}

	return auth, tmpDir, nil
}

// FetchChartOCI fetches a Helm chart from an OCI registry.
func FetchChartOCI(repoURL, chartName, version string, auth *AuthConfig, logger *log.Logger) (*chart.Chart, error) {
	settings := cli.New()
	loggerDefault := stdlog.Default()
	actionConfig, err := initActionConfig(settings, loggerDefault)
	if err != nil {
		return nil, fmt.Errorf("failed to init action config: %w", err)
	}

	var tmpCertDir string
	auth, tmpCertDir, err = getCertTmpPath(auth)
	if err != nil {
		return nil, err
	}
	// 删除证书临时目录
	defer func() {
		if err := os.RemoveAll(tmpCertDir); err != nil {
			logger.Warnf("failed to remove temp CA file %s : %v", filepath.Dir(auth.CertFilePath), err)
		}
	}()

	if !strings.HasPrefix(repoURL, "oci://") {
		repoURL = fmt.Sprintf("oci://%s", repoURL)
	}

	chartRef := fmt.Sprintf("%s/%s", strings.TrimRight(repoURL, "/"), chartName)
	registryClient, err := newRegistryClientTLS(settings, loggerDefault, chartRef, false, auth)
	if err != nil {
		return nil, fmt.Errorf("failed to created registry client: %w", err)
	}
	actionConfig.RegistryClient = registryClient

	// 创建临时目录用于存放chart
	tmpDir, err := os.MkdirTemp("", "helm-chart-*")
	if err != nil {
		logger.Error("failed to create temp chart dir: %w", err)
		return nil, err
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			logger.Warnf("failed to remove temp chart file: %v", err)
		}
	}()

	pullClient := action.NewPullWithOpts(
		action.WithConfig(actionConfig))
	pullClient.DestDir = tmpDir
	pullClient.Settings = settings
	pullClient.Version = version
	pullClient.InsecureSkipTLSverify = false

	_, err = pullClient.Run(chartRef)
	if err != nil {
		logger.Errorf("failed to pull oci chart: %v", err)
		return nil, fmt.Errorf("failed to pull chart: %w", err)
	}

	return loader.Load(filepath.Join(tmpDir, fmt.Sprintf("%s-%s.tgz", chartName, version)))
}

func initActionConfig(settings *cli.EnvSettings, logger *stdlog.Logger) (*action.Configuration, error) {
	return initActionConfigList(settings, logger, false)
}

func initActionConfigList(settings *cli.EnvSettings, logger *stdlog.Logger, allNamespaces bool) (*action.Configuration, error) {
	actionConfig := new(action.Configuration)

	namespace := func() string {
		if allNamespaces {
			return ""
		}
		return settings.Namespace()
	}()

	if err := actionConfig.Init(
		settings.RESTClientGetter(),
		namespace,
		helmDriver,
		logger.Printf); err != nil {
		return nil, err
	}

	return actionConfig, nil
}

func newRegistryClient(settings *cli.EnvSettings, plainHTTP bool) (*registry.Client, error) {
	opts := []registry.ClientOption{
		registry.ClientOptDebug(settings.Debug),
		registry.ClientOptEnableCache(true),
		registry.ClientOptWriter(os.Stderr),
		registry.ClientOptCredentialsFile(settings.RegistryConfig),
	}
	if plainHTTP {
		opts = append(opts, registry.ClientOptPlainHTTP())
	}

	registryClient, err := registry.NewClient(opts...)
	if err != nil {
		return nil, err
	}
	return registryClient, nil
}

func newRegistryClientTLS(settings *cli.EnvSettings, logger *stdlog.Logger, chartRef string, plainHTTP bool,
	auth *AuthConfig) (*registry.Client, error) {
	var err error
	var registryClient *registry.Client
	if auth.KeyFilePath != "" && auth.CertFilePath != "" || auth.CaFilePath != "" || auth.InsecureSkipTLSVerify {
		registryClient, err = registry.NewRegistryClientWithTLS(
			logger.Writer(),
			auth.CertFilePath,
			auth.KeyFilePath,
			auth.CaFilePath,
			auth.InsecureSkipTLSVerify,
			settings.RegistryConfig,
			settings.Debug)

		if err != nil {
			return nil, err
		}
	} else {
		registryClient, err = newRegistryClient(settings, plainHTTP)
		if err != nil {
			return nil, err
		}
	}

	if registryClient == nil {
		return nil, fmt.Errorf("failed to create chart registry client")
	}

	if len(auth.Password) > 0 && len(auth.Username) > 0 {
		registryHost := extractRegistryHost(chartRef)
		if registryHost == "" {
			return nil, fmt.Errorf("failed to extract chart registry host from OCI reference")
		}

		err = registryClient.Login(
			registryHost,
			registry.LoginOptBasicAuth(string(auth.Username), string(auth.Password)),
			registry.LoginOptTLSClientConfig(auth.CertFilePath, auth.KeyFilePath, auth.CaFilePath),
			registry.LoginOptInsecure(auth.InsecureSkipTLSVerify),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to login chart registry: %w", err)
		}
	}
	return registryClient, nil
}

// 从OCI引用中提取主机名
func extractRegistryHost(ociRef string) string {
	// 移除 oci:// 前缀
	ref := strings.TrimPrefix(ociRef, "oci://")

	// 提取第一个 / 之前的部分
	if idx := strings.Index(ref, "/"); idx != -1 {
		return ref[:idx]
	}

	return ref
}

// FetchChartTraditional 通过传统方式获取并加载Chart
func FetchChartTraditional(
	repoURL, chartName, version string,
	auth *AuthConfig, logger *log.Logger,
) (*chart.Chart, error) {
	var err error
	var tmpDir string
	auth, tmpDir, err = getCertTmpPath(auth)
	if err != nil {
		return nil, err
	}

	// 删除证书临时目录
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			logger.Warnf("failed to remove temp CA file: %v", err)
		}
	}()

	isSpell := false
	if !strings.HasPrefix(repoURL, "http://") && !strings.HasPrefix(repoURL, "https://") {
		isSpell = true
		repoURL = fmt.Sprintf("https://%s", repoURL)
	}

	chartPath, err := fetchChartFromTraditionalRepo(repoURL, chartName, version, auth)
	if err != nil {
		logger.Warnf("failed to fetch chart from traditional registry: %v", err)
		if !isSpell {
			return nil, fmt.Errorf("failed to locate chart: %w", err)
		}
		// 尝试使用http协议再拉一次
		logger.Info("trying to fetch chart from http protocol")
		repoURL = fmt.Sprintf("http://%s", strings.TrimPrefix(repoURL, "https://"))
		chartPath, err = fetchChartFromTraditionalRepo(repoURL, chartName, version, auth)
		if err != nil {
			return nil, fmt.Errorf("failed to locate chart: %w", err)
		}
	}

	return loader.Load(chartPath)
}

func fetchChartFromTraditionalRepo(repoURL, chartName, version string, auth *AuthConfig) (string, error) {
	settings := cli.New()
	cpo := &action.ChartPathOptions{
		RepoURL:               repoURL,
		Version:               version,
		Username:              string(auth.Username),
		Password:              string(auth.Password),
		CaFile:                auth.CaFilePath,
		CertFile:              auth.CertFilePath,
		KeyFile:               auth.KeyFilePath,
		InsecureSkipTLSverify: auth.InsecureSkipTLSVerify,
	}

	defer func() {
		cpo.CertFile, cpo.KeyFile, cpo.CaFile, cpo.Password = "", "", "", ""
	}()
	return cpo.LocateChart(chartName, settings)
}
