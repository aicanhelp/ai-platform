/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package initialize

import (
	_ "embed"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"text/template"

	"gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	"gopkg.openfuyao.cn/bkecommon/cluster/validation"
	templateutil "gopkg.openfuyao.cn/bkecommon/template"
)

const (
	// DefaultFileMode defines default file permission mode (0644: rw-r--r--)
	DefaultFileMode = 0644
)

var (
	//go:embed tmpl/bke-cluster.tmpl
	bkeClusterTmpl string
)

type BkeConfig v1beta1.BKEConfig

func NewBkeConfigFromClusterConfig(cluster *v1beta1.BKEConfig) (*BkeConfig, error) {
	conf := BkeConfig(*cluster)
	SetDefaultBKEConfig(&conf)
	return &conf, nil
}

func ConvertBkEConfig(conf *BkeConfig) (*v1beta1.BKEConfig, error) {
	cluster := v1beta1.BKEConfig(*conf)
	return &cluster, nil
}

func (bc *BkeConfig) Validate() error {
	return validation.ValidateBKEConfig(v1beta1.BKEConfig(*bc))
}

func NewExternalEtcdConfig() map[string]string {
	return map[string]string{
		"etcdEndpoints": "",
		"etcdCAFile":    "",
		"etcdCertFile":  "",
		"etcdKeyFile":   "",
	}
}

// GenerateClusterAPIConfigFIle generates the cluster-API configuration file.
func (bc *BkeConfig) GenerateClusterAPIConfigFIle(name, namespace string, externalEtcd map[string]string) (string, error) {
	workspace := "/bke"
	dirPath := path.Join(workspace, namespace)
	if err := os.MkdirAll(dirPath, DefaultFileMode); err != nil {
		return "", err
	}
	filePath := path.Join(dirPath, name+".yaml")
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, DefaultFileMode)
	if err != nil {
		return "", err
	}
	defer file.Close()
	tmpl, err := template.New("bke-cluster").Funcs(templateutil.UtilFuncMap()).Parse(bkeClusterTmpl)
	if err != nil {
		return "", err
	}
	padding, err := bc.buildPadding(name, namespace, externalEtcd)
	if err != nil {
		return "", err
	}
	if err := tmpl.Execute(file, padding); err != nil {
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return filePath, nil
}

// buildPadding builds the template padding data for cluster-API configuration file.
// Note: After BKENode split, nodes are no longer part of BkeConfig.
// SANs for master nodes should be provided externally when calling this function.
func (bc *BkeConfig) buildPadding(name, namespace string, externalEtcd map[string]string) (map[string]string, error) {
	repo := fmt.Sprintf("%s:%s", bc.Cluster.ImageRepo.Domain, bc.Cluster.ImageRepo.Port)
	if bc.Cluster.ImageRepo.Prefix != "" {
		repo += "/" + bc.Cluster.ImageRepo.Prefix
	}
	sans := bc.Cluster.APIServer.CertSANs
	dnsIP, err := GetClusterDNSIP(bc.Cluster.Networking.ServiceSubnet)
	if err != nil {
		return nil, err
	}
	sans = append(sans, dnsIP)

	padding := map[string]string{
		"name":      name,
		"namespace": namespace,
		"repo":      repo,
		// 将 master数量设置为1，worker数量设置为0，避免安装过程中出现问题,由bke的控制器去调整实际的副本数
		"masterReplicas":    strconv.Itoa(1),
		"workerReplicas":    strconv.Itoa(0),
		"kubernetesVersion": bc.Cluster.KubernetesVersion,
		"workerVersion":     bc.Cluster.KubernetesVersion,
		"servicesCIDR":      bc.Cluster.Networking.ServiceSubnet,
		"podsCIDR":          bc.Cluster.Networking.PodSubnet,
		"serviceDomain":     bc.Cluster.Networking.DNSDomain,
		"SANS":              generateSANS(sans),
		"externalEtcd":      "false",
	}
	if externalEtcd != nil {
		padding["externalEtcd"] = "true"
		for k, v := range externalEtcd {
			padding[k] = v
		}
	}
	return padding, nil
}

func (bc *BkeConfig) YumRepo() string {
	return fmt.Sprintf("http://%s:%s", bc.Cluster.HTTPRepo.Domain, bc.Cluster.HTTPRepo.Port)
}

func (bc *BkeConfig) ImageRepo() string {
	address := validation.GetImageRepoAddress(bc.Cluster.ImageRepo)
	if bc.Cluster.ImageRepo.Prefix != "" {
		return fmt.Sprintf("%s/%s/", address, bc.Cluster.ImageRepo.Prefix)

	} else {
		return fmt.Sprintf("%s/", address)
	}
}

// ChartRepo is the chart repository address.
func (bc *BkeConfig) ChartRepo() string {
	address := validation.GetImageRepoAddress(bc.Cluster.ChartRepo)
	if bc.Cluster.ChartRepo.Prefix != "" {
		return fmt.Sprintf("%s/%s/", address, bc.Cluster.ChartRepo.Prefix)
	}
	return fmt.Sprintf("%s/", address)
}

func generateSANS(sans []string) string {
	sa := map[string]uint8{
		"127.0.0.1":         0,
		"localhost":         0,
		DefaultClusterDNSIP: 0,
	}
	for _, s := range sans {
		sa[s] = 0
	}
	sn := ""
	for key, _ := range sa {
		sn += "- " + key + "\n          "
	}
	return sn
}

func commonFuncMaps() template.FuncMap {
	return template.FuncMap{
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
	}
}
