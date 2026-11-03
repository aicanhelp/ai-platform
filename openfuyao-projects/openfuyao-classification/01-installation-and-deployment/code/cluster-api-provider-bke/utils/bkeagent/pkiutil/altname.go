/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pkiutil

import (
	"fmt"
	"math/big"
	"net"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	netutil "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/net"
)

// GetAPIServerCertAltNamesFromBkeConfig returns the AltNames object for the API server certificate
// Deprecated: Use GetAPIServerCertAltNamesWithNodes instead for capbke controller
//
//	dnsDomain、serviceSubnet、certSANs、master node ip
func GetAPIServerCertAltNamesFromBkeConfig(bkeConfig *bkev1beta1.BKEConfig) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}
	// append all the master node or load balancer ip to altnames.IPs
	nodesData, err := cluster.GetNodesData(bkeConfig.Cluster.ContainerdConfigRef.Namespace, bkeConfig.Cluster.ContainerdConfigRef.Name)
	if err != nil {
		return nil, err
	}
	return GetAPIServerCertAltNamesWithNodes(bkeConfig, bkenode.Nodes(nodesData))
}

// GetAPIServerCertAltNamesWithNodes returns the AltNames for the API server certificate using provided nodes
// This is designed for capbke controller which already has nodes from NodeFetcher
func GetAPIServerCertAltNamesWithNodes(bkeConfig *bkev1beta1.BKEConfig, nodes bkenode.Nodes) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}
	altNames := &certutil.AltNames{}
	// append dnsDomain to altnames.DNSNames
	if bkeConfig.Cluster.Networking.DNSDomain != "" {
		altNames.DNSNames = append(altNames.DNSNames, fmt.Sprintf("kubernetes.default.svc.%s", bkeConfig.Cluster.Networking.DNSDomain))
	} else {
		altNames.DNSNames = append(altNames.DNSNames, "kubernetes.default.svc.cluster.local")
	}

	for _, node := range nodes.Master() {
		altNames.IPs = append(altNames.IPs, net.ParseIP(node.IP))
		altNames.DNSNames = append(altNames.DNSNames, node.Hostname)
	}

	if bkeConfig.Cluster.APIServer != nil && bkeConfig.Cluster.APIServer.CertSANs != nil {
		if err := AppendSANsToAltNames(altNames, bkeConfig.Cluster.APIServer.CertSANs, APIServerCertName); err != nil {
			return nil, err
		}
	}

	// append IP of the internal Kubernetes API service
	if bkeConfig.Cluster.Networking.ServiceSubnet != "" {
		ip, err := getAPIServerVirtualIP(bkeConfig.Cluster.Networking.ServiceSubnet, false)
		if err != nil {
			return nil, err
		}
		altNames.IPs = append(altNames.IPs, ip)
	}
	// remove repeat ips in altnames.IPs
	altNames.IPs = netutil.RemoveRepIP(altNames.IPs)
	altNames.DNSNames = netutil.RemoveRepDomain(altNames.DNSNames)
	return altNames, nil
}

// GetEtcdCertAltNamesFromBkeConfig returns the AltNames object for the etcd server certificate
// Deprecated: Use GetEtcdCertAltNamesWithNodes instead for capbke controller
func GetEtcdCertAltNamesFromBkeConfig(bkeConfig *bkev1beta1.BKEConfig, isServer bool) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}
	// append etcd.CertSANs to altnames.DNSNames of altnames.IPs from node.json
	nodesData, err := cluster.GetNodesData(bkeConfig.Cluster.ContainerdConfigRef.Namespace, bkeConfig.Cluster.ContainerdConfigRef.Name)
	if err != nil {
		return nil, err
	}
	return GetEtcdCertAltNamesWithNodes(bkeConfig, bkenode.Nodes(nodesData), isServer)
}

// GetEtcdCertAltNamesWithNodes returns the AltNames for the etcd certificate using provided nodes
func GetEtcdCertAltNamesWithNodes(bkeConfig *bkev1beta1.BKEConfig, nodes bkenode.Nodes, isServer bool) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}
	altNames := &certutil.AltNames{}
	for _, n := range nodes.Etcd() {
		altNames.IPs = append(altNames.IPs, net.ParseIP(n.IP))
		altNames.DNSNames = append(altNames.DNSNames, n.Hostname)
	}

	if bkeConfig.Cluster.Etcd != nil {
		if isServer {
			if err := AppendSANsToAltNames(altNames, bkeConfig.Cluster.Etcd.ServerCertSANs, EtcdServerCertName); err != nil {
				return nil, err
			}
		} else {
			if err := AppendSANsToAltNames(altNames, bkeConfig.Cluster.Etcd.PeerCertSANs, EtcdPeerCertName); err != nil {
				return nil, err
			}
		}
	}

	// remove repeat ips in altnames.IPs
	altNames.IPs = netutil.RemoveRepIP(altNames.IPs)
	altNames.DNSNames = netutil.RemoveRepDomain(altNames.DNSNames)
	return altNames, nil
}

// AppendSANsToAltNames adds the SANs to the AltNames of the leaf cert
func AppendSANsToAltNames(altNames *certutil.AltNames, SANs []string, certName string) error {
	for _, altname := range SANs {
		if err := appendSANEntry(altNames, altname); err != nil {
			return errors.Errorf(
				"%q was not added to the %q SAN, because it is not a valid IP or RFC-1123 compliant DNS entry",
				altname, certName,
			)
		}
	}

	altNames.IPs = netutil.RemoveRepIP(altNames.IPs)
	altNames.DNSNames = netutil.RemoveRepDomain(altNames.DNSNames)
	return nil
}

func appendSANEntry(altNames *certutil.AltNames, altname string) error {
	if ip := net.ParseIP(altname); ip != nil {
		altNames.IPs = append(altNames.IPs, ip)
		return nil
	}

	if isValidDNSName(altname) {
		altNames.DNSNames = append(altNames.DNSNames, altname)
		return nil
	}

	return fmt.Errorf("invalid SAN entry")
}

func isValidDNSName(name string) bool {
	return len(validation.IsDNS1123Subdomain(name)) == 0 ||
		len(validation.IsWildcardDNS1123Subdomain(name)) == 0
}

// getAPIServerVirtualIP returns the IP of the internal Kubernetes API service
func getAPIServerVirtualIP(svcSubnetList string, isDualStack bool) (net.IP, error) {
	// Parse the service CIDR
	_, svcSubnet, err := net.ParseCIDR(svcSubnetList)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to parse ServiceSubnet %v", svcSubnetList)
	}

	// Get the first IP address from the service CIDR (index 1)
	internalAPIServerVirtualIP, err := GetIndexedIP(svcSubnet, 1)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to get the first IP address from the given CIDR: %s", svcSubnet.String())
	}

	return internalAPIServerVirtualIP, nil
}

// GetIndexedIP returns a net.IP that is subnet.IP + index in the contiguous IP space.
func GetIndexedIP(subnet *net.IPNet, index int) (net.IP, error) {
	// 将IP转换为大整数（统一为16字节表示）
	baseIP := subnet.IP.To16()
	baseInt := big.NewInt(0).SetBytes(baseIP)

	// 添加偏移量
	offsetInt := big.NewInt(int64(index))
	resultInt := big.NewInt(0).Add(baseInt, offsetInt)

	// 将结果转换回IP（确保16字节）
	resultBytes := resultInt.Bytes()
	// 确保结果为16字节
	resultBytes = append(make([]byte, utils.IPByteLength), resultBytes...)
	ip := net.IP(resultBytes[len(resultBytes)-utils.IPByteLength:])

	if !subnet.Contains(ip) {
		return nil, fmt.Errorf("can't generate IP with index %d from subnet. subnet too small. subnet: %q",
			index, subnet)
	}

	return ip, nil
}
