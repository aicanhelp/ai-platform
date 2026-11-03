package pkiutil

import (
	"fmt"
	"math/big"
	"net"

	"github.com/pkg/errors"
	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	"k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"

	netutil "gopkg.openfuyao.cn/bkeagent/utils/net"
)

// GetAPIServerCertAltNamesFromBkeConfig returns the AltNames object for the API server certificate
//
//	dnsDomain、serviceSubnet、certSANs、master node ip
func GetAPIServerCertAltNamesFromBkeConfig(bkeConfig *bkev1beta1.BKEConfig) (*certutil.AltNames, error) {
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

	// append all the master node or load balancer ip to altnames.IPs
	nodes := bkenode.Nodes(bkeConfig.Nodes)
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
func GetEtcdCertAltNamesFromBkeConfig(bkeConfig *bkev1beta1.BKEConfig, isServer bool) (*certutil.AltNames, error) {
	if bkeConfig == nil {
		return nil, errors.New("bkeConfig is nil")
	}
	altNames := &certutil.AltNames{}
	// append etcd.CertSANs to altnames.DNSNames of altnames.IPs from node.json
	nodes := bkenode.Nodes(bkeConfig.Nodes)
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
		if ip := net.ParseIP(altname); ip != nil {
			altNames.IPs = append(altNames.IPs, ip)
		} else if len(validation.IsDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		} else if len(validation.IsWildcardDNS1123Subdomain(altname)) == 0 {
			altNames.DNSNames = append(altNames.DNSNames, altname)
		} else {
			return errors.Errorf("%q was not added to the %q SAN, because it is not a valid IP or RFC-1123 compliant DNS entry", altname, certName)
		}
	}
	altNames.IPs = netutil.RemoveRepIP(altNames.IPs)
	altNames.DNSNames = netutil.RemoveRepDomain(altNames.DNSNames)
	return nil
}

// getAPIServerVirtualIP returns the IP of the internal Kubernetes API service
func getAPIServerVirtualIP(svcSubnetList string, isDualStack bool) (net.IP, error) {
	svcSubnet, err := getKubernetesServiceCIDR(svcSubnetList, isDualStack)
	if err != nil {
		return nil, errors.Wrap(err, "unable to get internal Kubernetes Service IP from the given service CIDR")
	}
	internalAPIServerVirtualIP, err := GetIndexedIP(svcSubnet, 1)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to get the first IP address from the given CIDR: %s", svcSubnet.String())
	}
	return internalAPIServerVirtualIP, nil
}

// getKubernetesServiceCIDR returns the default Service CIDR for the Kubernetes internal service
func getKubernetesServiceCIDR(svcSubnetList string, isDualStack bool) (*net.IPNet, error) {
	// todo use dual stack for kubernetes service

	// internal IP address for the API server
	_, svcSubnet, err := net.ParseCIDR(svcSubnetList)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to parse ServiceSubnet %v", svcSubnetList)
	}
	return svcSubnet, nil
}

// GetIndexedIP returns a net.IP that is subnet.IP + index in the contiguous IP space.
func GetIndexedIP(subnet *net.IPNet, index int) (net.IP, error) {
	ip := addIPOffset(bigForIP(subnet.IP), index)
	if !subnet.Contains(ip) {
		return nil, fmt.Errorf("can't generate IP with index %d from subnet. subnet too small. subnet: %q", index, subnet)
	}
	return ip, nil
}

// bigForIP creates a big.Int based on the provided net.IP
func bigForIP(ip net.IP) *big.Int {
	// NOTE: Convert to 16-byte representation so we can
	// handle v4 and v6 values the same way.
	return big.NewInt(0).SetBytes(ip.To16())
}

// addIPOffset adds the provided integer offset to a base big.Int representing a net.IP
// NOTE: If you started with a v4 address and overflow it, you get a v6 result.
func addIPOffset(base *big.Int, offset int) net.IP {
	r := big.NewInt(0).Add(base, big.NewInt(int64(offset))).Bytes()
	r = append(make([]byte, 16), r...)
	return net.IP(r[len(r)-16:])
}
