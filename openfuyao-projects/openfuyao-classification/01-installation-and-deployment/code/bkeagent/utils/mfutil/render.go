package mfutil

import (
	"bytes"
	"embed"
	_ "embed"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/blang/semver/v4"
	"github.com/pkg/errors"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	bkenet "gopkg.openfuyao.cn/bkecommon/utils/net"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/clientutil"
	bkeetcd "gopkg.openfuyao.cn/bkeagent/utils/etcd"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
)

//
//go:embed  tmpl/*
var f embed.FS

const (
	keepalivedBaseTmpl = `
{{ range $instance := .instances }}
{{ $instance }}
{{ end }}
`
	// CertChainFileName defines certificate ca and chain name
	CertCAAndChainFileName = "ca-chain.crt"
)

func renderK8sAndStore(c *BKEComponent, tmpl []byte, cfg interface{}, funcMap *template.FuncMap) error {
	var t *template.Template
	var err error
	if funcMap != nil {
		log.Info("render with funcMap")
		t, err = template.New(c.Name).Funcs(*funcMap).Parse(string(tmpl))
		if err != nil {
			return err
		}
	} else {
		log.Info("render without funcMap")
		t, err = template.New(c.Name).Parse(string(tmpl))
		if err != nil {
			return err
		}
	}

	filePath := pathForManifest(c)
	writer, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer writer.Close()
	err = t.Execute(writer, cfg)
	if err != nil {
		return err
	}
	return nil
}

func renderHAAndStore(c *BKEHAComponent, tmpl []byte, cfg map[string]interface{}, filePath string, funcMap *template.FuncMap) error {
	var t *template.Template
	var err error
	if funcMap != nil {
		t, err = template.New(c.Name).Funcs(*funcMap).Parse(string(tmpl))
		if err != nil {
			return err
		}
	} else {
		t, err = template.New(c.Name).Parse(string(tmpl))
		if err != nil {
			return err
		}
	}

	writer, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		return err
	}
	defer writer.Close()
	err = t.Execute(writer, cfg)
	if err != nil {
		return err
	}
	return nil
}

func renderAPIServer(c *BKEComponent, cfg *BootScope) error {
	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-apiserver.yaml.tmpl")
	if err != nil {
		return err
	}
	file, err := fs.ReadFile(f, "tmpl/k8s/audit-policy.yaml")
	if err != nil {
		return err
	}
	// write the audit policy file,default to /etc/kubernetes/audit-policy.yaml
	if err := os.WriteFile(GetAuditPolicyFilePath(), file, 0644); err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), apiServerFuncMap()))
}

func renderEtcd(c *BKEComponent, cfg *BootScope) error {
	var etcdClient *etcd.Client
	var err error

	tmpl, err := fs.ReadFile(f, "tmpl/k8s/etcd.yaml.tmpl")
	if err != nil {
		return err
	}
	etcdPeerAddress := etcd.GetPeerURL(&kubeadmapi.APIEndpoint{AdvertiseAddress: cfg.HostIP})

	init, ok := cfg.Extra["Init"]
	if ok && !init.(bool) {
		client, err := clientutil.ClientSetFromManagerClusterSecret(cfg.Extra["mccs"].([]string)...)
		if err != nil {
			return errors.Wrapf(err, "Failed to get client")
		}
		etcdClient, err = etcd.NewFromCluster(client, cfg.BkeConfig.Cluster.CertificatesDir)
		if err != nil {
			return err
		}
		initialCluster, err := etcdClient.ListMembers()
		if err != nil {
			return err
		}
		// only add the new member if it doesn't already exists
		var exists bool
		log.Infof("checking if the etcd member already exists: %s", etcdPeerAddress)
		for i := range initialCluster {
			if initialCluster[i].PeerURL == etcdPeerAddress {
				exists = true
				if len(initialCluster[i].Name) == 0 {
					initialCluster[i].Name = cfg.HostName
				}
				break
			}
		}

		if exists {
			log.Infof("etcd member already exists: %q", etcdPeerAddress)
			cfg.Extra["EtcdInitialCluster"] = initialCluster
		} else {
			log.Infof("adding etcd member: %s", etcdPeerAddress)
			cfg.Extra["EtcdInitialCluster"], err = etcdClient.AddMember(cfg.HostName, etcdPeerAddress)
			if err != nil {
				return err
			}
			log.Infof("Updated etcd member list: %v", cfg.Extra["EtcdInitialCluster"])
		}
	} else {
		cfg.Extra["EtcdInitialCluster"] = []etcd.Member{{Name: cfg.HostName, PeerURL: etcdPeerAddress}}
	}

	cfg.Extra["EtcdAdvertiseUrls"] = etcd.GetClientURLByIP(cfg.HostIP)
	if err := renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), etcdFuncMap())); err != nil {
		return err
	}

	return nil
}

func renderController(c *BKEComponent, cfg *BootScope) error {
	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-controller-manager.yaml.tmpl")
	if err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), controllerFuncMap()))
}

func renderScheduler(c *BKEComponent, cfg *BootScope) error {
	// todo 还需要处理一个文件 gpu-admission.config
	if v, ok := cfg.Extra["gpuEnable"]; ok && v.(string) == "true" {
		policyFile, err := fs.ReadFile(f, "tmpl/k8s/scheduler-policy-config.json")
		if err != nil {
			return err
		}
		if err = os.WriteFile(GetSchedulerPolicyFilePath(), policyFile, 0755); err != nil {
			return err
		}
		admissionConfig, err := fs.ReadFile(f, "tmpl/k8s/gpu-admission.config")
		if err != nil {
			return err
		}
		if err = os.WriteFile(GetSchedulerAdmissionConfigFilePath(), admissionConfig, 0755); err != nil {
			return err
		}
	}

	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-scheduler.yaml.tmpl")
	if err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), schedulerFuncMap()))
}

// HA component yaml render func
func renderHAProxy(c *BKEHAComponent, cfg map[string]interface{}) error {
	if v, ok := cfg["haproxyConfigDir"]; ok {
		c.ConfPath = v.(string)
	}

	// step1 render conf file
	tmpl, err := fs.ReadFile(f, "tmpl/haproxy/haproxy.cfg.tmpl")
	if err != nil {
		return err
	}
	if err := renderHAAndStore(c, tmpl, cfg, pathForHAManifestConf(c), nil); err != nil {
		return err
	}
	// step2 render yaml file
	tmpl, err = fs.ReadFile(f, "tmpl/haproxy/haproxy.yaml.tmpl")
	if err != nil {
		return err
	}

	if err := renderHAAndStore(c, tmpl, cfg, pathForHAManifest(c), utilFuncMap()); err != nil {
		return err
	}
	return nil
}

func renderKeepalived(c *BKEHAComponent, cfg map[string]interface{}) error {
	if v, ok := cfg["keepAlivedConfigDir"]; ok {
		c.ConfPath = v.(string)
	}

	var checkTmpl []byte
	var scriptName string
	var err error

	isMasterHa := cfg["isMasterHa"].(bool)

	// step 1 render check script
	if isMasterHa {
		checkTmpl, err = fs.ReadFile(f, "tmpl/keepalived/check-master.sh.tmpl")
		if err != nil {
			return err
		}
		scriptName = "check-master.sh"
	} else {
		checkTmpl, err = fs.ReadFile(f, "tmpl/keepalived/check-ingress.sh.tmpl")
		if err != nil {
			return err
		}
		scriptName = "check-ingress.sh"
	}
	log.Infof("render keepalived check script: %s", scriptName)
	if err := renderHAAndStore(c, checkTmpl, cfg, pathForHAManifestScript(c, scriptName), nil); err != nil {
		return err
	}

	// step 2 render instance
	cfg["instances"] = []string{}

	if isMasterHa {
		log.Infof("render keepalived master VIP instance")
		t, err := template.New("keepalived.master.conf.tmpl").Funcs(*keepalivedConfFuncMap()).ParseFS(f, "tmpl/keepalived/keepalived.master.conf.tmpl")
		if err != nil {
			return err
		}
		content := bytes.NewBuffer(nil)
		if err := t.Execute(content, cfg); err != nil {
			return err
		}
		cfg["instances"] = append(cfg["instances"].([]string), content.String())
	} else {
		log.Infof("render keepalived ingress VIP instance")
		t, err := template.New("keepalived.ingress.conf.tmpl").Funcs(*keepalivedConfFuncMap()).ParseFS(f, "tmpl/keepalived/keepalived.ingress.conf.tmpl")
		if err != nil {
			return err
		}
		content := bytes.NewBuffer(nil)
		if err := t.Execute(content, cfg); err != nil {
			return err
		}
		cfg["instances"] = append(cfg["instances"].([]string), content.String())
	}

	// step 3 get base
	var baseTmpl *template.Template

	if utils.Exists(pathForHAManifestConf(c)) {
		file, err := os.OpenFile(pathForHAManifestConf(c), os.O_RDWR|os.O_APPEND, 0644)
		if err != nil {
			return err
		}
		defer file.Close()
		stat, err := file.Stat()
		if err != nil {
			return err
		}
		buffer := make([]byte, stat.Size())
		if _, err = file.Read(buffer); err != nil {
			return err
		}
		// remove old ingress instance or master instance
		if isMasterHa {
			buffer = regexp.MustCompile(MasterKeepalivedInstanceReg).ReplaceAll(buffer, []byte(""))
		} else {
			buffer = regexp.MustCompile(IngressKeepalivedInstanceReg).ReplaceAll(buffer, []byte(""))
		}

		// append keepalivedBaseTmpl to buffer
		buffer = append(buffer, []byte(keepalivedBaseTmpl)...)
		baseTmpl, err = template.New("keepalived.conf").Funcs(*keepalivedConfFuncMap()).Parse(string(buffer))
		if err != nil {
			return err
		}
	} else {
		baseTmpl, err = template.New("keepalived.base.conf.tmpl").Funcs(*keepalivedConfFuncMap()).ParseFS(f, "tmpl/keepalived/keepalived.base.conf.tmpl")
		if err != nil {
			return err
		}
	}

	log.Infof("render keepalived conf file")
	// step 4 render conf file
	writer, err := os.OpenFile(pathForHAManifestConf(c), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer writer.Close()

	if err := baseTmpl.Execute(writer, cfg); err != nil {
		return err
	}

	// step 5 render yaml file
	tmpl, err := fs.ReadFile(f, "tmpl/keepalived/keepalived.yaml.tmpl")
	if err != nil {
		return err
	}
	if err := renderHAAndStore(c, tmpl, cfg, pathForHAManifest(c), mergeFuncMap(keepalivedConfFuncMap(), utilFuncMap())); err != nil {
		return err
	}
	return nil
}

func keepalivedConfFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"randomString": func(n int) string {
			// 生成n位随机字符串
			var letters = []byte("abcdefghijklmnopqrstuvwxyz1234567890")
			result := make([]byte, n)
			rand.Seed(time.Now().UnixNano())
			for i := range result {
				result[i] = letters[rand.Intn(len(letters))]
			}
			return string(result)
		},
		"computeWeight": func(nodes []HANode) string {
			return strconv.Itoa(10 * len(nodes))
		},
		"isMaster": KeepalivedInstanceIsMaster,
		"priority": func(nodes []HANode) string {
			ips, err := bkenet.GetAllInterfaceIP()
			if err != nil {
				return "100"
			}
			for i, n := range nodes {
				// 跳过master
				if i == 0 {
					continue
				}
				for _, ip := range ips {
					if strings.Contains(ip, n.IP) {
						return strconv.Itoa(100 - 10*i)
					}
				}
			}
			return "100"
		},
	}
}

func KeepalivedInstanceIsMaster(nodes []HANode) bool {
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		return false
	}
	for _, ip := range ips {
		if strings.Contains(ip, nodes[0].IP) {
			return true
		}
	}
	return false
}

func GlobalFuncMap() *template.FuncMap {
	base := &template.FuncMap{
		"imageRepo": func(cfg *BootScope) string {
			bkeCfg := bkeinit.BkeConfig(*cfg.BkeConfig)
			return bkeCfg.ImageRepo()
		},
	}
	return mergeFuncMap(base, k8sVersionFuncMap())
}

func utilFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"randomString": func(n int) string {
			// 生成n位随机字符串
			var letters = []byte("abcdefghijklmnopqrstuvwxyz1234567890")
			result := make([]byte, n)
			rand.Seed(time.Now().UnixNano())
			for i := range result {
				result[i] = letters[rand.Intn(len(letters))]
			}
			return string(result)
		},
	}
}

const DefaultAPIServerAuthorizationMode = "Node,RBAC"

func apiServerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"advertiseAddress": func(cfg *BootScope) string {
			ip, _ := bkenet.GetExternalIP()
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.Host != "" {
				return cfg.CurrentNode.APIServer.Host
			}
			return ip
		},
		"etcdServers": func(cfg *BootScope) string {
			bkeNodes := bkenode.Nodes(cfg.BkeConfig.Nodes)
			etcdNodes := bkeNodes.Etcd()
			if etcdNodes.Length() == 0 {
				return etcd.GetClientURLByIP(cfg.HostIP)
			}
			etcdEndpoints := make([]string, 0, len(etcdNodes))
			for _, n := range etcdNodes {
				etcdEndpoints = append(etcdEndpoints, etcd.GetClientURLByIP(n.IP))
			}
			return strings.Join(etcdEndpoints, ",")
		},
		"apiServerPort": func(cfg *BootScope) int32 {
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.Port != 0 {
				return cfg.CurrentNode.APIServer.Port
			}
			return cfg.BkeConfig.Cluster.APIServer.Port
		},
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultAPIServerImageName, k8sVersion)
		},
		"clientCAFile": func(cfg *BootScope) string {
			caChainPath := fmt.Sprintf("%s/%s", cfg.BkeConfig.Cluster.CertificatesDir, CertCAAndChainFileName)
			if utils.Exists(caChainPath) {
				return caChainPath
			}
			return fmt.Sprintf("%s/%s", cfg.BkeConfig.Cluster.CertificatesDir, pkiutil.CACertName)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.ExtraArgs != nil {
				if _, ok := cfg.CurrentNode.APIServer.ExtraArgs["authorization-mode"]; !ok {
					cfg.CurrentNode.APIServer.ExtraArgs["authorization-mode"] = DefaultAPIServerAuthorizationMode
				}
				return getExtraArgs(cfg.CurrentNode.APIServer.ExtraArgs)
			}
			if _, ok := cfg.BkeConfig.Cluster.APIServer.ExtraArgs["authorization-mode"]; !ok {
				cfg.BkeConfig.Cluster.APIServer.ExtraArgs["authorization-mode"] = DefaultAPIServerAuthorizationMode
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.APIServer.ExtraArgs)
		},
		"upgradeWithOpenFuyao": isUpgradeWithOpenFuyao,
	}
}

func isUpgradeWithOpenFuyao(cfg *BootScope) bool {
	log.Info("get upgradeWithOpenFuyao param")
	if _, ok := cfg.Extra["upgradeWithOpenFuyao"]; !ok {
		log.Info("not found upgradeWithOpenFuyao")
		return false
	}
	log.Info("upgradeWithOpenFuyao param is ", cfg.Extra["upgradeWithOpenFuyao"].(bool))
	return cfg.Extra["upgradeWithOpenFuyao"].(bool)
}

func etcdFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"initialCluster": func(members []etcd.Member) string {
			if len(members) == 0 {
				ip, _ := bkenet.GetExternalIP()
				hostname := utils.HostName()
				return fmt.Sprintf("%s=https://%s:%d", hostname, ip, bkeetcd.EtcdListenPeerPort)
			}
			var result []string
			for _, member := range members {
				result = append(result, fmt.Sprintf("%s=%s", member.Name, member.PeerURL))
			}
			return strings.Join(result, ",")
		},
		"imageInfo": func(cfg *BootScope) string {
			var etcdVersion string
			if cfg.BkeConfig.Cluster.EtcdVersion != "" {
				etcdVersion = cfg.BkeConfig.Cluster.EtcdVersion
			} else {
				etcdVersion = bkeinit.DefaultEtcdImageTag
			}
			etcdVersion = strings.TrimPrefix(etcdVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultEtcdImageName, etcdVersion)
		},
		"dataDir": func(cfg *BootScope) string {
			if cfg.CurrentNode.Etcd != nil && cfg.CurrentNode.Etcd.DataDir != "" {
				return cfg.CurrentNode.Etcd.DataDir
			}
			return cfg.BkeConfig.Cluster.Etcd.DataDir
		},
		"etcdAdvertiseUrls": func(cfg *BootScope) string {
			if v, ok := cfg.Extra["EtcdAdvertiseUrls"]; ok {
				return v.(string)
			}
			return etcd.GetClientURLByIP(cfg.HostIP)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.Etcd != nil && cfg.CurrentNode.Etcd.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.Etcd.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.Etcd.ExtraArgs)
		},
	}

}

func controllerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultControllerManagerImageName, k8sVersion)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.ControllerManager != nil && cfg.CurrentNode.ControllerManager.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.ControllerManager.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.ControllerManager.ExtraArgs)
		},
		"getSubnetMask": func(cidr string) string {
			res := strings.Split(cidr, "/")
			if len(res) != 2 {
				return ""
			}
			// 转为int
			mask, err := strconv.Atoi(res[1])
			if err != nil {
				return ""
			}
			if mask > 24 {
				return res[1]
			}
			return "24"
		},
	}
}

func schedulerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultSchedulerImageName, k8sVersion)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.Scheduler != nil && cfg.CurrentNode.Scheduler.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.Scheduler.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.Scheduler.ExtraArgs)
		},
	}
}

// mergeFuncMap merge two funcMap
func mergeFuncMap(f1, f2 *template.FuncMap) *template.FuncMap {
	f := *f1
	for k, v := range *f2 {
		f[k] = v
	}
	return &f
}

func getExtraArgs(argsMap map[string]string) []string {
	var args []string
	for k, v := range argsMap {
		args = append(args, fmt.Sprintf("%s=%s", k, v))
	}
	return args
}

func k8sVersionFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"vgt": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.GT(dv)
		},
		"vlt": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.LT(dv)
		},
		"veq": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.EQ(dv)
		},
		"vgte": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.GTE(dv)
		},
		"vlte": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.LTE(dv)
		},
		"vne": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.NE(dv)
		},
	}
}

func parseVersion(version string) semver.Version {
	v, _ := semver.ParseTolerant(version)
	return v
}
