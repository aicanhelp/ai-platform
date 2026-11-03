package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	"k8s.io/client-go/kubernetes"
)

const monitoringAPIGroup = "monitoring.coreos.com"

// MonitoringCRDBackup 保存 monitoring.coreos.com CRD 及实例，供临时删除后复原。
type MonitoringCRDBackup struct {
	Dir      string
	CRDNames []string
}

// InvalidateDiscovery 清空 client-go discovery 缓存，避免删/复原 CRD 后读到旧 API 列表。
func InvalidateDiscovery(clientset kubernetes.Interface) {
	type invalidator interface {
		Invalidate()
	}
	if inv, ok := clientset.Discovery().(invalidator); ok {
		inv.Invalidate()
	}
}

// BackupMonitoringCRDs 备份集群中全部 monitoring.coreos.com CRD 及其实例（去掉 uid 等不可复用字段）。
func BackupMonitoringCRDs(ctx context.Context) (*MonitoringCRDBackup, error) {
	dir, err := os.MkdirTemp("", "infernex-pm003-monitoring-crd-")
	if err != nil {
		return nil, fmt.Errorf("mkdir backup dir: %w", err)
	}
	crdDir := filepath.Join(dir, "crds")
	crDir := filepath.Join(dir, "crs")
	if err := os.MkdirAll(crdDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(crDir, 0o755); err != nil {
		return nil, err
	}

	raw, err := runKubectl(ctx, "get", "crd", "-o", "json")
	if err != nil {
		return nil, fmt.Errorf("list crds: %w", err)
	}
	var list map[string]any
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("decode crd list: %w", err)
	}
	items, _ := list["items"].([]any)
	backup := &MonitoringCRDBackup{Dir: dir}
	for _, item := range items {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		spec, _ := obj["spec"].(map[string]any)
		group, _ := spec["group"].(string)
		if group != monitoringAPIGroup {
			continue
		}
		meta, _ := obj["metadata"].(map[string]any)
		name, _ := meta["name"].(string)
		if name == "" {
			continue
		}
		stripK8sMeta(obj)
		crdJSON, err := json.Marshal(obj)
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(crdDir, name+".json"), crdJSON, 0o644); err != nil {
			return nil, err
		}
		crRaw, err := runKubectl(ctx, "get", name, "-A", "-o", "json")
		if err != nil {
			return nil, fmt.Errorf("list instances of %s: %w", name, err)
		}
		var crList map[string]any
		if err := json.Unmarshal(crRaw, &crList); err != nil {
			return nil, fmt.Errorf("decode instances of %s: %w", name, err)
		}
		if crItems, ok := crList["items"].([]any); ok {
			cleaned := make([]any, 0, len(crItems))
			for _, cr := range crItems {
				crObj, ok := cr.(map[string]any)
				if !ok {
					continue
				}
				stripK8sMeta(crObj)
				cleaned = append(cleaned, crObj)
			}
			crList["items"] = cleaned
		}
		crJSON, err := json.Marshal(crList)
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(crDir, name+".json"), crJSON, 0o644); err != nil {
			return nil, err
		}
		backup.CRDNames = append(backup.CRDNames, name)
		GinkgoWriter.Printf("已备份 CRD %s\n", name)
	}
	if len(backup.CRDNames) == 0 {
		_ = os.RemoveAll(dir)
		return nil, fmt.Errorf("未找到 %s CRD，无法备份", monitoringAPIGroup)
	}
	GinkgoWriter.Printf("monitoring CRD 备份目录: %s (%d 个)\n", dir, len(backup.CRDNames))
	return backup, nil
}

// DeleteMonitoringCRDs 按备份清单删除 monitoring.coreos.com CRD（实例随 CRD 级联删除）。
func DeleteMonitoringCRDs(ctx context.Context, backup *MonitoringCRDBackup) error {
	if backup == nil {
		return nil
	}
	for _, name := range backup.CRDNames {
		GinkgoWriter.Printf("删除 CRD %s\n", name)
		if _, err := runKubectl(ctx, "delete", "crd", name, "--wait=true", "--timeout=180s", "--ignore-not-found=true"); err != nil {
			return fmt.Errorf("delete crd %s: %w", name, err)
		}
	}
	return nil
}

// RestoreMonitoringCRDs 先创建去 uid 的 CRD，Established 后再 restore 实例。
func RestoreMonitoringCRDs(ctx context.Context, backup *MonitoringCRDBackup) error {
	if backup == nil {
		return nil
	}
	defer func() {
		if backup.Dir != "" {
			_ = os.RemoveAll(backup.Dir)
		}
	}()
	crdDir := filepath.Join(backup.Dir, "crds")
	crDir := filepath.Join(backup.Dir, "crs")
	for _, name := range backup.CRDNames {
		path := filepath.Join(crdDir, name+".json")
		GinkgoWriter.Printf("复原 CRD %s\n", name)
		if _, err := runKubectl(ctx, "get", "crd", name); err == nil {
			if _, err := runKubectl(ctx, "apply", "--server-side", "--force-conflicts", "-f", path); err != nil {
				return fmt.Errorf("apply existing crd %s: %w", name, err)
			}
		} else if _, err := runKubectl(ctx, "create", "-f", path); err != nil {
			return fmt.Errorf("create crd %s: %w", name, err)
		}
	}
	deadline := time.Now().Add(2 * time.Minute)
	for _, name := range backup.CRDNames {
		for {
			if time.Now().After(deadline) {
				return fmt.Errorf("wait crd %s Established timeout", name)
			}
			if _, err := runKubectl(ctx, "wait", "--for=condition=Established", "crd/"+name, "--timeout=30s"); err == nil {
				break
			}
			time.Sleep(2 * time.Second)
		}
	}
	for _, name := range backup.CRDNames {
		path := filepath.Join(crDir, name+".json")
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var crList map[string]any
		if err := json.Unmarshal(raw, &crList); err != nil {
			return fmt.Errorf("decode restored instances of %s: %w", name, err)
		}
		items, _ := crList["items"].([]any)
		if len(items) == 0 {
			continue
		}
		GinkgoWriter.Printf("复原 %s 实例 %d 个\n", name, len(items))
		if _, err := runKubectl(ctx, "apply", "--server-side", "--force-conflicts", "-f", path); err != nil {
			return fmt.Errorf("restore instances of %s: %w", name, err)
		}
	}
	GinkgoWriter.Println("monitoring.coreos.com CRD 与实例已复原")
	return nil
}

func stripK8sMeta(obj map[string]any) {
	meta, _ := obj["metadata"].(map[string]any)
	if meta != nil {
		for _, k := range []string{
			"uid", "resourceVersion", "generation", "creationTimestamp",
			"managedFields", "deletionTimestamp", "deletionGracePeriodSeconds",
			"selfLink",
		} {
			delete(meta, k)
		}
		if ann, ok := meta["annotations"].(map[string]any); ok {
			delete(ann, "kubectl.kubernetes.io/last-applied-configuration")
			if len(ann) == 0 {
				delete(meta, "annotations")
			} else {
				meta["annotations"] = ann
			}
		}
		obj["metadata"] = meta
	}
	delete(obj, "status")
}

func runKubectl(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = strings.TrimSpace(stdout.String())
		}
		return stdout.Bytes(), fmt.Errorf("kubectl %s: %w (%s)", strings.Join(args, " "), err, msg)
	}
	return stdout.Bytes(), nil
}
