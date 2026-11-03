package capbke

import (
	"context"
	"crypto/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

func TestBKENode_Default(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	client := fake.NewClientBuilder().WithScheme(scheme).Build()
	webhook := &BKENode{Client: client}

	t.Run("set default port", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
		}
		err := webhook.Default(context.Background(), node)
		assert.NoError(t, err)
		assert.Equal(t, "22", node.Spec.Port)
	})

	t.Run("keep existing port", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "2222"},
		}
		err := webhook.Default(context.Background(), node)
		assert.NoError(t, err)
		assert.Equal(t, "2222", node.Spec.Port)
	})
}

func TestBKENode_ValidateCreate(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	t.Run("valid node", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{
				IP:       "192.168.1.1",
				Port:     "22",
				Username: "root",
				Password: "test",
				Role:     []string{"master"},
			},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateCreate(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("missing cluster label", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1"},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateCreate(context.Background(), node)
		assert.Error(t, err)
	})

	t.Run("create master node during initial deploy allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master", "etcd"}},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateCreate(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("create master node during scale-up not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master", "etcd"}},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(&bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
			}).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateCreate(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node scale-up is not allowed")
	})

	t.Run("create worker node during scale-up allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"node"}},
		}
		client := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(&bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
			}).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateCreate(context.Background(), node)
		assert.NoError(t, err)
	})
}

func TestBKENode_ValidateUpdate(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	t.Run("IP change not allowed", func(t *testing.T) {
		oldNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master"}},
		}
		newNode := oldNode.DeepCopy()
		newNode.Spec.IP = "192.168.1.2"

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldNode, newNode)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "IP cannot be changed")
	})

	t.Run("role change in pending state not allowed", func(t *testing.T) {
		oldNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodePending},
		}
		newNode := oldNode.DeepCopy()
		newNode.Spec.Role = []string{"worker"}

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldNode, newNode)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot change node role")
	})

	t.Run("worker to master role change not allowed", func(t *testing.T) {
		oldNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"node"}},
		}
		newNode := oldNode.DeepCopy()
		newNode.Spec.Role = []string{"master", "etcd"}

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldNode, newNode)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "changing node role to master is not allowed")
	})

	t.Run("master to master role unchanged allowed", func(t *testing.T) {
		oldNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"master", "etcd"}},
		}
		newNode := oldNode.DeepCopy()
		newNode.Spec.Hostname = "new-hostname"

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldNode, newNode)
		assert.NoError(t, err)
	})

	t.Run("worker to worker role unchanged allowed", func(t *testing.T) {
		oldNode := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1", Port: "22", Username: "root", Password: "test", Role: []string{"node"}},
		}
		newNode := oldNode.DeepCopy()
		newNode.Spec.Hostname = "new-hostname"

		client := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: client}
		_, err := webhook.ValidateUpdate(context.Background(), oldNode, newNode)
		assert.NoError(t, err)
	})
}

func TestBKENode_ValidateDelete(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	// 预创建 BKECluster 的 fake client，用于需要关联集群的测试
	withCluster := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(&bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
		}).Build()

	// 无 BKECluster 的 fake client
	withoutCluster := fake.NewClientBuilder().WithScheme(scheme).Build()

	t.Run("delete pending node not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodePending},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot delete node while in pending state")
	})

	t.Run("delete upgrading node not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodeUpgrading},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot delete node while upgrading")
	})

	t.Run("delete ready node allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete ready master node not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{Client: withCluster}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node deletion is not allowed")
	})

	t.Run("delete provisioned master node not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeProvisioned},
		}
		webhook := &BKENode{Client: withCluster}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node deletion is not allowed")
	})

	t.Run("delete ready master/worker node not allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master/node", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{Client: withCluster}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node deletion is not allowed")
	})

	t.Run("delete deleting master node allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodeDeleting},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete failed master node allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodeFailed},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete init failed master node allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status:     confv1beta1.BKENodeStatus{State: bkev1beta1.NodeInitFailed},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete ready worker node allowed", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
			Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"node"}},
			Status:     confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete master node allowed when bkecluster missing", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{Client: withoutCluster}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.NoError(t, err)
	})

	t.Run("delete master node blocked when bkecluster exists", func(t *testing.T) {
		node := &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: confv1beta1.NodeReady},
		}
		webhook := &BKENode{Client: withCluster}
		_, err := webhook.ValidateDelete(context.Background(), node)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node deletion is not allowed")
	})
}

func TestBKENode_ValidateDelete_BKEClusterDeleting(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	// 带 DeletionTimestamp 的 BKECluster，模拟集群删除中（真实场景下删除中的对象必有 finalizer）
	now := metav1.Now()
	withDeletingCluster := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(&bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "test-cluster",
				Namespace:         "default",
				DeletionTimestamp: &now,
				Finalizers:        []string{"bke.bocloud.com/clusters"},
			},
		}).Build()

	newNode := func(state confv1beta1.NodeState) *confv1beta1.BKENode {
		return &confv1beta1.BKENode{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node",
				Namespace: "default",
				Labels:    map[string]string{nodeutil.ClusterNameLabel: "test-cluster"},
			},
			Spec:   confv1beta1.BKENodeSpec{IP: "192.168.1.1", Role: []string{"master", "etcd"}},
			Status: confv1beta1.BKENodeStatus{State: state},
		}
	}

	// issue 312 核心场景：创建失败的 HA 集群中 Initializing 状态的 master 节点
	t.Run("initializing master node allowed when bkecluster is deleting", func(t *testing.T) {
		webhook := &BKENode{Client: withDeletingCluster}
		_, err := webhook.ValidateDelete(context.Background(), newNode(bkev1beta1.NodeInitializing))
		assert.NoError(t, err)
	})

	t.Run("ready master node allowed when bkecluster is deleting", func(t *testing.T) {
		webhook := &BKENode{Client: withDeletingCluster}
		_, err := webhook.ValidateDelete(context.Background(), newNode(confv1beta1.NodeReady))
		assert.NoError(t, err)
	})

	t.Run("pending node allowed when bkecluster is deleting", func(t *testing.T) {
		webhook := &BKENode{Client: withDeletingCluster}
		_, err := webhook.ValidateDelete(context.Background(), newNode(confv1beta1.NodePending))
		assert.NoError(t, err)
	})

	t.Run("upgrading node allowed when bkecluster is deleting", func(t *testing.T) {
		webhook := &BKENode{Client: withDeletingCluster}
		_, err := webhook.ValidateDelete(context.Background(), newNode(confv1beta1.NodeUpgrading))
		assert.NoError(t, err)
	})

	// 回归保护：集群未删除时 Initializing master 仍被拦截（缩容场景）
	t.Run("initializing master node blocked when bkecluster is not deleting", func(t *testing.T) {
		notDeleting := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(&bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
			}).Build()
		webhook := &BKENode{Client: notDeleting}
		_, err := webhook.ValidateDelete(context.Background(), newNode(bkev1beta1.NodeInitializing))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "master node deletion is not allowed")
	})

	// spec.reset 驱动的卸载路径：对象未被删除，仅 ClusterStatus 被控制器置为 Deleting
	// （前台 Web 删除与文档主推的后台删除均为此形态）
	t.Run("initializing master node allowed when only cluster status is deleting", func(t *testing.T) {
		resetCluster := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(&bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
				Status:     confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterDeleting},
			}).Build()
		webhook := &BKENode{Client: resetCluster}
		_, err := webhook.ValidateDelete(context.Background(), newNode(bkev1beta1.NodeInitializing))
		assert.NoError(t, err)
	})

	t.Run("isBKEClusterDeleting nil client does not panic", func(t *testing.T) {
		webhook := &BKENode{}
		node := newNode(confv1beta1.NodeReady)
		assert.NotPanics(t, func() {
			assert.False(t, webhook.isBKEClusterDeleting(context.Background(), node))
		})
	})

	t.Run("isBKEClusterDeleting cluster not found returns false", func(t *testing.T) {
		emptyClient := fake.NewClientBuilder().WithScheme(scheme).Build()
		webhook := &BKENode{Client: emptyClient}
		node := newNode(confv1beta1.NodeReady)
		assert.False(t, webhook.isBKEClusterDeleting(context.Background(), node))
	})

	t.Run("isBKEClusterDeleting missing cluster label returns false", func(t *testing.T) {
		webhook := &BKENode{Client: withDeletingCluster}
		node := newNode(confv1beta1.NodeReady)
		node.Labels = nil
		assert.False(t, webhook.isBKEClusterDeleting(context.Background(), node))
	})
}

func TestEncryptPasswordIfNeeded(t *testing.T) {
	// Register a valid 32-byte AES key provider so AesEncrypt does not fail
	// with an uninitialized provider. The provider is package-level, so it is
	// restored on cleanup.
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("failed to generate test key: %v", err)
	}
	security.SetKeyProvider(&staticKeyProvider{key: key})
	t.Cleanup(func() { security.SetKeyProvider(nil) })

	t.Run("plain password", func(t *testing.T) {
		result, err := encryptPasswordIfNeeded("plaintext")
		assert.NoError(t, err)
		assert.NotEqual(t, "plaintext", result)
	})

	t.Run("empty password", func(t *testing.T) {
		result, err := encryptPasswordIfNeeded("")
		assert.NoError(t, err)
		assert.NotEmpty(t, result)
	})
}

// staticKeyProvider is a trivial security.KeyProvider for tests.
type staticKeyProvider struct {
	key []byte
	err error
}

func (p *staticKeyProvider) Key(_ context.Context) ([]byte, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.key, nil
}

// TestBKENode_DefaultEncryptFailClosed verifies that Default rejects the
// request when password encryption fails, instead of storing the plaintext
// password in etcd.
func TestBKENode_DefaultEncryptFailClosed(t *testing.T) {
	security.SetKeyProvider(&staticKeyProvider{err: assert.AnError})
	t.Cleanup(func() { security.SetKeyProvider(nil) })

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	client := fake.NewClientBuilder().WithScheme(scheme).Build()
	webhook := &BKENode{Client: client}

	node := &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node", Namespace: "default"},
		Spec:       confv1beta1.BKENodeSpec{IP: "192.168.1.1", Password: "secret-password"},
	}
	err := webhook.Default(context.Background(), node)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to encrypt password")
	// The plaintext password must not survive in the object.
	assert.Equal(t, "secret-password", node.Spec.Password)
}

func TestBKENode_ValidateCreateInvalidType(t *testing.T) {
	webhook := &BKENode{}
	_, err := webhook.ValidateCreate(context.Background(), &bkev1beta1.BKECluster{})
	assert.Error(t, err)
}

func TestBKENode_ValidateUpdateInvalidType(t *testing.T) {
	webhook := &BKENode{}
	_, err := webhook.ValidateUpdate(context.Background(), &bkev1beta1.BKECluster{}, &bkev1beta1.BKECluster{})
	assert.Error(t, err)
}

func TestBKENode_ValidateDeleteInvalidType(t *testing.T) {
	webhook := &BKENode{}
	_, err := webhook.ValidateDelete(context.Background(), &bkev1beta1.BKECluster{})
	assert.Error(t, err)
}

func TestBKENode_DefaultInvalidType(t *testing.T) {
	webhook := &BKENode{}
	err := webhook.Default(context.Background(), &bkev1beta1.BKECluster{})
	assert.Error(t, err)
}
