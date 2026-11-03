package controllers

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/bkeagent/api/v1beta1"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(agentv1beta1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func TestUpdateCommandStatus(t *testing.T) {
	ctx := ctrl.SetupSignalHandler()
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}
	go func() {
		if err := mgr.Start(ctx); err != nil {
			setupLog.Error(err, "problem running manager")
			os.Exit(1)
		}
	}()
	time.Sleep(2 * time.Second)
	obj := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo",
			Namespace: "default",
		},
		Spec: agentv1beta1.CommandSpec{},
	}
	err = mgr.GetClient().Create(ctx, obj)
	if err != nil {
		fmt.Println(err.Error())
		if !strings.Contains(err.Error(), "exists") {
			t.Error(err)
		}
	}
	var wg sync.WaitGroup
	for i := 1; i < 100; i++ {
		wg.Add(1)
		go func(num int) {
			defer wg.Done()
			node := fmt.Sprintf("node-%d-%d", num, rand.IntnRange(1, 1000))
			for {
				time.Sleep(time.Duration(rand.IntnRange(1, 20)) * time.Second)
				obj2 := &agentv1beta1.Command{}
				err = mgr.GetAPIReader().Get(ctx, client.ObjectKey{Namespace: "default", Name: "demo"}, obj2)
				if err != nil {
					fmt.Println(err.Error())
					continue
				}
				if obj2.Status == nil {
					obj2.Status = map[string]*agentv1beta1.CommandStatus{}
				}
				if _, ok := obj2.Status[node]; ok {
					break
				}
				obj2.Status[node] = &agentv1beta1.CommandStatus{
					LastStartTime: &metav1.Time{Time: time.Now()},
					Succeeded:     rand.IntnRange(1, 20),
					Failed:        rand.IntnRange(1, 20),
				}
				err = mgr.GetClient().Status().Update(ctx, obj2)
				if err != nil {
					fmt.Println(fmt.Sprintf("node: %s, num: %d, error: %s", node, num, "failed"))
					continue
				}
				fmt.Println(fmt.Sprintf("node: %s, num: %d, update status success", node, num))
				break
			}
		}(i)
	}
	wg.Wait()
}

func TestSyncStatusUntilComplete(t *testing.T) {
	ctx := ctrl.SetupSignalHandler()
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}
	go func() {
		if err := mgr.Start(ctx); err != nil {
			setupLog.Error(err, "problem running manager")
			os.Exit(1)
		}
	}()
	time.Sleep(2 * time.Second)

	cr := &CommandReconciler{
		Client:    mgr.GetClient(),
		APIReader: mgr.GetAPIReader(),
		Scheme:    mgr.GetScheme(),
		NodeName:  "nodeName",
		Ctx:       ctx,
	}
	obj := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo2",
			Namespace: "default",
		},
		Spec: agentv1beta1.CommandSpec{},
	}
	err = mgr.GetClient().Create(ctx, obj)
	if err != nil {
		fmt.Println(err.Error())
		if !strings.Contains(err.Error(), "exists") {
			t.Error(err)
		}
	}
	obj2 := &agentv1beta1.Command{}
	err = mgr.GetClient().Get(ctx, client.ObjectKey{Namespace: "default", Name: "demo2"}, obj2)
	if err != nil {
		t.Error(err)
	}
	if obj2.Status == nil {
		obj2.Status = map[string]*agentv1beta1.CommandStatus{}
	}
	if _, ok := obj2.Status[cr.NodeName]; ok {
		return
	}
	obj2.Status[cr.NodeName] = &agentv1beta1.CommandStatus{
		LastStartTime: &metav1.Time{Time: time.Now()},
		Succeeded:     rand.IntnRange(1, 20),
		Failed:        rand.IntnRange(1, 20),
	}
	err = cr.syncStatusUntilComplete(obj2)
	if err != nil {
		t.Error(err)
	}
}
