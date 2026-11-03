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

package patchutil

import (
	"os"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestMain(m *testing.M) {
	_ = os.MkdirAll(os.TempDir(), 0o1777)
	os.Exit(m.Run())
}

type componentStatus struct {
	Name     string `json:"name"`
	Resource string `json:"resource"`
	Health   bool   `json:"componentHealth"`
	Message  string `json:"message"`
}

type productStatus struct {
	Name           string            `json:"name"`
	StartTime      *metav1.Time      `json:"startTime,omitempty"`
	UpdateTime     *metav1.Time      `json:"updateTime,omitempty"`
	CompletionTime *metav1.Time      `json:"completionTime,omitempty"`
	Health         bool              `json:"health"`
	Component      []componentStatus `json:"component,omitempty"`
	Reason         string            `json:"reason"`
}

func TestDiff(t *testing.T) {
	now := time.Now()
	oldobj := productStatus{
		Name: "test",
		StartTime: &metav1.Time{
			Time: now,
		},
		UpdateTime: &metav1.Time{
			Time: now.Add(time.Minute),
		},
		CompletionTime: &metav1.Time{
			Time: now.Add(time.Hour),
		},
		Health: true,
		Component: []componentStatus{
			{
				Name:     "test",
				Resource: "",
				Health:   true,
				Message:  "ok",
			},
		},
		Reason: "no reason",
	}

	newobj := productStatus{
		Name: "test",
		StartTime: &metav1.Time{
			Time: now,
		},
		UpdateTime: &metav1.Time{
			Time: now.Add(time.Minute),
		},
		CompletionTime: &metav1.Time{
			Time: now.Add(time.Hour),
		},
		Health: true,
		Component: []componentStatus{
			{
				Name:     "test",
				Resource: "",
				Health:   true,
				Message:  "ok",
			},
			{
				Name:     "test2",
				Resource: "",
				Health:   true,
				Message:  "ok",
			},
		},
		Reason: "no reason",
	}

	diff, err := Diff(oldobj, newobj)
	if err != nil {
		t.Fatal(err)
	}
	if len(diff) == 0 {
		t.Fatal("expected non-empty diff")
	}

	for _, d := range diff {
		switch d.Kind() {
		case "add":
			valueInterface, err := d.ValueInterface()
			if err != nil {
				t.Fatal(err)
			}
			value, err := json.Marshal(valueInterface)
			if err != nil {
				t.Fatal(err)
			}
			var component componentStatus
			if err := json.Unmarshal(value, &component); err != nil {
				t.Fatal(err)
			}
			t.Logf("component: %v, health: %v, msg: %v", component.Name, component.Health, component.Message)
		}
	}
}

func TestGetDiffPaths(t *testing.T) {
	oldobj := map[string]string{"a": "1"}
	newobj := map[string]string{"a": "2", "b": "3"}

	diff, err := Diff(oldobj, newobj)
	if err != nil {
		t.Fatal(err)
	}

	paths := GetDiffPaths(diff)
	if len(paths) == 0 {
		t.Fatal("expected diff paths")
	}
}
