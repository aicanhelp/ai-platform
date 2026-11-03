/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package fake

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/testing"

	v1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

// FakeColocationWorkloadMatchRules implements ColocationWorkloadMatchRuleInterface
type FakeColocationWorkloadMatchRules struct {
	Fake *FakeColocationV1
	ns   string
}

var colocationworkloadmatchrulesResource = v1.SchemeGroupVersion.WithResource("colocationworkloadmatchrules")

var colocationworkloadmatchrulesKind = v1.SchemeGroupVersion.WithKind("ColocationWorkloadMatchRule")

// Get takes name of the colocationWorkloadMatchRule, and returns the corresponding colocationWorkloadMatchRule object, and an error if there is any.
func (c *FakeColocationWorkloadMatchRules) Get(ctx context.Context, name string,
	options metav1.GetOptions) (*v1.ColocationWorkloadMatchRule, error) {
	obj, err := c.Fake.Invokes(
		testing.NewGetAction(colocationworkloadmatchrulesResource, c.ns, name), &v1.ColocationWorkloadMatchRule{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ColocationWorkloadMatchRule), err
}

// List takes label and field selectors, and returns the list of ColocationWorkloadMatchRules that match those selectors.
func (c *FakeColocationWorkloadMatchRules) List(ctx context.Context,
	opts metav1.ListOptions) (*v1.ColocationWorkloadMatchRuleList, error) {
	obj, err := c.Fake.Invokes(testing.NewListAction(colocationworkloadmatchrulesResource,
		colocationworkloadmatchrulesKind, c.ns, opts), &v1.ColocationWorkloadMatchRuleList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.ColocationWorkloadMatchRuleList{ListMeta: obj.(*v1.ColocationWorkloadMatchRuleList).ListMeta}
	for _, item := range obj.(*v1.ColocationWorkloadMatchRuleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested colocationWorkloadMatchRules.
func (c *FakeColocationWorkloadMatchRules) Watch(
	ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(colocationworkloadmatchrulesResource, c.ns, opts))

}

// Create takes the representation of a colocationWorkloadMatchRule and creates it.  Returns the server's representation of the colocationWorkloadMatchRule, and an error, if there is any.
func (c *FakeColocationWorkloadMatchRules) Create(ctx context.Context,
	colocationWorkloadMatchRule *v1.ColocationWorkloadMatchRule,
	opts metav1.CreateOptions) (*v1.ColocationWorkloadMatchRule, error) {
	obj, err := c.Fake.Invokes(testing.NewCreateAction(colocationworkloadmatchrulesResource,
		c.ns, colocationWorkloadMatchRule), &v1.ColocationWorkloadMatchRule{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ColocationWorkloadMatchRule), err
}

// Update takes the representation of a colocationWorkloadMatchRule and updates it. Returns the server's representation of the colocationWorkloadMatchRule, and an error, if there is any.
func (c *FakeColocationWorkloadMatchRules) Update(ctx context.Context,
	colocationWorkloadMatchRule *v1.ColocationWorkloadMatchRule,
	opts metav1.UpdateOptions) (*v1.ColocationWorkloadMatchRule, error) {
	obj, err := c.Fake.Invokes(testing.NewUpdateAction(colocationworkloadmatchrulesResource,
		c.ns, colocationWorkloadMatchRule), &v1.ColocationWorkloadMatchRule{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ColocationWorkloadMatchRule), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeColocationWorkloadMatchRules) UpdateStatus(ctx context.Context,
	colocationWorkloadMatchRule *v1.ColocationWorkloadMatchRule,
	opts metav1.UpdateOptions) (*v1.ColocationWorkloadMatchRule, error) {
	obj, err := c.Fake.Invokes(testing.NewUpdateSubresourceAction(colocationworkloadmatchrulesResource,
		"status", c.ns, colocationWorkloadMatchRule), &v1.ColocationWorkloadMatchRule{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ColocationWorkloadMatchRule), err
}

// Delete takes name of the colocationWorkloadMatchRule and deletes it. Returns an error if one occurs.
func (c *FakeColocationWorkloadMatchRules) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	_, err := c.Fake.Invokes(testing.NewDeleteActionWithOptions(
		colocationworkloadmatchrulesResource, c.ns, name, opts), &v1.ColocationWorkloadMatchRule{})

	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeColocationWorkloadMatchRules) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions,
	listOpts metav1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(colocationworkloadmatchrulesResource, c.ns, listOpts)

	_, err := c.Fake.Invokes(action, &v1.ColocationWorkloadMatchRuleList{})
	return err
}

// Patch applies the patch and returns the patched colocationWorkloadMatchRule.
func (c *FakeColocationWorkloadMatchRules) Patch(ctx context.Context, name string, pt types.PatchType, data []byte,
	opts metav1.PatchOptions, subresources ...string) (*v1.ColocationWorkloadMatchRule, error) {
	obj, err := c.Fake.Invokes(testing.NewPatchSubresourceAction(colocationworkloadmatchrulesResource,
		c.ns, name, pt, data, subresources...), &v1.ColocationWorkloadMatchRule{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ColocationWorkloadMatchRule), err
}
