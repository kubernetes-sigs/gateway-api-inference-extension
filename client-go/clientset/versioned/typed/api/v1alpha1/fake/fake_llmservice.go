/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// Code generated by client-gen. DO NOT EDIT.

package fake

import (
	context "context"
	json "encoding/json"
	fmt "fmt"

	v1alpha1 "inference.networking.x-k8s.io/llm-instance-gateway/api/v1alpha1"
	apiv1alpha1 "inference.networking.x-k8s.io/llm-instance-gateway/client-go/applyconfiguration/api/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeLLMServices implements LLMServiceInterface
type FakeLLMServices struct {
	Fake *FakeApiV1alpha1
	ns   string
}

var llmservicesResource = v1alpha1.SchemeGroupVersion.WithResource("llmservices")

var llmservicesKind = v1alpha1.SchemeGroupVersion.WithKind("LLMService")

// Get takes name of the lLMService, and returns the corresponding lLMService object, and an error if there is any.
func (c *FakeLLMServices) Get(ctx context.Context, name string, options v1.GetOptions) (result *v1alpha1.LLMService, err error) {
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewGetActionWithOptions(llmservicesResource, c.ns, name, options), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// List takes label and field selectors, and returns the list of LLMServices that match those selectors.
func (c *FakeLLMServices) List(ctx context.Context, opts v1.ListOptions) (result *v1alpha1.LLMServiceList, err error) {
	emptyResult := &v1alpha1.LLMServiceList{}
	obj, err := c.Fake.
		Invokes(testing.NewListActionWithOptions(llmservicesResource, llmservicesKind, c.ns, opts), emptyResult)

	if obj == nil {
		return emptyResult, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.LLMServiceList{ListMeta: obj.(*v1alpha1.LLMServiceList).ListMeta}
	for _, item := range obj.(*v1alpha1.LLMServiceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested lLMServices.
func (c *FakeLLMServices) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchActionWithOptions(llmservicesResource, c.ns, opts))

}

// Create takes the representation of a lLMService and creates it.  Returns the server's representation of the lLMService, and an error, if there is any.
func (c *FakeLLMServices) Create(ctx context.Context, lLMService *v1alpha1.LLMService, opts v1.CreateOptions) (result *v1alpha1.LLMService, err error) {
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewCreateActionWithOptions(llmservicesResource, c.ns, lLMService, opts), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// Update takes the representation of a lLMService and updates it. Returns the server's representation of the lLMService, and an error, if there is any.
func (c *FakeLLMServices) Update(ctx context.Context, lLMService *v1alpha1.LLMService, opts v1.UpdateOptions) (result *v1alpha1.LLMService, err error) {
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewUpdateActionWithOptions(llmservicesResource, c.ns, lLMService, opts), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeLLMServices) UpdateStatus(ctx context.Context, lLMService *v1alpha1.LLMService, opts v1.UpdateOptions) (result *v1alpha1.LLMService, err error) {
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceActionWithOptions(llmservicesResource, "status", c.ns, lLMService, opts), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// Delete takes name of the lLMService and deletes it. Returns an error if one occurs.
func (c *FakeLLMServices) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteActionWithOptions(llmservicesResource, c.ns, name, opts), &v1alpha1.LLMService{})

	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeLLMServices) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	action := testing.NewDeleteCollectionActionWithOptions(llmservicesResource, c.ns, opts, listOpts)

	_, err := c.Fake.Invokes(action, &v1alpha1.LLMServiceList{})
	return err
}

// Patch applies the patch and returns the patched lLMService.
func (c *FakeLLMServices) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.LLMService, err error) {
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(llmservicesResource, c.ns, name, pt, data, opts, subresources...), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied lLMService.
func (c *FakeLLMServices) Apply(ctx context.Context, lLMService *apiv1alpha1.LLMServiceApplyConfiguration, opts v1.ApplyOptions) (result *v1alpha1.LLMService, err error) {
	if lLMService == nil {
		return nil, fmt.Errorf("lLMService provided to Apply must not be nil")
	}
	data, err := json.Marshal(lLMService)
	if err != nil {
		return nil, err
	}
	name := lLMService.Name
	if name == nil {
		return nil, fmt.Errorf("lLMService.Name must be provided to Apply")
	}
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(llmservicesResource, c.ns, *name, types.ApplyPatchType, data, opts.ToPatchOptions()), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}

// ApplyStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating ApplyStatus().
func (c *FakeLLMServices) ApplyStatus(ctx context.Context, lLMService *apiv1alpha1.LLMServiceApplyConfiguration, opts v1.ApplyOptions) (result *v1alpha1.LLMService, err error) {
	if lLMService == nil {
		return nil, fmt.Errorf("lLMService provided to Apply must not be nil")
	}
	data, err := json.Marshal(lLMService)
	if err != nil {
		return nil, err
	}
	name := lLMService.Name
	if name == nil {
		return nil, fmt.Errorf("lLMService.Name must be provided to Apply")
	}
	emptyResult := &v1alpha1.LLMService{}
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(llmservicesResource, c.ns, *name, types.ApplyPatchType, data, opts.ToPatchOptions(), "status"), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(*v1alpha1.LLMService), err
}
