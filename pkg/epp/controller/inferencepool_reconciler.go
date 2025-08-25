/*
Copyright 2025 The Kubernetes Authors.

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

package controller

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// InferencePoolReconciler utilizes the controller runtime to reconcile Instance Gateway resources
// This implementation is just used for reading & maintaining data sync. The Gateway implementation
// will have the proper controller that will create/manage objects on behalf of the server pool.
type InferencePoolReconciler struct {
	client.Reader
	Datastore datastore.Datastore
	PoolGKNN  common.GKNN
}

func (c *InferencePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("group", c.PoolGKNN.Group).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferencePool")

	// 1. Initialize a generic client.Object based on the group.
	var obj client.Object
	switch c.PoolGKNN.Group {
	case v1.GroupName:
		obj = &v1.InferencePool{}
	case v1alpha2.GroupName:
		obj = &v1alpha2.InferencePool{}
	default:
		// Handle unsupported groups gracefully.
		err := fmt.Errorf("unsupported API group: %s", c.PoolGKNN.Group)
		logger.Error(err, "Cannot reconcile InferencePool")
		return ctrl.Result{}, err
	}

	// 2. Perform a single, generic fetch for the object.
	if err := c.Get(ctx, req.NamespacedName, obj); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("InferencePool not found. Clearing the datastore")
			c.Datastore.Clear()
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Unable to get InferencePool")
		return ctrl.Result{}, err
	}

	// 3. Perform common checks using the client.Object interface.
	if !obj.GetDeletionTimestamp().IsZero() {
		logger.Info("InferencePool is marked for deletion. Clearing the datastore")
		c.Datastore.Clear()
		return ctrl.Result{}, nil
	}

	// 4. Convert the fetched object to the canonical v1.InferencePool.
	v1infPool := &v1.InferencePool{}

	switch pool := obj.(type) {
	case *v1.InferencePool:
		// If it's already a v1 object, just use it.
		v1infPool = pool
	case *v1alpha2.InferencePool:
		var err error
		err = pool.ConvertTo(v1infPool)
		if err != nil {
			logger.Error(err, "Failed to convert XInferencePool to InferencePool")
			return ctrl.Result{}, err
		}
	default:
		return ctrl.Result{}, fmt.Errorf("unsupported API group: %s", c.PoolGKNN.Group)
	}

	if err := c.Datastore.PoolSet(ctx, c.Reader, v1infPool); err != nil {
		logger.Error(err, "Failed to update datastore")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (c *InferencePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	switch c.PoolGKNN.Group {
	case v1alpha2.GroupName:
		return ctrl.NewControllerManagedBy(mgr).
			For(&v1alpha2.InferencePool{}).
			Complete(c)
	case v1.GroupName:
		return ctrl.NewControllerManagedBy(mgr).
			For(&v1.InferencePool{}).
			Complete(c)
	default:
		return fmt.Errorf("unknown group %s", c.PoolGKNN.Group)
	}
}
