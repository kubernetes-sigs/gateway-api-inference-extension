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
	"reflect"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// InferencePoolReconciler utilizes the controller runtime to reconcile Instance Gateway resources
// This implementation is just used for reading & maintaining data sync. The Gateway implementation
// will have the proper controller that will create/manage objects on behalf of the server pool.
type InferencePoolReconciler struct {
	client.Client
	Record             record.EventRecorder
	PoolNamespacedName types.NamespacedName
	Datastore          datastore.Datastore
}

func (c *InferencePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("inferencePool", req.NamespacedName).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferencePool")

	infPool := &v1alpha2.InferencePool{}

	if err := c.Get(ctx, req.NamespacedName, infPool); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("InferencePool not found. Clearing the datastore")
			c.Datastore.Clear()
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Unable to get InferencePool")
		return ctrl.Result{}, err
	} else if !infPool.DeletionTimestamp.IsZero() {
		logger.Info("InferencePool is marked for deletion. Clearing the datastore")
		c.Datastore.Clear()
		return ctrl.Result{}, nil
	}

	c.updateDatastore(ctx, infPool)

	return ctrl.Result{}, nil
}

func (c *InferencePoolReconciler) updateDatastore(ctx context.Context, newPool *v1alpha2.InferencePool) {
	logger := log.FromContext(ctx)
	oldPool, err := c.Datastore.PoolGet()
	c.Datastore.PoolSet(newPool)
	if err != nil || !reflect.DeepEqual(newPool.Spec.Selector, oldPool.Spec.Selector) {
		logger.V(logutil.DEFAULT).Info("Updating inference pool endpoints", "selector", newPool.Spec.Selector)
		// A full resync is required to address two cases:
		// 1) At startup, the pod events may get processed before the pool is synced with the datastore,
		//    and hence they will not be added to the store since pool selector is not known yet
		// 2) If the selector on the pool was updated, then we will not get any pod events, and so we need
		//    to resync the whole pool: remove pods in the store that don't match the new selector and add
		//    the ones that may have existed already to the store.
		c.Datastore.PodResyncAll(ctx, c.Client, newPool)
	}
}

func (c *InferencePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferencePool{}).
		Complete(c)
}
