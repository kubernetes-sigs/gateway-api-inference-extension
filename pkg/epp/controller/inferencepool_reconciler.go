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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// InferencePoolReconciler utilizes the controller runtime to reconcile Instance Gateway resources
// This implementation is just used for reading & maintaining data sync. The Gateway implementation
// will have the proper controller that will create/manage objects on behalf of the server pool.
type InferencePoolReconciler struct {
	client.Reader
	Datastore datastore.Datastore
}

func (c *InferencePoolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)
	logger.Info("Reconciling InferencePool")

	v1infPool := &v1.InferencePool{}
	if err := c.Get(ctx, req.NamespacedName, v1infPool); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("InferencePool not found. Clearing the datastore")
			c.Datastore.Clear()
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("unable to get InferencePool - %w", err)
	}

	if !v1infPool.GetDeletionTimestamp().IsZero() {
		logger.Info("InferencePool is marked for deletion. Clearing the datastore")
		c.Datastore.Clear()
		return ctrl.Result{}, nil
	}

	if err := c.Datastore.PoolSet(ctx, c.Reader, v1infPool); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update datastore - %w", err)
	}

	return ctrl.Result{}, nil
}

func (c *InferencePoolReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.InferencePool{}).
		Complete(c)
}
