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
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type InferenceObjectiveReconciler struct {
	client.Reader
	Datastore datastore.Datastore
	PoolGKNN  common.GKNN
}

func (c *InferenceObjectiveReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceObjective")

	infObjective := &v1alpha2.InferenceObjective{}
	notFound := false
	if err := c.Get(ctx, req.NamespacedName, infObjective); err != nil {
		if !errors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("unable to get InferenceObjective - %w", err)
		}
		notFound = true
	}

	if notFound || !infObjective.DeletionTimestamp.IsZero() || infObjective.Spec.PoolRef.Name != v1alpha2.ObjectName(c.PoolGKNN.Name) || infObjective.Spec.PoolRef.Group != v1alpha2.Group(c.PoolGKNN.Group) {
		// InferenceObjective object got deleted or changed the referenced inferencePool.
		c.Datastore.ObjectiveDelete(req.NamespacedName)
		return ctrl.Result{}, nil
	}

	// Add or update if the InferenceObjective instance has a creation timestamp older than the existing entry of the model.
	logger = logger.WithValues("poolRef", infObjective.Spec.PoolRef)
	c.Datastore.ObjectiveSet(infObjective)
	logger.Info("Added/Updated InferenceObjective")

	return ctrl.Result{}, nil
}

func (c *InferenceObjectiveReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferenceObjective{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceObjective)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceObjective))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
		}).
		Complete(c)
}

func (c *InferenceObjectiveReconciler) eventPredicate(infObjective *v1alpha2.InferenceObjective) bool {
	return string(infObjective.Spec.PoolRef.Name) == c.PoolGKNN.Name && string(infObjective.Spec.PoolRef.Group) == c.PoolGKNN.Group
}
