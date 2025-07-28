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
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type InferenceObjectiveReconciler struct {
	client.Reader
	Datastore          datastore.Datastore
	PoolNamespacedName types.NamespacedName
}

func (c *InferenceObjectiveReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT).WithValues("InferenceObjective", req.NamespacedName)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceObjective")

	infObjective := &v1alpha2.InferenceObjective{}
	notFound := false
	if err := c.Get(ctx, req.NamespacedName, infObjective); err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "Unable to get InferenceObjective")
			return ctrl.Result{}, err
		}
		notFound = true
	}

	if notFound || !infObjective.DeletionTimestamp.IsZero() || infObjective.Spec.PoolRef.Name != v1alpha2.ObjectName(c.PoolNamespacedName.Name) {
		// InferenceObjective object got deleted or changed the referenced pool.
		err := c.handleObjectiveDeleted(ctx, req.NamespacedName)
		return ctrl.Result{}, err
	}

	// Add or update if the InferenceObjective instance has a creation timestamp older than the existing entry of the model.
	logger = logger.WithValues("poolRef", infObjective.Spec.PoolRef).WithValues("modelName", infObjective.Spec.ModelName)
	if !c.Datastore.ObjectiveSetIfOlder(infObjective) {
		logger.Info("Skipping InferenceObjective, existing instance has older creation timestamp")
	} else {
		logger.Info("Added/Updated InferenceObjective")
	}

	return ctrl.Result{}, nil
}

func (c *InferenceObjectiveReconciler) handleObjectiveDeleted(ctx context.Context, req types.NamespacedName) error {
	logger := log.FromContext(ctx)

	// We will lookup and delete the modelName associated with this object, and search for
	// other instances referencing the same modelName if exist, and store the oldest in
	// its place. This ensures that the InferenceObjective with the oldest creation
	// timestamp is active.
	existing := c.Datastore.ObjectiveDelete(req)
	if existing == nil {
		// No entry exists in the first place, nothing to do.
		return nil
	}
	logger.Info("InferenceObjective removed from datastore", "poolRef", existing.Spec.PoolRef, "modelName", existing.Spec.ModelName)

	// TODO(#409): replace this backfill logic with one that is based on InferenceObjective Ready conditions once those are set by an external controller.
	updated, err := c.Datastore.ObjectiveResync(ctx, c.Reader, existing.Spec.ModelName)
	if err != nil {
		return err
	}
	if updated {
		logger.Info("Model replaced.", "modelName", existing.Spec.ModelName)
	}
	return nil
}

func indexInferenceObjectivesByModelName(obj client.Object) []string {
	m, ok := obj.(*v1alpha2.InferenceObjective)
	if !ok {
		return nil
	}
	return []string{m.Spec.ModelName}
}

func (c *InferenceObjectiveReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	// Create an index on ModelName for InferenceObjective objects.
	indexer := mgr.GetFieldIndexer()
	if err := indexer.IndexField(ctx, &v1alpha2.InferenceObjective{}, datastore.ModelNameIndexKey, indexInferenceObjectivesByModelName); err != nil {
		return fmt.Errorf("setting index on ModelName for InferenceObjective: %w", err)
	}
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
	return string(infObjective.Spec.PoolRef.Name) == c.PoolNamespacedName.Name
}
