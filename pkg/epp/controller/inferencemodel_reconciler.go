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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	modelNameKey = "spec.modelName"
)

type InferenceModelReconciler struct {
	client.Client
	Scheme             *runtime.Scheme
	Record             record.EventRecorder
	Datastore          datastore.Datastore
	PoolNamespacedName types.NamespacedName
}

func (c *InferenceModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.Namespace != c.PoolNamespacedName.Namespace {
		return ctrl.Result{}, nil
	}
	logger := log.FromContext(ctx).V(logutil.DEFAULT).WithValues("inferenceModel", req.Name)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceModel")

	infModel := &v1alpha2.InferenceModel{}
	notFound := false
	if err := c.Get(ctx, req.NamespacedName, infModel); err != nil {
		if !errors.IsNotFound(err) {
			logger.Error(err, "Unable to get InferenceModel")
			return ctrl.Result{}, err
		}
		notFound = true
	}

	if notFound || !infModel.DeletionTimestamp.IsZero() || infModel.Spec.PoolRef.Name != c.PoolNamespacedName.Name {
		// InferenceModel object got deleted or changed the referenced pool.
		err := c.handleModelDeleted(ctx, req.NamespacedName)
		return ctrl.Result{}, err
	}

	// Add or update if the InferenceModel instance has a creation timestamp older than the existing entry of the model.
	logger = logger.WithValues("poolRef", infModel.Spec.PoolRef).WithValues("modelName", infModel.Spec.ModelName)
	if !c.Datastore.ModelSetIfOlder(infModel) {
		logger.Info("Skipping InferenceModel, existing instance has older creation timestamp")

	}
	logger.Info("Added/Updated InferenceModel")

	return ctrl.Result{}, nil
}

func (c *InferenceModelReconciler) handleModelDeleted(ctx context.Context, req types.NamespacedName) error {
	logger := log.FromContext(ctx)

	// We will lookup and delete the modelName associated with this object, and search for
	// other instances referencing the same modelName if exist, and store the oldest in
	// its place. This ensures that the InferenceModel with the oldest creation
	// timestamp is active.
	existing, exists := c.Datastore.ModelDelete(req)
	if !exists {
		// No entry exists in the first place, nothing to do.
		return nil
	}
	logger.Info("InferenceModel removed from datastore", "poolRef", existing.Spec.PoolRef, "modelName", existing.Spec.ModelName)

	// List all InferenceModels with a matching ModelName.
	var models v1alpha2.InferenceModelList
	if err := c.List(ctx, &models, client.MatchingFields{modelNameKey: existing.Spec.ModelName}, client.InNamespace(c.PoolNamespacedName.Namespace)); err != nil {
		return fmt.Errorf("listing models that match the modelName %s: %w", existing.Spec.ModelName, err)
	}
	if len(models.Items) == 0 {
		// No other instances of InferenceModels with this ModelName exists.
		return nil
	}

	var oldest *v1alpha2.InferenceModel
	for i := range models.Items {
		m := &models.Items[i]
		if m.Spec.ModelName != existing.Spec.ModelName || // The index should filter those out, but just in case!
			m.Spec.PoolRef.Name != c.PoolNamespacedName.Name || // We don't care about other pools, we could setup an index on this too!
			m.Name == existing.Name { // We don't care about the same object, it could be in the list if it was only marked for deletion, but not yet deleted.
			continue
		}
		if oldest == nil || m.ObjectMeta.CreationTimestamp.Before(&oldest.ObjectMeta.CreationTimestamp) {
			oldest = m
		}
	}
	if oldest != nil && c.Datastore.ModelSetIfOlder(oldest) {
		logger.Info("InferenceModel replaced.",
			"poolRef", oldest.Spec.PoolRef,
			"modelName", oldest.Spec.ModelName,
			"newInferenceModel", types.NamespacedName{Name: oldest.Name, Namespace: oldest.Namespace})
	}

	return nil
}

func indexInferenceModelsByModelName(obj client.Object) []string {
	m, ok := obj.(*v1alpha2.InferenceModel)
	if !ok {
		return nil
	}
	return []string{m.Spec.ModelName}
}

func (c *InferenceModelReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	// Create an index on ModelName for InferenceModel objects.
	indexer := mgr.GetFieldIndexer()
	if err := indexer.IndexField(ctx, &v1alpha2.InferenceModel{}, modelNameKey, indexInferenceModelsByModelName); err != nil {
		return fmt.Errorf("setting index on ModelName for InferenceModel: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferenceModel{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModel)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceModel)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceModel))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModel)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModel)) },
		}).
		Complete(c)
}

func (c *InferenceModelReconciler) eventPredicate(infModel *v1alpha2.InferenceModel) bool {
	return (infModel.Spec.PoolRef.Name == c.PoolNamespacedName.Name) && (infModel.GetNamespace() == c.PoolNamespacedName.Namespace)
}
