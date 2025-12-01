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

type InferenceModelRewriteReconciler struct {
	client.Reader
	Datastore datastore.Datastore
	PoolGKNN  common.GKNN
}

func (c *InferenceModelRewriteReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling InferenceModelRewrite")

	infModelRewrite := &v1alpha2.InferenceModelRewrite{}
	notFound := false
	if err := c.Get(ctx, req.NamespacedName, infModelRewrite); err != nil {
		if !errors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("unable to get InferenceModelRewrite - %w", err)
		}
		notFound = true
	}

	isDeleted := !infModelRewrite.DeletionTimestamp.IsZero()
	isPooRefUnmatch := infModelRewrite.Spec.PoolRef == nil ||
		infModelRewrite.Spec.PoolRef.Name != v1alpha2.ObjectName(c.PoolGKNN.Name) ||
		infModelRewrite.Spec.PoolRef.Group != v1alpha2.Group(c.PoolGKNN.Group)

	if notFound || isDeleted || isPooRefUnmatch {
		// InferenceModelRewrite object got deleted or changed the referenced pool.
		c.Datastore.ModelRewriteDelete(req.NamespacedName)
		return ctrl.Result{}, nil
	}

	// Add or update if the InferenceModelRewrite instance has a creation timestamp older than the existing entry of the model.
	logger = logger.WithValues("poolRef", infModelRewrite.Spec.PoolRef)
	c.Datastore.ModelRewriteSet(infModelRewrite)
	logger.Info("Added/Updated InferenceModelRewrite")

	return ctrl.Result{}, nil
}

func (c *InferenceModelRewriteReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferenceModelRewrite{}).
		WithEventFilter(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModelRewrite)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceModelRewrite)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceModelRewrite))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModelRewrite)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceModelRewrite)) },
		}).
		Complete(c)
}

func (c *InferenceModelRewriteReconciler) eventPredicate(infModelRewrite *v1alpha2.InferenceModelRewrite) bool {
	return infModelRewrite.Spec.PoolRef != nil && string(infModelRewrite.Spec.PoolRef.Name) == c.PoolGKNN.Name && string(infModelRewrite.Spec.PoolRef.Group) == c.PoolGKNN.Group
}
