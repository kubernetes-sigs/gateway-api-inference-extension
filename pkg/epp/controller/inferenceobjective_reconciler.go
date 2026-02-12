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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
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

	matchesPool, err := c.objectiveMatchesPool(ctx, infObjective)
	if err != nil {
		return ctrl.Result{}, err
	}

	if notFound || !infObjective.DeletionTimestamp.IsZero() || !matchesPool {
		logger.V(logutil.DEBUG).Info("InferenceObjective is being removed",
			"name", infObjective.Name,
			"notFound", notFound,
			"matchesPool", matchesPool)
		// InferenceObjective object got deleted or doesn't match our pool.
		c.Datastore.ObjectiveDelete(req.NamespacedName)
		return ctrl.Result{}, nil
	}

	// Add or update the InferenceObjective
	if infObjective.Spec.PoolRef != nil {
		logger = logger.WithValues("name", infObjective.Name, "poolRef", infObjective.Spec.PoolRef)
	} else if infObjective.Spec.PoolSelector != nil {
		logger = logger.WithValues("name", infObjective.Name, "poolSelector", infObjective.Spec.PoolSelector)
	}
	c.Datastore.ObjectiveSet(infObjective)
	logger.Info("Added/Updated InferenceObjective")

	return ctrl.Result{}, nil
}

// objectiveMatchesPool checks if the given InferenceObjective matches the pool this reconciler manages.
func (c *InferenceObjectiveReconciler) objectiveMatchesPool(ctx context.Context, obj *v1alpha2.InferenceObjective) (bool, error) {
	if obj.Spec.PoolRef != nil {
		return string(obj.Spec.PoolRef.Name) == c.PoolGKNN.Name &&
			string(obj.Spec.PoolRef.Group) == c.PoolGKNN.Group, nil
	}

	return c.poolMatchesSelector(ctx, obj.Spec.PoolSelector)
}

// poolMatchesSelector checks if our pool matches the given selector.
func (c *InferenceObjectiveReconciler) poolMatchesSelector(ctx context.Context, selector *v1alpha2.PoolSelector) (bool, error) {
	if selector == nil {
		return false, nil
	}

	// Check group/kind match first
	if string(selector.Group) != c.PoolGKNN.Group {
		return false, nil
	}
	if string(selector.Kind) != c.PoolGKNN.Kind {
		return false, nil
	}

	// Fetch the pool to get its labels
	var pool client.Object = &v1.InferencePool{}
	if c.PoolGKNN.Group == v1alpha2.GroupName {
		pool = &v1alpha2.InferencePool{}
	}
	poolKey := types.NamespacedName{
		Name:      c.PoolGKNN.Name,
		Namespace: c.PoolGKNN.Namespace,
	}
	if err := c.Get(ctx, poolKey, pool); err != nil {
		return false, fmt.Errorf("failed to get InferencePool for selector matching: %w", err)
	}

	// Convert selector to labels.Selector and match against pool labels
	labelSelector, err := poolSelectorToLabelSelector(selector)
	if err != nil {
		return false, fmt.Errorf("failed to convert poolSelector to label selector: %w", err)
	}

	return labelSelector.Matches(labels.Set(pool.GetLabels())), nil
}

// poolSelectorToLabelSelector converts a PoolSelector to a labels.Selector.
func poolSelectorToLabelSelector(selector *v1alpha2.PoolSelector) (labels.Selector, error) {
	labelSelector := labels.NewSelector()

	// Add matchLabels requirements
	for key, value := range selector.MatchLabels {
		req, err := labels.NewRequirement(string(key), selection.Equals, []string{string(value)})
		if err != nil {
			return nil, fmt.Errorf("invalid matchLabels requirement: %w", err)
		}
		labelSelector = labelSelector.Add(*req)
	}

	// Add matchExpressions requirements
	for _, expr := range selector.MatchExpressions {
		var op selection.Operator
		switch expr.Operator {
		case v1alpha2.LabelSelectorOpIn:
			op = selection.In
		case v1alpha2.LabelSelectorOpNotIn:
			op = selection.NotIn
		case v1alpha2.LabelSelectorOpExists:
			op = selection.Exists
		case v1alpha2.LabelSelectorOpDoesNotExist:
			op = selection.DoesNotExist
		default:
			return nil, fmt.Errorf("unknown operator: %s", expr.Operator)
		}

		values := make([]string, len(expr.Values))
		for i, v := range expr.Values {
			values[i] = string(v)
		}

		req, err := labels.NewRequirement(string(expr.Key), op, values)
		if err != nil {
			return nil, fmt.Errorf("invalid matchExpressions requirement: %w", err)
		}
		labelSelector = labelSelector.Add(*req)
	}

	return labelSelector, nil
}

func (c *InferenceObjectiveReconciler) SetupWithManager(mgr ctrl.Manager) error {
	var pool client.Object = &v1.InferencePool{}
	if c.PoolGKNN.Group == v1alpha2.GroupName {
		pool = &v1alpha2.InferencePool{}
	}
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha2.InferenceObjective{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			UpdateFunc: func(e event.UpdateEvent) bool {
				return c.eventPredicate(e.ObjectOld.(*v1alpha2.InferenceObjective)) || c.eventPredicate(e.ObjectNew.(*v1alpha2.InferenceObjective))
			},
			DeleteFunc:  func(e event.DeleteEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
			GenericFunc: func(e event.GenericEvent) bool { return c.eventPredicate(e.Object.(*v1alpha2.InferenceObjective)) },
		})).
		// Watch InferencePool for label changes to re-evaluate poolSelector objectives
		Watches(
			pool,
			handler.EnqueueRequestsFromMapFunc(c.mapPoolToObjectives),
		).
		Complete(c)
}

// mapPoolToObjectives maps InferencePool events to InferenceObjective reconcile requests.
// When the pool's labels change, we need to re-evaluate all objectives with poolSelector.
func (c *InferenceObjectiveReconciler) mapPoolToObjectives(ctx context.Context, obj client.Object) []reconcile.Request {
	logger := log.FromContext(ctx)

	// Only react to our pool
	if obj.GetName() != c.PoolGKNN.Name || obj.GetNamespace() != c.PoolGKNN.Namespace {
		return nil
	}

	// List all objectives in the namespace and trigger reconcile for those with poolSelector
	objectives := &v1alpha2.InferenceObjectiveList{}
	if err := c.List(ctx, objectives, client.InNamespace(obj.GetNamespace())); err != nil {
		logger.Error(err, "Failed to list InferenceObjectives for pool label change")
		return nil
	}

	var requests []reconcile.Request
	for _, obj := range objectives.Items {
		if obj.Spec.PoolSelector != nil {
			requests = append(requests, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      obj.Name,
					Namespace: obj.Namespace,
				},
			})
		}
	}

	if len(requests) > 0 {
		logger.V(logutil.DEBUG).Info("Pool changed, re-evaluating objectives with poolSelector", "count", len(requests))
	}

	return requests
}

func (c *InferenceObjectiveReconciler) eventPredicate(infObjective *v1alpha2.InferenceObjective) bool {
	if infObjective.Spec.PoolRef != nil {
		return string(infObjective.Spec.PoolRef.Name) == c.PoolGKNN.Name &&
			string(infObjective.Spec.PoolRef.Group) == c.PoolGKNN.Group
	}

	// poolSelector: only group/kind check here; full label matching happens in Reconcile
	// to allow retry on transient infPool fetch failure.
	if infObjective.Spec.PoolSelector != nil {
		return string(infObjective.Spec.PoolSelector.Group) == c.PoolGKNN.Group &&
			string(infObjective.Spec.PoolSelector.Kind) == c.PoolGKNN.Kind
	}

	return false
}
