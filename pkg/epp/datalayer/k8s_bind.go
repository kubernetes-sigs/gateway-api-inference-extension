/*
Copyright 2026 The Kubernetes Authors.

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

package datalayer

import (
	"context"
	"strings"

	"github.com/go-logr/logr"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

// BindNotificationSource registers a watcher/reconciler for the source's GVK.
// The framework core owns the cache and reconciliation; the source only receives
// deep-copied events via Notify.
func BindNotificationSource(src fwkdl.NotificationSource, mgr ctrl.Manager) error {
	gvk := src.GVK()
	log := mgr.GetLogger().WithName("notification-controller").WithValues("gvk", gvk.Kind)

	recon := &notificationReconciler{
		client: mgr.GetClient(),
		src:    src,
		gvk:    gvk,
		log:    log,
	}

	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)

	// Use the source's name to make the controller name unique
	// This allows multiple notification sources for the same GVK
	// (needed in tests, the Register() of data sources still imposes
	// one source per GVK).
	controllerName := "notify_" + strings.ToLower(gvk.Kind) + "_" + src.TypedName().Name
	
	return ctrl.NewControllerManagedBy(mgr).
		// Naming the controller allows you to see specific metrics/logs for this watch
		Named(controllerName).
		For(obj).
		// ResourceVersionChanged is safer for generic notifications than GenerationChanged,
		// as it catches metadata and status updates that the consumer might need.
		WithEventFilter(predicate.ResourceVersionChangedPredicate{}).
		Complete(recon)
}

// Reconciler for notifications. This is a generic reconciler that can be used for any GVK.
type notificationReconciler struct {
	client client.Client
	src    fwkdl.NotificationSource
	gvk    schema.GroupVersionKind
	log    logr.Logger
}

// Reconciler carries out the actual notifcation logic.
func (r *notificationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.log.WithValues("resource", req.NamespacedName, "gvk", r.gvk.String())

	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(r.gvk)

	err := r.client.Get(ctx, req.NamespacedName, u) // client.Get returns a unique instance
	event := fwkdl.NotificationEvent{}

	if err != nil {
		if apierrors.IsNotFound(err) { // Ensure the object passed to Notify has identity metadata
			u.SetName(req.Name)
			u.SetNamespace(req.Namespace)

			log.V(1).Info("sending delete notification")
			event.Type = fwkdl.EventDelete
			event.Object = u
			return ctrl.Result{}, r.src.Notify(ctx, event)
		}
		log.Error(err, "failed to fetch resource from cache")
		return ctrl.Result{}, err
	}

	log.V(1).Info("sending add/update notification")
	event.Type = fwkdl.EventAddOrUpdate
	event.Object = u
	if err := r.src.Notify(ctx, event); err != nil {
		log.Error(err, "notification source failed to process event")
		return ctrl.Result{}, err // Requeue on failure
	}

	return ctrl.Result{}, nil
}
