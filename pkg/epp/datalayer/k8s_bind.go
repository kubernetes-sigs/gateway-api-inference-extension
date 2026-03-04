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
func BindNotificationSource(src fwkdl.NotificationSource, extractors []fwkdl.NotificationExtractor, mgr ctrl.Manager) error {
	gvk := src.GVK()
	log := mgr.GetLogger().WithName("notification-controller").WithValues("gvk", gvk.Kind)

	reconciler := &notificationReconciler{
		client:     mgr.GetClient(),
		src:        src,
		extractors: extractors,
		gvk:        gvk,
		log:        log,
	}

	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(gvk)

	// use the source's name to make the controller name unique
	// This allows multiple notification sources for the same GVK
	// (needed in tests, Configure() sources still imposes
	// one source per GVK).
	controllerName := "notify_" + strings.ToLower(gvk.Kind) + "_" + src.TypedName().Name

	return ctrl.NewControllerManagedBy(mgr).
		// Naming the controller allows you to see specific metrics/logs for this watch
		Named(controllerName).
		For(obj).
		// ResourceVersionChanged is safer for generic notifications than GenerationChanged,
		// as it catches metadata and status updates that the consumer might need.
		WithEventFilter(predicate.ResourceVersionChangedPredicate{}).
		Complete(reconciler)
}

// Reconciler for notifications. This is a generic reconciler that can be used for any GVK.
type notificationReconciler struct {
	client     client.Client
	src        fwkdl.NotificationSource
	extractors []fwkdl.NotificationExtractor
	gvk        schema.GroupVersionKind
	log        logr.Logger
}

// Reconciler carries out the actual notifcation logic.
func (rn *notificationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := rn.log.WithValues("resource", req.NamespacedName, "gvk", rn.gvk.String())

	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(rn.gvk)

	event := &fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: u,
	}

	err := rn.client.Get(ctx, req.NamespacedName, u)
	if err != nil {
		if apierrors.IsNotFound(err) {
			u.SetName(req.Name)
			u.SetNamespace(req.Namespace)
			event.Type = fwkdl.EventDelete
		} else {
			log.Error(err, "failed to fetch resource from cache")
			return ctrl.Result{}, err
		}
	}

	return rn.dispatch(ctx, log, event)
}

func (rn *notificationReconciler) dispatch(ctx context.Context, log logr.Logger, event *fwkdl.NotificationEvent) (ctrl.Result, error) {
	log.V(1).Info("processing notification", "eventType", event.Type)

	processed, err := rn.src.Notify(ctx, *event)
	if err != nil {
		log.Error(err, "notifier failed to process event")
		return ctrl.Result{}, err
	}
	if processed == nil {
		return ctrl.Result{}, nil
	}

	for _, ext := range rn.extractors {
		if err := ext.ExtractNotification(ctx, *processed); err != nil {
			log.Error(err, "extractor failed", "extractor", ext.TypedName())
		}
	}

	return ctrl.Result{}, nil
}
