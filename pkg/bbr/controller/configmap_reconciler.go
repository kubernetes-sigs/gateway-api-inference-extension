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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type ConfigMapReconciler struct {
	client.Reader
	Datastore datastore.Datastore
}

func (c *ConfigMapReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).V(logutil.DEFAULT)
	ctx = ctrl.LoggerInto(ctx, logger)

	logger.Info("Reconciling ConfigMap")

	configmap := &corev1.ConfigMap{}
	err := c.Get(ctx, req.NamespacedName, configmap)
	if err != nil && !errors.IsNotFound(err) {
		return ctrl.Result{}, fmt.Errorf("unable to get ConfigMap - %w", err)
	}

	if errors.IsNotFound(err) || !configmap.DeletionTimestamp.IsZero() {
		// ConfigMap object got deleted or is marked for deletion.
		c.Datastore.ConfigMapDelete(configmap)
		return ctrl.Result{}, nil
	}

	// otherwise, add or update the entries of the configmap
	if err := c.Datastore.ConfigMapUpdateOrAddIfNotExist(configmap); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to add or update ConfigMap - %w", err)
	}

	return ctrl.Result{}, nil
}

func (c *ConfigMapReconciler) SetupWithManager(ctx context.Context, mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.ConfigMap{}).
		Complete(c)
}
