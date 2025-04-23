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

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	podutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pod"
)

type PodReconciler struct {
	client.Client
	Datastore datastore.Datastore
	Record    record.EventRecorder
}

func (c *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	if !c.Datastore.PoolHasSynced() {
		logger.V(logutil.TRACE).Info("Skipping reconciling Pod because the InferencePool is not available yet")
		// When the inferencePool is initialized it lists the appropriate pods and populates the datastore, so no need to requeue.
		return ctrl.Result{}, nil
	}

	logger.V(logutil.VERBOSE).Info("Pod being reconciled", "name", req.NamespacedName)

	pod := &corev1.Pod{}
	if err := c.Get(ctx, req.NamespacedName, pod); err != nil {
		if apierrors.IsNotFound(err) {
			c.Datastore.PodDelete(req.NamespacedName)
			return ctrl.Result{}, nil
		}
		logger.V(logutil.DEFAULT).Error(err, "Unable to get pod", "name", req.NamespacedName)
		return ctrl.Result{}, err
	}

	c.updateDatastore(logger, pod)
	return ctrl.Result{}, nil
}

func (c *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		Complete(c)
}

func (c *PodReconciler) updateDatastore(logger logr.Logger, pod *corev1.Pod) {
	namespacedName := types.NamespacedName{Name: pod.Name, Namespace: pod.Namespace}
	if !podutil.IsPodReady(pod) || !c.Datastore.PoolLabelsMatch(pod.Labels) {
		logger.V(logutil.DEBUG).Info("Pod removed or not added", "name", namespacedName)
		c.Datastore.PodDelete(namespacedName)
	} else {
		if c.Datastore.PodUpdateOrAddIfNotExist(pod) {
			logger.V(logutil.DEFAULT).Info("Pod added", "name", namespacedName)
		} else {
			logger.V(logutil.DEFAULT).Info("Pod already exists", "name", namespacedName)
		}
	}
}
