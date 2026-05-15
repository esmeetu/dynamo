/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	DynamoCheckpointWebhookName = "dynamocheckpoint-validating-webhook"
	dynamoCheckpointWebhookPath = "/validate-nvidia-com-v1alpha1-dynamocheckpoint"
)

type DynamoCheckpointHandler struct{}

func NewDynamoCheckpointHandler() *DynamoCheckpointHandler {
	return &DynamoCheckpointHandler{}
}

func (h *DynamoCheckpointHandler) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoCheckpointWebhookName)
	ckpt, err := castToDynamoCheckpoint(obj)
	if err != nil {
		return nil, err
	}
	logger.Info("validate create", "name", ckpt.Name, "namespace", ckpt.Namespace)
	return nil, validateDynamoCheckpoint(ckpt)
}

func (h *DynamoCheckpointHandler) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	logger := log.FromContext(ctx).WithName(DynamoCheckpointWebhookName)
	ckpt, err := castToDynamoCheckpoint(newObj)
	if err != nil {
		return nil, err
	}
	logger.Info("validate update", "name", ckpt.Name, "namespace", ckpt.Namespace)
	if !ckpt.DeletionTimestamp.IsZero() {
		return nil, nil
	}
	return nil, validateDynamoCheckpoint(ckpt)
}

func (h *DynamoCheckpointHandler) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	ckpt, err := castToDynamoCheckpoint(obj)
	if err != nil {
		return nil, err
	}
	log.FromContext(ctx).WithName(DynamoCheckpointWebhookName).Info("validate delete", "name", ckpt.Name, "namespace", ckpt.Namespace)
	return nil, nil
}

func (h *DynamoCheckpointHandler) RegisterWithManager(mgr manager.Manager) error {
	leaseAwareValidator := internalwebhook.NewLeaseAwareValidator(h, internalwebhook.GetExcludedNamespaces())
	observedValidator := observability.NewObservedValidator(leaseAwareValidator, consts.ResourceTypeDynamoCheckpoint)
	webhook := admission.
		WithCustomValidator(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoCheckpoint{}, observedValidator).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dynamoCheckpointWebhookPath, webhook)
	return nil
}

func validateDynamoCheckpointGMSSnapshot(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	// A DynamoCheckpoint is itself a Snapshot resource; service specs pass checkpoint.enabled instead.
	return checkpoint.ValidateGMSSnapshotGate("spec.gpuMemoryService", true, ckpt.Spec.GPUMemoryService)
}

// validateDynamoCheckpointCheckpointSidecars enforces DynamoCheckpoint-specific
// rules for the GMSCheckpointSpec overrides:
//
//  1. checkpoint.loader is rejected on a DynamoCheckpoint — Jobs only save;
//     restore happens on worker pods reconciled from a service's
//     ServiceCheckpointConfig, never on the DynamoCheckpoint Job itself.
//  2. checkpoint.{loader,saver} require gpuMemoryService.enabled=true (locked
//     design rule D3, also enforced on DGD via SharedSpecValidator).
func validateDynamoCheckpointCheckpointSidecars(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	if ckpt.Spec.GPUMemoryService == nil || ckpt.Spec.GPUMemoryService.Checkpoint == nil {
		return nil
	}
	cp := ckpt.Spec.GPUMemoryService.Checkpoint
	if cp.Loader != nil {
		return fmt.Errorf(
			"spec.gpuMemoryService.checkpoint.loader is not valid on a DynamoCheckpoint (Jobs only save; use checkpoint.saver, or set checkpoint.loader on the consuming service)")
	}
	if cp.Saver != nil && !ckpt.Spec.GPUMemoryService.Enabled {
		return fmt.Errorf(
			"spec.gpuMemoryService.checkpoint.saver requires gpuMemoryService.enabled=true")
	}
	return nil
}

func validateDynamoCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	if err := validateDynamoCheckpointGMSSnapshot(ckpt); err != nil {
		return err
	}
	return validateDynamoCheckpointCheckpointSidecars(ckpt)
}

func castToDynamoCheckpoint(obj runtime.Object) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	ckpt, ok := obj.(*nvidiacomv1alpha1.DynamoCheckpoint)
	if !ok {
		return nil, fmt.Errorf("expected DynamoCheckpoint but got %T", obj)
	}
	return ckpt, nil
}
