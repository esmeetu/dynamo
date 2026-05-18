/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// overlayServiceGMSRestoreLoader applies the service's restore-time loader
// override to an already GMS-enabled checkpoint.
func overlayServiceGMSRestoreLoader(info *checkpoint.CheckpointInfo, serviceGMS *nvidiacomv1beta1.GPUMemoryServiceSpec) {
	if info == nil || info.GPUMemoryService == nil || !info.GPUMemoryService.Enabled {
		return
	}

	alphaServiceGMS := dynamo.ToAlphaGPUMemoryService(serviceGMS)
	if alphaServiceGMS == nil ||
		!alphaServiceGMS.Enabled ||
		alphaServiceGMS.Checkpoint == nil ||
		alphaServiceGMS.Checkpoint.Loader == nil {
		return
	}

	out := info.GPUMemoryService.DeepCopy()
	if out.Checkpoint == nil {
		out.Checkpoint = &nvidiacomv1alpha1.GMSCheckpointSpec{}
	}
	out.Checkpoint.Loader = alphaServiceGMS.Checkpoint.Loader.DeepCopy()
	info.GPUMemoryService = out
}

// gmsSpecForAutoCheckpointSave returns the GMS save-time config for an
// auto-created DynamoCheckpoint. Loader overrides stay on the restoring service.
func gmsSpecForAutoCheckpointSave(serviceGMS *nvidiacomv1beta1.GPUMemoryServiceSpec) *nvidiacomv1alpha1.GPUMemoryServiceSpec {
	out := dynamo.ToAlphaGPUMemoryService(serviceGMS)
	if out == nil {
		return nil
	}

	out = out.DeepCopy()
	if out.Checkpoint != nil {
		out.Checkpoint.Loader = nil
		if out.Checkpoint.Saver == nil {
			out.Checkpoint = nil
		}
	}
	return out
}
