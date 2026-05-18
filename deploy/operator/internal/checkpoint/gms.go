/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"
	"path/filepath"
	"sort"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
)

const (
	GMSLoaderContainer = "gms-loader"
	GMSSaverContainer  = "gms-saver"

	gmsCheckpointLoaderModule = "gpu_memory_service.cli.snapshot.loader"
	gmsCheckpointSaverModule  = "gpu_memory_service.cli.snapshot.saver"

	// envCheckpointDir is the environment variable name for the GMS
	// checkpoint artifact directory on the snapshot PVC.
	envCheckpointDir = "GMS_CHECKPOINT_DIR"
)

// EnsureGMSRestoreSidecars adds the GMS server init sidecar and loader.
// The loader is a regular sidecar; the GMS RO lock — not init-phase ordering —
// gates the restored engine on weight load. Idempotent.
//
// If checkpointSpec.Loader is set, it is layered onto the default loader.
func EnsureGMSRestoreSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
	checkpointSpec *nvidiacomv1alpha1.GMSCheckpointSpec,
) {
	if podSpec == nil || mainContainer == nil {
		return
	}

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	for _, c := range podSpec.Containers {
		if c.Name == GMSLoaderContainer {
			return
		}
	}

	loader := gms.Container(GMSLoaderContainer, gmsCheckpointLoaderModule, mainContainer.Image)
	loader.VolumeMounts = append(loader.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	loader.Env = append(loader.Env, corev1.EnvVar{Name: envCheckpointDir, Value: resolveGMSArtifactDir(storage)})
	loader = applyGMSCheckpointClientSpec(loader, gmsCheckpointSpecLoader(checkpointSpec))
	podSpec.Containers = append(podSpec.Containers, loader)
}

// EnsureGMSCheckpointJobSidecars adds the GMS server init sidecar and saver
// as a regular Job container. Saver is a regular container (not init+sleep)
// so Job completion gates on tensor write.
//
// If checkpointSpec.Saver is set, it is layered onto the default saver.
func EnsureGMSCheckpointJobSidecars(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	storage snapshotprotocol.Storage,
	checkpointSpec *nvidiacomv1alpha1.GMSCheckpointSpec,
) error {
	if podSpec == nil || mainContainer == nil {
		return nil
	}
	if len(mainContainer.Resources.Claims) == 0 {
		return fmt.Errorf("gms checkpoint clients require main container resource claims (DRA must be enabled)")
	}
	if storage.PVCName == "" || storage.BasePath == "" || storage.Location == "" {
		return fmt.Errorf("gms checkpoint jobs require resolved checkpoint storage")
	}

	gmsArtifactDir := resolveGMSArtifactDir(storage)

	gms.EnsureServerSidecar(podSpec, mainContainer)
	snapshotprotocol.InjectCheckpointVolume(podSpec, storage.PVCName)

	saver := gms.Container(GMSSaverContainer, gmsCheckpointSaverModule, mainContainer.Image)
	saver.VolumeMounts = append(saver.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.CheckpointVolumeName, MountPath: storage.BasePath})
	saver.Env = append(saver.Env, corev1.EnvVar{Name: envCheckpointDir, Value: gmsArtifactDir})
	saver = applyGMSCheckpointClientSpec(saver, gmsCheckpointSpecSaver(checkpointSpec))
	podSpec.Containers = append(podSpec.Containers, saver)
	return nil
}

// applyGMSCheckpointClientSpec layers optional user fields onto the default GMS client
// container. Image and Command override; Env merges except GMS_SOCKET_DIR;
// EnvFromSecret and VolumeMounts append.
func applyGMSCheckpointClientSpec(base corev1.Container, spec *nvidiacomv1alpha1.GMSCheckpointClientSpec) corev1.Container {
	if spec == nil {
		return base
	}
	if spec.Image != "" {
		base.Image = spec.Image
	}
	if len(spec.Command) > 0 {
		base.Command = append([]string(nil), spec.Command...)
	}
	if spec.EnvFromSecret != nil && *spec.EnvFromSecret != "" {
		base.EnvFrom = append(base.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *spec.EnvFromSecret},
			},
		})
	}
	if len(spec.Envs) > 0 {
		base.Env = mergeEnvVars(base.Env, spec.Envs)
	}
	if len(spec.VolumeMounts) > 0 {
		base.VolumeMounts = append(base.VolumeMounts, spec.VolumeMounts...)
	}
	return base
}

func gmsCheckpointSpecLoader(cp *nvidiacomv1alpha1.GMSCheckpointSpec) *nvidiacomv1alpha1.GMSCheckpointClientSpec {
	if cp == nil {
		return nil
	}
	return cp.Loader
}

func gmsCheckpointSpecSaver(cp *nvidiacomv1alpha1.GMSCheckpointSpec) *nvidiacomv1alpha1.GMSCheckpointClientSpec {
	if cp == nil {
		return nil
	}
	return cp.Saver
}

// mergeEnvVars uses user-wins semantics except for operator-owned GMS_SOCKET_DIR.
func mergeEnvVars(base, overrides []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar, len(base)+len(overrides))
	for _, env := range base {
		envMap[env.Name] = env
	}
	for _, env := range overrides {
		if env.Name == gms.EnvSocketDir {
			continue
		}
		envMap[env.Name] = env
	}
	merged := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		merged = append(merged, env)
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Name < merged[j].Name
	})
	return merged
}

func resolveGMSArtifactDir(storage snapshotprotocol.Storage) string {
	// GMS data lives under /checkpoints/gms/<hash>/versions/<version>
	// separate from the CRIU tree (/checkpoints/<hash>/versions/<version>)
	// so the non-root saver can create directories at the PVC root.
	artifactVersion := filepath.Base(storage.Location)
	checkpointID := filepath.Base(filepath.Dir(filepath.Dir(storage.Location)))
	return filepath.Join(storage.BasePath, "gms", checkpointID, "versions", artifactVersion)
}
