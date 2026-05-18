/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func ptrTo[T any](v T) *T { return &v }

// gmsTestStorage mirrors the storage shape used by the existing GMS tests.
func gmsTestStorage() snapshotprotocol.Storage {
	return snapshotprotocol.Storage{
		PVCName:  "ckpt-pvc",
		BasePath: "/checkpoints",
		Location: "/checkpoints/" + testHash + "/versions/1",
	}
}

func gmsTestPodSpec() *corev1.PodSpec {
	return &corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    consts.MainContainerName,
			Image:   "test-image:latest",
			Command: []string{"python3"},
			Args:    []string{"-m", "dynamo.vllm"},
			Resources: corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{Name: "gpu"}},
			},
		}},
	}
}

// TestEnsureGMSRestoreSidecars_NilCheckpointSpec asserts the byte-identical
// contract: when GPUMemoryService.Checkpoint is nil the operator behaves
// exactly as before this PR — default command, default image (main container's
// image), GMS_CHECKPOINT_DIR env set, checkpoint PVC mounted at BasePath.
func TestEnsureGMSRestoreSidecars_NilCheckpointSpec(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, nil)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader, "default loader is injected even with nil checkpoint spec")

	assert.Equal(t, "test-image:latest", loader.Image, "default image is the main container's image")
	assert.Equal(t, []string{"python3", "-m", gmsCheckpointLoaderModule}, loader.Command,
		"default command is python3 -m gpu_memory_service.cli.snapshot.loader")
	assert.Empty(t, loader.EnvFrom, "no envFrom on default loader")

	mounts := map[string]string{}
	for _, m := range loader.VolumeMounts {
		mounts[m.Name] = m.MountPath
	}
	assert.Equal(t, "/checkpoints", mounts[snapshotprotocol.CheckpointVolumeName],
		"checkpoint PVC mounted at BasePath unchanged")
	assert.Equal(t, gms.SharedMountPath, mounts[gms.SharedVolumeName],
		"operator-owned shared volume mount preserved")

	env := map[string]string{}
	for _, e := range loader.Env {
		env[e.Name] = e.Value
	}
	assert.Equal(t, "/checkpoints/gms/"+testHash+"/versions/1", env[envCheckpointDir],
		"GMS_CHECKPOINT_DIR still operator-set when no user spec")
}

// TestEnsureGMSRestoreSidecars_ImageOverride asserts a user-supplied image
// replaces the default while the operator's mounts, env, and command stay.
func TestEnsureGMSRestoreSidecars_ImageOverride(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
			Image: "byo-loader:v2",
		},
	}

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, spec)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader)

	assert.Equal(t, "byo-loader:v2", loader.Image, "user image overrides default")
	assert.Equal(t, GMSLoaderContainer, loader.Name, "container name stays operator-owned")
	assert.Equal(t, []string{"python3", "-m", gmsCheckpointLoaderModule}, loader.Command,
		"command unchanged when user didn't override")

	env := map[string]string{}
	for _, e := range loader.Env {
		env[e.Name] = e.Value
	}
	assert.Equal(t, "/checkpoints/gms/"+testHash+"/versions/1", env[envCheckpointDir],
		"GMS_CHECKPOINT_DIR preserved")
	assert.NotEmpty(t, env[gms.EnvSocketDir], "GMS_SOCKET_DIR preserved")
}

// TestEnsureGMSRestoreSidecars_CommandOverride asserts the user's command
// fully replaces the default argv (no hidden "python3 -m" prefix per locked C2).
func TestEnsureGMSRestoreSidecars_CommandOverride(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	customCmd := []string{"python3", "-m", "gpu_memory_service.cli.snapshot.loader", "--max-workers", "16"}
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
			Command: customCmd,
		},
	}

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, spec)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader)
	assert.Equal(t, customCmd, loader.Command,
		"user command fully replaces default; no implicit prefix injected")
}

// TestEnsureGMSRestoreSidecars_EnvsMerge asserts user envs merge with
// operator-set envs. GMS_SOCKET_DIR remains operator-owned because it points at
// the injected UDS mount; GMS_CHECKPOINT_DIR is intentionally user-overridable
// for custom client implementations.
func TestEnsureGMSRestoreSidecars_EnvsMerge(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
			Envs: []corev1.EnvVar{
				{Name: "GMS_TRANSFER_BACKEND", Value: "nixl-gds"},
				{Name: envCheckpointDir, Value: "/override/path"},
				{Name: gms.EnvSocketDir, Value: "/bad/socket"},
			},
		},
	}

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, spec)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader)

	env := map[string]string{}
	for _, e := range loader.Env {
		env[e.Name] = e.Value
	}
	assert.Equal(t, "nixl-gds", env["GMS_TRANSFER_BACKEND"], "new user env appended")
	assert.Equal(t, "/override/path", env[envCheckpointDir],
		"user env wins on name collision with operator-set var")
	assert.Equal(t, gms.SharedMountPath, env[gms.EnvSocketDir], "operator GMS_SOCKET_DIR preserved")
}

// TestEnsureGMSRestoreSidecars_VolumeMountsAppend asserts user mounts are
// appended to (not replacing) the operator's gms-intrapod-control and
// checkpoint PVC mounts.
func TestEnsureGMSRestoreSidecars_VolumeMountsAppend(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
			VolumeMounts: []corev1.VolumeMount{
				{Name: "weights", MountPath: "/mnt/weights"},
			},
		},
	}

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, spec)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader)

	mounts := map[string]string{}
	for _, m := range loader.VolumeMounts {
		mounts[m.Name] = m.MountPath
	}
	assert.Equal(t, "/checkpoints", mounts[snapshotprotocol.CheckpointVolumeName],
		"operator checkpoint PVC mount preserved")
	assert.Equal(t, gms.SharedMountPath, mounts[gms.SharedVolumeName],
		"operator gms-intrapod-control mount preserved")
	assert.Equal(t, "/mnt/weights", mounts["weights"],
		"user mount appended alongside operator mounts")
}

// TestEnsureGMSRestoreSidecars_EnvFromSecret asserts a user-supplied
// envFromSecret is appended as an envFrom source.
func TestEnsureGMSRestoreSidecars_EnvFromSecret(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
			EnvFromSecret: ptrTo("s3-creds"),
		},
	}

	EnsureGMSRestoreSidecars(podSpec, &podSpec.Containers[0], storage, spec)

	loader := findContainer(podSpec, GMSLoaderContainer)
	require.NotNil(t, loader)
	require.Len(t, loader.EnvFrom, 1, "exactly one envFrom source appended")
	require.NotNil(t, loader.EnvFrom[0].SecretRef)
	assert.Equal(t, "s3-creds", loader.EnvFrom[0].SecretRef.Name)
}

// TestEnsureGMSCheckpointJobSidecars_NilCheckpointSpec asserts byte-identical
// saver injection when the user supplies no override.
func TestEnsureGMSCheckpointJobSidecars_NilCheckpointSpec(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()

	require.NoError(t, EnsureGMSCheckpointJobSidecars(podSpec, &podSpec.Containers[0], storage, nil))

	saver := findContainer(podSpec, GMSSaverContainer)
	require.NotNil(t, saver)
	assert.Equal(t, "test-image:latest", saver.Image)
	assert.Equal(t, []string{"python3", "-m", gmsCheckpointSaverModule}, saver.Command)

	env := map[string]string{}
	for _, e := range saver.Env {
		env[e.Name] = e.Value
	}
	assert.Equal(t, "/checkpoints/gms/"+testHash+"/versions/1", env[envCheckpointDir])
}

// TestEnsureGMSCheckpointJobSidecars_SaverOverride asserts the saver merge
// path matches the loader merge path (same applyGMSSidecarSpec helper).
func TestEnsureGMSCheckpointJobSidecars_SaverOverride(t *testing.T) {
	podSpec := gmsTestPodSpec()
	storage := gmsTestStorage()
	spec := &nvidiacomv1alpha1.GMSCheckpointSpec{
		Saver: &nvidiacomv1alpha1.GMSSidecarSpec{
			Image:         "byo-saver:v3",
			Command:       []string{"python3", "-m", "my.saver", "--max-workers", "32"},
			Envs:          []corev1.EnvVar{{Name: "GMS_SAVE_LOCK_TIMEOUT_MS", Value: "60000"}},
			VolumeMounts:  []corev1.VolumeMount{{Name: "weights", MountPath: "/mnt/weights"}},
			EnvFromSecret: ptrTo("s3-creds"),
		},
		// Loader on a Job-side path is ignored by Ensure...JobSidecars (and
		// rejected by the DynamoCheckpoint webhook); set it here to prove the
		// helper doesn't accidentally use the wrong field.
		Loader: &nvidiacomv1alpha1.GMSSidecarSpec{Image: "should-not-be-used:latest"},
	}

	require.NoError(t, EnsureGMSCheckpointJobSidecars(podSpec, &podSpec.Containers[0], storage, spec))

	saver := findContainer(podSpec, GMSSaverContainer)
	require.NotNil(t, saver)
	assert.Equal(t, "byo-saver:v3", saver.Image, "user saver image overrides default")
	assert.NotEqual(t, "should-not-be-used:latest", saver.Image,
		"Ensure...JobSidecars must not read Loader by accident")
	assert.Equal(t, []string{"python3", "-m", "my.saver", "--max-workers", "32"}, saver.Command)

	env := map[string]string{}
	for _, e := range saver.Env {
		env[e.Name] = e.Value
	}
	assert.Equal(t, "60000", env["GMS_SAVE_LOCK_TIMEOUT_MS"])
	assert.Equal(t, "/checkpoints/gms/"+testHash+"/versions/1", env[envCheckpointDir],
		"operator-set GMS_CHECKPOINT_DIR preserved")

	mounts := map[string]string{}
	for _, m := range saver.VolumeMounts {
		mounts[m.Name] = m.MountPath
	}
	assert.Equal(t, "/checkpoints", mounts[snapshotprotocol.CheckpointVolumeName])
	assert.Equal(t, "/mnt/weights", mounts["weights"])

	require.Len(t, saver.EnvFrom, 1)
	require.NotNil(t, saver.EnvFrom[0].SecretRef)
	assert.Equal(t, "s3-creds", saver.EnvFrom[0].SecretRef.Name)
}

// TestApplyGMSSidecarSpec_NilIsNoop directly covers the helper for paranoia.
func TestApplyGMSSidecarSpec_NilIsNoop(t *testing.T) {
	base := corev1.Container{
		Name:    GMSLoaderContainer,
		Image:   "base:1",
		Command: []string{"python3", "-m", gmsCheckpointLoaderModule},
		Env:     []corev1.EnvVar{{Name: "X", Value: "1"}},
	}
	out := applyGMSSidecarSpec(base, nil)
	assert.Equal(t, base, out, "nil spec is byte-identical no-op")
}

// TestApplyGMSSidecarSpec_EmptySpecIsNoop covers the "checkpoint.loader: {}"
// case explicitly — empty struct opts into the merge path but every field is
// empty, so the base container is returned unchanged.
func TestApplyGMSSidecarSpec_EmptySpecIsNoop(t *testing.T) {
	base := corev1.Container{
		Name:    GMSLoaderContainer,
		Image:   "base:1",
		Command: []string{"python3", "-m", gmsCheckpointLoaderModule},
	}
	out := applyGMSSidecarSpec(base, &nvidiacomv1alpha1.GMSSidecarSpec{})
	assert.Equal(t, "base:1", out.Image)
	assert.Equal(t, []string{"python3", "-m", gmsCheckpointLoaderModule}, out.Command)
	assert.Empty(t, out.EnvFrom)
}

// TestApplyGMSSidecarSpec_EmptyEnvFromSecretIgnored asserts an explicitly empty
// EnvFromSecret pointer does not add a malformed envFrom source. This matches
// the lean-config stance — operator-supplied config is trusted, but an empty
// secret name is not a legal Kubernetes reference and must not be rendered.
func TestApplyGMSSidecarSpec_EmptyEnvFromSecretIgnored(t *testing.T) {
	base := corev1.Container{Name: GMSLoaderContainer}
	out := applyGMSSidecarSpec(base, &nvidiacomv1alpha1.GMSSidecarSpec{
		EnvFromSecret: ptrTo(""),
	})
	assert.Empty(t, out.EnvFrom, "empty secret name does not add an envFrom source")
}
