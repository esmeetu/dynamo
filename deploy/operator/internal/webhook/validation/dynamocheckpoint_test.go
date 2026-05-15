/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestValidateDynamoCheckpointCheckpointSidecars(t *testing.T) {
	cases := []struct {
		name      string
		ckpt      *nvidiacomv1alpha1.DynamoCheckpoint
		wantErr   bool
		errSubstr string
	}{
		{
			name: "no gpuMemoryService is accepted",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{},
			},
			wantErr: false,
		},
		{
			name: "no checkpoint sub-spec is accepted",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
					GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
				},
			},
			wantErr: false,
		},
		{
			name: "checkpoint.saver with enabled=true is accepted",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
					GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
						Enabled: true,
						Checkpoint: &nvidiacomv1alpha1.GMSCheckpointSpec{
							Saver: &nvidiacomv1alpha1.GMSSidecarSpec{
								Image: "my-saver:latest",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "checkpoint.loader is rejected (Jobs only save)",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
					GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
						Enabled: true,
						Checkpoint: &nvidiacomv1alpha1.GMSCheckpointSpec{
							Loader: &nvidiacomv1alpha1.GMSSidecarSpec{
								Image: "my-loader:latest",
							},
						},
					},
				},
			},
			wantErr:   true,
			errSubstr: "checkpoint.loader is not valid on a DynamoCheckpoint",
		},
		{
			name: "checkpoint.loader with saver also set is still rejected",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
					GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
						Enabled: true,
						Checkpoint: &nvidiacomv1alpha1.GMSCheckpointSpec{
							Loader: &nvidiacomv1alpha1.GMSSidecarSpec{Image: "my-loader:latest"},
							Saver:  &nvidiacomv1alpha1.GMSSidecarSpec{Image: "my-saver:latest"},
						},
					},
				},
			},
			wantErr:   true,
			errSubstr: "checkpoint.loader is not valid on a DynamoCheckpoint",
		},
		{
			name: "checkpoint.saver without enabled=true is rejected",
			ckpt: &nvidiacomv1alpha1.DynamoCheckpoint{
				Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
					GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
						Enabled: false,
						Checkpoint: &nvidiacomv1alpha1.GMSCheckpointSpec{
							Saver: &nvidiacomv1alpha1.GMSSidecarSpec{Image: "my-saver:latest"},
						},
					},
				},
			},
			wantErr:   true,
			errSubstr: "requires gpuMemoryService.enabled=true",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDynamoCheckpointCheckpointSidecars(tc.ckpt)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.errSubstr)
				}
				if !strings.Contains(err.Error(), tc.errSubstr) {
					t.Fatalf("expected error containing %q, got %q", tc.errSubstr, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// TestValidateDynamoCheckpoint_Composition covers that validateDynamoCheckpoint
// chains the GMS snapshot feature-gate check with the sidecar rules. We use
// the env var consts.DynamoOperatorAllowGMSSnapshotEnvVar to enable the gate;
// without it the GMS+snapshot combination is rejected upstream and the more
// specific sidecar rule never runs.
func TestValidateDynamoCheckpoint_Composition(t *testing.T) {
	t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "1")

	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled: true,
				Checkpoint: &nvidiacomv1alpha1.GMSCheckpointSpec{
					Loader: &nvidiacomv1alpha1.GMSSidecarSpec{Image: "loader:latest"},
				},
			},
		},
	}

	err := validateDynamoCheckpoint(ckpt)
	if err == nil {
		t.Fatalf("expected loader rejection, got nil")
	}
	if !strings.Contains(err.Error(), "checkpoint.loader is not valid on a DynamoCheckpoint") {
		t.Fatalf("expected loader-rejection error, got %q", err.Error())
	}
}
