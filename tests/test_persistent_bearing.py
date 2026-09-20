import unittest

import torch

from cvphr.models.persistent_bearing import (
    HypothesisAlignedGeoMemory,
    MemoryConfig,
    PersistentBearing,
    PersistentBearingLoss,
    SymmetryAwareCircularHead,
)


class PersistentBearingTest(unittest.TestCase):
    def test_joint_volume_loss_and_backward(self):
        model = PersistentBearing(
            backbone_name="tiny_cnn",
            pretrained=False,
            feature_dim=16,
            num_angle_bins=12,
            topk=3,
        )
        patches = torch.randn(2, 5, 3, 64, 64)
        coords = torch.tensor([[0.2, -0.3], [-0.5, 0.7]])
        theta = torch.tensor([[30.0], [-120.0]])
        theta_rad = torch.deg2rad(theta[:, 0])
        direction = torch.stack(
            (torch.cos(theta_rad), torch.sin(theta_rad)), dim=1
        )

        output = model.forward_full(patches, return_features=True)
        losses = PersistentBearingLoss()(output, coords, theta, direction)
        losses["loss"].backward()

        self.assertEqual(output["logits"].shape, (2, 12, 5, 5))
        self.assertEqual(output["topk"]["poses"].shape, (2, 3, 3))
        self.assertEqual(output["uav_descriptor"].shape, (2, 16))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(model.logit_scale.grad)

    def test_large_mlp_variant_preserves_joint_pose_contract(self):
        model = PersistentBearing(
            backbone_name="tiny_cnn",
            pretrained=False,
            feature_dim=32,
            num_angle_bins=8,
            topk=2,
            adapter_mlp_depth=2,
            adapter_mlp_ratio=2,
            shared_mlp_depth=1,
            shared_mlp_ratio=2,
        )
        patches = torch.randn(1, 5, 3, 64, 64)
        output = model.forward_full(patches)
        output["logits"].square().mean().backward()

        adapter_weight = model.satellite_adapter.channel_mlp[0].mlp[0].weight
        shared_weight = model.shared_metric_mlp[0].mlp[0].weight
        self.assertEqual(model.model_name, "persistent_bearing_large")
        self.assertEqual(output["logits"].shape, (1, 8, 5, 5))
        self.assertEqual(output["topk"]["poses"].shape, (1, 2, 3))
        self.assertIsNotNone(adapter_weight.grad)
        self.assertIsNotNone(shared_weight.grad)

    def test_symmetry_aware_circular_mixture_loss_and_backward(self):
        model = PersistentBearing(
            backbone_name="tiny_cnn",
            pretrained=False,
            feature_dim=16,
            num_angle_bins=12,
            topk=3,
            heading_distribution="von_mises_mixture",
            heading_hidden_dim=16,
            heading_top_modes=2,
        )
        patches = torch.randn(2, 5, 3, 64, 64)
        coords = torch.tensor([[0.2, -0.3], [-0.5, 0.7]])
        theta = torch.tensor([[179.0], [-179.0]])
        theta_rad = torch.deg2rad(theta[:, 0])
        direction = torch.stack(
            (torch.cos(theta_rad), torch.sin(theta_rad)), dim=1
        )

        output = model.forward_full(patches, return_features=True)
        losses = PersistentBearingLoss()(output, coords, theta, direction)
        losses["loss"].backward()

        mixture = output["heading_mixture"]
        self.assertEqual(mixture["weights"].shape, (2, 12))
        self.assertEqual(mixture["mode_directions"].shape, (2, 2, 2))
        self.assertTrue(
            torch.allclose(
                mixture["directions"].norm(dim=-1),
                torch.ones(2, 12),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.isfinite(losses["heading_mixture_nll"]))
        self.assertTrue(torch.isfinite(losses["circular"]))
        self.assertTrue(torch.isfinite(losses["unit_circle"]))
        self.assertIsNotNone(model.heading_head.refiner[-1].weight.grad)

    def test_antipodal_peaks_have_high_symmetry_score(self):
        centers = torch.arange(12) * (2.0 * torch.pi / 12) - torch.pi
        head = SymmetryAwareCircularHead(
            angle_centers=centers,
            hidden_dim=8,
            top_modes=2,
        )
        evidence = torch.full((1, 12), -20.0)
        evidence[0, 0] = 0.0
        evidence[0, 6] = 0.0
        output = head(evidence)

        self.assertGreater(float(output["symmetry_score"][0]), 0.99)
        angle_delta = torch.abs(
            torch.atan2(
                torch.sin(output["mode_angles"][0, 0] - output["mode_angles"][0, 1]),
                torch.cos(output["mode_angles"][0, 0] - output["mode_angles"][0, 1]),
            )
        )
        self.assertAlmostEqual(float(angle_delta), float(torch.pi), places=4)

        uniform = head(torch.zeros(1, 12))
        self.assertAlmostEqual(
            float(uniform["symmetry_score"][0]), 2.0 / 12.0, places=4
        )

    def test_hypothesis_memory_promotes_consistent_observations(self):
        memory = HypothesisAlignedGeoMemory(
            MemoryConfig(
                max_hypotheses=2,
                promote_after=2,
                write_risk_threshold=0.6,
            )
        )
        poses = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.8, 3.0]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.95, 0.05])
        descriptor = torch.tensor([1.0, 0.0, 0.0, 0.0])

        first = memory.step(poses, scores, descriptor)
        second = memory.step(poses, scores, descriptor)

        self.assertFalse(bool(first["lost"]))
        self.assertTrue(bool(first["wrote_memory"]))
        self.assertGreaterEqual(int(second["confirmed_entries"]), 1)
        self.assertLess(float(second["risk"]), 0.6)


if __name__ == "__main__":
    unittest.main()
