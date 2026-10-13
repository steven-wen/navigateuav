import math
import unittest

import torch

from cvphr.models.continuous_se2 import ContinuousSE2Bearing, ContinuousSE2Loss
from cvphr.models.persistent_bearing import PersistentBearing


def _tiny_kwargs():
    return {
        "backbone_name": "tiny_cnn",
        "pretrained": False,
        "freeze_backbone": True,
        "feature_dim": 32,
        "adapter_mlp_depth": 1,
        "shared_mlp_depth": 1,
        "mlp_dropout": 0.0,
        "match_dim": 16,
        "refinement_dim": 16,
        "refinement_steps": 2,
        "check_finite": True,
    }


class ContinuousSE2Test(unittest.TestCase):
    def test_procrustes_recovers_continuous_pose(self):
        query = ContinuousSE2Bearing._pixel_center_grid(
            4, 4, torch.device("cpu"), torch.float32
        )
        angle = torch.tensor([0.73])
        cosine, sine = torch.cos(angle), torch.sin(angle)
        crop_scale = 0.5
        target_position = torch.tensor([[0.35, -0.55]])
        translation = target_position * (1.0 - crop_scale)
        x, y = query[:, 0], query[:, 1]
        matched = torch.stack(
            (
                translation[:, 0:1]
                + crop_scale * (cosine[:, None] * x - sine[:, None] * y),
                translation[:, 1:2]
                + crop_scale * (sine[:, None] * x + cosine[:, None] * y),
            ),
            dim=-1,
        )
        position, direction = ContinuousSE2Bearing._solve_similarity_transform(
            query,
            matched,
            torch.ones(1, query.shape[0]),
            crop_scale,
            1e-6,
        )
        expected_direction = torch.stack((cosine, sine), dim=1)
        self.assertTrue(torch.allclose(position, target_position, atol=1e-5))
        self.assertTrue(
            torch.allclose(direction, expected_direction, atol=1e-5)
        )

    def test_grid_sample_is_differentiable_through_pose(self):
        satellite = torch.randn(2, 8, 16, 16, requires_grad=True)
        position = torch.tensor(
            [[0.2, -0.3], [-0.4, 0.1]], requires_grad=True
        )
        angles = torch.tensor([0.4, -1.1], requires_grad=True)
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        aligned, grid = ContinuousSE2Bearing._warp_satellite(
            satellite, position, direction, (8, 8)
        )
        self.assertEqual(aligned.shape, (2, 8, 8, 8))
        self.assertEqual(grid.shape, (2, 8, 8, 2))
        aligned.square().mean().backward()
        self.assertTrue(torch.isfinite(position.grad).all())
        self.assertTrue(torch.isfinite(angles.grad).all())
        self.assertGreater(float(position.grad.abs().sum()), 0.0)
        self.assertGreater(float(angles.grad.abs().sum()), 0.0)

    def test_forward_loss_backward_is_finite(self):
        torch.manual_seed(5)
        model = ContinuousSE2Bearing(**_tiny_kwargs())
        patches = torch.randn(2, 5, 3, 64, 64)
        coords = torch.empty(2, 2).uniform_(-0.8, 0.8)
        angles = torch.tensor([0.35, -2.2])
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        output = model.forward_full(patches, return_attention=True)
        self.assertEqual(output["position"].shape, (2, 2))
        self.assertEqual(output["direction"].shape, (2, 2))
        self.assertEqual(output["pose_stages"].shape, (2, 3, 4))
        self.assertTrue(torch.isfinite(output["position"]).all())
        self.assertTrue(torch.isfinite(output["direction"]).all())
        losses = ContinuousSE2Loss()(output, coords, direction)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name, module in (
            ("satellite_adapter", model.satellite_adapter),
            ("uav_adapter", model.uav_adapter),
            ("match_projection", model.match_projection),
            ("refiner_head", model.refiner.head),
        ):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertTrue(math.isfinite(gradient), name)
            self.assertGreater(gradient, 0.0, name)

    def test_v1_frontend_checkpoint_loading(self):
        source = PersistentBearing(
            backbone_name="tiny_cnn",
            pretrained=False,
            freeze_backbone=True,
            feature_dim=32,
            adapter_mlp_depth=1,
            shared_mlp_depth=1,
        )
        target = ContinuousSE2Bearing(**_tiny_kwargs())
        report = target.load_v1_frontend(
            {"model_state_dict": source.state_dict()}
        )
        self.assertGreater(report["loaded_tensors"], 0)
        self.assertEqual(report["skipped"], [])
        self.assertTrue(
            torch.equal(
                source.satellite_adapter.projections[0][0].weight,
                target.satellite_adapter.projections[0][0].weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
