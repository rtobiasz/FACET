from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from facet.io.checkpoint import checkpoint_state_from_payload, create_checkpoint_payload
from facet.training.optimizer_state import (
    PLANAR_OPTIMIZER_GROUP_NAMES,
    build_planar_optimizer,
    compact_planar_conditioning_optimizer_state,
)
from facet.model.components import (
    PlanarGaussianModel,
    materialize_renderer_conditioning,
)
from facet.model.geometry import (
    PlanarCompatibilityConstraint,
    angles_to_planar_quaternions,
)


def values(count=2):
    shapes = {
        "RGB": (count, 3), "A": (count, 1), "k": (count, 1),
        "w1_uv": (64, 8), "w1_v": (64, 24), "w1_conditioning": (64, 96),
        "b1": (64,), "w2": (64, 64), "b2": (64,), "w3": (16, 64),
        "b3": (16,), "conditioning_variable": (count, 96), "features": (13154,),
        "mean_xy": (count, 2), "theta": (count,), "s": (count, 2),
    }
    return {name: torch.randn(shape) for name, shape in shapes.items()}


class PlanarProductionGeometryTest(unittest.TestCase):
    def test_materializes_geometry_without_storing_xyz_or_quaternion(self):
        model = PlanarGaussianModel(values(), plane_z=0.25, requires_grad=True)
        self.assertNotIn("m", model._parameters)
        self.assertNotIn("q", model._parameters)
        torch.testing.assert_close(model.m[:, 2], torch.full((2,), 0.25))
        torch.testing.assert_close(model.q[:, 1:3], torch.zeros((2, 2)))
        self.assertEqual(tuple(model.model_tensors()), model.PARAMETER_NAMES)
        self.assertIn("m", model.export_geometry())
        self.assertNotIn("theta", model.export_geometry())

    def test_native_gradient_assignment_applies_chain_rule(self):
        model = PlanarGaussianModel(values(1), requires_grad=True)
        model.theta.data.fill_(0.6)
        shared = {
            name: torch.ones_like(getattr(model, name))
            for name in model.PARAMETER_NAMES
            if name not in ("mean_xy", "theta")
        }
        bundle = type("Gradients", (), {"as_dict": lambda self: {
            **shared,
            "m": torch.tensor([[2.0, 3.0, 99.0]]),
            "q": torch.tensor([[5.0, 7.0, 11.0, 13.0]]),
        }})()
        model.assign_native_gradients(bundle)
        torch.testing.assert_close(model.mean_xy.grad, torch.tensor([[2.0, 3.0]]))
        expected = 0.5 * (
            -torch.sin(torch.tensor(0.3)) * 5.0
            + torch.cos(torch.tensor(0.3)) * 13.0
        )
        torch.testing.assert_close(model.theta.grad, expected.reshape(1))

    def test_conditioning_ablation_masks_values_and_native_gradients(self):
        model = PlanarGaussianModel(
            values(2), requires_grad=True, active_conditioning_dimensions=16
        )
        self.assertEqual(model.conditioning_variable.shape, (2, 16))
        self.assertEqual(model.w1_conditioning.shape, (64, 16))
        self.assertEqual(model.renderer_conditioning().shape, (2, 96))
        self.assertEqual(model.renderer_w1_conditioning().shape, (64, 96))
        self.assertTrue(torch.all(model.renderer_conditioning()[:, 16:] == 0))
        self.assertTrue(torch.all(model.renderer_w1_conditioning()[:, 16:] == 0))
        shared = {
            name: torch.ones_like(getattr(model, name))
            for name in model.PARAMETER_NAMES
            if name not in (
                "mean_xy", "theta", "conditioning_variable", "w1_conditioning"
            )
        }
        bundle = type("Gradients", (), {"as_dict": lambda self: {
            **shared,
            "conditioning_variable": torch.ones((2, 96)),
            "w1_conditioning": torch.ones((64, 96)),
            "m": torch.ones((2, 3)),
            "q": torch.ones((2, 4)),
        }})()
        model.assign_native_gradients(bundle)
        self.assertTrue(torch.all(model.conditioning_variable.grad == 1))
        self.assertTrue(torch.all(model.w1_conditioning.grad == 1))

    def test_zero_dimensional_conditioning_is_physically_compact(self):
        model = PlanarGaussianModel(
            values(3), requires_grad=True, active_conditioning_dimensions=0
        )
        self.assertEqual(model.conditioning_variable.shape, (3, 0))
        self.assertEqual(model.w1_conditioning.shape, (64, 0))
        self.assertEqual(model.renderer_conditioning().shape, (3, 96))
        self.assertEqual(model.renderer_w1_conditioning().shape, (64, 96))
        self.assertEqual(model.native_conditioning().shape, (3, 96))
        self.assertTrue(model.native_conditioning().is_contiguous())
        self.assertEqual(model.native_conditioning().numel(), 3 * 96)
        self.assertEqual(float(model.native_conditioning().abs().sum()), 0.0)
        self.assertEqual(model.conditioning_variable.numel(), 0)
        self.assertEqual(model.w1_conditioning.numel(), 0)
        self.assertEqual(model.conditioning_variable.untyped_storage().nbytes(), 0)
        self.assertEqual(model.w1_conditioning.untyped_storage().nbytes(), 0)
        payload = create_checkpoint_payload(
            model.model_tensors(), optimizer_state=None, iteration=0,
            training_time_seconds=0.0, mode="image_2d"
        )
        self.assertEqual(payload["model"]["conditioning_variable"].shape, (3, 0))
        self.assertEqual(payload["model"]["w1_conditioning"].shape, (64, 0))

    def test_renderer_conditioning_materializes_compact_dimensions(self):
        zero_dimensional = torch.empty((7, 0))
        materialized = materialize_renderer_conditioning(zero_dimensional)
        self.assertEqual(materialized.shape, (7, 96))
        self.assertTrue(materialized.is_contiguous())
        self.assertEqual(float(materialized.abs().sum()), 0.0)

        compact = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        padded = materialize_renderer_conditioning(compact)
        self.assertEqual(padded.shape, (7, 96))
        torch.testing.assert_close(padded[:, :3], compact)
        self.assertEqual(float(padded[:, 3:].abs().sum()), 0.0)

    def test_legacy_optimizer_state_is_compacted_for_zero_dimensional_resume(self):
        rates = {name: 1e-2 for name in PLANAR_OPTIMIZER_GROUP_NAMES}
        legacy = PlanarGaussianModel(values(1), requires_grad=True)
        legacy_optimizer = build_planar_optimizer(legacy, rates)
        for parameter in legacy.parameters():
            parameter.grad = torch.ones_like(parameter)
        legacy_optimizer.step()

        compact = PlanarGaussianModel(
            legacy.model_tensors(), requires_grad=True,
            active_conditioning_dimensions=0,
        )
        compact_optimizer = build_planar_optimizer(compact, rates)
        compact_optimizer.load_state_dict(
            compact_planar_conditioning_optimizer_state(
                legacy_optimizer.state_dict(), 0
            )
        )
        for parameter in compact.parameters():
            parameter.grad = torch.zeros_like(parameter)
        compact_optimizer.step()
        groups = {
            group["name"]: group for group in compact_optimizer.param_groups
        }
        for name in ("conditioning_variable", "w1_conditioning"):
            parameter = groups[name]["params"][0]
            self.assertEqual(
                compact_optimizer.state[parameter]["exp_avg"].shape,
                parameter.shape,
            )
            self.assertEqual(
                compact_optimizer.state[parameter]["exp_avg"].untyped_storage().nbytes(),
                0,
            )

    def test_optimizer_one_step_and_resume_are_exact(self):
        model = PlanarGaussianModel(values(1), requires_grad=True)
        rates = {name: 1e-2 for name in PLANAR_OPTIMIZER_GROUP_NAMES}
        optimizer = build_planar_optimizer(model, rates)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        payload = create_checkpoint_payload(
            model.model_tensors(), optimizer_state=optimizer.state_dict(),
            iteration=1, training_time_seconds=2.0, mode="image_2d"
        )
        state = checkpoint_state_from_payload(payload, "cpu", require_rgb=True)
        restored = PlanarGaussianModel(state.model, requires_grad=True)
        restored_optimizer = build_planar_optimizer(restored, rates)
        restored_optimizer.load_state_dict(state.optimizer_state)
        left_state = optimizer.state_dict()["state"].values()
        right_state = restored_optimizer.state_dict()["state"].values()
        for left, right in zip(left_state, right_state):
            torch.testing.assert_close(left["exp_avg"], right["exp_avg"])
            torch.testing.assert_close(left["exp_avg_sq"], right["exp_avg_sq"])

    def test_scalar_angle_is_no_worse_on_tiny_orientation_benchmark(self):
        target = angles_to_planar_quaternions(torch.tensor([1.1]))
        theta = torch.nn.Parameter(torch.tensor([-0.7]))
        projected_q = torch.nn.Parameter(
            angles_to_planar_quaternions(torch.tensor([-0.7]))
        )
        scalar_optimizer = torch.optim.Adam([theta], lr=0.08)
        projected_optimizer = torch.optim.Adam([projected_q], lr=0.08)
        means = torch.nn.Parameter(torch.zeros((1, 3)))
        constraint = PlanarCompatibilityConstraint()
        projected_model = type(
            "Projected", (), {"m": means, "q": projected_q}
        )()

        for _ in range(25):
            scalar_optimizer.zero_grad(set_to_none=True)
            scalar_loss = torch.mean(
                (angles_to_planar_quaternions(theta) - target) ** 2
            )
            scalar_loss.backward()
            scalar_optimizer.step()

            projected_optimizer.zero_grad(set_to_none=True)
            projected_loss = torch.mean((projected_q - target) ** 2)
            projected_loss.backward()
            projected_optimizer.step()
            constraint.apply(projected_model, projected_optimizer)

        scalar_loss = torch.mean(
            (angles_to_planar_quaternions(theta) - target) ** 2
        )
        projected_loss = torch.mean((projected_q - target) ** 2)
        self.assertLessEqual(float(scalar_loss), float(projected_loss) + 1e-6)

    def test_mode_and_orientation_mismatch_are_rejected(self):
        payload = create_checkpoint_payload(
            values(), optimizer_state=None, iteration=0,
            training_time_seconds=0, mode="image_2d"
        )
        payload["orientation"] = "quaternion"
        with self.assertRaisesRegex(ValueError, "scalar_angle"):
            checkpoint_state_from_payload(payload, "cpu")


if __name__ == "__main__":
    unittest.main()
