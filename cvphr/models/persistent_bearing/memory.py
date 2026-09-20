import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@dataclass
class MemoryConfig:
    max_hypotheses: int = 5
    position_sigma: float = 0.20
    heading_sigma_rad: float = math.radians(20.0)
    memory_overlap_sigma: float = 0.35
    memory_weight: float = 0.5
    promote_after: int = 3
    max_tentative: int = 4
    max_confirmed: int = 32
    write_risk_threshold: float = 0.45
    lost_risk_threshold: float = 0.75
    min_branch_weight: float = 0.05


@dataclass
class MemoryEntry:
    pose: torch.Tensor
    descriptor: torch.Tensor
    confidence: float
    frame_id: int

    def clone(self) -> "MemoryEntry":
        return MemoryEntry(
            pose=self.pose.clone(),
            descriptor=self.descriptor.clone(),
            confidence=self.confidence,
            frame_id=self.frame_id,
        )


@dataclass
class HypothesisBranch:
    pose: torch.Tensor
    log_weight: float
    tentative: List[MemoryEntry] = field(default_factory=list)
    confirmed: List[MemoryEntry] = field(default_factory=list)
    consistent_writes: int = 0

    def clone(self) -> "HypothesisBranch":
        return HypothesisBranch(
            pose=self.pose.clone(),
            log_weight=self.log_weight,
            tentative=[entry.clone() for entry in self.tentative],
            confirmed=[entry.clone() for entry in self.confirmed],
            consistent_writes=self.consistent_writes,
        )


class HypothesisAlignedGeoMemory:
    """Online multi-hypothesis pose belief with guarded per-mode memory banks.

    Pose x/y values must use one route-level coordinate system. If model outputs
    local RSB coordinates, convert them to global map coordinates before calling
    step().
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.branches: List[HypothesisBranch] = []
        self.frame_id = 0

    def reset(self) -> None:
        self.branches = []
        self.frame_id = 0

    @staticmethod
    def _prepare_descriptor(descriptor: torch.Tensor) -> torch.Tensor:
        descriptor = descriptor.detach().to(device="cpu", dtype=torch.float32)
        descriptor = descriptor.reshape(-1)
        return F.normalize(descriptor, dim=0, eps=1e-6)

    @staticmethod
    def _prepare_candidates(
        poses: torch.Tensor,
        scores: torch.Tensor,
    ):
        poses = poses.detach().to(device="cpu", dtype=torch.float32)
        scores = scores.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"Expected poses [K,3], got {poses.shape}")
        if poses.shape[0] != scores.shape[0]:
            raise ValueError("Candidate pose and score counts differ")
        if poses.shape[0] == 0:
            raise ValueError("At least one pose candidate is required")
        poses[:, 2] = _wrap_angle(poses[:, 2])
        scores = scores.clamp_min(1e-9)
        scores = scores / scores.sum()
        return poses, scores

    def _transition_log_likelihood(
        self,
        previous_pose: torch.Tensor,
        current_pose: torch.Tensor,
        motion: torch.Tensor,
    ) -> float:
        predicted = previous_pose + motion
        predicted[2] = _wrap_angle(predicted[2])
        position_error = (current_pose[:2] - predicted[:2]).square().sum()
        heading_error = _wrap_angle(current_pose[2] - predicted[2]).square()
        value = -0.5 * (
            position_error / (self.config.position_sigma ** 2)
            + heading_error / (self.config.heading_sigma_rad ** 2)
        )
        return float(value.item())

    def _memory_score(
        self,
        branch: HypothesisBranch,
        current_pose: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> float:
        entries = branch.confirmed + branch.tentative[-self.config.max_tentative :]
        if not entries:
            return 0.0

        weighted_scores = []
        weights = []
        for entry in entries:
            distance_sq = (current_pose[:2] - entry.pose[:2]).square().sum()
            overlap = torch.exp(
                -0.5
                * distance_sq
                / (self.config.memory_overlap_sigma ** 2)
            )
            similarity = torch.dot(descriptor, entry.descriptor).clamp(-1.0, 1.0)
            weight = overlap * max(entry.confidence, 1e-3)
            weighted_scores.append(weight * similarity)
            weights.append(weight)

        denominator = torch.stack(weights).sum().clamp_min(1e-6)
        return float((torch.stack(weighted_scores).sum() / denominator).item())

    @staticmethod
    def _risk(weights: torch.Tensor) -> Dict[str, float]:
        weights = weights / weights.sum().clamp_min(1e-9)
        entropy = -(
            weights * weights.clamp_min(1e-9).log()
        ).sum() / math.log(max(weights.numel(), 2))
        top_two = weights.topk(min(2, weights.numel())).values
        margin = (
            top_two[0] - top_two[1]
            if top_two.numel() == 2
            else top_two[0]
        )
        risk = 0.7 * entropy + 0.3 * (1.0 - margin)
        return {
            "entropy": float(entropy.item()),
            "margin": float(margin.item()),
            "risk": float(risk.clamp(0.0, 1.0).item()),
        }

    def _write_entry(
        self,
        branch: HypothesisBranch,
        descriptor: torch.Tensor,
        confidence: float,
    ) -> None:
        branch.tentative.append(
            MemoryEntry(
                pose=branch.pose.clone(),
                descriptor=descriptor.clone(),
                confidence=confidence,
                frame_id=self.frame_id,
            )
        )
        branch.tentative = branch.tentative[-self.config.max_tentative :]
        branch.consistent_writes += 1

        if branch.consistent_writes >= self.config.promote_after:
            branch.confirmed.extend(branch.tentative)
            branch.confirmed = branch.confirmed[-self.config.max_confirmed :]
            branch.tentative = []
            branch.consistent_writes = 0

    def step(
        self,
        poses: torch.Tensor,
        scores: torch.Tensor,
        descriptor: torch.Tensor,
        motion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        poses, scores = self._prepare_candidates(poses, scores)
        descriptor = self._prepare_descriptor(descriptor)
        motion_tensor = (
            torch.zeros(3, dtype=torch.float32)
            if motion is None
            else motion.detach().to(device="cpu", dtype=torch.float32).reshape(3)
        )
        motion_tensor[2] = _wrap_angle(motion_tensor[2])

        if not self.branches:
            count = min(self.config.max_hypotheses, poses.shape[0])
            weights = scores[:count]
            weights = weights / weights.sum()
            self.branches = [
                HypothesisBranch(
                    pose=poses[index].clone(),
                    log_weight=float(weights[index].log().item()),
                )
                for index in range(count)
            ]
        else:
            candidate_log_weights = []
            candidate_parents = []
            for candidate_index, current_pose in enumerate(poses):
                parent_scores = []
                for branch in self.branches:
                    transition = self._transition_log_likelihood(
                        branch.pose,
                        current_pose,
                        motion_tensor,
                    )
                    memory_score = self._memory_score(
                        branch,
                        current_pose,
                        descriptor,
                    )
                    parent_scores.append(
                        branch.log_weight
                        + transition
                        + self.config.memory_weight * memory_score
                    )
                parent_tensor = torch.tensor(parent_scores, dtype=torch.float32)
                candidate_log_weights.append(
                    scores[candidate_index].log()
                    + torch.logsumexp(parent_tensor, dim=0)
                )
                candidate_parents.append(int(parent_tensor.argmax().item()))

            candidate_log_weights = torch.stack(candidate_log_weights)
            candidate_log_weights = (
                candidate_log_weights
                - torch.logsumexp(candidate_log_weights, dim=0)
            )
            count = min(self.config.max_hypotheses, poses.shape[0])
            selected = candidate_log_weights.topk(count).indices

            new_branches = []
            for candidate_index in selected.tolist():
                parent_index = candidate_parents[candidate_index]
                branch = self.branches[parent_index].clone()
                branch.pose = poses[candidate_index].clone()
                branch.log_weight = float(
                    candidate_log_weights[candidate_index].item()
                )
                new_branches.append(branch)
            self.branches = new_branches

        log_weights = torch.tensor(
            [branch.log_weight for branch in self.branches],
            dtype=torch.float32,
        )
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        weights = log_weights.exp()
        for index, branch in enumerate(self.branches):
            branch.log_weight = float(log_weights[index].item())

        risk = self._risk(weights)
        wrote = False
        if risk["risk"] <= self.config.write_risk_threshold:
            for weight, branch in zip(weights.tolist(), self.branches):
                if weight >= self.config.min_branch_weight:
                    self._write_entry(
                        branch,
                        descriptor,
                        confidence=weight,
                    )
                    wrote = True
        else:
            for branch in self.branches:
                branch.consistent_writes = 0
                branch.tentative = []

        poses_out = torch.stack([branch.pose for branch in self.branches])
        best_index = int(weights.argmax().item())
        result = {
            "pose": poses_out[best_index].clone(),
            "poses": poses_out,
            "weights": weights,
            "entropy": torch.tensor(risk["entropy"]),
            "margin": torch.tensor(risk["margin"]),
            "risk": torch.tensor(risk["risk"]),
            "lost": torch.tensor(
                risk["risk"] >= self.config.lost_risk_threshold
            ),
            "wrote_memory": torch.tensor(wrote),
            "confirmed_entries": torch.tensor(
                sum(len(branch.confirmed) for branch in self.branches)
            ),
        }
        self.frame_id += 1
        return result
