import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, Space

from repo_anonymized.shared.actor import PiForward
from repo_anonymized.shared.tensor_utils import TensorOrDict


class ACNForward(NamedTuple):
    pi_forward: PiForward
    v: torch.Tensor


class ActorCriticNetwork(nn.Module, ABC):
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        return self._distribution_and_value(
            obs, action=action, action_masks=action_masks
        )

    def distribution_and_value(
        self, obs: torch.Tensor, action_masks: Optional[TensorOrDict] = None
    ) -> ACNForward:
        return self._distribution_and_value(obs, action_masks=action_masks)

    @abstractmethod
    def _distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[TensorOrDict] = None,
        action_masks: Optional[TensorOrDict] = None,
    ) -> ACNForward:
        ...

    @abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        ...

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        ...

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return ()

    @abstractmethod
    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        ...

    def unfreeze(self):
        self.freeze(False, False, freeze_backbone=False)


def default_hidden_sizes(obs_space: Space) -> Sequence[int]:
    if isinstance(obs_space, Box):
        if len(obs_space.shape) == 3:  # type: ignore
            # By default feature extractor to output has no hidden layers
            return []
        elif len(obs_space.shape) == 1:  # type: ignore
            return [64, 64]
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
    elif isinstance(obs_space, Discrete):
        return [64]
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")
