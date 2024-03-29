import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from repo_anonymized.shared.tensor_utils import NumpyOrDict, TensorOrDict, numpy_to_tensor
from repo_anonymized.wrappers.normalize import NormalizeObservation, NormalizeReward
from repo_anonymized.wrappers.vectorable_wrapper import VecEnv, VecEnvObs, find_wrapper

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
}

VEC_NORMALIZE_FILENAME = "vecnormalize.pkl"
MODEL_FILENAME = "model.pth"
NORMALIZE_OBSERVATION_FILENAME = "norm_obs.npz"
NORMALIZE_REWARD_FILENAME = "norm_reward.npz"

PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, env: VecEnv, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.vec_normalize = unwrap_vec_normalize(env)
        self.norm_observation = find_wrapper(env, NormalizeObservation)
        self.norm_reward = find_wrapper(env, NormalizeReward)
        self.device = None

    def to(
        self: PolicySelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> PolicySelf:
        super().to(device, dtype, non_blocking)
        self.device = device
        return self

    @abstractmethod
    def act(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        ...

    def save_weights(self, path: str) -> None:
        torch.save(
            self.state_dict(),
            os.path.join(path, MODEL_FILENAME),
        )

    def load_weights(self, path: str) -> None:
        self.load_state_dict(
            torch.load(os.path.join(path, MODEL_FILENAME), map_location=self.device)
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        if self.vec_normalize:
            self.vec_normalize.save(os.path.join(path, VEC_NORMALIZE_FILENAME))
        if self.norm_observation:
            self.norm_observation.save(
                os.path.join(path, NORMALIZE_OBSERVATION_FILENAME)
            )
        if self.norm_reward:
            self.norm_reward.save(os.path.join(path, NORMALIZE_REWARD_FILENAME))
        self.save_weights(path)

    def load(self, path: str) -> None:
        # VecNormalize load occurs in env.py
        self.load_weights(path)
        if self.norm_observation:
            self.norm_observation.load(
                os.path.join(path, NORMALIZE_OBSERVATION_FILENAME)
            )
        if self.norm_reward:
            self.norm_reward.load(os.path.join(path, NORMALIZE_REWARD_FILENAME))

    def load_from(self: PolicySelf, policy: PolicySelf) -> PolicySelf:
        self.load_state_dict(policy.state_dict())
        if self.norm_observation:
            assert policy.norm_observation
            self.norm_observation.load_from(policy.norm_observation)
        if self.norm_reward:
            assert policy.norm_reward
            self.norm_reward.load_from(policy.norm_reward)
        return self

    def __deepcopy__(self: PolicySelf, memo: Dict[int, Any]) -> PolicySelf:
        cls = self.__class__
        cpy = cls.__new__(cls)

        memo[id(self)] = cpy

        for k, v in self.__dict__.items():
            if k == "env":
                setattr(cpy, k, v)  # Don't deepcopy Env
            else:
                setattr(cpy, k, deepcopy(v, memo))

        return cpy

    def reset_noise(self) -> None:
        pass

    def _as_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        assert self.device
        return numpy_to_tensor(a, self.device)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sync_normalization(self, destination_env) -> None:
        current = destination_env
        while current != current.unwrapped:
            if isinstance(current, VecNormalize):
                assert self.vec_normalize
                current.ret_rms = deepcopy(self.vec_normalize.ret_rms)
                if hasattr(self.vec_normalize, "obs_rms"):
                    current.obs_rms = deepcopy(self.vec_normalize.obs_rms)
            elif isinstance(current, NormalizeObservation):
                assert self.norm_observation
                current.rms = deepcopy(self.norm_observation.rms)
            elif isinstance(current, NormalizeReward):
                assert self.norm_reward
                current.rms = deepcopy(self.norm_reward.rms)
            current = getattr(current, "venv", getattr(current, "env", current))
            if not current:
                raise AttributeError(
                    f"{type(current)} doesn't include env or venv attribute"
                )
