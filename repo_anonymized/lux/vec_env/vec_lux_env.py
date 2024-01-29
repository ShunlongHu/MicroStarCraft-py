from typing import Dict, List, Optional, TypeVar

import gym
import numpy as np
from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from repo_anonymized.lux.rewards import LuxRewardWeights
from repo_anonymized.lux.wrappers.lux_env_gridnet import LuxEnvGridnet
from repo_anonymized.wrappers.vectorable_wrapper import (
    VecEnvMaskedResetReturn,
    VecEnvObs,
    VecEnvStepReturn,
)

VecLuxEnvSelf = TypeVar("VecLuxEnvSelf", bound="VecLuxEnv")


class VecLuxEnv(VectorEnv):
    def __init__(
        self,
        num_envs: int,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ice_distance_buffer: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        self.envs = [
            LuxEnvGridnet(
                gym.make("LuxAI_S2-v0", collect_stats=True, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
                verify=verify,
                factory_ice_distance_buffer=factory_ice_distance_buffer,
            )
            for _ in range(num_envs // 2)
        ]
        single_env = self.envs[0]
        map_dim = single_env.unwrapped.env_cfg.map_size
        self.num_map_tiles = map_dim * map_dim
        single_observation_space = single_env.single_observation_space
        self.action_plane_space = single_env.action_plane_space
        single_action_space = single_env.single_action_space
        self.metadata = single_env.metadata
        super().__init__(num_envs, single_observation_space, single_action_space)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        step_returns = [
            env.step(action[2 * idx : 2 * idx + 2]) for idx, env in enumerate(self.envs)
        ]
        obs = np.concatenate([sr[0] for sr in step_returns])
        rewards = np.concatenate([sr[1] for sr in step_returns])
        dones = np.concatenate([sr[2] for sr in step_returns])
        infos = [info for sr in step_returns for info in sr[3]]
        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        env_obervations = [env.reset() for env in self.envs]
        return np.concatenate(env_obervations)

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        mapped_mask = env_mask[::2]
        assert np.all(
            mapped_mask == env_mask[1::2]
        ), "env_mask must be the same for player 1 and 2: {env_mask}"
        masked_envs = [env for env, m in zip(self.envs, mapped_mask) if m]
        return VecEnvMaskedResetReturn(
            obs=np.concatenate([env.reset() for env in masked_envs]),
            action_mask=np.concatenate([env.get_action_mask() for env in masked_envs]),
        )

    def seed(self, seed: Optional[int] = None):
        seed_rng = np.random.RandomState(seed)
        for e, s in zip(
            self.envs, seed_rng.randint(0, np.iinfo(np.int32).max, len(self.envs))
        ):
            e.seed(s)

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close()

    @property
    def unwrapped(self: VecLuxEnvSelf) -> VecLuxEnvSelf:
        return self

    def render(self, mode="human", **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode, **kwargs)
        if mode == "human":
            for env in self.envs:
                env.render(mode=mode, **kwargs)
        elif mode == "rgb_array":
            imgs = self._get_images()
            bigimg = tile_images(imgs)
            return bigimg

    def _get_images(self) -> List[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def get_action_mask(self) -> np.ndarray:
        return np.concatenate([env.get_action_mask() for env in self.envs])

    @property
    def reward_weights(self) -> LuxRewardWeights:
        return self.envs[0].reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        for env in self.envs:
            env.reward_weights = reward_weights
