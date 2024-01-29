from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from repo_anonymized.shared.policy.policy import Policy
from repo_anonymized.shared.tensor_utils import batch_dict_keys
from repo_anonymized.wrappers.vectorable_wrapper import (
    VecEnvMaskedResetReturn,
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class AbstractSelfPlayReferenceWrapper(VectorableWrapper, ABC):
    next_obs: VecEnvObs
    next_action_masks: Optional[np.ndarray]

    def __init__(self, env) -> None:
        super().__init__(env)
        assert self.env.num_envs % 2 == 0
        self.num_envs = self.env.num_envs // 2

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        env = self.env  # type: ignore
        all_actions = np.zeros((env.num_envs,) + actions.shape[1:], dtype=actions.dtype)

        policy_assignments, learner_indexes = self._assignment_and_indexes()

        all_actions[learner_indexes] = actions
        for policy in set(p for p in policy_assignments if p):
            policy_indexes = [policy == p for p in policy_assignments]
            all_actions[policy_indexes] = policy.act(
                self.next_obs[policy_indexes],  # type: ignore
                deterministic=False,
                action_masks=batch_dict_keys(self.next_action_masks[policy_indexes])
                if self.next_action_masks is not None
                else None,
            )
        self.next_obs, rew, done, info = env.step(all_actions)
        self.next_action_masks = env.get_action_mask()

        rew = rew[learner_indexes]
        info = [_info for _info, is_learner in zip(info, learner_indexes) if is_learner]

        return self.next_obs[learner_indexes], rew, done[learner_indexes], info

    def reset(self) -> VecEnvObs:
        self.next_obs = super().reset()
        self.next_action_masks = self.env.get_action_mask()  # type: ignore
        _, indexes = self._assignment_and_indexes()
        return self.next_obs[indexes]  # type: ignore

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        _, learner_indexes = self._assignment_and_indexes()
        mapped_mask = np.zeros_like(learner_indexes)
        mapped_mask[learner_indexes] = env_mask
        assert np.all(
            mapped_mask[::2] == mapped_mask[1::2]
        ), f"Expected mapped_mask to be the same for player 1 and 2: {mapped_mask}"
        return self.env.masked_reset(mapped_mask)

    def get_action_mask(self) -> Optional[np.ndarray]:
        _, indexes = self._assignment_and_indexes()
        return (
            self.next_action_masks[indexes]
            if self.next_action_masks is not None
            else None
        )

    @abstractmethod
    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        ...


class SelfPlayReferenceWrapper(AbstractSelfPlayReferenceWrapper):
    policies_getter_fn: Optional[Callable[[], Sequence[Policy]]]

    def __init__(self, env: VectorableWrapper, window: int) -> None:
        super().__init__(env)
        assert env.num_envs % 2 == 0
        self.num_envs = env.num_envs // 2
        self.window = window

    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        assert self.policies_getter_fn, f"policies_getter_fn must be set"
        assignments: List[Optional[Policy]] = [None] * self.env.num_envs  # type: ignore
        policies = list(reversed(self.policies_getter_fn()))
        for i in range(self.num_envs):
            policy = policies[i % len(policies)]
            assignments[2 * i + (i % 2)] = policy
        return assignments, [p is None for p in assignments]

    def close(self) -> None:
        self.policies_getter_fn = None
