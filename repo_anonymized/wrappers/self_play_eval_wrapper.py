from typing import List, Optional, Tuple

import numpy as np

from repo_anonymized.shared.policy.policy import Policy
from repo_anonymized.wrappers.self_play_reference_wrapper import (
    AbstractSelfPlayReferenceWrapper,
)
from repo_anonymized.wrappers.self_play_wrapper import SelfPlayWrapper


class SelfPlayEvalWrapper(AbstractSelfPlayReferenceWrapper):
    train_wrapper: Optional[SelfPlayWrapper]

    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        assert self.train_wrapper, "Must have assigned train_wrapper"
        assignments: List[Optional[Policy]] = [None] * self.env.num_envs  # type: ignore
        policies = list(reversed(self.train_wrapper.policies))
        for i in range(self.num_envs):
            policy = policies[i % len(policies)]
            assignments[2 * i + (i % 2)] = policy
        return assignments, [p is None for p in assignments]
