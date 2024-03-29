from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

import gym
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from repo_anonymized.shared.callbacks import Callback
from repo_anonymized.shared.policy.policy import Policy
from repo_anonymized.wrappers.vectorable_wrapper import VecEnv

AlgorithmSelf = TypeVar("AlgorithmSelf", bound="Algorithm")


class Algorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        policy: Policy,
        device: torch.device,
        tb_writer: SummaryWriter,
        **kwargs,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.device = device
        self.tb_writer = tb_writer

    @abstractmethod
    def learn(
        self: AlgorithmSelf,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> AlgorithmSelf:
        ...
