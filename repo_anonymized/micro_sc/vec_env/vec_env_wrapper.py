from dataclasses import astuple
from typing import Optional

import gym
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from repo_anonymized.microrts.vec_env.microrts_socket_env import MicroRTSSocketEnv
from repo_anonymized.microrts.vec_env.microrts_space_transform import (
    MicroRTSSpaceTransform,
)
from repo_anonymized.microrts.wrappers.microrts_stats_recorder import (
    MicrortsStatsRecorder,
)
from repo_anonymized.runner.config import Config, EnvHyperparams
from repo_anonymized.wrappers.action_mask_stats_recorder import ActionMaskStatsRecorder
from repo_anonymized.wrappers.action_mask_wrapper import MicrortsMaskWrapper
from repo_anonymized.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from repo_anonymized.wrappers.episode_stats_writer import EpisodeStatsWriter
from repo_anonymized.wrappers.hwc_to_chw_observation import HwcToChwObservation
from repo_anonymized.wrappers.is_vector_env import IsVectorEnv
from repo_anonymized.wrappers.score_reward_wrapper import ScoreRewardWrapper
from repo_anonymized.wrappers.self_play_wrapper import SelfPlayWrapper
from repo_anonymized.wrappers.vectorable_wrapper import VecEnv
from repo_anonymized.micro_sc.vec_env.vec_env import *
import repo_anonymized.micro_sc.vec_env.vec_env

from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def sample(action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = -1e6
    permutedAction = action.permute(0, 2, 3, 1)
    logits = torch.where(mask == 1, permutedAction, neg_inf)
    resultantAction = torch.zeros([*permutedAction.size()[:-1], 0])
    lastDim = 0
    for i in ACTION_SIZE:
        dist = Categorical(logits=logits[:, :, :, lastDim:lastDim + i])
        resultantAction = torch.concat([resultantAction, dist.sample().unsqueeze(-1)], dim=-1)
        lastDim += i
    return resultantAction
class VecEnvScRandom:
    def __init__(self, vecEnv: VecEnvSc):
        self.env = vecEnv
        self.unwrapped = self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_plane_space = self.env.action_plane_space

    def reset(self):
        ob, masks = self.env.reset()
        self.isDone = torch.zeros(self.env.num_workers)
        self.opponentMask = masks[1].permute(0, 2, 3, 1)
        return (ob[0], masks[0].permute(0, 2, 3, 1))

    def step(self, actions):
        randActions = torch.rand(actions.size())
        result = sample(randActions, self.opponentMask)
        obs, re, masks, isDone, t = self.env.step(actions.permute(0, 3, 1, 2), result.permute(0, 3, 1, 2))
        self.opponentMask = masks[1].permute(0, 2, 3, 1)
        reallyDone = torch.clamp_min_(isDone - self.isDone, 0)
        self.isDone = isDone
        return obs[0], masks[0].permute(0, 2, 3, 1), re[0] * (1-reallyDone), reallyDone, {}

    def render(self):
        pass


def make_micro_sc_env(config: Config,
                      hparams: EnvHyperparams,
                      training: bool = True,
                      render: bool = False,
                      normalize_load_path: Optional[str] = None,
                      tb_writer: Optional[SummaryWriter] = None,) ->VecEnv:
    (
        _,  # env_type
        n_envs,
        _,  # frame_stack
        make_kwargs,
        _,  # no_reward_timeout_steps
        _,  # no_reward_fire_steps
        _,  # vec_env_class
        _,  # normalize
        _,  # normalize_kwargs,
        rolling_length,
        _,  # train_record_video
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        _,  # normalize_type
        _,  # mask_actions
        bots,
        self_play_kwargs,
        selfplay_bots,
        additional_win_loss_reward,
        map_paths,
        score_reward_kwargs,
        is_agent,
        valid_sizes,
        paper_planes_sizes,
        fixed_size,
        terrain_overrides,
        time_budget_ms,
        video_frames_per_second,
        _,  # reference_bot,
        _, # self_play_reference_kwargs,
        _,  # additional_win_loss_smoothing_factor,
    ) = astuple(hparams)
    vecEnv = VecEnvSc(n_envs, torch.device("cpu"), 0, False, True, 0, 5, 5, 100)
    return VecEnvScRandom(vecEnv)
