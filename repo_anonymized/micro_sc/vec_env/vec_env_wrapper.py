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
    logits = torch.where(mask == 1, action, neg_inf)
    resultantAction = torch.zeros([*action.size()[:-1], 0])
    lastDim = 0
    for i in ACTION_SIZE:
        dist = Categorical(logits=logits[:, :, :, lastDim:lastDim + i])
        resultantAction = torch.concat([resultantAction, dist.sample().unsqueeze(-1)], dim=-1)
        lastDim += i
    return resultantAction
class VecEnvScRandom:
    def __init__(self, vecEnv: VecEnvSc, maxSteps):
        self.env = vecEnv
        self.unwrapped = self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_plane_space = self.env.action_plane_space
        self.num_envs = self.env.num_workers
        self.metadata = {"semantics.async": False, "render_modes": ["rgb_array"]}
        self.maxSteps = maxSteps
        self.render_mode = 'rgb_array'

    def reset(self):
        ob, masks = self.env.reset()
        self.opponentMask = masks[1].permute(0, 2, 3, 1)
        self.mask = masks[0].permute(0, 2, 3, 1) == 1
        self.lastObs = ob[0].numpy()
        self.stepCnt = 0
        return ob[0].numpy()

    def step(self, actions):
        actions = torch.from_numpy(actions.reshape(actions.shape[0], GAME_H, GAME_W, -1))
        randActions = torch.rand([*actions.shape[:-1], sum(ACTION_SIZE)])
        result = sample(randActions, self.opponentMask)
        obs, masks, re, isDone, t = self.env.step(actions.permute(0, 3, 1, 2), result.permute(0, 3, 1, 2))
        self.opponentMask = masks[1].permute(0, 2, 3, 1)
        self.mask = masks[0].permute(0, 2, 3, 1) == 1
        self.lastObs = obs[0].numpy()
        self.stepCnt += 1
        if self.stepCnt >= self.maxSteps:
            isDone[:] = 1
        return obs[0].numpy(), re[0].numpy(), isDone.numpy(), [{"rewards": float(re[0][i, 1])} for i in range(self.env.num_workers)]

    def get_action_mask(self):
        return self.mask.numpy()

    def render(self, mode="rgb_array"):
        retVal = np.zeros((GAME_W, GAME_H, 3))
        a = self.lastObs[0, ObPlane.OWNER_1, :, :]
        b = self.lastObs[0, ObPlane.OWNER_2, :, :]
        n = self.lastObs[0, ObPlane.OWNER_NONE, :, :]
        retVal[:, :, 0] = b
        retVal[:, :, 1] = a
        retVal[:, :, 2] = n
        retVal = (retVal+1)/2 * 255
        img = plt.imshow(retVal.astype(np.dtype('uint8')))
        return img.make_image(None)[0][:, :, :-1]

    def close(self):
        return


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
    return VecEnvScRandom(vecEnv, make_kwargs.get('max_steps'))
