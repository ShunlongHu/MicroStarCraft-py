import itertools
import os
import shutil
from collections import deque
from copy import deepcopy
from time import perf_counter
from typing import Callable, Deque, Dict, List, Optional, Sequence, Union

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from repo_anonymized.shared.callbacks import Callback
from repo_anonymized.shared.policy.policy import Policy
from repo_anonymized.shared.stats import Episode, EpisodeAccumulator, EpisodesStats
from repo_anonymized.shared.tensor_utils import batch_dict_keys
from repo_anonymized.wrappers.self_play_reference_wrapper import SelfPlayReferenceWrapper
from repo_anonymized.wrappers.vec_episode_recorder import VecEpisodeRecorder
from repo_anonymized.wrappers.vectorable_wrapper import VecEnv, find_wrapper


class EvaluateAccumulator(EpisodeAccumulator):
    def __init__(
        self,
        num_envs: int,
        goal_episodes: int,
        print_returns: bool = True,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
    ):
        super().__init__(num_envs)
        self.completed_episodes_by_env_idx = [[] for _ in range(num_envs)]
        self.goal_episodes_per_env = int(np.ceil(goal_episodes / num_envs))
        self.print_returns = print_returns
        if ignore_first_episode:
            first_done = set()

            def should_record_done(idx: int) -> bool:
                has_done_first_episode = idx in first_done
                first_done.add(idx)
                return has_done_first_episode

            self.should_record_done = should_record_done
        else:
            self.should_record_done = lambda idx: True
        self.additional_keys_to_log = additional_keys_to_log

    def on_done(self, ep_idx: int, episode: Episode, info: Dict) -> None:
        if self.additional_keys_to_log:
            episode.info = {k: info[k] for k in self.additional_keys_to_log}
        if (
            self.should_record_done(ep_idx)
            and len(self.completed_episodes_by_env_idx[ep_idx])
            >= self.goal_episodes_per_env
        ):
            return
        self.completed_episodes_by_env_idx[ep_idx].append(episode)
        if self.print_returns:
            print(
                f"Episode {len(self)} | "
                f"Score {episode.score} | "
                f"Length {episode.length}"
            )

    def __len__(self) -> int:
        return sum(len(ce) for ce in self.completed_episodes_by_env_idx)

    @property
    def episodes(self) -> List[Episode]:
        return list(itertools.chain(*self.completed_episodes_by_env_idx))

    def is_done(self) -> bool:
        return all(
            len(ce) == self.goal_episodes_per_env
            for ce in self.completed_episodes_by_env_idx
        )


def evaluate(
    env: VecEnv,
    policy: Policy,
    n_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    print_returns: bool = True,
    ignore_first_episode: bool = False,
    additional_keys_to_log: Optional[List[str]] = None,
    score_function: str = "mean-std",
) -> EpisodesStats:
    policy.sync_normalization(env)
    policy.eval()

    episodes = EvaluateAccumulator(
        env.num_envs,
        n_episodes,
        print_returns,
        ignore_first_episode,
        additional_keys_to_log=additional_keys_to_log,
    )

    obs = env.reset()
    get_action_mask = getattr(env, "get_action_mask", None)
    while not episodes.is_done():
        act = policy.act(
            obs,
            deterministic=deterministic,
            action_masks=batch_dict_keys(get_action_mask())
            if get_action_mask
            else None,
        )
        obs, rew, done, info = env.step(act)
        episodes.step(rew, done, info)
        if render:
            env.render()
    stats = EpisodesStats(
        episodes.episodes,
        score_function=score_function,
    )
    if print_returns:
        print(stats)
    return stats


class EvalCallback(Callback):
    prior_policies: Optional[Deque[Policy]]

    def __init__(
        self,
        policy: Policy,
        env: VecEnv,
        tb_writer: SummaryWriter,
        best_model_path: Optional[str] = None,
        step_freq: Union[int, float] = 50_000,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        only_record_video_on_best: bool = True,
        video_env: Optional[VecEnv] = None,
        video_dir: Optional[str] = None,
        max_video_length: int = 3600,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
        score_function: str = "mean-std",
        wandb_enabled: bool = False,
        score_threshold: Optional[float] = None,
        skip_evaluate_at_start: bool = False,
        only_checkpoint_initial_policy: bool = False,
        latest_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.tb_writer = tb_writer
        self.best_model_path = best_model_path
        self.step_freq = int(step_freq)
        self.n_episodes = n_episodes
        self.save_best = save_best
        self.deterministic = deterministic
        self.stats: List[EpisodesStats] = []
        self.best = None

        self.only_record_video_on_best = only_record_video_on_best
        assert (video_env is not None) == (video_dir is not None)
        self.video_env = video_env
        self.video_dir = video_dir
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        self.max_video_length = max_video_length
        self.ignore_first_episode = ignore_first_episode
        self.additional_keys_to_log = additional_keys_to_log
        self.score_function = score_function
        self.wandb_enabled = wandb_enabled
        self.score_threshold = score_threshold
        self.skip_evaluate_at_start = skip_evaluate_at_start
        self.latest_model_path = latest_model_path

        self_play_reference_wrapper = find_wrapper(env, SelfPlayReferenceWrapper)
        if self_play_reference_wrapper:
            self.prior_policies = deque(maxlen=self_play_reference_wrapper.window)
            self.only_checkpoint_initial_policy = only_checkpoint_initial_policy
            self.checkpoint_policy()

            def policies_getter_fn() -> Sequence[Policy]:
                assert self.prior_policies
                return self.prior_policies

            self_play_reference_wrapper.policies_getter_fn = policies_getter_fn
            if video_env:
                video_sprw = find_wrapper(video_env, SelfPlayReferenceWrapper)
                assert (
                    video_sprw
                ), f"video_env should have SelfPlayReferenceWrapper given eval env does"
                video_sprw.policies_getter_fn = policies_getter_fn
        else:
            self.prior_policies = None

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        desired_num_stats = self.timesteps_elapsed // self.step_freq
        if not self.skip_evaluate_at_start:
            desired_num_stats += 1
        if desired_num_stats > len(self.stats):
            self.evaluate()
        return True

    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        start_time = perf_counter()
        eval_stat = evaluate(
            self.env,
            self.policy,
            n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            print_returns=print_returns or False,
            ignore_first_episode=self.ignore_first_episode,
            additional_keys_to_log=self.additional_keys_to_log,
            score_function=self.score_function,
        )
        end_time = perf_counter()
        self.tb_writer.add_scalar(
            "eval/steps_per_second",
            eval_stat.length.sum() / (end_time - start_time),
            self.timesteps_elapsed,
        )
        self.policy.train(True)
        print(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")

        self.stats.append(eval_stat)

        if self.score_threshold is not None:
            is_best = eval_stat.score.score() >= self.score_threshold
            strictly_better = eval_stat.score.score() > self.score_threshold
        else:
            is_best = not self.best or eval_stat >= self.best
            strictly_better = not self.best or eval_stat > self.best

        if self.latest_model_path:
            self.policy.save(self.latest_model_path)
        if is_best:
            self.best = eval_stat
            if self.save_best:
                assert self.best_model_path
                self.policy.save(self.best_model_path)
                print("Saved best model")
                if self.wandb_enabled:
                    import wandb

                    best_model_name = os.path.split(self.best_model_path)[-1]
                    shutil.make_archive(
                        os.path.join(wandb.run.dir, best_model_name),  # type: ignore
                        "zip",
                        self.best_model_path,
                    )
            self.best.write_to_tensorboard(
                self.tb_writer, "best_eval", self.timesteps_elapsed
            )
        if self.video_env and (not self.only_record_video_on_best or strictly_better):
            assert self.video_env and self.video_dir
            best_video_base_path = os.path.join(
                self.video_dir, str(self.timesteps_elapsed)
            )
            video_wrapped = VecEpisodeRecorder(
                self.video_env,
                best_video_base_path,
                max_video_length=self.max_video_length,
            )
            video_stats = evaluate(
                video_wrapped,
                self.policy,
                1,
                deterministic=self.deterministic,
                print_returns=False,
                score_function=self.score_function,
            )
            print(f"Saved video: {video_stats}")

        eval_stat.write_to_tensorboard(self.tb_writer, "eval", self.timesteps_elapsed)
        self.checkpoint_policy()
        return eval_stat

    def checkpoint_policy(self):
        if self.prior_policies is not None:
            if self.only_checkpoint_initial_policy and len(self.prior_policies) > 0:
                return
            self.prior_policies.append(deepcopy(self.policy))
