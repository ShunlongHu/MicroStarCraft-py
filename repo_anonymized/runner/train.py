# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import logging
import os
import sys

from repo_anonymized.rollout.guided_learner_rollout import GuidedLearnerRolloutGenerator
from repo_anonymized.rollout.random_guided_learner_rollout import (
    RandomGuidedLearnerRolloutGenerator,
)
from repo_anonymized.rollout.reference_ai_rollout import ReferenceAIRolloutGenerator
from repo_anonymized.rollout.sync_step_rollout import SyncStepRolloutGenerator
from repo_anonymized.shared.callbacks.self_play_callback import SelfPlayCallback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import yaml
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from repo_anonymized.runner.config import Config, EnvHyperparams, RunArgs
from repo_anonymized.runner.running_utils import (
    ALGOS,
    get_device,
    hparam_dict,
    load_hyperparams,
    make_policy,
    plot_eval_callback,
    set_device_optimizations,
    set_seeds,
)
from repo_anonymized.shared.callbacks.callback import Callback
from repo_anonymized.shared.callbacks.eval_callback import EvalCallback
from repo_anonymized.shared.callbacks.hyperparam_transitions import HyperparamTransitions
from repo_anonymized.shared.callbacks.reward_decay_callback import RewardDecayCallback
from repo_anonymized.shared.stats import EpisodesStats
from repo_anonymized.shared.vec_env import make_env, make_eval_env
from repo_anonymized.wrappers.self_play_wrapper import SelfPlayWrapper
from repo_anonymized.wrappers.vectorable_wrapper import find_wrapper


@dataclass
class TrainArgs(RunArgs):
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None


def train(args: TrainArgs):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(args)
    hyperparams = load_hyperparams(args.algo, args.env)
    logging.info(hyperparams)
    config = Config(args, hyperparams, os.getcwd())

    wandb_enabled = bool(args.wandb_project_name)
    if wandb_enabled:
        wandb.tensorboard.patch(
            root_logdir=config.tensorboard_summary_path, pytorch=True
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=asdict(hyperparams),
            name=config.run_name(),
            monitor_gym=True,
            save_code=True,
            tags=args.wandb_tags,
            group=args.wandb_group,
        )
        wandb.config.update(args)

    tb_writer = SummaryWriter(config.tensorboard_summary_path.replace(':', '_'))

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_env(
        config, EnvHyperparams(**config.env_hyperparams), tb_writer=tb_writer
    )
    device = get_device(config, env)
    set_device_optimizations(device, **config.device_hyperparams)
    policy = make_policy(config, env, device, **config.policy_hyperparams)
    algo = ALGOS[args.algo](policy, device, tb_writer, **config.algo_hyperparams)

    num_parameters = policy.num_parameters()
    num_trainable_parameters = policy.num_trainable_parameters()
    if wandb_enabled:
        wandb.run.summary["num_parameters"] = num_parameters  # type: ignore
        wandb.run.summary["num_trainable_parameters"] = num_trainable_parameters  # type: ignore
    else:
        print(
            f"num_parameters = {num_parameters} ; "
            f"num_trainable_parameters = {num_trainable_parameters}"
        )

    self_play_wrapper = find_wrapper(env, SelfPlayWrapper)
    eval_env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        self_play_wrapper=self_play_wrapper,
    )
    video_env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        override_hparams={"n_envs": 1},
        self_play_wrapper=self_play_wrapper,
    )
    eval_callback = EvalCallback(
        policy,
        eval_env,
        tb_writer,
        best_model_path=config.model_dir_path(best=True),
        **config.eval_callback_params(),
        video_env=video_env,
        video_dir=config.videos_path,
        additional_keys_to_log=config.additional_keys_to_log,
        wandb_enabled=wandb_enabled,
        latest_model_path=config.model_dir_path(best=False),
    )
    callbacks: List[Callback] = [eval_callback]
    if config.hyperparams.reward_decay_callback:
        callbacks.append(
            RewardDecayCallback(
                config, env, **(config.hyperparams.reward_decay_callback_kwargs or {})
            )
        )
    if config.hyperparams.hyperparam_transitions_kwargs:
        callbacks.append(
            HyperparamTransitions(
                config,
                env,
                algo,
                **config.hyperparams.hyperparam_transitions_kwargs,
            )
        )
    if self_play_wrapper:
        callbacks.append(SelfPlayCallback(policy, self_play_wrapper))

    rollout_hyperparams = {
        **config.rollout_hyperparams,
        "subaction_mask": config.policy_hyperparams.get("subaction_mask", None),
    }
    if config.rollout_type:
        if config.rollout_type == "sync":
            rollout_generator_cls = SyncStepRolloutGenerator
        elif config.rollout_type == "reference":
            rollout_generator_cls = ReferenceAIRolloutGenerator
        elif config.rollout_type in {"guided", "guided_random"}:
            if config.rollout_type == "guided_random":
                rollout_generator_cls = RandomGuidedLearnerRolloutGenerator
            elif config.rollout_type == "guided":
                rollout_generator_cls = GuidedLearnerRolloutGenerator
            else:
                raise ValueError(f"{config.rollout_type} not recognized rollout_type")
            guide_policy_hyperparams = {
                **config.policy_hyperparams,
                **rollout_hyperparams.get("guide_policy", {}),
            }
            rollout_hyperparams["guide_policy"] = make_policy(
                config, env, device, **guide_policy_hyperparams
            )
        else:
            raise ValueError(f"{config.rollout_type} not recognized rollout_type")
    else:
        rollout_generator_cls = SyncStepRolloutGenerator

    rollout_generator = rollout_generator_cls(
        policy,
        env,
        **rollout_hyperparams,
    )
    algo.learn(config.n_timesteps, rollout_generator, callbacks=callbacks)

    policy.save(config.model_dir_path(best=False))

    eval_stats = eval_callback.evaluate(n_episodes=10, print_returns=True)

    plot_eval_callback(eval_callback, tb_writer, config.run_name())

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    if eval_callback.best:
        log_dict["best_eval"] = eval_callback.best._asdict()
    log_dict.update(asdict(hyperparams))
    log_dict.update(vars(args))
    with open(config.logs_path, "a") as f:
        yaml.dump({config.run_name(): log_dict}, f)

    best_eval_stats: EpisodesStats = eval_callback.best  # type: ignore
    tb_writer.add_hparams(
        hparam_dict(hyperparams, vars(args)),
        {
            "hparam/best_mean": best_eval_stats.score.mean,
            "hparam/best_result": best_eval_stats.score.mean
            - best_eval_stats.score.std,
            "hparam/last_mean": eval_stats.score.mean,
            "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
        },
        None,
        config.run_name(),
    )

    tb_writer.close()

    if wandb_enabled:
        shutil.make_archive(
            os.path.join(wandb.run.dir, config.model_dir_name()),  # type: ignore
            "zip",
            config.model_dir_path(),
        )
        wandb.finish()
