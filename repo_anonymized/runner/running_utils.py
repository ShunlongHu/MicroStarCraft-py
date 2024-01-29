import argparse
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Type, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import yaml
from gym.spaces import Box, Discrete
from torch.utils.tensorboard.writer import SummaryWriter

from repo_anonymized.a2c.a2c import A2C
from repo_anonymized.acbc.acbc import ACBC
from repo_anonymized.dqn.dqn import DQN
from repo_anonymized.dqn.policy import DQNPolicy
from repo_anonymized.ppo.ppo import PPO
from repo_anonymized.runner.config import Config, Hyperparams
from repo_anonymized.runner.wandb_load import load_player
from repo_anonymized.shared.algorithm import Algorithm
from repo_anonymized.shared.callbacks.eval_callback import EvalCallback
from repo_anonymized.shared.policy.actor_critic import ActorCritic
from repo_anonymized.shared.policy.policy import Policy
from repo_anonymized.shared.vec_env.utils import import_for_env_id, is_microrts
from repo_anonymized.vpg.policy import VPGActorCritic
from repo_anonymized.vpg.vpg import VanillaPolicyGradient
from repo_anonymized.wrappers.vectorable_wrapper import VecEnv, single_observation_space

ALGOS: Dict[str, Type[Algorithm]] = {
    "dqn": DQN,
    "vpg": VanillaPolicyGradient,
    "ppo": PPO,
    "a2c": A2C,
    "acbc": ACBC,
}
POLICIES: Dict[str, Type[Policy]] = {
    "dqn": DQNPolicy,
    "vpg": VPGActorCritic,
    "ppo": ActorCritic,
    "a2c": ActorCritic,
    "acbc": ActorCritic,
}

HYPERPARAMS_PATH = "hyperparams"


def base_parser(multiple: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default=["dqn"],
        type=str,
        choices=list(ALGOS.keys()),
        nargs="+" if multiple else 1,
        help="Abbreviation(s) of algorithm(s)",
    )
    parser.add_argument(
        "--env",
        default=["CartPole-v1"],
        type=str,
        nargs="+" if multiple else 1,
        help="Name of environment(s) in gym",
    )
    parser.add_argument(
        "--seed",
        default=[1],
        type=int,
        nargs="*" if multiple else "?",
        help="Seeds to run experiment. Unset will do one run with no set seed",
    )
    return parser


def load_hyperparams(algo: str, env_id: str) -> Hyperparams:
    hp = load_hyperparams_by_env_id(algo, env_id) or load_hyperparams_by_algo(
        algo, env_id
    )
    if hp is None:
        raise ValueError(f"({algo},{env_id}) hyperparameters not specified")
    return hp


def load_hyperparams_by_algo(algo: str, env_id: str) -> Optional[Hyperparams]:
    root_path = Path(__file__).parent.parent
    hyperparams_path = os.path.join(root_path, HYPERPARAMS_PATH, f"{algo}.yml")
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    if env_id in hyperparams_dict:
        return Hyperparams(**hyperparams_dict[env_id])

    import_for_env_id(env_id)
    spec = gym.spec(env_id)
    entry_point_name = str(spec.entry_point)  # type: ignore
    if "AtariEnv" in entry_point_name and "_atari" in hyperparams_dict:
        return Hyperparams(**hyperparams_dict["_atari"])
    elif "gym_microrts" in entry_point_name and "_microrts" in hyperparams_dict:
        return Hyperparams(**hyperparams_dict["_microrts"])

    return None


def load_hyperparams_by_env_id(algo: str, env_id: str) -> Optional[Hyperparams]:
    root_path = Path(__file__).parent.parent
    env_prefix = env_id.split("-")[0]
    hyperparams_path = os.path.join(
        root_path, HYPERPARAMS_PATH, f"{algo}-{env_prefix}.yml"
    )
    if not os.path.exists(hyperparams_path):
        return None
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    if env_id in hyperparams_dict:
        return Hyperparams(**hyperparams_dict[env_id])
    return None


def get_device(config: Config, env: VecEnv) -> torch.device:
    device = config.device
    # cuda by default
    if device == "auto":
        device = "cuda"
        # Apple MPS is a second choice (sometimes)
        if device == "cuda" and not torch.cuda.is_available():
            device = "mps"
        # If no MPS, fallback to cpu
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        # Simple environments like Discreet and 1-D Boxes might also be better
        # served with the CPU.
        if device == "mps":
            obs_space = single_observation_space(env)
            if isinstance(obs_space, Discrete):
                device = "cpu"
            elif isinstance(obs_space, Box) and len(obs_space.shape) == 1:
                device = "cpu"
            if is_microrts(config):
                device = "cpu"
    logging.info(f"Device: {device}")
    return torch.device(device)


def set_device_optimizations(
    device: torch.device, set_float32_matmul_precision: Optional[str] = None
) -> None:
    if device.type == "cuda":
        if set_float32_matmul_precision:
            logging.info(
                f"Setting torch.set_float32_matmul_precision to {set_float32_matmul_precision}"
            )
            torch.set_float32_matmul_precision(set_float32_matmul_precision)


def set_seeds(seed: Optional[int], use_deterministic_algorithms: bool) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Stop warning and it would introduce stochasticity if I was using TF
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def make_policy(
    config: Config,
    env: VecEnv,
    device: torch.device,
    load_path: Optional[str] = None,
    load_run_path: Optional[str] = None,
    load_run_path_best: bool = True,
    **kwargs,
) -> Policy:
    policy = POLICIES[config.algo](env, **kwargs).to(device)
    if not load_path and load_run_path:
        import wandb

        api = wandb.Api()
        _, _, load_path = load_player(
            api, load_run_path, config.args, config.root_dir, load_run_path_best
        )
        assert load_path
    if load_path:
        policy.load(load_path)
    return policy


def plot_eval_callback(callback: EvalCallback, tb_writer: SummaryWriter, run_name: str):
    figure = plt.figure()
    cumulative_steps = [
        (idx + 1) * callback.step_freq for idx in range(len(callback.stats))
    ]
    plt.plot(
        cumulative_steps,
        [s.score.mean for s in callback.stats],
        "b-",
        label="mean",
    )
    plt.plot(
        cumulative_steps,
        [s.score.mean - s.score.std for s in callback.stats],
        "g--",
        label="mean-std",
    )
    plt.fill_between(
        cumulative_steps,
        [s.score.min for s in callback.stats],  # type: ignore
        [s.score.max for s in callback.stats],  # type: ignore
        facecolor="cyan",
        label="range",
    )
    plt.xlabel("Steps")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Eval {run_name}")
    tb_writer.add_figure("eval", figure)


Scalar = Union[bool, str, float, int, None]


def hparam_dict(
    hyperparams: Hyperparams, args: Dict[str, Union[Scalar, list]]
) -> Dict[str, Scalar]:
    flattened = args.copy()
    for k, v in flattened.items():
        if isinstance(v, list):
            flattened[k] = json.dumps(v)
    for k, v in asdict(hyperparams).items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                key = f"{k}/{sk}"
                if isinstance(sv, dict) or isinstance(sv, list):
                    flattened[key] = str(sv)
                else:
                    flattened[key] = sv
        elif isinstance(v, list):
            flattened[k] = json.dumps(v)
        else:
            flattened[k] = v  # type: ignore
    return flattened  # type: ignore
