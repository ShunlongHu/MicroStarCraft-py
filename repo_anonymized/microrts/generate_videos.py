import os
from dataclasses import dataclass

from repo_anonymized.runner.evaluate import EvalArgs, evaluate_model


@dataclass
class VideoParams:
    wandb_run_path: str
    map_name: str
    video_fps: int = 150
    video_suffix: str = ""


MAP_NAME_TO_PATH = {
    "basesWorkers8x8A": "maps/8x8/basesWorkers8x8A.xml",
    "FourBasesWorkers8x8": "maps/8x8/FourBasesWorkers8x8.xml",
    "basesWorkers16x16A": "maps/16x16/basesWorkers16x16A.xml",
    "TwoBasesBarracks16x16": "maps/16x16/TwoBasesBarracks16x16.xml",
    "NoWhereToRun9x8": "maps/NoWhereToRun9x8.xml",
    "DoubleGame24x24": "maps/DoubleGame24x24.xml",
    "BWDistantResources32x32": "maps/BWDistantResources32x32.xml",
    "BloodBath": "maps/BroodWar/(4)BloodBath.scmB.xml",
}


VIDEOS = [
    VideoParams(
        "repo_anonymized-benchmarks/1ilo9yae",
        "basesWorkers8x8A",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/1ilo9yae",
        "FourBasesWorkers8x8",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/1ilo9yae",
        "basesWorkers16x16A",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/1ilo9yae",
        "TwoBasesBarracks16x16",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/vmns9sbe",
        "NoWhereToRun9x8",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/unnxtprk",
        "DoubleGame24x24",
        video_fps=30,
    ),
    VideoParams(
        "repo_anonymized-benchmarks/x4tg80vk",
        "BWDistantResources32x32",
        video_fps=30,
    ),
    VideoParams("repo_anonymized-benchmarks/nh5pdv4o", "BloodBath"),
    VideoParams(
        "repo_anonymized-2023/jl8zkpfr",
        "BWDistantResources32x32",
        video_fps=60,
        video_suffix="-squnet",
    ),
]

if __name__ == "__main__":
    for vp in VIDEOS:
        args = EvalArgs(
            algo="ppo",
            env="",
            render=False,
            n_episodes=1,
            wandb_run_path=vp.wandb_run_path,
            video_path=os.path.expanduser(
                f"~/Desktop/{vp.map_name}-AnonymizedAI-Mayari{vp.video_suffix}"
            ),
            override_hparams={
                "bots": {"mayari": 1},
                "map_paths": [MAP_NAME_TO_PATH[vp.map_name]],
                "video_frames_per_second": vp.video_fps,
            },
            thop=True,
        )
        evaluate_model(args, os.getcwd())
