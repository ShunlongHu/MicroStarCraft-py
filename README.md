# repo_anonymized

_Note: Anonymized for blind review. Links are broken, and code is possibly broken. Final
version will link to public repo._

Implementations of reinforcement learning algorithms.

- Competitions
  - IEEE-CoG2023 MicroRTS
    competition at repo_anonymized/microrts:
    Technical details in repo_anonymized/microrts/technical-description.md.
  - Lux AI Season 2
- GitHub repo links WandB benchmark reports and Huggingface models for
  - Basic OpenAI gym envs, PyBullet, and Atari games
  - procgen (starpilot, hard)

## Prerequisites: Weights & Biases (WandB)

Training and benchmarking assumes you have a Weights & Biases project to upload runs to.
By default training goes to a repo-anonymized project while benchmarks go to
repo-anonymized-benchmarks. During training and benchmarking runs, videos of the best
models and the model weights are uploaded to WandB.

Before doing anything below, you'll need to create a wandb account and run `wandb
login`.

## Setup and Usage

### Lambda Labs instance for benchmarking

Benchmark runs are uploaded to WandB, which can be made into reports. So far I've found Lambda
Labs A10 instances to be a good balance of performance (14 hours to train PPO in 14
environments [5 basic gym, 4 PyBullet, CarRacing-v0, and 4 Atari] across 3 seeds) vs
cost ($0.60/hr).

```
git clone REPO_URL
cd repo-anonymized
# git checkout BRANCH_NAME if running on non-main branch
bash ./scripts/setup.sh
wandb login
bash ./scripts/benchmark.sh [-a {"ppo"}] [-e ENVS] [-j {6}] [-p {repo-anonymized-benchmarks}] [-s {"1 2 3"}]
```

Benchmarking runs are by default upload to a repo-anonymized-benchmarks project. Runs upload
videos of the running best model and the weights of the best and last model.
Benchmarking runs are tagged with a shorted commit hash (i.e., `benchmark_5598ebc`) and
hostname (i.e., `host_192-9-145-26`)

#### Publishing models to Huggingface

Publishing benchmarks to Huggingface requires logging into Huggingface with a
write-capable API token:

```
git config --global credential.helper store
huggingface-cli login
# --virtual-display likely must be specified if running on a remote machine.
python benchmark_publish.py --wandb-tags HOST_TAG COMMIT_TAG --wandb-report-url WANDB_REPORT_URL [--virtual-display]
```

#### Hyperparameter tuning with Optuna

Hyperparameter tuning can be done with the `tuning/tuning.sh` script, which runs
multiple processes of optimize.py. Start by doing all the setup meant for training
before running `tuning/tuning.sh`:

```
# Setup similar to training above
wandb login
bash scripts/tuning.sh -a ALGO -e ENV -j N_JOBS -s NUM_SEEDS
```

### macOS

#### Installation

My local development has been on an M1 Mac. These instructions might not be complete,
but these are the approximate setup and usage I've been using:

1. Install libraries with homebrew

```
brew install swig
brew install --cask xquartz
```

2. Download and install Miniconda for arm64

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

3. Create a conda environment from this repo's environment.yml

```
conda env create -f environment.yml -n repo_anonymized
conda activate repo_anonymized
```

4. Install other dependencies with poetry

```
poetry install
```

#### Usage

Training, benchmarking, and watching the agents playing the environments can be done
locally:

```
python train.py [-h] [--algo {ppo}] [--env ENV [ENV ...]] [--seed [SEED ...]] [--wandb-project-name WANDB_PROJECT_NAME] [--wandb-tags [WANDB_TAGS ...]] [--pool-size POOL_SIZE] [-virtual-display]
```

train.py by default uploads to the repo-anonymized WandB project. Training creates videos
of the running best model, which will cause popups. Creating the first video requires a
display, so you shouldn't shutoff the display until the video of the initial model is
created (1-5 minutes depending on environment). The --virtual-display flag should allow
headless mode, but that hasn't been reliable on macOS.

```
python enjoy.py [-h] [--algo {ppo}] [--env ENV] [--seed SEED] [--render RENDER] [--best BEST] [--n_episodes N_EPISODES] [--deterministic-eval DETERMINISTIC_EVAL] [--no-print-returns]
# OR
python enjoy.py [--wandb-run-path WANDB_RUN_PATH]
```

The first enjoy.py where you specify algo, env, and seed loads a model you locally
trained with those parameters and renders the agent playing the environment.

The second enjoy.py downloads the model and hyperparameters from a WandB run.

## Hyperparameters

These are specified in yaml files in the hyperparams directory by game (`atari` is a
special case for all Atari games).

## gym-microRTS Setup

```
python -m pip install -e '.[microrts]'
```

Requires Java SDK to also be installed.
