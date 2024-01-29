# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from repo_anonymized.runner.evaluate import EvalArgs, evaluate_model
from repo_anonymized.runner.running_utils import base_parser


def enjoy() -> None:
    parser = base_parser(multiple=False)
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument(
        "--no-print-returns", action="store_true", help="Limit printing"
    )
    # wandb-run-path overrides base RunArgs
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.add_argument(
        "--video-path", type=str, help="Path to save video of all plays"
    )
    parser.add_argument("--override-hparams", default=None, type=str)
    parser.add_argument("--visualize-model-path", default=None, type=str)
    parser.add_argument(
        "--thop", action="store_true", help="Output MACs and num parameters"
    )
    args = parser.parse_args()
    args.algo = args.algo[0]
    args.env = args.env[0]
    args.seed = args.seed[0]
    args.override_hparams = (
        json.loads(args.override_hparams) if args.override_hparams else None
    )
    args = EvalArgs(**vars(args))

    evaluate_model(args, os.getcwd())


if __name__ == "__main__":
    enjoy()
