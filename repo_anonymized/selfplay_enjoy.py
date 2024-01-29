# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from repo_anonymized.runner.running_utils import base_parser
from repo_anonymized.runner.selfplay_evaluate import SelfplayEvalArgs, selfplay_evaluate


def selfplay_enjoy() -> None:
    parser = base_parser(multiple=False)
    parser.add_argument(
        "--wandb-run-paths",
        type=str,
        nargs="*",
        help="WandB run paths to load players from. Must be 0 or 2",
    )
    parser.add_argument(
        "--model-file-paths",
        type=str,
        help="File paths to load players from. Must be 0 or 2",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n-envs", default=1, type=int)
    parser.add_argument("--n-episodes", default=1, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument(
        "--no-print-returns", action="store_true", help="Limit printing"
    )
    parser.add_argument(
        "--video-path", type=str, help="Path to save video of all plays"
    )
    args = parser.parse_args()
    args.algo = args.algo[0]
    args.env = args.env[0]
    args.seed = args.seed[0]
    args = SelfplayEvalArgs(**vars(args))

    selfplay_evaluate(args, os.getcwd())


if __name__ == "__main__":
    selfplay_enjoy()
