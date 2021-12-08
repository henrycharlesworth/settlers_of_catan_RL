import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', type=float, default=3e-4
    )
    parser.add_argument(
        '--use-linear-lr-decay', action='store_true', default=False
    )
    parser.add_argument(
        '--eps', type=float, default=1e-5, help="ADAM optimiser epsilon"
    )
    parser.add_argument(
        '--gamma', type=float, default=0.999, help="discount factor"
    )
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95
    )
    parser.add_argument(
        '--entropy-coef', type=float, default=0.05
    )
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5
    )
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5
    )
    parser.add_argument(
        '--seed', type=int, default=0
    )
    parser.add_argument(
        '--num-processes', type=int, default=128
    )
    parser.add_argument(
        '--num-envs-per-process', type=int, default=5
    )
    parser.add_argument(
        '--num-steps', type=int, default=200
    )
    parser.add_argument(
        '--truncated-seq-len', type=int, default=10
    )
    parser.add_argument(
        '--ppo-epoch', type=int, default=10
    )
    parser.add_argument(
        '--num-mini-batch', type=int, default=64
    )
    parser.add_argument(
        '--clip-param', type=float, default=0.2
    )
    parser.add_argument(
        '--total-env-steps', type=int, default=5e9
    )
    parser.add_argument(
        '--recompute-returns', action='store_true', default=False
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False
    )
    parser.add_argument(
        '--cuda-deterministic', action='store_true', default=False
    )
    parser.add_argument(
        '--num-policies-to-store', type=int, default=2000
    )
    parser.add_argument(
        '--add-policy-every', type=int, default=5, help='add new policy every this many updates'
    )
    parser.add_argument(
        '--update-opponent-policies-every', type=int, default=1
    )
    parser.add_argument(
        '--eval-every', type=int, default=25
    )
    parser.add_argument(
        '--num-eval-episodes', type=int, default=128
    )
    parser.add_argument(
        '--num-eval-processes', type=int, default=16
    )
    parser.add_argument(
        '--load-from-checkpoint', action='store_true', default=False
    )
    parser.add_argument(
        '--load-file-path', type=str, default=""
    )
    parser.add_argument(
        '--expt-id', type=str, default="default"
    ),
    parser.add_argument(
        '--update_timeout', type=int, default=15
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
