#!/usr/bin/env python3
"""
Hidden Markov Model (HMM) CLI application.

Example usage::

    # discrete HMM
    python -m cmla.scripts.hmm_cli train --data-file hmm.json --states 2 --output model.json
    python -m cmla.scripts.hmm_cli viterbi --model model.json --observations "0 1 0 1"

    # HMM with Gaussian mixture emission (GMM-HMM)
    python -m cmla.scripts.hmm_cli train --type gmm --states 3 --mixtures 2 --data-file sequences.csv --output gmm_hmm.json
    python -m cmla.scripts.hmm_cli forward --model gmm_hmm.json --data-file sequences.csv

Data file formats:
    .json/.pkl/.pickle  output of sampler_cli ("sample" holds the sequences)
    other (e.g. .csv)   one frame per line, comma separated; blank line separates sequences
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from cmla.models.emission import DiscreteEmission
from cmla.models.hmm import HMM, hmm_baum_welch, hmm_viterbi_training
from cmla.models.hmm_init import init_gmm_hmm
from cmla.models.sampler import load_sequences_with_blank


def load_sequences(args, observation_type: str) -> list[np.ndarray]:
    """Read observation sequences from --observations or --data-file.

    Args:
        args: parsed arguments
        observation_type (str): "discrete" or "gmm"

    Returns:
        list[np.ndarray]: (T,) int arrays for discrete, (T, D) float arrays for gmm
    """
    if args.observations:
        if observation_type != "discrete":
            raise ValueError("--observations is only for discrete HMM")
        return [np.array(args.observations.split(), dtype=int)]
    if args.data_file is None:
        raise ValueError("specify --data-file or --observations")

    suffix = args.data_file.suffix.lower()
    if suffix == ".json":
        with open(args.data_file, encoding="utf-8") as f:
            raw = json.load(f)["sample"]
    elif suffix in (".pkl", ".pickle"):
        with open(args.data_file, "rb") as f:
            raw = pickle.load(f)["sample"]
    else:
        raw = load_sequences_with_blank(str(args.data_file))

    if observation_type == "discrete":
        seqs = [np.asarray(x).reshape(-1) for x in raw]
        if any(not np.array_equal(x, np.round(x)) for x in seqs):
            raise ValueError("discrete HMM requires integer observations")
        return [x.astype(int) for x in seqs]
    seqs = [np.asarray(x, dtype=float) for x in raw]
    return [x.reshape(len(x), -1) for x in seqs]


def observation_type_of(hmm: HMM) -> str:
    return "discrete" if isinstance(hmm.emission, DiscreteEmission) else "gmm"


ALGORITHM_NAMES = {
    "baum-welch": "Baum-Welch training (forward-backward algorithm)",
    "viterbi": "Viterbi training (Viterbi algorithm)",
}


def train(args):
    """Train an HMM and save it."""
    if args.iterations < 0:
        raise ValueError("number of iterations must be >= 0")
    np.random.seed(args.seed)
    if args.model:
        hmm = HMM.load(str(args.model))
        seqs = load_sequences(args, observation_type_of(hmm))
        print(f"Loaded initial model from {args.model}")
    else:
        seqs = load_sequences(args, args.type)
        if args.type == "discrete":
            num_symbols = args.symbols or int(max(x.max() for x in seqs)) + 1
            hmm = HMM(args.states, num_symbols)
        else:
            feature_dim = seqs[0].shape[1]
            hmm = HMM(
                args.states,
                feature_dim,
                observation_type="gmm",
                num_mixtures=args.mixtures,
            )
            init_gmm_hmm(hmm, seqs, method=args.init)
    print(
        f"Model: {hmm.emission} ({len(seqs)} sequences, {sum(map(len, seqs))} frames)",
        flush=True,  # before log messages on stderr
    )

    print(
        f"Training: {ALGORITHM_NAMES[args.algorithm]}, {args.iterations} iterations",
        flush=True,
    )
    if args.algorithm == "baum-welch":
        hmm_baum_welch(
            hmm, seqs, itr_limit=args.iterations, checkpoint_dir=args.checkpoint_dir
        )
    else:
        hmm_viterbi_training(hmm, seqs, itr_limit=args.iterations)

    # per-iteration values are computed before each update; evaluate the final model
    total = sum(hmm.log_likelihood(x) for x in seqs)
    print(
        f"Training completed: {args.iterations} iterations, "
        f"E[log P(X)] = {total / len(seqs):.4f} per sequence, "
        f"{total / sum(map(len, seqs)):.4f} per frame"
    )

    hmm.save(str(args.output))
    print(f"Model saved to {args.output}")


def viterbi(args):
    """Most likely state sequence of each observation sequence."""
    hmm = HMM.load(str(args.model))
    seqs = load_sequences(args, observation_type_of(hmm))
    results = []
    for i, x in enumerate(seqs):
        path, log_prob = hmm.viterbi_search(x)
        path = [int(s) for s in path]
        results.append({"path": path, "log_prob": float(log_prob)})
        print(f"seq {i}: log P(X, S*)={log_prob:.4f} path={' '.join(map(str, path))}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


def forward(args):
    """Log-likelihood of each observation sequence."""
    hmm = HMM.load(str(args.model))
    seqs = load_sequences(args, observation_type_of(hmm))
    log_probs = [hmm.log_likelihood(x) for x in seqs]
    for i, (x, ll) in enumerate(zip(seqs, log_probs)):
        print(f"seq {i}: T={len(x)} log P(X)={ll:.4f}")
    total = float(np.sum(log_probs))
    print(f"total log P(X)={total:.4f}, per frame={total / sum(map(len, seqs)):.4f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"log_prob": log_probs, "total": total}, f, indent=2)
        print(f"Results saved to {args.output}")


def add_data_arguments(parser):
    parser.add_argument("--data-file", "-f", type=Path, help="Input data file")
    parser.add_argument(
        "--observations",
        "-obs",
        type=str,
        help="One discrete observation sequence (space-separated integers)",
    )


def create_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(
        description="Hidden Markov Model analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # logging options, accepted after any subcommand
    log_options = argparse.ArgumentParser(add_help=False)
    log_group = log_options.add_mutually_exclusive_group()
    log_group.add_argument(
        "--verbose", "-v", action="store_true", help="Show all log messages"
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Do not show log-likelihood of each training iteration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train HMM model", parents=[log_options]
    )
    add_data_arguments(train_parser)
    train_parser.add_argument(
        "--model", "-m", type=Path, help="Initial model (otherwise created from data)"
    )
    train_parser.add_argument(
        "--type",
        choices=["discrete", "gmm"],
        default="discrete",
        help="Emission type of a new model (default: discrete)",
    )
    train_parser.add_argument(
        "--states", "-s", type=int, default=2, help="Number of hidden states"
    )
    train_parser.add_argument(
        "--symbols",
        type=int,
        help="Number of observation symbols (discrete). Default: max symbol + 1",
    )
    train_parser.add_argument(
        "--mixtures", type=int, default=2, help="Number of Gaussians per state (gmm)"
    )
    train_parser.add_argument(
        "--init",
        choices=["kmeans", "uniform_segment"],
        default="kmeans",
        help="Initialization of a new GMM-HMM (default: kmeans)",
    )
    train_parser.add_argument(
        "--algorithm",
        choices=["baum-welch", "viterbi"],
        default="baum-welch",
        help="Training algorithm (default: baum-welch)",
    )
    train_parser.add_argument(
        "--iterations", "-n", type=int, default=20, help="Training iterations"
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=Path, help="Save Baum-Welch checkpoints here"
    )
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("trained_hmm_model.json"),
        help="Output model file (.json or .pkl)",
    )
    train_parser.set_defaults(func=train)

    for name, func, help_text in [
        ("viterbi", viterbi, "Most likely state sequence (Viterbi algorithm)"),
        ("forward", forward, "Log-likelihood of sequences (forward algorithm)"),
    ]:
        sub = subparsers.add_parser(name, help=help_text, parents=[log_options])
        sub.add_argument(
            "--model", "-m", type=Path, required=True, help="HMM model file"
        )
        add_data_arguments(sub)
        sub.add_argument("--output", "-o", type=Path, help="Output file (JSON)")
        sub.set_defaults(func=func)
    return parser


def main(argv=None):
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        if not args.quiet:  # training progress of HMM only
            logging.getLogger("cmla.models.hmm").setLevel(logging.INFO)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
