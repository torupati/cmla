#!/usr/bin/env python3
"""
Hidden Markov Model (HMM) CLI application.

Example usage:
    python scripts/hmm_cli.py --train --data-file observations.txt
    python scripts/hmm_cli.py --viterbi --observations "0 1 0 1"
    python scripts/hmm_cli.py --help
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cmla.models.hmm import HMM


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hidden Markov Model analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train HMM model")
    mode_group.add_argument(
        "--viterbi", action="store_true", help="Run Viterbi algorithm"
    )
    mode_group.add_argument(
        "--forward", action="store_true", help="Run forward algorithm"
    )

    # Common arguments
    parser.add_argument(
        "--model-file", "-m", type=Path, help="HMM model file (JSON format)"
    )

    parser.add_argument("--data-file", "-f", type=Path, help="Input data file")

    parser.add_argument(
        "--observations",
        "-obs",
        type=str,
        help="Observation sequence (space-separated integers)",
    )

    parser.add_argument(
        "--states",
        "-s",
        type=int,
        default=2,
        help="Number of hidden states (default: 2)",
    )

    parser.add_argument("--output", "-o", type=Path, help="Output file for results")

    args = parser.parse_args()

    # Load or create HMM model
    if args.model_file and args.model_file.exists():
        print(f"Loading HMM model from {args.model_file}")
        with open(args.model_file, "r") as f:
            model_data = json.load(f)
        hmm = HMM(
            num_states=len(model_data["transition_matrix"]),
            num_observations=len(model_data["observation_matrix"][0]),
        )
        hmm.transition_matrix = np.array(model_data["transition_matrix"])
        hmm.observation_matrix = np.array(model_data["observation_matrix"])
        hmm.initial_state_probability = np.array(
            model_data["initial_state_probability"]
        )
    else:
        print(f"Creating new HMM model with {args.states} states")
        # Create a simple HMM for demonstration
        hmm = HMM(num_states=args.states, num_observations=2)
        # Initialize with random parameters
        hmm.transition_matrix = np.random.rand(args.states, args.states)
        hmm.transition_matrix = hmm.transition_matrix / hmm.transition_matrix.sum(
            axis=1, keepdims=True
        )
        hmm.observation_matrix = np.random.rand(args.states, 2)
        hmm.observation_matrix = hmm.observation_matrix / hmm.observation_matrix.sum(
            axis=1, keepdims=True
        )
        hmm.initial_state_probability = np.ones(args.states) / args.states

    # Get observations
    if args.observations:
        observations = list(map(int, args.observations.split()))
    elif args.data_file and args.data_file.exists():
        observations = np.loadtxt(args.data_file, dtype=int).tolist()
    else:
        print("Error: Please specify --observations or --data-file", file=sys.stderr)
        sys.exit(1)

    print(f"Observations: {observations}")

    # Execute requested operation
    try:
        if args.train:
            print("Training HMM model...")
            hmm.train_baum_welch([observations], max_iterations=100)
            print("Training completed!")

            # Save trained model
            model_data = {
                "transition_matrix": hmm.transition_matrix.tolist(),
                "observation_matrix": hmm.observation_matrix.tolist(),
                "initial_state_probability": hmm.initial_state_probability.tolist(),
            }

            output_file = args.output or Path("trained_hmm_model.json")
            with open(output_file, "w") as f:
                json.dump(model_data, f, indent=2)
            print(f"Model saved to {output_file}")

        elif args.viterbi:
            print("Running Viterbi algorithm...")
            path, prob = hmm.viterbi(observations)
            print(f"Most likely state sequence: {path}")
            print(f"Probability: {prob}")

        elif args.forward:
            print("Running forward algorithm...")
            alpha, prob = hmm.forward(observations)
            print(f"Forward probability: {prob}")
            print(f"Alpha matrix shape: {alpha.shape}")

    except Exception as e:
        print(f"Error during operation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
