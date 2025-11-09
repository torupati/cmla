# Command line interface to generate sample data
# for Gaussian Mixture Model and Hidden Markov Model
# Usage example:
# python tools/sample_generator.py GMM 1000 out_gmm.pickle --cluster 4 --dimension 2
# python tools/sample_generator.py HMM 100 out_hmm.pickle --avelen 10
# python tools/sample_generator.py MM 100 out_mm.pickle
import json
import logging
import pickle

import numpy as np

from cmla.models.hmm import HMM
from cmla.models.sampler import (
    generate_gmm_samples,
    generate_sample_parameter,
    sample_lengths,
    sample_multiple_markov_process,
    sampling_from_hmm,
    save_sequences_with_blank,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    handlers=[logging.FileHandler("sample_generator.log"), logging.StreamHandler()],
    level=logging.INFO,
)


def generate_gmm_observation_func(args):
    """Generate observations from a Gaussian Mixture Model.

    Args:
        args: Command line arguments containing parameters for GMM generation.
    """
    _logger.info(f"{args=}")

    # Set random seed for reproducible results
    np.random.seed(args.random_seed)

    init_prob, mean_vectors, covariances = generate_sample_parameter(
        args.cluster, args.dimension
    )

    sample_data, labels = generate_gmm_samples(
        args.N, init_prob, mean_vectors, covariances
    )
    _logger.info(
        "generated sample size: n=%d d=%d", sample_data.shape[0], sample_data.shape[1]
    )

    if args.csv:
        csv_file = args.out_file.replace(".json", ".csv").replace(".pkl", ".csv")
        np.savetxt(csv_file, sample_data, delimiter=",")
        _logger.info("CSV output: %s", csv_file)

    if args.pickle:
        kmeans_param_dict = {
            "Mu": mean_vectors,
            "Sigma": covariances,
            "Pi": init_prob,
            "covariance_type": "diag",
            "trainvars": "outside",
            "dist_mode": "linear",
        }
        pickle_file = args.out_file.replace(".json", ".pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(
                {
                    "model_param": kmeans_param_dict,
                    "sample": sample_data,
                    "model_type": "KmeansClustering",
                },
                f,
            )
        _logger.info("Pickle output: %s", pickle_file)

    # Always save as JSON (default format)
    kmeans_param_dict = {
        "Mu": mean_vectors.tolist(),
        "Sigma": covariances.tolist(),
        "Pi": init_prob.tolist(),
        "covariance_type": "diag",
        "trainvars": "outside",
        "dist_mode": "linear",
    }
    with open(args.out_file, "w") as f:
        json.dump(
            {
                "model_param": kmeans_param_dict,
                "sample": sample_data.tolist(),
                "labels": labels,
                "model_type": "KmeansClustering",
            },
            f,
            indent=2,
        )
    _logger.info("JSON output: %s", args.out_file)


def main_mm(args):
    _logger.info(f"Markov Process: {args=}")

    init_state = np.array([0.1, 0.9])
    state_tran = np.array([[0.9, 0.1], [0.5, 0.5]])

    sample_data = sample_multiple_markov_process(args.N, init_state, state_tran)
    _logger.info("generated sample size: n=%d", len(sample_data))

    if args.csv:
        save_sequences_with_blank(args.out_file, sample_data)
        _logger.info("CSV output: %s", args.out_file)
    else:
        markov_model_dict = {
            "init_prob": init_state.tolist(),
            "tran_prob": state_tran.tolist(),
            "model_type": "MarkovProcess",
            "number_of_process": args.N,
        }

        # Save as JSON
        with open(args.out_file, "w") as f:
            json.dump(
                {
                    "model_param": markov_model_dict,
                    "sample": sample_data,  # sample_data is already a list
                    "model_type": "MarkovProcess",
                },
                f,
                indent=2,
            )
        _logger.info("JSON output: %s", args.out_file)


def main_hmm(args):
    _logger.info(f"{args=}")

    hmm_param = HMM(2, 5)
    hmm_param.init_state = np.array([0.1, 0.9])
    hmm_param.state_tran = np.array([[0.9, 0.1], [0.5, 0.5]])
    hmm_param.obs_prob = np.array(
        [[0.50, 0.20, 0.20, 0.10, 0.00], [0.00, 0.10, 0.40, 0.40, 0.10]]
    )
    x_lengths = sample_lengths(args.avelen, args.N)
    st, x = sampling_from_hmm(x_lengths, hmm_param)

    # Save HMM data using the model's built-in save method
    hmm_param.save_hmm_and_data(args.out_file, x, st)
    _logger.info(f"output: {args.out_file}")


def create_parser():
    """Create and return the argument parser"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("N", type=int, help="number of sample")
    parser.add_argument(
        "out_file", type=str, help="output file name", default="sample_data.json"
    )
    parser.add_argument(
        "--csv", action="store_true", help="output csv format. default: json format"
    )
    parser.add_argument(
        "--pickle",
        action="store_true",
        help="output pickle format. default: json format",
    )

    subparsers = parser.add_subparsers(
        title="model",
        description="probabilistic models(gmm,mm,hmm)",
        help="select one from GMM,HMM,MM",
        required=True,
    )
    # Markov Process
    parser_mm = subparsers.add_parser("MM", help="Markov models")
    # parser_mm.add_argument('bar', type=int, help='bar help')
    parser_mm.set_defaults(func=main_mm)

    # Gaussian Mixture Model
    parser_gmm = subparsers.add_parser("GMM", help="Gaussian Mixture models")
    parser_gmm.add_argument("--cluster", type=int, help="number of cluster", default=4)
    parser_gmm.add_argument("--dimension", type=int, help="vector dimension", default=2)
    parser_gmm.add_argument("--random-seed", type=int, help="random seed", default=0)
    # parser.add_argument('-v', '--verbose', action='store_true')
    parser_gmm.set_defaults(func=generate_gmm_observation_func)

    # Hidden Markov Model
    parser_hmm = subparsers.add_parser("HMM", help="Hidden markov models")
    parser_hmm.add_argument(
        "--avelen", type=int, help="average sample lengths", default=10
    )
    parser_hmm.set_defaults(func=main_hmm)

    return parser


def main():
    """Main entry point for the CLI"""
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
