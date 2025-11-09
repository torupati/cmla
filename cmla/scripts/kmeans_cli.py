#!/usr/bin/env python3
"""
K-means clustering CLI application.

Example usage:
    python scripts/kmeans_cli.py train --clusters 3 --data-file data.csv --output model.json
    python scripts/kmeans_cli.py predict --model model.json --data-file new_data.csv
    python scripts/kmeans_cli.py evaluate --model model.json --data-file test_data.csv
    python scripts/kmeans_cli.py --help
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cmla.models.kmeans import KmeansCluster, kmeans_clustering


def train_kmeans(args):
    """Train a K-means model and save it."""
    # Generate or load training data
    if args.random_data:
        print(f"Generating {args.samples} random data points...")
        data = np.random.randn(args.samples, 2) * 2
        data[: args.samples // 3] += [5, 5]  # Create clusters
        data[args.samples // 3 : 2 * args.samples // 3] += [-3, 2]
    elif args.data_file:
        if not args.data_file.exists():
            print(f"Error: Data file {args.data_file} not found", file=sys.stderr)
            sys.exit(1)
        data = np.loadtxt(args.data_file, delimiter=",")
    else:
        print("Error: Please specify --data-file or --random-data", file=sys.stderr)
        sys.exit(1)

    print(f"Training data shape: {data.shape}")
    print(f"Training K-means with {args.clusters} clusters...")

    # Run K-means clustering using the existing function
    try:
        # Initialize centroids randomly
        n_samples, n_features = data.shape
        initial_centroids = np.random.randn(args.clusters, n_features)

        # Train using the existing clustering function
        final_centroids, labels = kmeans_clustering(
            data, initial_centroids, max_iter=args.max_iter, tol=args.tolerance
        )

        print("Training completed!")
        print("Centroids:")
        for i, centroid in enumerate(final_centroids):
            print(f"  Cluster {i}: {centroid}")

        # Calculate inertia for model evaluation
        inertia = 0
        for i, centroid in enumerate(final_centroids):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroid) ** 2)

        print(f"Final inertia: {inertia:.4f}")

        # Save model if output specified
        if args.output:
            model_data = {
                "model_type": "KMeans",
                "n_clusters": args.clusters,
                "centroids": final_centroids.tolist(),
                "max_iter": args.max_iter,
                "tolerance": args.tolerance,
                "training_data_shape": list(data.shape),
                "training_labels": labels.tolist(),
                "inertia": inertia,
            }

            with open(args.output, "w") as f:
                json.dump(model_data, f, indent=2)
            print(f"Model saved to {args.output}")

        return final_centroids, data, labels

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


def predict_kmeans(args):
    """Load a trained model and predict on new data."""
    # Load model
    if not args.model.exists():
        print(f"Error: Model file {args.model} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.model, "r") as f:
        model_data = json.load(f)

    if model_data.get("model_type") != "KMeans":
        print("Error: Invalid model file format", file=sys.stderr)
        sys.exit(1)

    # Load prediction data
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} not found", file=sys.stderr)
        sys.exit(1)

    data = np.loadtxt(args.data_file, delimiter=",")
    print(f"Prediction data shape: {data.shape}")

    # Make predictions using centroids
    try:
        centroids = np.array(model_data["centroids"])
        n_clusters, n_features = centroids.shape

        # Create a KmeansCluster instance for prediction
        kmeans = KmeansCluster(num_clusters=n_clusters, D=n_features, trainable=False)
        kmeans.Mu = centroids

        # Predict labels
        labels = kmeans.predict(data)

        print("Prediction completed!")
        print(f"Predicted {len(np.unique(labels))} different clusters")

        # Save predictions if output specified
        if args.output:
            results = {
                "predictions": labels.tolist(),
                "data": data.tolist(),
                "model_file": str(args.model),
                "centroids_used": centroids.tolist(),
            }

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Predictions saved to {args.output}")
        else:
            print("Predicted labels:", labels[:10], "..." if len(labels) > 10 else "")

        return labels

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


def evaluate_kmeans(args):
    """Evaluate a trained model on test data."""
    # Load model
    if not args.model.exists():
        print(f"Error: Model file {args.model} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.model, "r") as f:
        model_data = json.load(f)

    # Load test data
    if not args.data_file.exists():
        print(f"Error: Data file {args.data_file} not found", file=sys.stderr)
        sys.exit(1)

    data = np.loadtxt(args.data_file, delimiter=",")
    print(f"Evaluation data shape: {data.shape}")

    # Evaluate model
    try:
        centroids = np.array(model_data["centroids"])
        n_clusters, n_features = centroids.shape

        # Create a KmeansCluster instance for evaluation
        kmeans = KmeansCluster(num_clusters=n_clusters, D=n_features, trainable=False)
        kmeans.Mu = centroids

        # Predict and evaluate
        labels = kmeans.predict(data)

        # Calculate inertia (within-cluster sum of squares)
        inertia = 0
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)

        print("Evaluation completed!")
        print(f"Number of clusters: {n_clusters}")
        print(f"Inertia (within-cluster sum of squares): {inertia:.4f}")
        print("Data points per cluster:")

        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} points")

        # Save evaluation results if output specified
        if args.output:
            results = {
                "model_file": str(args.model),
                "test_data_shape": list(data.shape),
                "inertia": inertia,
                "cluster_counts": {
                    int(label): int(count)
                    for label, count in zip(unique_labels, counts)
                },
                "predictions": labels.tolist(),
                "centroids": centroids.tolist(),
            }

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to {args.output}")

        return inertia, labels

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="K-means clustering analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        help="K-means operations",
        required=True,
        dest="command",
    )

    # Train subparser
    train_parser = subparsers.add_parser(
        "train",
        help="Train a K-means model",
        description="Train a K-means clustering model on provided data",
    )

    train_parser.add_argument(
        "--clusters", "-k", type=int, default=3, help="Number of clusters (default: 3)"
    )
    train_parser.add_argument(
        "--data-file", "-f", type=Path, help="Input training data file (CSV format)"
    )
    train_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file for trained model (JSON format)",
    )
    train_parser.add_argument(
        "--random-data",
        action="store_true",
        help="Generate random data for demonstration",
    )
    train_parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for random data (default: 100)",
    )
    train_parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum number of iterations (default: 300)",
    )
    train_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for convergence (default: 1e-4)",
    )
    train_parser.set_defaults(func=train_kmeans)

    # Predict subparser
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict using trained model",
        description="Use a trained K-means model to predict cluster labels for new data",
    )

    predict_parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Trained model file (JSON format)",
    )
    predict_parser.add_argument(
        "--data-file",
        "-f",
        type=Path,
        required=True,
        help="Input data file for prediction (CSV format)",
    )
    predict_parser.add_argument(
        "--output", "-o", type=Path, help="Output file for predictions (JSON format)"
    )
    predict_parser.set_defaults(func=predict_kmeans)

    # Evaluate subparser
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained model",
        description="Evaluate a trained K-means model on test data",
    )

    evaluate_parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Trained model file (JSON format)",
    )
    evaluate_parser.add_argument(
        "--data-file",
        "-f",
        type=Path,
        required=True,
        help="Test data file (CSV format)",
    )
    evaluate_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for evaluation results (JSON format)",
    )
    evaluate_parser.set_defaults(func=evaluate_kmeans)

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Execute the appropriate function based on the subcommand
    args.func(args)


if __name__ == "__main__":
    main()
