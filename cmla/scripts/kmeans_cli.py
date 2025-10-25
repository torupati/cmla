#!/usr/bin/env python3
"""
K-means clustering CLI application.

Example usage:
    python scripts/kmeans_cli.py --clusters 3 --data-file data.csv
    python scripts/kmeans_cli.py --help
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from cmla.models.kmeans import kmeans_clustering


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="K-means clustering analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--clusters", "-k", type=int, default=3, help="Number of clusters (default: 3)"
    )

    parser.add_argument(
        "--data-file", "-f", type=Path, help="Input data file (CSV format)"
    )

    parser.add_argument("--output", "-o", type=Path, help="Output file for results")

    parser.add_argument(
        "--random-data",
        action="store_true",
        help="Generate random data for demonstration",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for random data (default: 100)",
    )

    args = parser.parse_args()

    # Generate or load data
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
        parser.print_help()
        sys.exit(1)

    print(f"Data shape: {data.shape}")
    print(f"Running K-means with {args.clusters} clusters...")

    # Run K-means clustering
    try:
        centroids, labels = kmeans_clustering(data, args.clusters)

        print("Clustering completed!")
        print("Centroids:")
        for i, centroid in enumerate(centroids):
            print(f"  Cluster {i}: {centroid}")

        # Save results if output specified
        if args.output:
            results = {
                "centroids": centroids.tolist(),
                "labels": labels.tolist(),
                "data": data.tolist(),
            }
            import json

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error during clustering: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
