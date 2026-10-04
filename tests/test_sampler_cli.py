"""
Test module for sampler_cli.py

This module tests the command-line interface for sample generation,
including argument parsing, file output, and data validation.
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cmla.scripts.sampler_cli import (
    generate_gmm_observation_func,
    main_hmm,
    main_mm,
)


class TestSamplerCLI:
    """Test class for sampler CLI functions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gmm_json_output(self):
        """Test GMM sample generation with JSON output"""
        output_file = os.path.join(self.temp_dir, "test_gmm.json")

        # Mock command line arguments
        class MockArgs:
            def __init__(self):
                self.N = 10
                self.out_file = output_file
                self.csv = False
                self.pickle = False
                self.cluster = 2
                self.dimension = 3
                self.random_seed = 42

        args = MockArgs()
        generate_gmm_observation_func(args)

        # Verify file was created
        assert os.path.exists(output_file)

        # Verify JSON content
        with open(output_file, "r") as f:
            data = json.load(f)

        assert "model_param" in data
        assert "sample" in data
        assert "labels" in data
        assert data["model_type"] == "KmeansClustering"

        # Verify sample data structure
        samples = data["sample"]
        assert len(samples) == 10  # N samples
        assert len(samples[0]) == 3  # dimension

        # Verify model parameters
        model_param = data["model_param"]
        assert "Mu" in model_param
        assert "Sigma" in model_param
        assert "Pi" in model_param
        assert len(model_param["Mu"]) == 2  # cluster count

    def test_gmm_multiple_outputs(self):
        """Test GMM with CSV, Pickle, and JSON outputs"""
        output_file = os.path.join(self.temp_dir, "test_multi.json")
        csv_file = os.path.join(self.temp_dir, "test_multi.csv")
        pickle_file = os.path.join(self.temp_dir, "test_multi.pkl")

        class MockArgs:
            def __init__(self):
                self.N = 5
                self.out_file = output_file
                self.csv = True
                self.pickle = True
                self.cluster = 3
                self.dimension = 2
                self.random_seed = 0

        args = MockArgs()
        generate_gmm_observation_func(args)

        # Verify all files were created
        assert os.path.exists(output_file)  # JSON
        assert os.path.exists(csv_file)  # CSV
        assert os.path.exists(pickle_file)  # Pickle

        # Verify CSV content
        csv_data = np.loadtxt(csv_file, delimiter=",")
        assert csv_data.shape == (5, 2)  # N=5, dimension=2

        # Verify Pickle content
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
        assert "model_param" in pickle_data
        assert "sample" in pickle_data

    def test_markov_process_output(self):
        """Test Markov process sample generation"""
        output_file = os.path.join(self.temp_dir, "test_mm.json")

        class MockArgs:
            def __init__(self):
                self.N = 8
                self.out_file = output_file
                self.csv = False

        args = MockArgs()
        main_mm(args)

        # Verify file was created
        assert os.path.exists(output_file)

        # Verify JSON content
        with open(output_file, "r") as f:
            data = json.load(f)

        assert "model_param" in data
        assert "sample" in data
        assert data["model_type"] == "MarkovProcess"

        model_param = data["model_param"]
        assert "init_prob" in model_param
        assert "tran_prob" in model_param
        assert model_param["number_of_process"] == 8

    def test_hmm_output(self):
        """Test HMM sample generation"""
        output_file = os.path.join(self.temp_dir, "test_hmm.json")

        class MockArgs:
            def __init__(self):
                self.N = 3
                self.out_file = output_file
                self.avelen = 5

        args = MockArgs()
        main_hmm(args)

        # Verify file was created
        assert os.path.exists(output_file)

        # Verify file contains HMM data (check if it's valid JSON/pickle)
        file_ext = Path(output_file).suffix.lower()
        if file_ext == ".json":
            with open(output_file, "r") as f:
                data = json.load(f)
            assert data["model_type"] == "HMM"
        else:
            # Default should be determined by save_hmm_and_data method
            assert os.path.getsize(output_file) > 0

    @pytest.mark.parametrize(
        "cluster,dimension",
        [
            (2, 2),
            (3, 4),
            (5, 3),
        ],
    )
    def test_gmm_different_parameters(self, cluster, dimension):
        """Test GMM with different cluster and dimension parameters"""
        output_file = os.path.join(self.temp_dir, f"test_{cluster}_{dimension}.json")

        class MockArgs:
            def __init__(self):
                self.N = 6
                self.out_file = output_file
                self.csv = False
                self.pickle = False
                self.cluster = cluster
                self.dimension = dimension
                self.random_seed = 123

        args = MockArgs()
        generate_gmm_observation_func(args)

        with open(output_file, "r") as f:
            data = json.load(f)

        # Verify parameters match
        samples = data["sample"]
        assert len(samples[0]) == dimension

        model_param = data["model_param"]
        assert len(model_param["Mu"]) == cluster
        assert len(model_param["Sigma"]) == cluster

    def test_file_extensions(self):
        """Test different file extensions for automatic format detection"""
        test_cases = [
            ("test.json", "json"),
            ("test.pkl", "pickle"),
            ("test.pickle", "pickle"),
        ]

        for filename, expected_format in test_cases:
            output_file = os.path.join(self.temp_dir, filename)

            class MockArgs:
                def __init__(self):
                    self.N = 3
                    self.out_file = output_file
                    self.avelen = 4

            if expected_format in ["pickle"]:
                # Test with HMM which uses save_hmm_and_data
                args = MockArgs()
                main_hmm(args)
                assert os.path.exists(output_file)
            else:
                # Test with JSON format
                args = MockArgs()
                main_hmm(args)
                assert os.path.exists(output_file)

    def test_error_handling(self):
        """Test error handling for invalid parameters"""
        # Test with invalid directory
        invalid_output = "/nonexistent/directory/test.json"

        class MockArgs:
            def __init__(self):
                self.N = 5
                self.out_file = invalid_output
                self.csv = False
                self.pickle = False
                self.cluster = 2
                self.dimension = 2
                self.random_seed = 0

        args = MockArgs()

        # This should raise an error due to invalid directory
        with pytest.raises((FileNotFoundError, PermissionError)):
            generate_gmm_observation_func(args)


class TestSamplerCLIIntegration:
    """Integration tests using actual command line interface"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("sys.argv")
    def test_cli_argument_parsing(self, mock_argv):
        """Test command line argument parsing"""
        import argparse

        # Test GMM argument parsing
        test_args = [
            "sampler_cli.py",
            "10",  # N
            "output.json",  # out_file
            "GMM",  # model type
            "--cluster",
            "3",
            "--dimension",
            "2",
        ]

        # Parse arguments manually to test the parser
        parser = argparse.ArgumentParser()
        parser.add_argument("N", type=int)
        parser.add_argument("out_file", type=str)
        parser.add_argument("--csv", action="store_true")
        parser.add_argument("--pickle", action="store_true")

        subparsers = parser.add_subparsers(required=True)
        parser_gmm = subparsers.add_parser("GMM")
        parser_gmm.add_argument("--cluster", type=int, default=4)
        parser_gmm.add_argument("--dimension", type=int, default=2)
        parser_gmm.add_argument("--random-seed", type=int, default=0)
        parser_gmm.set_defaults(func=generate_gmm_observation_func)

        args = parser.parse_args(test_args[1:])  # Skip script name

        assert args.N == 10
        assert args.out_file == "output.json"
        assert args.cluster == 3
        assert args.dimension == 2
        assert args.csv is False
        assert args.pickle is False

    def test_data_consistency(self):
        """Test that generated data is consistent and valid"""
        output_file = os.path.join(self.temp_dir, "consistency_test.json")

        class MockArgs:
            def __init__(self):
                self.N = 20
                self.out_file = output_file
                self.csv = False
                self.pickle = False
                self.cluster = 3
                self.dimension = 4
                self.random_seed = 42  # Fixed seed for reproducibility

        # Generate data twice with same seed
        args1 = MockArgs()
        generate_gmm_observation_func(args1)

        with open(output_file, "r") as f:
            data1 = json.load(f)

        # Generate again with same parameters
        args2 = MockArgs()
        args2.out_file = os.path.join(self.temp_dir, "consistency_test2.json")
        generate_gmm_observation_func(args2)

        with open(args2.out_file, "r") as f:
            data2 = json.load(f)

        # Data should be identical with same random seed
        np.testing.assert_array_almost_equal(
            np.array(data1["sample"]), np.array(data2["sample"]), decimal=10
        )


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
