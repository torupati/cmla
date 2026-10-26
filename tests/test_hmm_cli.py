"""Tests of hmm_cli, the sampler HMM-GMM subcommand and HMM plots."""

import json

import matplotlib
import numpy as np
import pytest

from cmla.models.emission import GMMEmission
from cmla.models.hmm import HMM, load_hmm_and_data
from cmla.scripts import hmm_cli
from cmla.scripts.sampler_cli import create_parser as create_sampler_parser

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """hmm_cli writes hmm_cli.log to the current directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def gmm_hmm_data(tmp_path):
    """GMM-HMM sample written by `sampler_cli N out.json --csv HMM-GMM`."""
    out_file = tmp_path / "gmm_hmm.json"
    args = create_sampler_parser().parse_args(
        [
            "15",
            str(out_file),
            "--csv",
            "HMM-GMM",
            "--states",
            "2",
            "--mixtures",
            "2",
            "--dimension",
            "3",
            "--avelen",
            "30",
            "--random-seed",
            "1",
        ]
    )
    args.func(args)
    return out_file


@pytest.fixture
def discrete_data(tmp_path):
    np.random.seed(0)
    hmm = HMM(2, 3)
    hmm.state_tran = np.array([[0.9, 0.1], [0.2, 0.8]])
    hmm.obs_prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    from cmla.models.sampler import sampling_from_hmm

    st, x = sampling_from_hmm([30] * 10, hmm)
    out_file = tmp_path / "discrete.json"
    hmm.save_hmm_and_data(str(out_file), x, st)
    return out_file


def test_sampler_hmm_gmm(gmm_hmm_data):
    hmm, x, st = load_hmm_and_data(str(gmm_hmm_data))
    assert isinstance(hmm.emission, GMMEmission)
    assert hmm.emission.means.shape == (2, 2, 3)
    assert len(x) == 15 and len(st) == 15
    assert all(np.asarray(seq).shape[1] == 3 for seq in x)
    assert [len(s) for s in st] == [len(seq) for seq in x]
    assert gmm_hmm_data.with_suffix(".csv").exists()


def test_train_gmm_from_csv(gmm_hmm_data, tmp_path, capsys):
    model_file = tmp_path / "model.json"
    hmm_cli.main(
        [
            "train",
            "--type",
            "gmm",
            "--states",
            "2",
            "--mixtures",
            "2",
            "--data-file",
            str(gmm_hmm_data.with_suffix(".csv")),
            "--iterations",
            "5",
            "--output",
            str(model_file),
        ]
    )
    hmm = HMM.load(str(model_file))
    assert isinstance(hmm.emission, GMMEmission)
    assert hmm.emission.means.shape == (2, 2, 3)
    assert "Model saved" in capsys.readouterr().out
    assert not (tmp_path / "models").exists()  # no checkpoints by default


def test_forward_and_viterbi_gmm(gmm_hmm_data, tmp_path):
    out_file = tmp_path / "forward.json"
    hmm_cli.main(
        [
            "forward",
            "-m",
            str(gmm_hmm_data),
            "-f",
            str(gmm_hmm_data),
            "-o",
            str(out_file),
        ]
    )
    result = json.loads(out_file.read_text())
    assert len(result["log_prob"]) == 15
    assert np.isclose(result["total"], np.sum(result["log_prob"]))

    out_file = tmp_path / "viterbi.json"
    hmm_cli.main(
        [
            "viterbi",
            "-m",
            str(gmm_hmm_data),
            "-f",
            str(gmm_hmm_data),
            "-o",
            str(out_file),
        ]
    )
    paths = [r["path"] for r in json.loads(out_file.read_text())]
    _, x, st = load_hmm_and_data(str(gmm_hmm_data))
    accuracy = np.mean(np.concatenate(paths) == np.concatenate(st))
    assert accuracy > 0.9  # true model decodes its own sample


def test_train_discrete_and_viterbi(discrete_data, tmp_path, capsys):
    model_file = tmp_path / "model.pkl"
    hmm_cli.main(["train", "-f", str(discrete_data), "-n", "10", "-o", str(model_file)])
    hmm = HMM.load(str(model_file))
    assert hmm.obs_prob.shape == (2, 3)

    hmm_cli.main(["viterbi", "-m", str(model_file), "-obs", "0 0 2 2"])
    assert "path=" in capsys.readouterr().out


def test_train_with_checkpoints(discrete_data, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    hmm_cli.main(
        [
            "train",
            "-f",
            str(discrete_data),
            "-n",
            "2",
            "--checkpoint-dir",
            str(ckpt_dir),
            "-o",
            str(tmp_path / "m.json"),
        ]
    )
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    assert len(ckpts) == 1
    HMM.load(str(ckpts[0]))


@pytest.mark.parametrize(
    "argv",
    [
        ["forward", "-m", "missing.json", "-obs", "0 1"],
        ["train", "--type", "gmm", "-obs", "0 1"],  # --observations is discrete only
        ["train"],  # no data
    ],
)
def test_errors_exit_1(argv, capsys):
    with pytest.raises(SystemExit) as e:
        hmm_cli.main(argv)
    assert e.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_discrete_model_rejects_bad_data(discrete_data, gmm_hmm_data, capsys):
    with pytest.raises(SystemExit):
        hmm_cli.main(["forward", "-m", str(discrete_data), "-obs", "0 5"])
    assert "must be in [0, 2]" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        hmm_cli.main(["forward", "-m", str(discrete_data), "-f", str(gmm_hmm_data)])
    assert "integer" in capsys.readouterr().err


def test_plots(gmm_hmm_data, discrete_data, tmp_path):
    from cmla.plots.hmm_plot import plot_checkpoint_dir, plot_emission

    hmm, x, st = load_hmm_and_data(str(gmm_hmm_data))
    fig = plot_emission(
        hmm, x=np.concatenate(x), states=np.concatenate(st), dims=(0, 2)
    )
    assert len(fig.axes[0].patches) == 4  # 2 states x 2 mixtures

    out_file = tmp_path / "discrete.png"
    plot_checkpoint_dir(str(discrete_data), str(out_file))
    assert out_file.exists()


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("baum-welch", "Training: Baum-Welch training (forward-backward algorithm), 2"),
        ("viterbi", "Training: Viterbi training (Viterbi algorithm), 2"),
    ],
)
def test_train_shows_algorithm(gmm_hmm_data, tmp_path, capsys, algorithm, expected):
    hmm_cli.main(
        [
            "train",
            "--type",
            "gmm",
            "--states",
            "2",
            "-f",
            str(gmm_hmm_data.with_suffix(".csv")),
            "--algorithm",
            algorithm,
            "-n",
            "2",
            "-o",
            str(tmp_path / "m.json"),
        ]
    )
    assert expected in capsys.readouterr().out


def test_negative_iterations(capsys):
    with pytest.raises(SystemExit):
        hmm_cli.main(["train", "-obs", "0 1", "-n", "-1"])
    assert ">= 0" in capsys.readouterr().err


def test_log_file(gmm_hmm_data, tmp_path, capsys):
    argv = [
        "train",
        "--type",
        "gmm",
        "--states",
        "2",
        "-f",
        str(gmm_hmm_data),
        "-n",
        "2",
    ]
    hmm_cli.main(argv + ["-o", str(tmp_path / "m.json")])
    out = capsys.readouterr()
    log = (tmp_path / "hmm_cli.log").read_text()
    assert "command: hmm_cli train" in log
    assert "cmla.models.hmm: iteration 1:" in log
    assert "cmla.models.hmm_init" in log  # file has all INFO messages
    assert "cmla.scripts.hmm_cli: Model saved to" in log
    assert "cmla.models.hmm_init" not in out.err  # terminal shows progress only
    assert "iteration 1:" in out.err

    # appended; -q affects the terminal only; custom path
    hmm_cli.main(argv + ["-o", str(tmp_path / "m.json"), "-q"])
    assert "iteration" not in capsys.readouterr().err
    assert (tmp_path / "hmm_cli.log").read_text().count("command: hmm_cli") == 2
    hmm_cli.main(
        [
            "forward",
            "-m",
            str(gmm_hmm_data),
            "-f",
            str(gmm_hmm_data),
            "--log-file",
            "x.log",
        ]
    )
    assert "command: hmm_cli forward" in (tmp_path / "x.log").read_text()


def test_log_file_records_error_and_can_be_disabled(tmp_path):
    with pytest.raises(SystemExit):
        hmm_cli.main(["forward", "-m", "missing.json", "-obs", "0"])
    assert "ERROR cmla.scripts.hmm_cli" in (tmp_path / "hmm_cli.log").read_text()
    (tmp_path / "hmm_cli.log").unlink()
    with pytest.raises(SystemExit):
        hmm_cli.main(["forward", "-m", "missing.json", "-obs", "0", "--log-file", ""])
    assert not (tmp_path / "hmm_cli.log").exists()
