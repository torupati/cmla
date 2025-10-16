# Hidden Markov Models implementaton for learning purpose
# Reference: "Pattern Recognition and Machine Learning" by C. M. Bishop, Chapter 13

import pickle
from logging import getLogger
from os import makedirs, path
from typing import List

import numpy as np
from numpy import log, zeros

logger = getLogger(__name__)

eps = 1.0e-128  # to avoid log(0)


class HMM:
    """
    HMM parameter definition
    """

    def __init__(
        self,
        num_hidden_states: int,
        feature_dim: int,
        observation_type: str = "discrete",
    ):
        """Define a Hidden Markov Model (HMM) parameter.

        Args:
            num_hidden_states (int): Number of hidden state
            feature_dim (int): Category number (dimension) of observation
        """
        if num_hidden_states < 1:
            raise ValueError(f"num_hidden_states must be > 0. got {num_hidden_states}")
        self.init_state = np.array(
            [1 / num_hidden_states] * num_hidden_states
        )  # initial state probability Pr(s[t=0]==i)
        self.state_tran = np.ones((num_hidden_states, num_hidden_states)) * (
            1 / num_hidden_states
        )  # state transition probability, Pr(s[t+1]=j | s[t]=i)
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be > 0. got {feature_dim}")

        self.obs_prob = np.zeros(
            (num_hidden_states, feature_dim)
        )  # state emission probability, Pr(y|s[t]=i)
        for m in range(num_hidden_states):
            self.obs_prob[m, :] = np.random.uniform(0, 1, feature_dim)
            self.obs_prob[m, :] = self.obs_prob[m, :] / self.obs_prob[m, :].sum()

        # training variables (keep sufficient statistics for parameter update)
        self._ini_state_stat = np.zeros(num_hidden_states)
        self._state_tran_stat = np.zeros((num_hidden_states, num_hidden_states))
        self._obs_count = np.zeros((num_hidden_states, feature_dim))
        self._training_count = 0
        self._training_total_log_likelihood = 0.0

    def __repr__(self) -> str:
        return (
            f"HMM (#state={self.num_hidden_states}) \n"
            + " [initial state probability]\n"
            + f"{self.init_state.shape}\n"
            + " [transition probability]\n"
            + f"{self.state_tran.shape}\n"
            + "observation probability\n"
            + f" {self.obs_prob.shape}"
        )

    @property
    def num_hidden_states(self) -> int:
        """Number of hidden states. read-only.

        Returns:
            int: number of hidden states
        """
        return self.state_tran.shape[0]

    def randomize_state_transition_probabilities(self):
        _vals = np.random.uniform(size=self.num_hidden_states)
        self.init_state = _vals / sum(_vals)  # _vals > 0 is garanteered.

        self.state_tran = np.random.uniform(
            size=(self.num_hidden_states, self.num_hidden_states)
        )
        for m in range(self.num_hidden_states):
            self.state_tran[m, :] = self.state_tran[m, :] / sum(self.state_tran[m, :])
        assert np.allclose(self.state_tran.sum(axis=1), 1.0)
        assert np.all(self.state_tran >= 0.0)
        assert np.all(self.state_tran <= 1.0)
        assert np.allclose(self.init_state.sum(), 1.0)
        assert np.all(self.init_state >= 0.0)
        assert np.all(self.init_state <= 1.0)
        logger.debug(f"randomized state_tran=\n{self.state_tran}")
        logger.debug(f"randomized init_state=\n{self.init_state}")
        return

    def randomize_observation_probabilities(self):
        self.obs_prob = np.random.uniform(
            size=(self.num_hidden_states, self.obs_prob.shape[1])
        )
        for m in range(self.num_hidden_states):
            self.obs_prob[m, :] = self.obs_prob[m, :] / sum(self.obs_prob[m, :])
        assert np.allclose(self.obs_prob.sum(axis=1), 1.0)
        assert np.all(self.obs_prob >= 0.0)
        assert np.all(self.obs_prob <= 1.0)
        logger.debug(f"randomized obs_prob=\n{self.obs_prob}")
        return

    def viterbi_search(self, obss):
        """Viterbi search of discrete observation HMM. Likelihood is in log scale.

        - Finds the most-probable (Viterbi) path through the HMM states given observation.
        - Trellis (search space) is allocated in this method and release after the computation.

        Args:
            obss (List[int]): given observation sequence(descreat signal), y[t]

        Returns:
            best_path (List[int]): most probable state sequence, s[t]
            log_prob (float): log P(X|best_path) P(best_path)
        """

        T = len(obss)
        # Note that this implementation (keeping all observation of each state, each time step) is naieve because
        # it is not necessary to keep at the same time and memory exhasting.
        # (1) log P(x[t]|s[t]) is only required at time step t in viterbi search
        # (2) Probability can be stored in log scale in advance.
        _log_obsprob = np.zeros((T, self.num_hidden_states))
        for t in range(T):
            x_t = np.zeros(self.obs_prob.shape[1])
            x_t[obss[t]] = 1.0
            for s in range(self.num_hidden_states):
                _obs_prob = self.obs_prob[s, :]
                _obs_prob[_obs_prob < 1.0e-100] = 1.0e-100
                _log_obsprob[t, s] = np.dot(x_t, np.log(_obs_prob))

        _trellis_prob = np.ones((self.num_hidden_states, T), dtype=float) * np.log(
            eps
        )  # log scale
        _trellis_bp = np.zeros((self.num_hidden_states, T), dtype=int)

        _log_init_state = []
        for x in self.init_state:
            v = -10000.0
            if x > 1.0e-300:
                v = np.log(x)
            _log_init_state.append(v)
        _trellis_prob[:, 0] = _log_init_state + _log_obsprob[0, :]
        _trellis_bp[:, 0] = 0
        for t in range(1, T):  # for each time step t
            for j in range(self.num_hidden_states):  # for each state s[t-1]=i to s[t]=j
                # _probs[i] = P(x[1:t]|s[t-1]=i,s[t]=j)
                _probs = (
                    _trellis_prob[:, t - 1]
                    + log(self.state_tran[:, j] + eps)
                    + _log_obsprob[t, j]
                )
                # calculate path from each s[t-1]. _probs is array
                _trellis_bp[j, t] = _probs.argmax()
                _trellis_prob[j, t] = _probs[_trellis_bp[j, t]]
                # print(f'back pointer[t={t}, s={j}]={_trellis_bp[j,t]}')
            # print('t=', t)
            # print('trellis=', _trellis_prob[:,t])
        # print(_trellis_prob)
        # back traincing
        best_path: List[int] = []
        t, s = T - 1, _trellis_prob[:, -1].argmax()
        while t >= 0:
            # print("back trace t={} s={}".format(t,s))
            best_path.append(s)
            s = _trellis_bp[s, t]
            t = t - 1
        best_path.reverse()
        log_prob = _trellis_prob[:, -1].max()  # max log P(x[1:T]|s[1:T]|) P(s[1:T])
        return best_path, log_prob

    def forward_viterbi(self, obss: List[int]):
        """Push training sequence to get probability of latent varialble condition by input.

        Args:
            obss (List[int]): _description_
        Returns: Probability of latent state.
            gamma_1: g(t,s) = P(S[t]=s|X)
            gamma_1: g(t,s,s') = P(S[t]=s,S[t+1]=s'|X)
        """
        T = len(obss)
        best_path, log_likelihood = self.viterbi_search(obss)
        assert len(best_path) == len(obss)
        self._training_total_log_likelihood += log_likelihood

        _gamma1 = zeros([T, self.num_hidden_states])  # element is binary
        _gamma2 = zeros(
            [T - 1, self.num_hidden_states, self.num_hidden_states]
        )  # g(t, s, s') element is binary
        for t, s in enumerate(best_path):
            _gamma1[t, s] = 1.0  # gamma(t,s)=P(S[t]=s|X)
            if t < T - 1:
                _gamma2[t, s, best_path[t + 1]] = (
                    1.0  # gamma(t,s,s')=P(S[t]=s, s[t+1]=s'|X)
                )
        #
        return _gamma1, _gamma2, log_likelihood

    def calc_logobss(self, obss):
        """Calculate log observation probabilities.

        Args:
            obss (List[int]): observation sequence

        Returns:
            np.ndarray: (T, M)-shape array, log observation probabilities
        """
        T = len(obss)
        _log_obsprob = np.zeros((T, self.num_hidden_states))  # log P(x[t]|s[t]=i)
        for t in range(T):
            # create one-hot vector for observation.
            x_t = np.zeros(self.obs_prob.shape[1])
            x_t[obss[t]] = 1.0
            for s in range(self.num_hidden_states):
                _log_obsprob[t, s] = np.dot(x_t, np.log(self.obs_prob[s, :]))
        return _log_obsprob

    def forward_algorithm(self, obsprob) -> (np.ndarray, np.ndarray):
        """HMM forward algorithm
        Compute forward variable alpha and scaling factor alpha_scale
        in linear scale (not log scale).

        alpha[t,s] = P(s[t]|x[1],...,x[t])
        alpha_scale[t] = Sum_t s[t] P(x[1],...,x[t], s[t])

        Args:
            obsprob (np.ndarray): observation probabilities at time t, state s
        Returns:
            _alpha (np.ndarray): forward variable, shape (T,M)
            _alpha_scale (np.ndarray): scaling factor, shape (T,)
        """
        T, _M = obsprob.shape
        if obsprob.shape[1] != self.num_hidden_states:
            raise ValueError(
                f"obsprob.shape[1] ({obsprob.shape[1]}) must be same as num_hidden_states ({self.num_hidden_states})"
            )
        _alpha_scale = np.ones(T) * np.nan
        _alpha = np.zeros(
            (T, self.num_hidden_states), dtype=float
        )  # linear scale, not log scale
        # alpha[t=0,s] = pi[s] * b(x[t=0],s)
        _alpha[0, :] = self.init_state * obsprob[0, :]
        _alpha_scale[0] = _alpha[0, :].sum()
        _alpha[0, :] = _alpha[0, :] / _alpha_scale[0]
        for t in range(1, T):  # for each time step t = 1, ..., T-1
            for i in range(self.num_hidden_states):  # for each state s[t]=i
                # _alpha[t, i] = 0.0
                # for _i0 in range(self.num_hidden_states):
                #    _alpha[t, i] += (
                #        _alpha[t - 1, _i0] * self.state_tran[_i0, i] * obsprob[t, i]
                #    )
                _alpha[t, :] = (_alpha[t - 1, :] @ self.state_tran) * obsprob[t, :]
            _alpha_scale[t] = _alpha[t, :].sum()
            _alpha[t, :] = _alpha[t, :] / _alpha_scale[t]
            # P(s[t]=s | X[t]=x[t], S)
        return _alpha, _alpha_scale

    def calculate_prob(self, obss) -> np.ndarray:
        """Caluclate log probabilities of given observation

        b(t, m) = P(x[t] | State=m)

        Args:
            obss (_type_): _description_

        Returns:
            np.ndarray: (T, M)-shape array, log observation probabilities
        """
        T = len(obss)
        _obsprob = np.zeros((T, self.num_hidden_states))
        for t in range(T):
            for s in range(self.num_hidden_states):
                _obsprob[t, s] = self.obs_prob[s, obss[t]]
        return _obsprob

    def forward_backward_algorithm_linear(self, obss):
        """Push training sequence to get probability of latent varialble condition by input.

        Args:
            obss (List[int]): _description_
        Returns: Probability of latent state.
            gamma_1: g(t,s) = P(S[t]=s|X)
            gamma_1: g(t,s,s') = P(S[t]=s,S[t+1]=s'|X)
        """
        T = len(obss)
        # _obsprob = np.exp(self.calc_logobss(obss))
        _obsprob = self.calculate_prob(obss)
        _alpha, _alpha_scale = self.forward_algorithm(_obsprob)

        _log_prob = 0.0
        for t in range(T):
            _log_prob += np.log(_alpha_scale[t])  # sum_s log P(x[1:T],s[T]=s)
        self._training_total_log_likelihood += _log_prob
        # print('alpha=', _alpha)

        _beta = (
            np.ones((T, self.num_hidden_states)) * np.nan
        )  # linear scale, not log scale
        _beta[T - 1, :] = np.ones(self.num_hidden_states)
        _g1 = np.ones((T, self.num_hidden_states)) * np.nan
        _g2 = np.ones((T - 1, self.num_hidden_states, self.num_hidden_states)) * np.nan
        _g1[T - 1, :] = _alpha[T - 1, :] * _beta[T - 1, :]
        # print('gamma=', _g1[T-1,:])
        # print(f'sum(gamma[t={T-1}])=', _g1[T-1,:].sum(), ' (last)')
        for t in range(T - 1, 0, -1):  # for each time step t = T-1, ..., 0
            # P(s[t-1],s[t]=s,X[t:T]=x[t:T])
            # for i in range(self.num_hidden_states):
            #    _beta[t - 1, i] = 0.0
            #    for _j in range(self.num_hidden_states):
            #        _beta[t - 1, i] += (
            #            self.state_tran[i, _j] * _obsprob[t, _j] * _beta[t, _j]
            #        )
            _beta[t - 1, :] = self.state_tran @ (_obsprob[t, :] * _beta[t, :])
            _beta[t - 1, :] = _beta[t - 1, :] / _alpha_scale[t]

            # merge forward probability and backward probability
            _g1[t - 1, :] = _alpha[t - 1, :] * _beta[t - 1, :]
            assert (_g1[t - 1, :].sum() - 1.0) < 1.0e-9
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):  # transition s[t-1] to s[t]
                    _g2[t - 1, i, j] = (
                        _alpha[t - 1, i]
                        * self.state_tran[i, j]
                        * _obsprob[t, j]
                        * _beta[t, j]
                    )
            _g2[t - 1, :, :] = _g2[t - 1, :, :] / _alpha_scale[t]
            # print('value=', (np.dot(_alpha[t-1,:], self.state_tran) * _obsprob[t-1,:]).shape)
            # print('gzi=', _g2[t-1,:,:])
            # print(f'sum(gzai[t={t-1}])', _g2[t-1,:,:].sum(axis=1))
            # input()
            for i in range(self.num_hidden_states):
                assert (_g1[t - 1, i] - _g2[t - 1, i, :].sum()) < 1.0e-06
        return _g1, _g2, _log_prob

    def push_sufficient_statistics(
        self, obss: np.ndarray, g1: np.ndarray, g2: np.ndarray
    ) -> int:
        """Update suffience statistics for parameters by given observation and latent state probability.
        This function is used in both Viterbi traning and Baum-Welch algorithm.

        Args:
            obss (numpy.ndarray): A shape-(T,D) array, observation X given
            g1 (numpy.ndarray): A shape-(T, M) array, gamma(t, s, s') = P(S[t]=s, S[t+1]=s'|X)
            g2 (numpy.ndarray): A shape-(T, M, M) array, gamma(t, s) = P(S[t]=s|X)
        """
        T = len(obss)
        self._ini_state_stat = self._ini_state_stat + g1[0]
        for t in range(T - 1):
            self._state_tran_stat = self._state_tran_stat + g2[t, :, :]
        for t in range(T):
            # make one-hot vector for observation
            o_t = np.zeros(self.obs_prob.shape[1])
            o_t[obss[t]] = 1
            for _m in range(self.num_hidden_states):
                self._obs_count[_m, :] = self._obs_count[_m, :] + g1[t, _m] * o_t
        self._training_count += 1

    def update_parameters(self):
        """Update HMM parameters by sufficiency statistics.
        Normalize sufficiency statistics and update parameters.
        After update, sufficiency statistics are reset to zero.

        Returns:
            float: total log likelihood of training data
        """
        _init_state = self._ini_state_stat / sum(self._ini_state_stat)
        _init_state[_init_state < eps] = (
            eps  # if probability is lower then eps, set eps to void log(0)
        )
        self.init_state = _init_state
        for m in range(self.num_hidden_states):  # normalize each state
            if sum(self._state_tran_stat[m, :]) > 0.0:
                self.state_tran[m, :] = self._state_tran_stat[m, :] / sum(
                    self._state_tran_stat[m, :]
                )
            if sum(self._obs_count[m, :]) > 0.0:
                self.obs_prob[m, :] = self._obs_count[m, :] / sum(self._obs_count[m, :])

        # reset training variables
        self._ini_state_stat = np.zeros(self.num_hidden_states)
        self._state_tran_stat = np.zeros(
            (self.num_hidden_states, self.num_hidden_states)
        )
        self._obs_count = np.zeros(
            (self.num_hidden_states, self.obs_prob.shape[1])
        )  # descrete observation
        self._training_count = 0

        tll = self._training_total_log_likelihood
        self._training_total_log_likelihood = 0.0
        return tll


def hmm_viterbi_training(hmm, obss_seqs, itr_limit: int = 10) -> dict:
    """
    HMM training using Viterbi training algorithm.

    Args:
        hmm (HMM): HMM parameter
        obss_seqs (list[np.ndarray]): observation sequences
        itr_limit (int): maximum iteration number

    Returns:
        dict: training history
    """
    itr_count = 0
    training_history = {"step": [], "log_likelihood": []}
    prev_likelihood = np.nan
    while itr_count < itr_limit:
        for x in obss_seqs:
            g1, g2, ll = hmm.forward_viterbi(x)
            hmm.push_sufficient_statistics(x, g1, g2)
        total_likelihood = hmm.update_parameters()
        logger.info(
            "itr {} E[logP(X)]={}".format(itr_count, total_likelihood / len(obss_seqs))
        )
        training_history["step"].append(itr_count)
        training_history["log_likelihood"].append(total_likelihood / len(obss_seqs))

        if itr_count > 0:
            assert prev_likelihood <= total_likelihood
        prev_likelihood = total_likelihood
        itr_count += 1
    return training_history


def hmm_baum_welch(hmm, obss_seqs, itr_limit: int = 100) -> dict:
    """HMM training using EM algorithm.

    Args:
        hmm (HMM): HMM parameter
        obss_seqs (list[np.ndarray]): observation sequences
        itr_limit (int): maximum iteration number
    Returns:
        dict: training history
    """
    itr_count = 0
    _save_model = True
    outdir = "models/checkpoints/"
    if _save_model:
        makedirs(outdir, exist_ok=True)
    ll_history = {
        "step": [],
        "log_likelihood": [],
        "total_obs_num": [],
        "total_seq_num": [],
    }
    prev_likelihood = np.nan
    while itr_count < itr_limit:
        total_obs_num = 0
        for x in obss_seqs:
            _gamma, _xi, tll = hmm.forward_backward_algorithm_linear(x)
            hmm.push_sufficient_statistics(x, _gamma, _xi)
            total_obs_num += len(x)
        total_likelihood = hmm.update_parameters()
        print(
            "itr {} E[logP(X)]={}".format(itr_count, total_likelihood / len(obss_seqs))
        )
        ll_history["step"].append(itr_count)
        ll_history["log_likelihood"].append(total_likelihood)
        ll_history["total_obs_num"].append(total_obs_num)
        ll_history["total_seq_num"].append(len(obss_seqs))
        # save model
        if _save_model and itr_count % 30 == 0:
            ckpt_file = path.join(outdir, f"hmm_checkpoint_{itr_count:06d}.ckpt")
            with open(ckpt_file, "wb") as f:
                # todo: save model as dict
                hmm_param_dict = {
                    "init_state": hmm.init_state,
                    "state_tran": hmm.state_tran,
                    "obs_prob": hmm.obs_prob,
                    "n_state": hmm.num_hidden_states,
                    "n_obs": hmm.obs_prob.shape[1],
                }
                pickle.dump(
                    {
                        "model": hmm_param_dict,
                        "model_type": "HMM",
                        "total_likelihood": total_likelihood,
                        "total_sequence_num": len(obss_seqs),
                        "total_obs_num": total_obs_num,
                        "iteration": itr_count,
                    },
                    f,
                )
                print(ckpt_file)

        # print('------ after Baum welch trianing ------')
        if itr_count > 0:
            assert prev_likelihood <= total_likelihood
        prev_likelihood = total_likelihood
        itr_count += 1
    return ll_history


def pickle_hmm_and_data_by_dict(out_file: str, hmm: HMM, x: np.ndarray, st: np.ndarray):
    """Save HMM model and data to pickle file.
    Args:
        out_file (str): output file name
        hmm (HMM): HMM model
        x (np.ndarray): observation sequence
        st (np.ndarray): latent state sequence
    """
    hmm_param_dict = {
        "init_state": hmm.init_state,
        "state_tran": hmm.state_tran,
        "obs_prob": hmm.obs_prob,
        "n_state": hmm.M,
        "n_obs": hmm.D,
    }
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "model_param": hmm_param_dict,
                "sample": x,
                "latent": st,
                "model_type": "HMM",
            },
            f,
        )


def load_hmm_and_data_from_pickle(in_file: str):
    """Load HMM model and data from pickle file.
    Args:
        in_file (str): input file name
    Returns:
        hmm (HMM): HMM model
        x (np.ndarray): observation sequence
        st (np.ndarray): latent state sequence
    """
    with open(in_file, "rb") as f:
        data = pickle.load(f)
        model_param = data.get("model_param", None)
        if model_param is None:
            raise ValueError(f"model_param not found in {in_file}")
        n_state = model_param.get("n_state", None)
        n_obs = model_param.get("n_obs", None)
        if n_state is None or n_obs is None:
            raise ValueError(f"n_state or n_obs not found in model_param of {in_file}")
        hmm = HMM(n_state, n_obs)
        hmm.init_state = model_param.get("init_state", hmm.init_state)
        hmm.state_tran = model_param.get("state_tran", hmm.state_tran)
        hmm.obs_prob = model_param.get("obs_prob", hmm.obs_prob)
        x = data.get("sample", None)
        st = data.get("latent", None)
        return hmm, x, st
