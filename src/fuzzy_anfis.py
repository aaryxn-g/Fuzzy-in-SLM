"""
fuzzy_anfis.py
==============
A small, self-contained ANFIS (Adaptive Neuro-Fuzzy Inference System) written in
plain NumPy. No deep-learning library is needed, so every step can be read and
explained line by line.

What this is
------------
ANFIS is a Takagi-Sugeno fuzzy system whose parameters are trained by gradient
descent, exactly like a small neural network. That is why it is called
"neuro-fuzzy": the fuzzy rules stay human-readable, but the membership functions
and rule outputs are learned from data.

Our use
-------
Inputs  : the model-confidence signals extracted from the small language model
          (entropy, top-1 margin, attention spread, layer instability).
Output  : a single confidence score in [0, 1] = the fuzzy system's estimate of
          "is this answer correct?".
Target  : the 0/1 correctness label of each answer.

Five classic ANFIS layers
--------------------------
L1 Fuzzify   : each input gets Gaussian membership functions (e.g. low / high).
L2 Rule      : firing strength of a rule = product of its membership values.
L3 Normalise : divide each firing strength by the total.
L4 Consequent: each rule has a (linear) output, weighted by its normalised firing.
L5 Output    : sum of the weighted rule outputs, squashed to [0,1] by a sigmoid.

Implementation note: the forward pass is in `ANFIS.forward`, the learned
parameters are the MF centres `c`, MF widths `s`, and the consequents `P`, `b`.
Gradients are derived by hand in `ANFIS._backward` and checked numerically in
`__main__`.
"""

import itertools
import numpy as np


def _sigmoid(z):
    # Numerically stable logistic function.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class ANFIS:
    """First-order Takagi-Sugeno ANFIS with Gaussian membership functions.

    Parameters
    ----------
    n_inputs : int
        Number of input signals (J).
    n_mf : int
        Membership functions per input (M). With M=2 you get "low"/"high"
        and M**J rules. Keep this small so the rule base stays readable.
    first_order : bool
        True  -> rule output is linear in the inputs (classic Jang ANFIS).
        False -> rule output is a single constant (zero-order / simplest to read).
    l2 : float
        Small weight decay on the consequent parameters for stability.
    seed : int
        Random seed for reproducible initialisation.
    """

    def __init__(self, n_inputs=4, n_mf=2, first_order=True, l2=1e-4, seed=0):
        self.J = n_inputs
        self.M = n_mf
        self.first_order = first_order
        self.l2 = l2
        rng = np.random.default_rng(seed)

        # Layer 1 parameters: Gaussian centres and widths, one set per input.
        # Centres spread across the standardised input range [-1.5, 1.5].
        self.c = np.tile(np.linspace(-1.5, 1.5, self.M), (self.J, 1))      # (J, M)
        self.c = self.c + 0.01 * rng.standard_normal(self.c.shape)
        self.s = np.full((self.J, self.M), 1.2)                            # (J, M)

        # Enumerate rules: every combination of one MF index per input.
        # sel[r, j] = which MF of input j rule r uses.
        self.sel = np.array(list(itertools.product(range(self.M), repeat=self.J)))
        self.R = self.sel.shape[0]                                         # R = M**J

        # Layer 4 parameters: consequent of each rule.
        # f_r(x) = x . P[r] + b[r].  Zero-order fixes P = 0 and learns only b.
        self.P = 0.01 * rng.standard_normal((self.R, self.J))              # (R, J)
        self.b = np.zeros(self.R)                                          # (R,)

        self._cache = None

    # ---------------------------------------------------------------- forward
    def forward(self, X):
        """Run the five ANFIS layers. X is (N, J) standardised. Returns p (N,)."""
        N = X.shape[0]

        # L1: membership of each input under each MF -> mu (N, J, M)
        # mu[n, j, m] = exp(-0.5 * ((x_nj - c_jm) / s_jm)^2)
        u = (X[:, :, None] - self.c[None, :, :]) / self.s[None, :, :]      # (N, J, M)
        mu = np.exp(-0.5 * u ** 2)                                         # (N, J, M)

        # Gather the membership each rule needs: mu_sel (N, R, J)
        # For input j, rule r selects MF sel[r, j].
        n_idx = np.arange(N)[:, None, None]
        j_idx = np.arange(self.J)[None, None, :]
        r_sel = self.sel[None, :, :]                                       # (1, R, J)
        mu_sel = mu[n_idx, j_idx, r_sel]                                   # (N, R, J)

        # L2: firing strength = product over inputs -> w (N, R)
        w = np.prod(mu_sel, axis=2)                                        # (N, R)

        # L3: normalise firing strengths -> wn (N, R)
        S = w.sum(axis=1, keepdims=True) + 1e-12
        wn = w / S                                                         # (N, R)

        # L4: rule consequents f_r(x) -> f (N, R)
        if self.first_order:
            f = X @ self.P.T + self.b[None, :]                            # (N, R)
        else:
            f = np.broadcast_to(self.b[None, :], (N, self.R))            # (N, R)

        # L5: aggregate and squash to a confidence in [0, 1]
        z = np.sum(wn * f, axis=1)                                         # (N,)
        p = _sigmoid(z)                                                    # (N,)

        self._cache = dict(X=X, mu=mu, mu_sel=mu_sel, w=w, S=S, wn=wn,
                           f=f, z=z, p=p)
        return p

    # --------------------------------------------------------------- backward
    def _backward(self, y):
        """Hand-derived gradients of mean BCE loss w.r.t. all parameters."""
        cache = self._cache
        X, mu = cache["X"], cache["mu"]
        w, S, wn, f, p = cache["w"], cache["S"], cache["wn"], cache["f"], cache["p"]
        N = X.shape[0]

        # Loss = mean binary cross-entropy. d(BCE)/dz = (p - y)/N.
        dz = (p - y) / N                                                   # (N,)

        # z = sum_r wn_r f_r
        dwn = dz[:, None] * f                                              # (N, R)
        df = dz[:, None] * wn                                              # (N, R)

        # Consequents: f_r = x.P_r + b_r
        gP = df.T @ X + self.l2 * self.P if self.first_order else np.zeros_like(self.P)
        gb = df.sum(axis=0)                                                # (R,)

        # Normalisation: wn_r = w_r / S, S = sum_k w_k
        # dL/dw_k = (1/S) * (dwn_k - sum_r dwn_r * wn_r)
        inner = np.sum(dwn * wn, axis=1, keepdims=True)                    # (N, 1)
        dw = (dwn - inner) / S                                             # (N, R)

        # Firing: w_r = prod_j mu_sel[:, r, j].  dL/dmu_sel = dw_r * w_r / mu_sel
        mu_sel = cache["mu_sel"]
        dmu_sel = dw[:, :, None] * (w[:, :, None] / (mu_sel + 1e-12))      # (N, R, J)

        # Scatter dmu_sel back onto mu (N, J, M), then onto Gaussian params.
        gc = np.zeros_like(self.c)
        gs = np.zeros_like(self.s)
        for j in range(self.J):
            x_j = X[:, j][:, None]                                         # (N, 1)
            for m in range(self.M):
                mask = (self.sel[:, j] == m)                              # rules using MF m on input j
                if not mask.any():
                    continue
                # Sum the incoming gradient for mu[:, j, m] across those rules.
                g_mu = dmu_sel[:, mask, j].sum(axis=1)[:, None]           # (N, 1)
                c_jm = self.c[j, m]
                s_jm = self.s[j, m]
                mu_jm = mu[:, j, m][:, None]                               # (N, 1)
                uu = (x_j - c_jm) / s_jm
                # dmu/dc = mu * u / s ; dmu/ds = mu * u^2 / s
                gc[j, m] += np.sum(g_mu * mu_jm * uu / s_jm)
                gs[j, m] += np.sum(g_mu * mu_jm * uu ** 2 / s_jm)

        return dict(c=gc, s=gs, P=gP, b=gb)

    # ------------------------------------------------------------------- loss
    @staticmethod
    def bce(p, y):
        eps = 1e-9
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # ------------------------------------------------------------------ train
    def fit(self, X, y, epochs=800, lr=0.05, X_val=None, y_val=None,
            patience=80, eval_every=10, verbose=False):
        """Full-batch Adam with optional early stopping.

        X standardised (N, J), y in {0,1}. If X_val/y_val are given, the
        validation loss is tracked and the parameters from the best validation
        epoch are kept (this is what stops the fuzzy gate from overfitting the
        training set and ruining its calibration on new data).
        """
        params = ["c", "s", "P", "b"]
        m = {k: np.zeros_like(getattr(self, k)) for k in params}
        v = {k: np.zeros_like(getattr(self, k)) for k in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        use_val = X_val is not None and y_val is not None
        best = {"val": np.inf, "epoch": 0,
                "params": {k: getattr(self, k).copy() for k in params}}
        since_improve = 0

        for t in range(1, epochs + 1):
            p = self.forward(X)
            loss = self.bce(p, y) + 0.5 * self.l2 * np.sum(self.P ** 2)
            grads = self._backward(y)
            for k in params:
                g = grads[k]
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g ** 2
                mhat = m[k] / (1 - b1 ** t)
                vhat = v[k] / (1 - b2 ** t)
                setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))
            # Keep widths positive and sane.
            self.s = np.clip(self.s, 0.2, 5.0)
            history.append(loss)

            if use_val and (t % eval_every == 0):
                vb = self.bce(self.predict_proba(X_val), y_val)
                if vb < best["val"] - 1e-5:
                    best = {"val": vb, "epoch": t,
                            "params": {k: getattr(self, k).copy() for k in params}}
                    since_improve = 0
                else:
                    since_improve += eval_every
                    if since_improve >= patience:
                        break
            if verbose and (t % 50 == 0 or t == 1):
                msg = f"epoch {t:4d}  loss {loss:.4f}"
                if use_val:
                    msg += f"  val {best['val']:.4f} (best@{best['epoch']})"
                print(msg)

        if use_val:
            for k in params:                 # restore best-validation parameters
                setattr(self, k, best["params"][k])
            self.best_epoch_ = best["epoch"]
        self.history_ = history
        return self

    def predict_proba(self, X):
        return self.forward(X)

    # ------------------------------------------------------- rule readability
    def rule_table(self, feature_names, mf_names=None):
        """Return a human-readable description of each rule and its tendency.

        For a first-order model we report the consequent bias b_r as the rule's
        baseline confidence tendency (via sigmoid), which is what makes the rule
        base readable.
        """
        if mf_names is None:
            mf_names = ([f"MF{m}" for m in range(self.M)] if self.M != 2
                        else ["low", "high"])
        rows = []
        for r in range(self.R):
            antecedent = " AND ".join(
                f"{feature_names[j]} is {mf_names[self.sel[r, j]]}"
                for j in range(self.J)
            )
            tendency = float(_sigmoid(np.array([self.b[r]]))[0])
            rows.append(dict(rule=r, antecedent=antecedent,
                             bias_confidence=round(tendency, 3)))
        return rows


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    # Numerical gradient check on a tiny random problem, so the team can trust
    # the hand-derived backprop.
    rng = np.random.default_rng(1)
    Xt = rng.standard_normal((16, 4))
    yt = (rng.random(16) < _sigmoid(Xt[:, 0] - Xt[:, 2])).astype(float)

    net = ANFIS(n_inputs=4, n_mf=2, first_order=True, seed=3)
    net.forward(Xt)
    ana = net._backward(yt)

    def loss_now():
        p = net.forward(Xt)
        return net.bce(p, yt) + 0.5 * net.l2 * np.sum(net.P ** 2)

    eps = 1e-6
    max_err = 0.0
    for name in ["c", "s", "P", "b"]:
        arr = getattr(net, name)
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps; lp = loss_now()
            arr[idx] = orig - eps; lm = loss_now()
            arr[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        err = np.max(np.abs(num - ana[name]))
        max_err = max(max_err, err)
        print(f"grad check {name:2s}: max abs diff {err:.2e}")
    print("overall max gradient error:", f"{max_err:.2e}")
    assert max_err < 1e-5, "gradient check failed"
    print("gradient check passed")
