"""Statistical two-sample baselines for detecting that two systems differ.

These are what a practitioner reaches for today when asked whether a simulator
changed, or whether a learned model matches its training environment: compare
the state distributions and see whether a test rejects. They are strong
detectors and the comparison is meant to be fair, so the discriminator gets a
proper train/test split and the kernel test gets a permutation null rather than
an asymptotic approximation.

What none of them can do is say *what* differs. A rejected null is a scalar; it
does not name the constraint that broke, and it certainly does not hand back the
numerical value of the parameter responsible. That asymmetry, not the detection
rate, is the point of the comparison.
"""

import numpy as np


def _subsample(X, n, rng):
    if X.shape[0] <= n:
        return X
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx]


def _standardise(X, Y):
    """Standardise both samples by the pooled statistics of the pair.

    Per-sample standardisation would erase exactly the mean and scale shifts the
    tests exist to find, so the scaler is fitted on the concatenation.
    """
    Z = np.vstack([X, Y])
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, (Y - mu) / sd


# ------------------------------------------------------------------- MMD ---

def _rbf(A, B, gamma):
    d2 = (np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :]
          - 2 * A @ B.T)
    return np.exp(-gamma * np.maximum(d2, 0))


def mmd2(X, Y, gamma):
    """Unbiased squared MMD."""
    n, m = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = _rbf(X, X, gamma), _rbf(Y, Y, gamma), _rbf(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.mean())


def _mmd2_from_kernel(K, a, diag):
    """Unbiased squared MMD from a precomputed kernel and a 0/1 membership vector.

    Identical in value to `mmd2` on the corresponding split, but a matrix-vector
    product instead of three fresh kernel evaluations, which is what makes a
    permutation budget of ten thousand affordable at this sample size.
    """
    b = 1.0 - a
    u = K @ a
    v = K @ b
    n, m = float(a.sum()), float(b.sum())
    if n < 2 or m < 2:
        return 0.0
    kxx = float(a @ u) - float(a @ diag)
    kyy = float(b @ v) - float(b @ diag)
    kxy = float(a @ v)
    return kxx / (n * (n - 1)) + kyy / (m * (m - 1)) - 2 * kxy / (n * m)


#: Permutation and subsample budgets for the kernel test.
#:
#: The smallest p-value a permutation test can return is 1/(n_perm+1), which
#: bounds how significant the test can be however large the effect. Ten thousand
#: permutations put that floor at 1e-4, two orders below the alpha the
#: comparison is scored at, so the budget is not what decides the verdict. The
#: subsample size is what the O(n_perm * n_sub^2) cost will bear; the kernel is
#: built once and permuted by index, so the budget buys resolution rather than
#: time.
MMD_N_PERM = 10_000
MMD_N_SUB = 2_000


def _median_heuristic_gamma(Z, rng, max_points=200):
    """RBF inverse bandwidth from an unlabelled pooled sample of ``Z``."""
    band_idx = rng.choice(len(Z), size=min(max_points, len(Z)), replace=False)
    Z_band = Z[band_idx]
    d2 = np.sum((Z_band[:, None, :] - Z_band[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return 1.0 / float(med)


def mmd_permutation_test(X, Y, n_perm=MMD_N_PERM, n_sub=MMD_N_SUB, seed=0,
                         chunk=250):
    """Permutation p-value for the RBF-kernel MMD, median-heuristic bandwidth.

    The bandwidth is estimated from an unlabelled subsample of the pooled
    observations.  It is consequently fixed under every label permutation,
    which preserves the permutation null.  Estimating it from the original
    ``X`` arm would make kernel selection depend on the labels being shuffled.
    """
    rng = np.random.default_rng(seed)
    X, Y = _standardise(X, Y)
    X, Y = _subsample(X, n_sub, rng), _subsample(Y, n_sub, rng)
    Z = np.vstack([X, Y]).astype(np.float32)
    # A fixed-size pooled subsample keeps the median heuristic inexpensive
    # without letting the observed group labels choose the kernel.
    gamma = _median_heuristic_gamma(Z, rng)

    K = _rbf(Z, Z, gamma).astype(np.float32)
    diag = np.diag(K).astype(np.float32).copy()
    N, n = Z.shape[0], X.shape[0]

    a0 = np.zeros(N, dtype=np.float32)
    a0[:n] = 1.0
    obs = _mmd2_from_kernel(K, a0, diag)

    count = 0
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        A = np.zeros((N, b), dtype=np.float32)
        for j in range(b):
            A[rng.permutation(N)[:n], j] = 1.0
        U = K @ A                                  # one BLAS-3 call per chunk
        na, nb = A.sum(0), N - A.sum(0)
        Ac = 1.0 - A
        kxx = np.einsum("ij,ij->j", A, U) - diag @ A
        kxy = np.einsum("ij,ij->j", Ac, U)
        Uc = K @ Ac
        kyy = np.einsum("ij,ij->j", Ac, Uc) - diag @ Ac
        stats = (kxx / (na * (na - 1)) + kyy / (nb * (nb - 1))
                 - 2 * kxy / (na * nb))
        count += int(np.sum(stats >= obs))
        done += b
    return float(obs), (count + 1) / (n_perm + 1)


# -------------------------------------------------------------------- KS ---

def _ks_stat(a, b):
    all_v = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), all_v, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), all_v, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def ks_per_dimension(X, Y, n_sub=2000, seed=0):
    """Bonferroni-corrected KS p-value and statistic for every column.

    `ks_bonferroni` reduces this to the single best column, which is all a
    detector needs. Keeping the per-column vector is what lets the same test be
    scored on localisation when the columns are named residuals rather than raw
    state coordinates.

    Columns constant in both arms are dropped from the family, as there, and
    reported with p = 1.
    """
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, n_sub, rng), _subsample(Y, n_sub, rng)
    raw = []
    for j in range(X.shape[1]):
        xj, yj = X[:, j], Y[:, j]
        if np.ptp(xj) == 0.0 and np.ptp(yj) == 0.0:
            raw.append(None)
            continue
        r = ks_2samp(xj, yj, method="asymp")
        raw.append((float(r.pvalue), float(r.statistic)))
    k = max(1, sum(1 for r in raw if r is not None))
    return [(1.0, 0.0) if r is None else (min(1.0, r[0] * k), r[1])
            for r in raw]


def ks_bonferroni(X, Y, n_sub=2000, seed=0):
    """Largest per-dimension KS statistic, with a Bonferroni-corrected p-value.

    Returns the statistic and p-value of the *same* dimension, the one with the
    smallest p, so the two reported numbers describe one test rather than two.

    The tail is evaluated by SciPy rather than by the truncated Kolmogorov
    series. Summing that series by hand is wrong in exactly the case this
    baseline meets most often: at D = 0 the alternating terms cancel to 0.0
    instead of 1.0, so a dimension that is bit-identical between the two systems
    reports maximal significance. Reacher's dz coordinate is identically zero in
    both arms of every healthy run, which is enough to make the whole test
    reject on a system with nothing wrong with it.

    Dimensions that are constant in both arms carry no distributional
    information and are dropped before the correction rather than counted in
    it, which is the standard treatment and avoids inflating the Bonferroni
    family with dimensions no test was actually run on.
    """
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, n_sub, rng), _subsample(Y, n_sub, rng)
    tested = []
    for j in range(X.shape[1]):
        xj, yj = X[:, j], Y[:, j]
        if np.ptp(xj) == 0.0 and np.ptp(yj) == 0.0:
            continue
        r = ks_2samp(xj, yj, method="asymp")
        tested.append((float(r.pvalue), float(r.statistic)))
    if not tested:
        return 0.0, 1.0
    best_p, best_d = min(tested)
    return best_d, float(min(1.0, best_p * len(tested)))


# --------------------------------------------------------- discriminator ---

def auc(scores, labels):
    """Rank-based AUC, ties averaged."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def discriminator_test(X, Y, hidden=64, epochs=300, seed=0, n_sub=3000,
                       patience=20):
    """Held-out AUC of an MLP trained to separate the two samples, with a null.

    The strongest of the baselines and the one closest to what a practitioner
    would actually build. An AUC near 0.5 means the two systems are
    indistinguishable at the level of single states.

    An AUC becomes a decision only against a threshold, and a threshold chosen
    by hand is not comparable with the corrected p-values the other methods
    report. The null is available without retraining, because the AUC is
    evaluated on data the network never saw: if the two samples come from one
    distribution the held-out labels are exchangeable given the scores, and the
    permutation distribution of the AUC is the Mann-Whitney null. That is the
    classifier two-sample test, and it puts every baseline on one footing.

    The split is three-way and training stops on the validation AUC. Two-way
    with a fixed epoch count is what this was, and it fails in the paired
    regime, where the two arms are near-duplicate rows carrying opposite
    labels: the network memorises the training rows, generalises the *inverse*
    relation to held-out data, and returns an AUC below 0.5. Measured that way
    a bit-identical pair of arms scored 0.29, which a one-sided test reports as
    p = 1 and a two-sided one as overwhelming evidence of a difference. Neither
    is a measurement of the baseline's power. Early stopping removes the
    memorisation, after which AUC on identical arms sits at chance and the
    one-sided test, which is the right form for a classifier two-sample test,
    is honest.

    Returns (auc, p_value).
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, Y = _standardise(X, Y)
    X, Y = _subsample(X, n_sub, rng), _subsample(Y, n_sub, rng)

    Z = np.vstack([X, Y]).astype(np.float32)
    y = np.concatenate([np.zeros(len(X)), np.ones(len(Y))]).astype(np.float32)
    perm = rng.permutation(len(Z))
    Z, y = Z[perm], y[perm]
    n_tr, n_va = int(0.5 * len(Z)), int(0.2 * len(Z))
    Ztr, ytr = Z[:n_tr], y[:n_tr]
    Zva, yva = Z[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    Zte, yte = Z[n_tr + n_va:], y[n_tr + n_va:]

    net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Zt, yt = torch.from_numpy(Ztr), torch.from_numpy(ytr)
    Zv = torch.from_numpy(Zva)

    best_auc, best_state, since = -1.0, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(Zt).squeeze(-1), yt).backward()
        opt.step()
        with torch.no_grad():
            sv = net(Zv).squeeze(-1).numpy()
        a_va = auc(sv, yva)
        if a_va > best_auc:
            best_auc, since = a_va, 0
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    with torch.no_grad():
        s = net(torch.from_numpy(Zte)).squeeze(-1).numpy()
    a = auc(s, yte)

    from scipy.stats import mannwhitneyu
    pos, neg = s[yte == 1], s[yte == 0]
    if len(pos) < 3 or len(neg) < 3:
        return a, 1.0
    try:
        p = float(mannwhitneyu(pos, neg, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return a, p


# ------------------------------------------------------- paired difference ---

#: Absolute floor on a paired coordinate discrepancy, the exact analogue of the
#: screen's RESIDUAL_FLOOR and set to the same value for the same reason. A rank
#: test has no scale: given block means differing in the seventeenth significant
#: figure it separates them completely and reports a broken system. Whatever is
#: granted to the screen has to be granted here, or the comparison measures the
#: floor and not the method.
PAIRED_FLOOR = 1e-8


def paired_difference_test(X, Y, X0, Y0, n_block=32, floor=PAIRED_FLOOR):
    """Bonferroni-corrected paired test on per-coordinate block-mean discrepancy.

    The other three tests are two-sample tests on marginals, and pooling the
    arms before calling them throws the pairing away. In the paired regime the
    two arms are logged from identical initial conditions on a common sampling
    grid, so row i of one corresponds to row i of the other, and a test using
    that correspondence is strictly better informed than one that does not.
    Scoring only unpaired baselines against a screen that is itself paired
    would credit the algebra with what is really a difference in the
    information each method was handed.

    The construction deliberately mirrors the screen's. For each coordinate the
    rows are cut into contiguous blocks and the mean absolute discrepancy
    |Y - X| is taken within each block; the same quantity is computed for a
    healthy paired pair (X0, Y0); and a one-sided Mann-Whitney test asks whether
    the test blocks are stochastically larger, Bonferroni corrected across
    coordinates. That is the screen's own test with the per-generator residual
    replaced by a per-coordinate discrepancy, so what separates the two is the
    feature and nothing else.

    A signed test in place of the absolute one has almost no power here, since a
    parameter error moves a trajectory in a direction that oscillates with the
    dynamics and the signed block means average towards zero.

    This baseline can name an offending *coordinate*, which the other three
    cannot. It still cannot name a constraint: a constraint is a polynomial in
    several coordinates, and on any parameter or structural fault the error
    reaches every coordinate within a few steps.

    Returns (statistic, corrected p, per-coordinate corrected p-values), the
    statistic being the ratio of median discrepancies on the separating
    coordinate. Arms whose shapes do not correspond are not paired and return
    (nan, 1.0, []).
    """
    from scipy.stats import mannwhitneyu

    if (X.shape != Y.shape or X0.shape != Y0.shape or X.shape[1] != X0.shape[1]
            or X.shape[0] < n_block * 2 or X0.shape[0] < n_block * 2):
        return float("nan"), 1.0, []

    def block_means(A, B):
        d = np.abs(B - A)
        cut = (A.shape[0] // n_block) * n_block
        return d[:cut].reshape(n_block, -1, d.shape[1]).mean(axis=1)

    bt, b0 = block_means(X, Y), block_means(X0, Y0)

    tested, per_col = [], [None] * X.shape[1]
    for j in range(bt.shape[1]):
        a, b = bt[:, j], b0[:, j]
        if np.max(a) <= floor and np.max(b) <= floor:
            continue
        try:
            p = float(mannwhitneyu(a, b, alternative="greater").pvalue)
        except ValueError:
            continue
        med0 = float(np.median(b))
        ratio = float(np.median(a)) / med0 if med0 > 0 else np.inf
        tested.append((p, ratio, j))
    if not tested:
        return 0.0, 1.0, [1.0] * X.shape[1]

    k = len(tested)
    for p, _, j in tested:
        per_col[j] = min(1.0, p * k)
    per_col = [1.0 if v is None else v for v in per_col]
    best_p, best_ratio, _ = min(tested)
    return float(best_ratio), float(min(1.0, best_p * k)), per_col


def discriminator_auc(X, Y, **kw):
    """The AUC alone, for callers that do not want the null."""
    return discriminator_test(X, Y, **kw)[0]