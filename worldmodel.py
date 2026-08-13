"""Learned transition models, and the algebraic consistency term.

The world model is deliberately ordinary: a small MLP trained on one-step
transitions by mean-squared error. Nothing about the diagnostic needs it to be
anything else, and using a plain model is what makes the result about the
diagnostic rather than about an architecture.

The consistency term of the paper's world-model section is implemented here
because it is one line of the loss and because the experiment needs to compare
models trained with and without it.
"""

import numpy as np
import sympy as sp
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Predicts the *increment* s' - s rather than s' itself.

    Standard practice for dynamics models and worth stating: predicting the
    state directly makes the identity map the easy solution, and a model that
    has learned the identity scores well on one-step error while being useless
    over a horizon. Predicting the increment removes that particular way of
    looking good for the wrong reason.
    """

    def __init__(self, n_obs, n_act, hidden=128, depth=2):
        super().__init__()
        layers, d = [], n_obs + n_act
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, n_obs)]
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], dim=-1))


def polys_to_torch(polys, variables):
    """Compile sympy polynomials into a batched torch residual function.

    Returns f(S) -> (batch, len(polys)). Kept as an explicit monomial sum rather
    than going through lambdify so the result is differentiable and stays on
    whatever device the states are on.
    """
    terms = []
    for p in polys:
        poly = sp.Poly(sp.expand(p), *variables)
        terms.append([(float(c), tuple(m))
                      for m, c in zip(poly.monoms(), poly.coeffs())])

    def f(S):
        outs = []
        for term_list in terms:
            acc = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
            for coeff, exps in term_list:
                t = torch.full_like(acc, coeff)
                for k, e in enumerate(exps):
                    if e:
                        t = t * S[:, k] ** e
                acc = acc + t
            outs.append(acc)
        return torch.stack(outs, dim=-1)

    return f


def consistency_loss(pred, vanishing_fn, conserved_fn=None, prev=None):
    """Penalise leaving the constraint manifold, and failing to conserve.

    The vanishing part is the paper's L_cons: each generator should evaluate to
    zero at the predicted state. The conserved part is different in kind, since
    a conserved quantity has no particular value to be driven to; what is
    penalised is the *change* in it across the predicted transition, which is
    the passive case of the balance law with zero power injection.
    """
    loss = (vanishing_fn(pred) ** 2).mean()
    if conserved_fn is not None and prev is not None:
        loss = loss + ((conserved_fn(pred) - conserved_fn(prev)) ** 2).mean()
    return loss


def split_indices(n, val_frac, generator, traj_len=None):
    """Train/validation row indices, held out by trajectory where possible.

    A permutation of rows puts validation transitions between training
    transitions of the same trajectory, one step apart, so a model that has
    memorised its neighbours scores well on them and the one-step validation
    error reads lower than it is. That error is one of the two predictors the
    world-model experiment compares, so the optimism runs towards a conclusion
    the experiment is meant to test. Holding out whole trajectories removes it.

    `traj_len` is the number of transitions per trajectory; rows are assumed to
    arrive in trajectory order, which is how `make_dataset` writes them. Falls
    back to a row permutation when the length is unknown or divides the sample
    into fewer than five trajectories, where a trajectory-wise split would put
    an unusable fraction of the data in validation.
    """
    if traj_len and n % traj_len == 0 and n // traj_len >= 5:
        n_traj = n // traj_len
        order = torch.randperm(n_traj, generator=generator)
        cut = max(1, int(round((1 - val_frac) * n_traj)))
        def rows(ts):
            return torch.cat([torch.arange(int(t) * traj_len,
                                           (int(t) + 1) * traj_len)
                              for t in ts])
        return rows(order[:cut]), rows(order[cut:])
    perm = torch.randperm(n, generator=generator)
    cut = int((1 - val_frac) * n)
    return perm[:cut], perm[cut:]


def train(S, A, S2, n_obs, n_act, hidden=128, depth=2, epochs=400, lr=1e-3,
          lam=0.0, vanishing=None, conserved=None, seed=0, batch=512,
          val_frac=0.2, weight_decay=0.0, traj_len=None):
    """Train one model; return (model, history).

    `lam` weights the algebraic consistency term. At lam = 0 this is an ordinary
    MSE dynamics model, which is the baseline the experiment needs.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(S)
    tr, va = split_indices(n, val_frac, g, traj_len=traj_len)

    S, A, S2 = (torch.as_tensor(x, dtype=torch.float32) for x in (S, A, S2))
    model = MLP(n_obs, n_act, hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    hist = []
    for ep in range(epochs):
        idx = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            pred = model(S[b], A[b])
            loss = ((pred - S2[b]) ** 2).mean()
            if lam > 0 and vanishing is not None:
                loss = loss + lam * consistency_loss(
                    pred, vanishing, conserved, S[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                vm = ((model(S[va], A[va]) - S2[va]) ** 2).mean().item()
            hist.append((ep, vm))
    with torch.no_grad():
        val_mse = ((model(S[va], A[va]) - S2[va]) ** 2).mean().item()
    return model, dict(history=hist, val_mse=val_mse)


@torch.no_grad()
def rollout(model, s0, actions):
    """Closed-loop rollout of the model from `s0` under a fixed action sequence."""
    s = torch.as_tensor(s0, dtype=torch.float32).unsqueeze(0)
    out = [s.squeeze(0).numpy()]
    for a in actions:
        a_t = torch.as_tensor(a, dtype=torch.float32).reshape(1, -1)
        s = model(s, a_t)
        out.append(s.squeeze(0).numpy())
    return np.array(out)
