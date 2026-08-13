"""PPO, and potential-based shaping from a discovered conserved quantity.

The shaping construction is the one the paper argues for and it is worth being
precise about why the obvious alternative is useless. Rewarding the agent for
the invariant residual it induces gives a signal that is identically zero in any
real environment, because the environment never violates its own physics. What
carries information is not that energy is conserved but *what its value is*, so
the potential is the distance from the energy the task requires:

    Phi(s) = -|E(s) - E*|,   r' = r + gamma Phi(s') - Phi(s)

Being potential-based, this leaves the set of optimal policies unchanged
\\citep{ng1999shaping}, which is what separates it from an arbitrary bonus and
is the property the experiment checks rather than assumes.

PPO is implemented here rather than imported so the whole study runs from the
same virtual environment as the algebra, with no RL framework pinned to a
different numpy.
"""

import numpy as np
import sympy as sp
import torch
import torch.nn as nn


# ------------------------------------------------------------- potentials ---

def energy_potential(poly, variables, target):
    """Phi(s) = -|p(s) - target| for a recovered conserved quantity `p`."""
    f = _compile(poly, variables)

    def phi(obs):
        return -np.abs(f(obs) - target)
    return phi


def _compile(poly, variables):
    p = sp.Poly(sp.expand(poly), *variables)
    terms = [(float(c), tuple(m)) for m, c in zip(p.monoms(), p.coeffs())]

    def f(obs):
        obs = np.atleast_2d(obs)
        acc = np.zeros(obs.shape[0])
        for c, exps in terms:
            t = np.full(obs.shape[0], c)
            for k, e in enumerate(exps):
                if e:
                    t = t * obs[:, k] ** e
            acc += t
        return acc if acc.shape[0] > 1 else acc[0]
    return f


# -------------------------------------------------------------------- PPO ---

class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, n_act))
        self.v = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


def ppo(env_fn, phi=None, gamma=0.99, lam=0.95, total_steps=150_000,
        rollout_len=1024, epochs=6, minibatch=256, clip=0.2, lr=3e-4,
        seed=0, novelty=None):
    """Minimal PPO. Returns (returns_log, steps_log).

    `phi` is the shaping potential, applied as gamma*Phi(s') - Phi(s) so that
    the guarantee of \\citet{ng1999shaping} applies exactly. `novelty` is an
    optional intrinsic bonus, used for the RND baseline, and is added to the
    reward *without* the potential-based structure, which is precisely why it
    carries no policy-invariance guarantee.

    Reported returns are always the *unshaped* environment returns. Reporting
    shaped returns would make any bonus look like an improvement by definition.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = env_fn()
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n

    net = ActorCritic(n_obs, n_act)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    ep_ret, returns_log, steps_log = 0.0, [], []
    step = 0

    while step < total_steps:
        O = np.zeros((rollout_len, n_obs), np.float32)
        A = np.zeros(rollout_len, np.int64)
        LP = np.zeros(rollout_len, np.float32)
        R = np.zeros(rollout_len, np.float32)
        D = np.zeros(rollout_len, np.float32)
        Vv = np.zeros(rollout_len + 1, np.float32)

        for t in range(rollout_len):
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, v = net(ot)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            O[t], A[t] = obs, int(a)
            LP[t], Vv[t] = float(dist.log_prob(a)), float(v)

            nxt, r, term, trunc, _ = env.step(int(a))
            ep_ret += r
            shaped = float(r)
            if phi is not None:
                # Terminal states get Phi = 0 by convention, which is what keeps
                # the telescoping sum policy-invariant across episode boundaries.
                phi_next = 0.0 if term else float(phi(nxt))
                shaped += gamma * phi_next - float(phi(obs))
            if novelty is not None:
                shaped += novelty(nxt)
            R[t], D[t] = shaped, float(term or trunc)

            obs = nxt
            step += 1
            if term or trunc:
                returns_log.append(ep_ret)
                steps_log.append(step)
                ep_ret = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            Vv[rollout_len] = float(net(torch.as_tensor(
                obs, dtype=torch.float32).unsqueeze(0))[1])

        adv = np.zeros(rollout_len, np.float32)
        gae = 0.0
        for t in reversed(range(rollout_len)):
            nonterm = 1.0 - D[t]
            delta = R[t] + gamma * Vv[t + 1] * nonterm - Vv[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + Vv[:rollout_len]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        Ot = torch.as_tensor(O)
        At = torch.as_tensor(A)
        LPt = torch.as_tensor(LP)
        advt = torch.as_tensor(adv)
        rett = torch.as_tensor(ret)

        for _ in range(epochs):
            idx = rng.permutation(rollout_len)
            for i in range(0, rollout_len, minibatch):
                b = idx[i:i + minibatch]
                logits, v = net(Ot[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(At[b])
                ratio = torch.exp(lp - LPt[b])
                l1 = ratio * advt[b]
                l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advt[b]
                loss = (-torch.min(l1, l2).mean()
                        + 0.5 * ((v - rett[b]) ** 2).mean()
                        - 0.01 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

    env.close()
    return np.array(returns_log), np.array(steps_log)


# ----------------------------------------------------------- RND baseline ---

class RND:
    """Random Network Distillation bonus \\citep{burda2019rnd}, scaled online."""

    def __init__(self, n_obs, hidden=64, lr=1e-3, seed=0, coef=0.05):
        torch.manual_seed(seed + 7)
        self.target = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 32))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 32))
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.coef, self.run = coef, 1.0

    def __call__(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        err = ((self.pred(o) - self.target(o)) ** 2).mean()
        self.opt.zero_grad()
        err.backward()
        self.opt.step()
        e = float(err.detach())
        self.run = 0.99 * self.run + 0.01 * e
        return self.coef * e / (self.run + 1e-8)
