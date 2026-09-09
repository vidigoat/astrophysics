"""Bayesian errors-in-variables regression of log M_BH on (log M*, log M_h) for the real samples.

Model: true (x1, x2) ~ bivariate normal (mean mu, covariance S); observed xi_obs = xi + N(0, e_xi);
y_obs = a x1 + b x2 + c + N(0, sqrt(s_int^2 + e_y^2)).  The latent x are marginalised analytically
(Gaussian), so the likelihood is a 3-d Gaussian per object; parameters (a, b, c, s_int, mu, S) are
sampled with a simple adaptive Metropolis chain.  Reports posterior medians and 68% intervals.
"""
import csv, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))


def load_marasco():
    R = list(csv.DictReader(open(os.path.join(ROOT, 'Data', 'obs', 'marasco2021.csv'))))
    y = np.array([float(r['logMBH']) for r in R]); x2 = np.array([float(r['logMh']) for r in R]); x1 = np.array([float(r['logMstar']) for r in R])
    ey = np.array([float(r['e_logMBH']) for r in R]); e2 = np.array([float(r['e_logMh']) for r in R]); efs = np.array([float(r['e_logfstar']) for r in R])
    e1 = np.sqrt(np.maximum(efs ** 2 - e2 ** 2, 0.10 ** 2))
    return y, x1, x2, ey, e1, e2


def load_gaspari(nonbcg=True):
    G = list(csv.DictReader(open(os.path.join(ROOT, 'Data', 'obs', 'gaspari2019.csv'))))
    keep = [not (r['central'].startswith('BCG') or r['central'].startswith('BGG')) for r in G] if nonbcg else [True] * len(G)
    G = [r for r, k in zip(G, keep) if k]
    y = np.array([float(r['logMBH']) for r in G]); x2 = np.array([float(r['logM500']) for r in G]); x1 = np.array([float(r['logLK']) for r in G]) + np.log10(0.75)
    ey = np.array([float(r['e_logMBH']) for r in G]); e2 = np.maximum(1.5 * np.array([float(r['e_logTxc']) for r in G]), 0.10); e1 = np.maximum(np.array([float(r['e_logLK']) for r in G]), 0.10)
    return y, x1, x2, ey, e1, e2


def loglike(p, y, x1, x2, ey, e1, e2):
    a, b, c, ls, m1, m2, ls1, ls2, rho = p
    s_int = np.exp(ls); s1 = np.exp(ls1); s2 = np.exp(ls2)
    if not (-0.999 < rho < 0.999): return -np.inf
    # latent covariance
    S = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    A = np.array([a, b])
    mean = np.array([m1, m2, a * m1 + b * m2 + c])
    ll = 0.0
    for i in range(len(y)):
        C = np.zeros((3, 3))
        C[:2, :2] = S + np.diag([e1[i] ** 2, e2[i] ** 2])
        C[:2, 2] = S @ A; C[2, :2] = C[:2, 2]
        C[2, 2] = A @ S @ A + s_int ** 2 + ey[i] ** 2
        d = np.array([x1[i], x2[i], y[i]]) - mean
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return -np.inf
        z = np.linalg.solve(L, d)
        ll += -0.5 * z @ z - np.log(np.diag(L)).sum()
    return ll


def run(y, x1, x2, ey, e1, e2, nstep=30000, seed=0):
    rng = np.random.default_rng(seed)
    # init from OLS
    X = np.column_stack([np.ones(len(y)), x1, x2]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    p = np.array([beta[1], beta[2], beta[0], np.log(0.3), x1.mean(), x2.mean(), np.log(x1.std()), np.log(x2.std()), np.corrcoef(x1, x2)[0, 1]])
    step = np.array([0.025, 0.02, 0.25, 0.03, 0.015, 0.015, 0.03, 0.03, 0.01])
    lp = loglike(p, y, x1, x2, ey, e1, e2); chain = []; acc = 0
    for it in range(nstep):
        q = p + step * rng.normal(size=len(p))
        lq = loglike(q, y, x1, x2, ey, e1, e2)
        if np.log(rng.uniform()) < lq - lp:
            p, lp = q, lq; acc += 1
        if it >= nstep // 3 and it % 10 == 0: chain.append(p.copy())
    return np.array(chain), acc / nstep


if __name__ == '__main__':
    for name, data in [('Marasco 2021 (55)', load_marasco()), ('Gaspari 2019 non-BCG (56)', load_gaspari(True)), ('Gaspari 2019 all (85)', load_gaspari(False))]:
        ch, acc = run(*data, nstep=int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
        a, b, c, ls = ch[:, 0], ch[:, 1], ch[:, 2], ch[:, 3]
        q = lambda v: (np.median(v), np.percentile(v, 16), np.percentile(v, 84))
        print(f'{name:28s} acc={acc:.2f}  a = {q(a)[0]:+.2f} [{q(a)[1]:+.2f},{q(a)[2]:+.2f}]   b = {q(b)[0]:+.2f} [{q(b)[1]:+.2f},{q(b)[2]:+.2f}]   '
              f'intrinsic scatter = {np.median(np.exp(ls)):.2f} dex   P(b < 0.5) = {np.mean(b < 0.5):.4f}   P(b > 5/3) = {np.mean(b > 5/3):.2f}')
