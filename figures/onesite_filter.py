"""One-site illustration for the slide "One site" of Lecture 5.

nu ~ exp(-lam phi^4/12) N(0, 1/q0) on one site, mass-regulator profile
K_q = q, C_q = 1/q.  Bridge: Phi_q = ((q-q0) Phi_inf + B_{q-q0}) / q.
Forecast psi_q = E[Phi_inf | Phi_q] (posterior tilted by
exp(q Phi_q y - (q-q0) y^2/2)); approximate forecast Y_q solves
q Y + (lam/3) Y^3 = q Phi_q (explicit force (lam/3) y^3); error Z = psi - Y.
Writes onesite_filter.pdf next to this script.  Fixed seed.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, ACCENT, GREY, TEAL = "#1B2A41", "#B5402E", "#8A8F98", "#1F7A72"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 200})

lam, q0, seed, M = 6.0, 1.0, 7, 200
rng = np.random.default_rng(seed)
y = np.linspace(-6, 6, 4001)
prior = -lam * y**4 / 12 - q0 * y**2 / 2
w = np.exp(prior - prior.max()); cdf = np.cumsum(w); cdf /= cdf[-1]
z = np.linspace(0, 5, 20001); Cstar = 3 * np.max(z * (1 + 3 * z**2) ** -3)

def forecast(qq, x):
    lw = prior + qq * x * y - (qq - q0) * y**2 / 2
    ww = np.exp(lw - lw.max()); return (ww * y).sum() / ww.sum()

def approx(qq, x):
    r = np.roots([lam / 3, 0, qq, -qq * x]); return r[np.abs(r.imag) < 1e-9].real[0]

def path(q):
    phi_inf = np.interp(rng.random(), cdf, y)
    s = q - q0
    B = np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(np.diff(s))))])
    Phi = ((q - q0) * phi_inf + B) / q
    psi = np.array([forecast(a, b) for a, b in zip(q, Phi)])
    Y = np.array([approx(a, b) for a, b in zip(q, Phi)])
    return phi_inf, Phi, psi, Y

qa = q0 + np.concatenate([[0.0], np.geomspace(1e-3, 1e4, 2000)])
phi_inf, Phi, psi, Y = path(qa)
qb = q0 + np.geomspace(1e-2, 1e4, 120)
e_obs, e_for, e_Z = [], [], []
for _ in range(M):
    pi, P, ps, YY = path(np.concatenate([[q0], qb]))
    e_obs.append((P[1:] - ps[1:]) ** 2); e_for.append((ps[1:] - pi) ** 2); e_Z.append(np.abs(ps[1:] - YY[1:]))
e_obs = np.sqrt(np.mean(e_obs, 0)); e_for = np.sqrt(np.mean(e_for, 0)); e_Z = np.max(e_Z, 0)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.7))
a1.axhline(phi_inf, color=ACCENT, lw=1.0, ls=":", label=r"$\Phi_\infty$ (hidden)")
a1.plot(qa, Phi, color=GREY, lw=0.8, label=r"$\Phi_q$ (observation)")
a1.plot(qa, psi, color=NAVY, lw=1.4, label=r"$\psi_q=\mathbb{E}_q\Phi_\infty$ (forecast)")
a1.set_xscale("log"); a1.set_xlabel(r"resolution $q$")
a1.set_title(r"one sample path, $\lambda=%g$, $q_0=%g$" % (lam, q0), fontsize=9)
a1.legend(frameon=False, fontsize=7.5, loc="lower right")
a2.loglog(qb, e_obs, color=GREY, lw=1.4, label=r"$\psi_q-\Phi_q=C_qF_q$ (rms)")
a2.loglog(qb, e_for, color=NAVY, lw=1.4, label=r"$\psi_q-\Phi_\infty$ (rms)")
a2.loglog(qb, e_Z, color=TEAL, lw=1.4, label=r"$Z_q=\psi_q-Y_q$ (max over paths)")
a2.loglog(qb, 2 * Cstar * (lam / 3)**0.5 * qb**-1.5, color=ACCENT, lw=1.0, ls="--", label=r"bound $2C_\ast(\lambda/3)^{1/2}q^{-3/2}$")
a2.set_xlabel(r"resolution $q$")
a2.set_title(r"sizes over %d paths" % M, fontsize=9)
a2.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "onesite_filter.pdf")); print("wrote onesite_filter.pdf")
