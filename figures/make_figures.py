"""Figures for the SISSA lectures on constructive QFT (Lectures 1-3).

Run with:  python3 make_figures.py
Writes PDF (and a few PNG) files into the directory of this script.
Only numpy and matplotlib are needed. All random samples use fixed seeds.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, ACCENT, GREY = "#1B2A41", "#B5402E", "#8A8F98"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def gaussian_field_2d(N, m, profile, rng=None, noise=None):
    """Real Gaussian field on the unit torus with Fourier covariance profile(|k|^2)."""
    k = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    if noise is None:
        noise = np.fft.fft2(rng.standard_normal((N, N)))
    phi = np.real(np.fft.ifft2(noise * np.sqrt(profile(k2, m))))
    return phi - phi.mean(), noise  # zero mode removed for display


# ------------------------------------------------------------------ L1
def fig_ou():
    rng = np.random.default_rng(1)
    m, dt = 1.0, 0.01
    a, s = np.exp(-m * dt), np.sqrt((1 - np.exp(-2 * m * dt)) / (2 * m))
    n_long = 400_000
    x = np.empty(n_long)
    x[0] = rng.standard_normal() / np.sqrt(2 * m)
    xi = rng.standard_normal(n_long)
    for i in range(1, n_long):
        x[i] = a * x[i - 1] + s * xi[i]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    tau = np.arange(2000) * dt
    ax1.plot(tau, x[:2000], color=NAVY, lw=0.7)
    ax1.set_xlabel(r"imaginary time $\tau$")
    ax1.set_ylabel(r"$q_\tau$")
    ax1.set_title("stationary Ornstein--Uhlenbeck path, $m=1$", fontsize=9)
    lags = np.arange(0, 400, 10)
    emp = [np.mean(x[: n_long - l] * x[l:]) for l in lags]
    tl = np.linspace(0, 4, 200)
    ax2.plot(tl, np.exp(-m * tl) / (2 * m), color=ACCENT, lw=1.2,
             label=r"$e^{-m\tau}/2m$")
    ax2.plot(lags * dt, emp, "o", ms=2.5, color=NAVY, label="empirical")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_title("two-point function", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "ou_path.pdf")


# ------------------------------------------------------------------ L2
def fig_gff2d():
    rng = np.random.default_rng(7)
    phi, _ = gaussian_field_2d(512, 4.0, lambda k2, m: 1.0 / (k2 + m**2), rng)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    v = np.quantile(np.abs(phi), 0.995)
    ax.imshow(phi, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
    fig.savefig(os.path.join(HERE, "gff2d.png"), dpi=220)
    plt.close(fig)
    print("wrote gff2d.png")


def fig_markov():
    """Domain Markov property of the lattice GFF: phi = h + psi in a disc."""
    N, m = 240, 0.03
    rng = np.random.default_rng(11)
    k = 2 * np.pi * np.fft.fftfreq(N)
    lam = m**2 + 4 * np.sin(k[:, None] / 2) ** 2 + 4 * np.sin(k[None, :] / 2) ** 2
    phi = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) / np.sqrt(lam)))
    yy, xx = np.mgrid[:N, :N]
    D = (xx - N / 2) ** 2 + (yy - N / 2) ** 2 < (0.3 * N) ** 2

    def A(u):  # (m^2 - Delta) on the torus
        return (4 + m**2) * u - (np.roll(u, 1, 0) + np.roll(u, -1, 0)
                                 + np.roll(u, 1, 1) + np.roll(u, -1, 1))

    # (m^2-Delta)h = 0 in D, h = phi off D: solve for the interior values by CG
    out = np.where(D, 0.0, phi)
    b = -np.where(D, A(out), 0.0)
    x = np.zeros_like(phi)
    r = b - np.where(D, A(x), 0.0)
    p, rs = r.copy(), np.sum(r * r)
    for _ in range(5000):
        Ap = np.where(D, A(p), 0.0)
        a = rs / np.sum(p * Ap)
        x += a * p
        r -= a * Ap
        rs_new = np.sum(r * r)
        if rs_new < 1e-20:
            break
        p, rs = r + rs_new / rs * p, rs_new
    h = out + x
    psi = np.where(D, phi - h, 0.0)

    v = np.quantile(np.abs(phi), 0.995)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4))
    titles = [r"$\varphi$",
              r"$\mathbb{E}[\varphi\,|\,\text{outside}]$: $A$-harmonic in $\Lambda$",
              r"$\varphi-\mathbb{E}[\varphi\,|\,\text{outside}]$: Dirichlet GFF"]
    t = np.linspace(0, 2 * np.pi, 400)
    for ax, f, tt in zip(axs, [phi, h, psi], titles):
        ax.imshow(f, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.plot(N / 2 + 0.3 * N * np.cos(t), N / 2 + 0.3 * N * np.sin(t),
                color="k", lw=0.8)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
        ax.set_title(tt, fontsize=8)
    fig.savefig(os.path.join(HERE, "markov_gff.png"), dpi=220)
    plt.close(fig)
    print("wrote markov_gff.png")


def K0(x):
    u = np.linspace(0, 12, 6001)
    return np.trapezoid(np.exp(-np.multiply.outer(x, np.cosh(u))), u, axis=-1)


def fig_green():
    r = np.logspace(-3, 1, 300)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.loglog(r, np.exp(-r) / 2, color=GREY, lw=1.2, label=r"$d=1$: $e^{-r}/2$")
    ax.loglog(r, K0(r) / (2 * np.pi), color=NAVY, lw=1.2,
              label=r"$d=2$: $K_0(r)/2\pi$")
    ax.loglog(r, np.exp(-r) / (4 * np.pi * r), color=ACCENT, lw=1.2,
              label=r"$d=3$: $e^{-r}/4\pi r$")
    ax.set_xlabel(r"$r=|x|$ ($m=1$)")
    ax.set_ylabel(r"$G_m(x)$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Green function of $m^2-\\Delta$", fontsize=9)
    save(fig, "green_functions.pdf")


def fig_wick_constant():
    m = 1.0
    Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    c = []
    for N in Ns:
        j = np.arange(N)
        lam1 = 4 * N**2 * np.sin(np.pi * j / N) ** 2
        lam = lam1[:, None] + lam1[None, :] + m**2
        c.append(np.sum(1.0 / lam))
    c = np.array(c)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(np.log(Ns), c, "o", color=NAVY, ms=4,
            label=r"$c_N=\mathbb{E}\,\varphi_u^2$ (lattice)")
    fit = c[-1] - np.log(Ns[-1]) / (2 * np.pi)
    xs = np.linspace(np.log(Ns[0]), np.log(Ns[-1]), 50)
    ax.plot(xs, xs / (2 * np.pi) + fit, color=ACCENT, lw=1.2,
            label=r"$\frac{1}{2\pi}\log N+\mathrm{const}$")
    ax.set_xlabel(r"$\log N$ (lattice $(N^{-1}\mathbb{Z}/\mathbb{Z})^2$, $m=1$)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("divergence of the Wick constant in $d=2$", fontsize=9)
    save(fig, "wick_constant.pdf")


def fig_she():
    rng = np.random.default_rng(3)
    m, N, nt, Tmax, Lc = 1.0, 256, 400, 2.0, 8.0
    dt = Tmax / nt
    kk = 2 * np.pi * np.arange(1, N // 2) / Lc
    lam = np.concatenate([[m**2], kk**2 + m**2, kk**2 + m**2])
    a = np.exp(-lam * dt)
    s = np.sqrt((1 - a**2) / lam)  # noise sqrt(2)*xi: stationary variance 1/lam
    modes = np.zeros(lam.size)
    x = np.linspace(0, Lc, N, endpoint=False)
    basis = np.vstack([np.ones(N) / np.sqrt(Lc)]
                      + [np.sqrt(2 / Lc) * np.cos(k * x) for k in kk]
                      + [np.sqrt(2 / Lc) * np.sin(k * x) for k in kk])
    field = np.empty((nt, N))
    for i in range(nt):
        modes = a * modes + s * rng.standard_normal(lam.size)
        field[i] = np.einsum("i,ij->j", modes, basis)  # avoids spurious BLAS warnings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    v = np.quantile(np.abs(field), 0.995)
    ax1.imshow(field.T, aspect="auto", origin="lower", cmap="RdBu_r",
               vmin=-v, vmax=v, extent=[0, Tmax, 0, Lc])
    ax1.set_xlabel("stochastic time $t$")
    ax1.set_ylabel(r"$x\in\mathbb{R}/8\mathbb{Z}$")
    ax1.set_title(r"$\partial_t\Phi=-(m^2-\Delta)\Phi+\sqrt{2}\,\xi$, $\Phi_0=0$",
                  fontsize=9)
    ts = np.linspace(0, Tmax, 200)
    ks = [0.0, np.pi / 2, np.pi]
    labels = ["$k=0$", r"$k=\pi/2$", r"$k=\pi$"]
    for k, col, lab in zip(ks, [NAVY, ACCENT, GREY], labels):
        L = k**2 + m**2
        ax2.plot(ts, 1 - np.exp(-2 * L * ts), color=col, lw=1.2, label=lab)
    nrep = 4000
    for k, col in zip(ks[:2], [NAVY, ACCENT]):
        L = k**2 + m**2
        tt = np.array([0.1, 0.3, 0.6, 1.0, 1.5])
        y = np.zeros(nrep)
        emp, t0 = [], 0.0
        for t1 in tt:
            h = t1 - t0
            y = np.exp(-L * h) * y + np.sqrt((1 - np.exp(-2 * L * h)) / L) * rng.standard_normal(nrep)
            emp.append(np.mean(y**2) * L)
            t0 = t1
        ax2.plot(tt, emp, "o", ms=3, color=col)
    ax2.set_xlabel("$t$")
    ax2.set_title(r"$\lambda_k\,\mathbb{E}|\hat\Phi_t(k)|^2$", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "she_relaxation.pdf")


# ------------------------------------------------------------------ L3
def fig_multiscale():
    rng = np.random.default_rng(11)
    N, m = 512, 4.0
    noise = np.fft.fft2(rng.standard_normal((N, N)))
    ts = [8, 32, 128, None]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.85))
    for ax, t in zip(axs, ts):
        if t is None:
            prof = lambda k2, m: 1.0 / (k2 + m**2)
            title = r"$t=\infty$ (lattice cutoff)"
        else:
            prof = lambda k2, m, t=t: np.exp(-(k2 + m**2) / t**2) / (k2 + m**2)
            title = rf"$t={t}$"
        phi, _ = gaussian_field_2d(N, m, prof, noise=noise)
        v = np.quantile(np.abs(phi), 0.995)
        ax.imshow(phi, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.savefig(os.path.join(HERE, "multiscale_gff.png"), dpi=220)
    plt.close(fig)
    print("wrote multiscale_gff.png")


def fig_polchinski_0d():
    rng = np.random.default_rng(5)
    lam, mu, T, nt = 0.5, 1.5, 5.0, 250
    VT = lambda x: lam * x**4 - mu * x**2
    q = lambda t: 1 - np.exp(-t)
    qd = lambda t: np.exp(-t)
    xg = np.linspace(-4, 4, 801)
    dx = xg[1] - xg[0]
    tg = np.linspace(0, T, nt + 1)
    F = np.empty((nt + 1, xg.size))
    for i, t in enumerate(tg):
        sres = q(T) - q(t)
        if sres < 1e-6:
            V = VT(xg)
        else:
            z = np.linspace(-7 * np.sqrt(sres), 7 * np.sqrt(sres), 701)
            w = np.exp(-z**2 / (2 * sres))
            E = (np.exp(-VT(xg[:, None] + z[None, :])) * w[None, :]).sum(axis=1)
            V = -np.log(E)
        F[i] = -np.gradient(V, dx)
    npaths = 20000
    phi = np.zeros(npaths)
    keep = np.zeros((nt + 1, 40))
    dt = T / nt
    for i in range(nt):
        f = np.interp(phi, xg, F[i])
        phi = phi + qd(tg[i]) * f * dt + np.sqrt(qd(tg[i]) * dt) * rng.standard_normal(npaths)
        keep[i + 1] = phi[:40]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
    for i, col in zip([0, nt // 5, nt // 2, nt], [GREY, NAVY, ACCENT, "k"]):
        axs[0].plot(xg, F[i], color=col, lw=1.0, label=rf"$t={tg[i]:.1f}$")
    axs[0].set_xlim(-2.5, 2.5), axs[0].set_ylim(-6, 6)
    axs[0].set_xlabel("$x$"), axs[0].set_title(r"force $F_t(x)=-V_t'(x)$", fontsize=9)
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].plot(tg, keep, color=NAVY, lw=0.4, alpha=0.7)
    axs[1].set_xlabel("resolution $t$"), axs[1].set_title(r"paths of $\Phi_t$", fontsize=9)
    dens = np.exp(-VT(xg) - xg**2 / (2 * q(T)))
    dens /= dens.sum() * dx
    axs[2].hist(phi, bins=80, density=True, color=NAVY, alpha=0.5)
    axs[2].plot(xg, dens, color=ACCENT, lw=1.2)
    axs[2].set_xlim(-2.5, 2.5)
    axs[2].set_xlabel("$x$"), axs[2].set_title(r"law of $\Phi_T$ vs $\nu_T$", fontsize=9)
    save(fig, "polchinski_0d.pdf")


def fig_regulator(sym="A", name="exp_regulator.pdf"):
    k = np.linspace(0, 12, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for t, col in zip([1, 2, 4], [GREY, NAVY, ACCENT]):
        A = k**2 + 1
        ax.plot(k, A / np.expm1(A / t**2), color=col, lw=1.2, label=rf"$t={t}$")
        ax.axhline(t**2, color=col, lw=0.5, ls=":")
    ax.set_xlabel(r"$|k|$ ($m=1$)")
    ax.set_title(rf"$\mathscr{{R}}_t(k)={sym}/(e^{{{sym}/t^2}}-1)$, ${sym}=|k|^2+m^2$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, name)


# ------------------------------------------------- L3: illustrations
# Palette of the "stochastic observables" talk: navy (field), teal (noise,
# conditional law), gold (force, forecast).
SNAVY, TEAL, GOLD, CREAM = "#1B2A49", "#1F7A72", "#B8862B", "#FBF7EE"
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

CM_FIELD = LinearSegmentedColormap.from_list("field", [TEAL, CREAM, SNAVY])
CM_FORCE = LinearSegmentedColormap.from_list("force", ["#5B6B8C", CREAM, GOLD])


def _frame(ax, col=SNAVY, lw=0.8):
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True), sp.set_color(col), sp.set_linewidth(lw)


def fig_fbsde_flow():
    """Forward field / backward force, on T^2 (force to first order in lambda)."""
    rng = np.random.default_rng(11)
    N, m, lam = 256, 2.0, 1.0
    noise = np.fft.fft2(rng.standard_normal((N, N)))
    ts = [10, 30, 100, 400]
    labels = [r"$t_1$", r"$t_2$", r"$t_3$", r"$T$"]
    fields, forces = [], []
    for t in ts:
        prof = lambda k2, m, t=t: np.exp(-(k2 + m**2) / t**2) / (k2 + m**2)
        phi, _ = gaussian_field_2d(N, m, prof, noise=noise)
        c = phi.var()
        phi, c = phi / np.sqrt(c), 1.0  # display units
        fields.append(phi)
        forces.append(-lam / 3 * (phi**3 - 3 * c * phi))  # -(lam/3) H_3(X_t; c_t)
    fig = plt.figure(figsize=(6.6, 4.25))
    fig.patch.set_facecolor("white")
    W, H, x0, gap, yF, yB = 0.15, 0.225, 0.03, 0.08, 0.43, 0.10
    for i, (phi, f) in enumerate(zip(fields, forces)):
        x = x0 + i * (W + gap)
        lab = labels[i].strip("$")
        ax = fig.add_axes([x, yF, W, H])
        ax.imshow(phi, cmap=CM_FIELD, vmin=-3, vmax=3, interpolation="bilinear")
        _frame(ax, SNAVY)
        ax.set_title(r"$\Phi_{%s}$" % lab, fontsize=10, color=SNAVY, pad=3)
        ax = fig.add_axes([x, yB, W, H])
        v = np.quantile(np.abs(f), 0.97)
        ax.imshow(f, cmap=CM_FORCE, vmin=-v, vmax=v, interpolation="bilinear")
        _frame(ax, GOLD, 1.4 if i == 3 else 0.8)
        ax.set_xlabel(r"$F_{%s}$" % lab, fontsize=10, color=GOLD, labelpad=3)
        # vertical link: the field at resolution t gives the force at t
        xc = x + W / 2
        fig.add_artist(FancyArrowPatch((xc, yF - 0.012), (xc, yB + H + 0.012),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
                       color=GOLD, lw=1.0))
        fig.text(xc + 0.012, (yF + yB + H) / 2, r"$-DV_{%s}$" % lab, color=GOLD,
                 fontsize=8.5, ha="left", va="center")
        if i < 3:
            xm = x + W + gap / 2
            fig.text(xm, yF + H / 2 + 0.03, "+", ha="center", va="center",
                     fontsize=14, color=TEAL)
            fig.text(xm, yF + H / 2 - 0.03, r"$\dot Q^{1/2}\mathrm{d}W$", ha="center",
                     va="center", fontsize=7.5, color=TEAL)
            fig.add_artist(FancyArrowPatch((x + W + gap - 0.008, yB + H / 2),
                           (x + W + 0.008, yB + H / 2), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=9, color=GOLD, lw=1.0))
    xT = x0 + 3 * (W + gap) + W
    # forward and reversed dynamics of the field
    fig.text((x0 + xT) / 2, 0.915,
             r"forward: $\mathrm{d}\Phi_t=\dot Q_tF_t\,\mathrm{d}t+\dot Q_t^{1/2}\mathrm{d}W_t$"
             "  (finer modes are revealed)", ha="center", va="center", fontsize=9, color=SNAVY)
    fig.add_artist(FancyArrowPatch((x0, 0.875), (xT, 0.875), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=14, color=SNAVY, lw=2.0))
    fig.add_artist(FancyArrowPatch((xT, 0.82), (x0, 0.82), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=12, color=SNAVY, lw=1.2,
                   linestyle=(0, (4, 2.5))))
    fig.text((x0 + xT) / 2, 0.78,
             r"reversed, $\hat\Phi_s=\Phi_{T-s}$: $\mathrm{d}\hat\Phi_s=-\dot Q_{T-s}Q_{T-s}^{-1}"
             r"\hat\Phi_s\,\mathrm{d}s+\dot Q_{T-s}^{1/2}\mathrm{d}\hat W_s$"
             "  (linear: Gaussian blurring of $\\Phi_T\\sim\\nu_T$)",
             ha="center", va="center", fontsize=8.5, color=SNAVY)
    fig.add_artist(FancyArrowPatch((xT, 0.045), (x0, 0.045), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=14, color=TEAL, lw=2.0))
    fig.text((x0 + xT) / 2, 0.0,
             r"backward: $F_t=-DV_t(\Phi_t)=\mathbb{E}_t[-DV_T(\Phi_T)]$  (effective force at each scale)",
             ha="center", fontsize=9, color=TEAL)
    fig.savefig(os.path.join(HERE, "fbsde_flow.png"), dpi=220)
    plt.close(fig)
    print("wrote fbsde_flow.png")


def fig_localisation():
    """Two-component field: forecast, conditional covariance, localisation."""
    rng = np.random.default_rng(3)
    T = 4.0
    q = lambda t: 1 - np.exp(-t)
    VT = lambda x: 1.4 * (x[..., 0]**2 - 1)**2 + 1.2 * (x[..., 1] - 0.7 * x[..., 0])**2
    def DVT(x):
        a = x[..., 1] - 0.7 * x[..., 0]
        return np.stack([5.6 * x[..., 0] * (x[..., 0]**2 - 1) - 1.68 * a, 2.4 * a], -1)
    gq = np.linspace(-3.5, 3.5, 351)
    Y = np.stack(np.meshgrid(gq, gq, indexing="ij"), -1).reshape(-1, 2)
    VY, DY = VT(Y), DVT(Y)

    def cond(x, S):
        """Mean, covariance and force of mu_t^x (grid quadrature)."""
        if S < 0.01:
            return x - S * DVT(x), S * np.eye(2), -DVT(x)
        e = VY + ((Y - x)**2).sum(1) / (2 * S)
        w = np.exp(-(e - e.min()))
        w /= w.sum()
        mean = (w[:, None] * Y).sum(0)
        d = Y - mean
        cov = np.einsum("n,ni,nj->ij", w, d, d)
        F = -(w[:, None] * DY).sum(0)
        return mean, cov, F

    nt, ts_show = 800, [0.12, 0.6, 1.6]
    dt = T / nt
    tg = np.linspace(0, T, nt + 1)
    phi = np.zeros((nt + 1, 2))
    for i in range(nt):
        S = q(T) - q(tg[i])
        _, _, F = cond(phi[i], S)
        qd = np.exp(-tg[i])
        phi[i + 1] = phi[i] + qd * F * dt + np.sqrt(qd * dt) * rng.standard_normal(2)
    g = np.linspace(-2.2, 2.2, 300)
    G1, G2 = np.meshgrid(g, g)
    X = np.stack([G1, G2], -1)
    dens = np.exp(-VT(X) - (G1**2 + G2**2) / (2 * q(T)))
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.45))
    for ax, ts in zip(axs, ts_show):
        i = int(round(ts / dt))
        S = q(T) - q(tg[i])
        mean, cov, F = cond(phi[i], S)
        assert np.allclose(mean, phi[i] + S * F, atol=2e-3)  # psi = Phi + S F
        ax.set_facecolor(CREAM)
        ax.contour(G1, G2, dens, levels=6, colors=GOLD, linewidths=0.35, alpha=0.6)
        # conditional cloud: importance resampling
        y = phi[i] + np.sqrt(S) * rng.standard_normal((6000, 2))
        w = np.exp(-(VT(y) - VT(y).min())); w /= w.sum()
        pick = rng.choice(6000, 220, p=w)
        ax.scatter(y[pick, 0], y[pick, 1], s=2.5, color=TEAL, alpha=0.45, lw=0)
        ev, evec = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        ax.add_patch(Ellipse(mean, 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=ang,
                             fc=TEAL, alpha=0.12, ec=TEAL, lw=0.8))
        ax.plot(phi[: i + 1, 0], phi[: i + 1, 1], color=SNAVY, lw=0.7, alpha=0.8)
        ax.annotate("", xy=phi[-1], xytext=phi[i], arrowprops=dict(
            arrowstyle="-|>", ls="--", lw=0.7, color=SNAVY, shrinkA=2, shrinkB=3))
        ax.annotate("", xy=mean, xytext=phi[i], arrowprops=dict(
            arrowstyle="-|>", lw=1.4, color=GOLD, shrinkA=2, shrinkB=2))
        ax.plot(*phi[i], "o", ms=4.5, color=SNAVY, zorder=5)
        ax.plot(*mean, "o", ms=5, color=GOLD, mec="white", mew=0.8, zorder=6)
        ax.plot(*phi[-1], "o", ms=4.5, mfc="white", mec=SNAVY, mew=1.0, zorder=5)
        ax.set_xlim(-2.2, 2.2), ax.set_ylim(-2.2, 2.2), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color("#D9CFB8")
        ax.set_title(rf"$t={ts}$", fontsize=9, color=SNAVY)
    axs[0].text(-2.05, 1.95, r"$\Phi_t$", color=SNAVY, fontsize=8, va="top")
    axs[0].text(-2.05, 1.6, r"$\psi_t=\Phi_t+S_tF_t$", color=GOLD, fontsize=8, va="top")
    axs[0].text(-2.05, 1.25, r"$\mu_t^{\Phi_t}$, $\mathrm{Cov}_t$", color=TEAL, fontsize=8, va="top")
    axs[0].text(-2.05, 0.9, r"$\circ\ \Phi_T$", color=SNAVY, fontsize=8, va="top")
    fig.tight_layout(w_pad=0.6)
    save(fig, "localisation.pdf")


def fig_shear():
    """Zero dimensions: the shear maps the graph of -V_t' to the graph of -Gamma_t'."""
    lam, a, T, t = 0.5, 1.5, 5.0, 1.2
    VT = lambda x: lam * x**4 - a * x**2
    q = lambda s: 1 - np.exp(-s)
    S = q(T) - q(t)
    xg = np.linspace(-3.5, 3.5, 1401)
    dx = xg[1] - xg[0]
    z = np.linspace(-7 * np.sqrt(S), 7 * np.sqrt(S), 801)
    w = np.exp(-z**2 / (2 * S))
    V = -np.log((np.exp(-VT(xg[:, None] + z[None, :])) * w[None, :]).sum(1))
    F = -np.gradient(V, dx)
    psi = xg + S * F
    assert np.all(np.diff(psi) > 0)  # graph over psi
    Gam = V - 0.5 * S * F**2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax in (a1, a2):
        ax.set_facecolor(CREAM)
    sel = (np.abs(xg) < 2.2)
    a1.plot(xg[sel], F[sel], color=SNAVY, lw=1.4, label=r"$p=F_t(x)=-V_t'(x)$")
    sel2 = (np.abs(psi) < 2.2)
    a1.plot(psi[sel2], F[sel2], color=GOLD, lw=1.4, label=r"$p=-\Gamma_t'(\psi)$")
    for x0 in np.linspace(-1.8, 1.8, 13):
        j = np.argmin(np.abs(xg - x0))
        a1.annotate("", xy=(psi[j], F[j]), xytext=(xg[j], F[j]), arrowprops=dict(
            arrowstyle="-|>", lw=0.6, color=TEAL, shrinkA=0, shrinkB=0, mutation_scale=6))
    a1.axhline(0, color="#CFC6B0", lw=0.5), a1.axvline(0, color="#CFC6B0", lw=0.5)
    a1.set_xlim(-2.2, 2.2), a1.set_ylim(-3.2, 3.2)
    a1.set_xlabel(r"field $x$ / forecast $\psi$"), a1.set_ylabel(r"force $p$")
    a1.set_title(r"shear $\Sigma_t:(x,p)\mapsto(x+S_tp,\,p)$", fontsize=9)
    a1.legend(frameon=False, fontsize=7, loc="upper right")
    sh = lambda y: y - y[np.abs(xg) < 2].min()
    a2.plot(xg, sh(VT(xg)), color="#8A8F98", lw=1.0, ls=":", label=r"$V_T$")
    a2.plot(xg, sh(V), color=SNAVY, lw=1.4, label=r"$V_t$ (Wilson)")
    a2.plot(psi, Gam - Gam[np.abs(psi) < 2].min(), color=GOLD, lw=1.4,
            label=r"$\Gamma_t$ (Wetterich)")
    a2.set_xlim(-2.2, 2.2), a2.set_ylim(-0.3, 3.0)
    a2.set_xlabel(r"$x$, $\psi$")
    a2.set_title(rf"potentials at $t={t}$ ($T={T:g}$)", fontsize=9)
    a2.legend(frameon=False, fontsize=7, loc="upper center")
    fig.tight_layout(w_pad=1.5)
    save(fig, "shear.pdf")


def fig_locality():
    """Covariation of point observables: overlap of the half-kernels."""
    r = 0.5
    xs, ys = np.array([-r / 2, 0.0]), np.array([r / 2, 0.0])
    g = np.linspace(-1.1, 1.1, 400)
    G1, G2 = np.meshgrid(g, g)
    fig = plt.figure(figsize=(6.6, 2.1))
    tl = [2.5, 4.5, 8.0]
    for i, t in enumerate(tl):
        ax = fig.add_axes([0.01 + i * 0.205, 0.08, 0.19, 0.78])
        kx = np.exp(-t**2 * ((G1 - xs[0])**2 + (G2 - xs[1])**2) / 2)
        ky = np.exp(-t**2 * ((G1 - ys[0])**2 + (G2 - ys[1])**2) / 2)
        img = np.ones(G1.shape + (3,)) * np.array(to_rgb(CREAM))
        for k, col in ((kx, SNAVY), (ky, TEAL)):
            al = 0.75 * k[..., None]
            img = img * (1 - al) + al * np.array(to_rgb(col))
        ov = (kx * ky / (kx * ky).max())[..., None] * np.exp(-0.0)
        al = 0.9 * ov * min(1, np.exp(-t**2 * r**2 / 4) * 4)
        img = img * (1 - al) + al * np.array(to_rgb(GOLD))
        ax.imshow(img, extent=[g[0], g[-1], g[0], g[-1]], origin="lower")
        ax.plot(*xs, "o", ms=3, color="white"), ax.plot(*ys, "o", ms=3, color="white")
        _frame(ax, "#D9CFB8")
        ax.set_title(rf"$s={t:g}/|x-y|$" if False else rf"scale $1/s={1/t:.2f}$", fontsize=8)
    ax = fig.add_axes([0.70, 0.22, 0.28, 0.62])
    s = np.linspace(0.05, 14, 600)
    m = 1.0
    for rr, col in ((0.25, GOLD), (0.5, TEAL), (1.0, SNAVY)):
        qd = 2 / s * np.exp(-m**2 / s**2) / (4 * np.pi) * np.exp(-s**2 * rr**2 / 4)
        ax.plot(s, qd, color=col, lw=1.2, label=rf"$|x-y|={rr:g}$")
    ax.set_xlabel(r"resolution $s$")
    ax.set_title(r"$\mathrm{d}\langle \mathbf{X}(x),\mathbf{X}(y)\rangle_s/\mathrm{d}s=\dot Q_s(x-y)$",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_yticks([])
    fig.savefig(os.path.join(HERE, "locality.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote locality.png")


def fig_brownian():
    """Brownian observables: Hermite martingale B^2-t and exponential martingale."""
    rng = np.random.default_rng(5)
    n, Tm, M = 2000, 10.0, 40
    t = np.linspace(0, Tm, n + 1)
    B = np.concatenate([np.zeros((M, 1)),
                        np.cumsum(rng.standard_normal((M, n)) * np.sqrt(Tm / n), 1)], 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for b in B:
        a1.plot(t, b**2 - t, color=SNAVY, lw=0.5, alpha=0.35)
    a1.fill_between(t, -np.sqrt(2) * t, np.sqrt(2) * t, color=TEAL, alpha=0.10, lw=0)
    a1.plot(t, 0 * t, color=TEAL, lw=1.2, label=r"mean $0$")
    a1.plot(t, np.sqrt(2) * t, color=TEAL, lw=0.8, ls=":", label=r"$\pm$ sd $=\sqrt{2}\,t$")
    a1.plot(t, -np.sqrt(2) * t, color=TEAL, lw=0.8, ls=":")
    a1.set_ylim(-14, 30), a1.set_xlim(0, Tm)
    a1.set_xlabel(r"$t$"), a1.set_title(r"$B_t^2-t$: no terminal value")
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    lam = 1.0
    Z = np.exp(lam * B - lam**2 * t / 2)
    for z in Z:
        a2.plot(t, z, color=SNAVY, lw=0.5, alpha=0.35)
    a2.set_yscale("log")
    a2.plot(t, 0 * t + 1, color=TEAL, lw=1.2, label=r"$\mathbb{E}Z_t=1$")
    a2.plot(t, np.exp(-lam**2 * t / 2), color=GOLD, lw=1.2, ls="--",
            label=r"median $\mathrm{e}^{-t/2}$")
    a2.set_ylim(1e-6, 1e2), a2.set_xlim(0, Tm)
    a2.set_xlabel(r"$t$"), a2.set_title(r"$Z_t=\mathrm{e}^{B_t-t/2}\to0$ a.s.")
    a2.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "brownian_observables.pdf")


if __name__ == "__main__":
    fig_ou()
    fig_gff2d()
    fig_markov()
    fig_green()
    fig_wick_constant()
    fig_she()
    fig_multiscale()
    fig_polchinski_0d()
    fig_regulator()
    fig_fbsde_flow()
    fig_localisation()
    fig_shear()
    fig_locality()
    fig_brownian()
    fig_regulator("L", "exp_regulator_L.pdf")  # talk-observables-fbsde
