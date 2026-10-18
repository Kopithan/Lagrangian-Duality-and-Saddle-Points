import numpy as np
import matplotlib.pyplot as plt

from common import (AXIS, BLUE, BLUE_RAMP, INK, INK2, MU_STAR, MUTED, P_STAR,
                    WASH, X_STAR, dual, lagrangian, mark_point, save,
                    setup_style, x_of_mu)

MU_VALUES = [0, 1, 2, 3]


def print_table_q1a():
    print("\nQ1(a) minimizer of L(mu, x) for each mu")
    print("  mu   x(mu)   L(mu, x(mu))   feasible (x >= 1)?")
    for mu in MU_VALUES:
        xm = x_of_mu(mu)
        status = "yes" if xm >= 1 else "no"
        print(f"  {mu:<4} {xm:<7.2f} {lagrangian(mu, xm):<14.2f} {status}")


def fig_q1a():
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    x = np.linspace(-1, 3, 500)

    ax.axvspan(1, 3, color=WASH, zorder=0, lw=0)
    ax.text(2.95, -0.22, r"feasible region ($x \geq 1$)", ha="right",
            va="bottom", color=INK2, fontsize=8.5)

    xp = np.linspace(0, 1.8, 200)
    ax.plot(xp, 2 * xp - xp ** 2, color=MUTED, lw=1.1, ls=(0, (1, 2)),
            zorder=2, label=r"minimum points, $y = 2x - x^2$")

    for mu, c in zip(MU_VALUES, BLUE_RAMP):
        ax.plot(x, lagrangian(mu, x), color=c, label=rf"$\mu = {mu}$", zorder=3)
        xm = x_of_mu(mu)
        mark_point(ax, xm, lagrangian(mu, xm), color=c, size=6.5)

    ax.annotate(r"$x(\mu)=\mu/2$ moves right", xy=(1.5, 0.75), xytext=(2.0, 0.42),
                color=INK, fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                shrinkA=2, shrinkB=5))

    ax.set_xlim(-1, 3)
    ax.set_ylim(-0.4, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L(\mu, x)$")
    ax.legend(loc="upper center", ncol=1, borderaxespad=0.2)
    save(fig, "q1a_lagrangian_vs_x")


def fig_q1b():
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    mu = np.linspace(0, 5, 400)

    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.plot(mu, dual(mu), color=BLUE, zorder=3)
    ax.plot([MU_STAR, MU_STAR], [-1.5, 1], color=MUTED, lw=0.9, ls=(0, (1, 2)))
    ax.plot([0, MU_STAR], [1, 1], color=MUTED, lw=0.9, ls=(0, (1, 2)))
    mark_point(ax, MU_STAR, dual(MU_STAR))

    ax.annotate(r"maximum $(\mu^{\ast}, d^{\ast}) = (2,\,1)$", xy=(MU_STAR, 1),
                xytext=(2.55, 1.28), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=5))
    ax.text(4.15, -0.55, r"$d(\mu) = \mu - \mu^2/4$", color=INK, fontsize=9.5,
            ha="right")

    ax.set_xlim(0, 5)
    ax.set_ylim(-1.5, 1.6)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$d(\mu)$")
    save(fig, "q1b_dual_function")


def fig_q1c_capped():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))
    x = np.linspace(-1, 3, 500)

    ax.axvspan(-1, 1, color=WASH, zorder=0, lw=0)
    ax.text(-0.3, 11.6, "infeasible", ha="center", va="top", color=INK2, fontsize=8)

    for M, c in zip([0, 2, 5, 20], BLUE_RAMP):
        gM = x ** 2 + M * np.maximum(0, 1 - x)
        ax.plot(x, gM, color=c, lw=1.5, label=rf"$M = {M}$")

    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 12)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$g_M(x)$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.53, 1.0), borderaxespad=0.1,
              handlelength=1.4)
    save(fig, "q1c_primal_capped")


def fig_q1c_true():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))

    ax.axvspan(-1, 1, color=WASH, zorder=0, lw=0)
    for xa in [-0.55, 0.0, 0.55]:
        ax.annotate("", xy=(xa, 11.6), xytext=(xa, 7.6),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    ax.text(0, 6.4, r"$g(x) = +\infty$", ha="center", fontsize=10)
    ax.text(0, 5.1, r"infeasible ($x<1$)", ha="center", color=INK2, fontsize=8)

    x = np.linspace(1, 3, 300)
    ax.plot(x, x ** 2, color=BLUE, zorder=3)
    ax.plot([1, 1], [1, 12], color=MUTED, lw=0.9, ls=(0, (1, 2)))
    ax.text(2.1, 7.0, r"$g(x) = x^2$", fontsize=10, ha="right")
    mark_point(ax, X_STAR, P_STAR)
    ax.annotate(r"min at $(1,\,1)$", xy=(1, 1), xytext=(1.45, 0.25),
                fontsize=8.5, va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))

    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 12)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$g(x)$")
    save(fig, "q1c_primal_function")


def numeric_check():
    x = np.linspace(-5, 5, 100001)          # step 1e-4
    mus = np.linspace(0, 6, 601)
    d_grid = np.array([lagrangian(m, x).min() for m in mus])
    err = np.max(np.abs(d_grid - dual(mus)))
    mu_best = mus[np.argmax(d_grid)]

    feasible = x[x >= 1]
    p_grid = (feasible ** 2).min()
    print("\nQ1 numerical check")
    print(f"  max |grid min of L - (mu - mu^2/4)| = {err:.1e}")
    print(f"  grid argmax of d(mu) = {mu_best:.3f},  max value = {d_grid.max():.6f}")
    print(f"  grid min of x^2 over x >= 1 = {p_grid:.6f}  (p* = 1)")


if __name__ == "__main__":
    setup_style()
    print_table_q1a()
    print("\nMaking Q1 figures")
    fig_q1a()
    fig_q1b()
    fig_q1c_capped()
    fig_q1c_true()
    numeric_check()
