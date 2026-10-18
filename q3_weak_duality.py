import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (AQUA, BLUE, INK, INK2, MU_STAR, ORANGE, P_STAR,
                    dual, lagrangian, mark_point, save, setup_style)

ENVELOPE = "#d6d5ce"
LINE_GREY = "#b9b8b0"


def print_table_q3a():
    print("\nQ3(a) weak duality check, p* = 1")
    print("  mu   x(mu)   d(mu)    p* - d(mu)   d(mu) <= p* ?")
    for mu in [0, 1, 2, 3]:
        d = dual(mu)
        print(f"  {mu:<4} {mu / 2:<7.2f} {d:<8.2f} {P_STAR - d:<12.2f} {d <= P_STAR}")


def fig_q3a():
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    mu = np.linspace(0, 4.5, 400)

    ax.fill_between(mu, dual(mu), P_STAR, color=BLUE, alpha=0.08, lw=0, zorder=1)
    ax.axhline(P_STAR, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.plot(mu, dual(mu), color=BLUE, zorder=3)

    for m in [0, 1, 3]:
        ax.plot([m, m], [dual(m), P_STAR], color=INK2, lw=0.9, zorder=2)
        mark_point(ax, m, dual(m), color=BLUE, size=6.5)
        side = -1 if m == 1 else 1
        ax.text(m + 0.07 * side, (dual(m) + P_STAR) / 2, f"gap {P_STAR - dual(m):g}",
                va="center", ha="right" if side < 0 else "left", fontsize=8.5)
    mark_point(ax, MU_STAR, dual(MU_STAR))
    ax.annotate("gap 0", xy=(MU_STAR, 1), xytext=(MU_STAR, 1.28), ha="center",
                fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))

    handles = [Line2D([], [], color=BLUE, label=r"dual function $d(\mu)$"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)),
                      label=r"optimal value $p^{\ast} = 1$"),
               Patch(color=BLUE, alpha=0.08, label=r"gap $p^{\ast} - d(\mu)$")]
    ax.legend(handles=handles, loc="lower left", borderaxespad=0.3)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("value")
    save(fig, "q3a_weak_duality")


def envelope_legend(ax, extra=None):
    handles = [Line2D([], [], color=ENVELOPE, lw=6, label="lower envelope"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)),
                      label=r"true $d(\mu)$")]
    if extra is not None:
        handles.insert(0, extra)
    ax.legend(handles=handles, loc="lower left", borderaxespad=0.2,
              handlelength=1.8)


def fig_q3b_three():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))
    mu = np.linspace(0, 5, 501)
    lines = [(0, BLUE), (1, ORANGE), (2, AQUA)]
    values = [lagrangian(mu, x0) * np.ones_like(mu) for x0, _ in lines]
    env = np.minimum.reduce(values)

    ax.plot(mu, env, color=ENVELOPE, lw=6, solid_capstyle="round", zorder=1)
    for (x0, c), y in zip(lines, values):
        ax.plot(mu, y, color=c, lw=1.4, zorder=3)
    ax.plot(mu, dual(mu), color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    for m in [0, 2, 4]:
        ax.plot(m, dual(m), "o", ms=4.5, mfc=INK, mec="white", mew=1.0, zorder=5)

    ax.text(4.97, 3.0, r"$x=0$: $L=\mu$", ha="right", va="center", fontsize=8.5)
    ax.text(4.95, 1.12, r"$x=1$: $L=1$", ha="right", va="bottom", fontsize=8.5)
    ax.text(0.1, 4.3, r"$x=2$: $L=4-\mu$", ha="left", va="center", fontsize=8.5)

    envelope_legend(ax)
    ax.set_xlim(0, 5)
    ax.set_ylim(-2.0, 4.6)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu, x)$")
    save(fig, "q3b_envelope_three")


def fig_q3b_many():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))
    mu = np.linspace(0, 5, 501)
    xs = np.arange(-1, 3.001, 0.25)
    values = [lagrangian(mu, x0) for x0 in xs]
    env = np.minimum.reduce(values)

    for y in values:
        ax.plot(mu, y, color=LINE_GREY, lw=0.6, zorder=2)
    ax.plot(mu, env, color=ENVELOPE, lw=6, solid_capstyle="round", zorder=1)
    ax.plot(mu, dual(mu), color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)

    extra = Line2D([], [], color=LINE_GREY, lw=0.8, label=r"17 lines, $x=-1,\ldots,3$")
    envelope_legend(ax, extra)
    ax.set_xlim(0, 5)
    ax.set_ylim(-2.0, 4.6)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu, x)$")
    save(fig, "q3b_envelope_many")


def numeric_check():
    mu = np.linspace(0, 10, 10001)
    gap = P_STAR - dual(mu)
    print("\nQ3 numerical check")
    print(f"  smallest gap p* - d(mu) on mu in [0, 10] = {gap.min():.2e} "
          f"at mu = {mu[np.argmin(gap)]:.3f}")
    print(f"  max |(p* - d(mu)) - (1 - mu/2)^2| = {np.max(np.abs(gap - (1 - mu / 2) ** 2)):.1e}")

    env3 = np.minimum.reduce([lagrangian(mu, x0) for x0 in [0, 1, 2]])
    xs = np.arange(-1, 3.001, 0.25)
    env17 = np.minimum.reduce([lagrangian(mu, x0) for x0 in xs])
    sec = lambda y: np.diff(y, 2)
    m = mu <= 5
    print(f"  min of (3-line envelope - d)  = {np.min(env3 - dual(mu)):.2e}  (>= 0: envelope is above d)")
    print(f"  max of (17-line envelope - d) on [0, 5] = {np.max((env17 - dual(mu))[m]):.4f}")
    tol = 1e-12
    print(f"  d(mu) concave (all second differences <= 0)? {sec(dual(mu)).max() <= tol}")
    print(f"  3-line envelope concave?                     {sec(env3).max() <= tol}")


if __name__ == "__main__":
    setup_style()
    print_table_q3a()
    print("\nMaking Q3 figures")
    fig_q3a()
    fig_q3b_three()
    fig_q3b_many()
    numeric_check()
