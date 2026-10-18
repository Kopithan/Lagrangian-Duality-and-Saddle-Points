import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from common import (AQUA, BLUE, INK, INK2, MU_STAR, MUTED, ORANGE, X_STAR,
                    dual, lagrangian, mark_point, save, setup_style)


def fig_L_vs_mu():
    fig, ax = plt.subplots(figsize=(2.95, 2.4))
    mu = np.linspace(0, 5, 200)
    ax.plot(mu, lagrangian(mu, X_STAR) * np.ones_like(mu), color=BLUE)
    mark_point(ax, MU_STAR, lagrangian(MU_STAR, X_STAR))
    ax.text(2.5, 2.35, r"$L(\mu, 1) = 1$ for every $\mu$", ha="center", fontsize=9)
    ax.annotate(r"$L(\mu^{\ast}, x^{\ast}) = 1$", xy=(MU_STAR, 1),
                xytext=(2.6, 0.3), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5.2)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu, x^{\ast})$")
    save(fig, "q2a_L_vs_mu")


def fig_L_vs_x():
    fig, ax = plt.subplots(figsize=(2.95, 2.4))
    x = np.linspace(-1, 3, 300)
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.plot(x, lagrangian(MU_STAR, x), color=BLUE, zorder=3)
    mark_point(ax, X_STAR, lagrangian(MU_STAR, X_STAR))
    ax.text(2.45, 3.35, r"$(x-1)^2 + 1$", ha="right", fontsize=9.5)
    ax.annotate(r"$L(\mu^{\ast}, x^{\ast}) = 1$", xy=(X_STAR, 1),
                xytext=(1.45, 0.3), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 5.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$L(\mu^{\ast}, x)$")
    save(fig, "q2a_L_vs_x")


def fig_saddle_contour():
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    x = np.linspace(-1, 3, 401)
    mu = np.linspace(0, 5, 401)
    X, MU = np.meshgrid(x, mu)
    Z = lagrangian(MU, X)

    cmap = LinearSegmentedColormap.from_list("div", ["#86b6ef", "#f7f6f3", "#f0a3a2"])
    norm = TwoSlopeNorm(vmin=-1, vcenter=1, vmax=11)
    cf = ax.contourf(X, MU, Z, levels=np.linspace(-1, 11, 49), cmap=cmap, norm=norm)
    try:
        cf.set_rasterized(True)
    except AttributeError:
        for coll in cf.collections:
            coll.set_rasterized(True)
    cs = ax.contour(X, MU, Z, levels=[0, 2, 4, 6, 8], colors=MUTED,
                    linewidths=0.6)
    spots = [(2.0, 4.0), (-0.5, 1.17), (2.5, 2.83), (-0.5, 2.5), (2.5, 1.5),
             (-0.7, 3.24), (2.7, 0.76), (-0.85, 3.93), (2.85, 0.4)]
    ax.clabel(cs, fmt="%g", fontsize=7, inline_spacing=2, manual=spots)

    ax.plot([1, 1], [0, 5], color=INK, lw=1.7, zorder=4)
    ax.plot([-1, 3], [0, 4], color=INK, lw=1.7, zorder=4)
    ax.plot([-1, 3], [MU_STAR, MU_STAR], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    mark_point(ax, X_STAR, MU_STAR, size=8)

    for (xt, mt, s) in [(0.3, 0.55, r"$L<1$"), (-0.35, 3.9, r"$L>1$"),
                        (2.35, 1.0, r"$L>1$"), (1.6, 4.45, r"$L<1$")]:
        ax.text(xt, mt, s, ha="center", va="center", fontsize=10)

    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\mu$")
    ax.grid(False)

    handles = [
        Line2D([], [], color=INK, lw=1.7, label=r"level set $L = 1$:" "\n"
               r"$x = 1$ and $\mu = x + 1$"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 3)),
               label=r"slice $\mu = \mu^{\ast} = 2$"),
        Line2D([], [], marker="o", ls="none", ms=8, mfc=ORANGE, mec="white",
               mew=1.5, label=r"saddle point $(1,\,2)$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0),
              borderaxespad=0, labelspacing=1.0)
    save(fig, "q2a_saddle_contour")


def dual_modified(mu):
    return -mu - mu ** 2 / 4


def fig_constraint_compare():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))
    x = np.linspace(-2, 3, 400)
    ax.plot(x, x ** 2, color=INK, lw=1.5, zorder=3)
    ax.text(-1.0, 4.25, r"$f(x)=x^2$", fontsize=9.5, ha="center")

    ax.plot([1, 3], [-0.9, -0.9], color=BLUE, lw=4, solid_capstyle="butt")
    ax.plot([-1, 3], [-1.9, -1.9], color=AQUA, lw=4, solid_capstyle="butt")
    ax.text(2.95, -0.75, r"feasible: $x \geq 1$", ha="right", va="bottom", fontsize=8)
    ax.text(2.95, -1.75, r"feasible: $x \geq -1$", ha="right", va="bottom", fontsize=8)
    ax.plot([1, 1], [-0.9, 1], color=BLUE, lw=0.9, ls=(0, (1, 2)))
    ax.plot([-1, -1], [-1.9, 1], color=AQUA, lw=0.9, ls=(0, (1, 2)))

    mark_point(ax, 1, 1)
    mark_point(ax, 0, 0)
    ax.annotate("active", xy=(1, 1), xytext=(1.55, 2.3), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    ax.annotate("inactive", xy=(0, 0), xytext=(-0.45, 1.6), fontsize=8.5,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2.5, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.yaxis.set_label_coords(-0.13, 0.667)
    save(fig, "q2c_constraint_compare")


def fig_dual_compare():
    fig, ax = plt.subplots(figsize=(2.95, 2.6))
    mu = np.linspace(0, 4, 300)
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
    ax.plot(mu, dual(mu), color=BLUE, label=r"$x \geq 1$", zorder=3)
    ax.plot(mu, dual_modified(mu), color=AQUA, label=r"$x \geq -1$", zorder=3)
    mark_point(ax, 2, 1)
    mark_point(ax, 0, 0)
    ax.annotate(r"$\mu^{\ast} = 2$", xy=(2, 1), xytext=(2.45, 1.25), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    ax.annotate(r"$\mu^{\ast} = 0$", xy=(0, 0), xytext=(1.1, -0.45), fontsize=9,
                va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=4))
    ax.set_xlim(0, 4)
    ax.set_ylim(-3, 1.7)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$d(\mu)$")
    ax.legend(loc="lower right", borderaxespad=0.2, handlelength=1.5)
    save(fig, "q2c_dual_compare")


# ----------------------------------------------------------------------
def numeric_check():
    mus = np.linspace(0, 50, 50001)
    xs = np.linspace(-10, 10, 200001)
    L_star = lagrangian(MU_STAR, X_STAR)
    left_gap = np.max(lagrangian(mus, X_STAR)) - L_star
    right_gap = np.min(lagrangian(MU_STAR, xs)) - L_star
    print("\nQ2 numerical check (original problem)")
    print(f"  L(mu*, x*) = {L_star:.4f}")
    print(f"  max over mu of L(mu, x*) - L(mu*, x*) = {left_gap:.2e}  (should be <= 0)")
    print(f"  min over x  of L(mu*, x) - L(mu*, x*) = {right_gap:.2e}  (should be >= 0)")
    print(f"  complementary slackness mu* h(x*) = {MU_STAR * (1 - X_STAR):.1f}")

    Lm = lambda m, x: x ** 2 + m * (-1 - x)
    d_mod = np.array([Lm(m, xs).min() for m in mus[:4001]])
    print("\nQ2 numerical check (modified problem, x >= -1)")
    print(f"  h(0) = {-1 - 0:.1f}  -> constraint inactive at x = 0")
    print(f"  grid argmax of modified d(mu) = {mus[np.argmax(d_mod)]:.3f}, "
          f"max value = {d_mod.max():.4f}")
    print(f"  max over mu of L(mu, 0) - L(0, 0) = {np.max(Lm(mus, 0.0)) - Lm(0, 0.0):.2e}")
    print(f"  min over x  of L(0, x)  - L(0, 0) = {np.min(Lm(0, xs)) - Lm(0, 0.0):.2e}")


if __name__ == "__main__":
    setup_style()
    print("Making Q2 figures")
    fig_L_vs_mu()
    fig_L_vs_x()
    fig_saddle_contour()
    fig_constraint_compare()
    fig_dual_compare()
    numeric_check()
