"""Build every figure used by docs/项目全景与知识点手册.md.

Run::

    python scripts/build_manual_figures.py

Schematic diagrams (01-04, 09-12, 14) are drawn for teaching. Data-driven
charts (05, 06, 08, 13) are computed from data/processed/freddie_unified.csv,
so nothing in the manual is hand-typed: re-run the pipeline and then this
script and the numbers stay in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
PROCESSED = ROOT / "data" / "processed" / "freddie_unified.csv"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#C9CFD6",
        "axes.labelcolor": "#1F2933",
        "text.color": "#1F2933",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "axes.grid": True,
        "grid.color": "#E8EBEF",
        "grid.linewidth": 0.8,
        "font.size": 10,
    }
)

INK = "#1F2933"
MUT = "#6B7280"
BLUE_D, BLUE, BLUE_L = "#1F4E79", "#3C7EB0", "#DCE9F5"
TEAL_D, TEAL, TEAL_L = "#14574A", "#2A9D8F", "#D8F0EA"
AMBER_D, AMBER, AMBER_L = "#8A5A12", "#C98A2B", "#FBEBD0"
RED_D, RED, RED_L = "#8C2F33", "#C0504D", "#F7DCDC"
GREY_L, LINE = "#F2F4F7", "#C9CFD6"


def _rounded(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.14):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )


def _arrow(ax, p0, p1, color=MUT, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=14,
            linewidth=lw, color=color, linestyle=ls, zorder=1,
            shrinkA=2, shrinkB=2,
        )
    )


def _blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


# --------------------------------------------------------------------------- 01
def fig_pipeline() -> None:
    fig, ax = _blank((9.2, 6.6))
    steps = [
        ("① 数据", "Freddie Mac 逐笔房贷", "331,122 笔 · 7 个 vintage", BLUE_L, BLUE_D),
        ("② 标签", "什么算「违约」", "36 个月内逾期 90 天以上", BLUE_L, BLUE_D),
        ("③ 特征", "只用放款当时就知道的信息", "分箱 + WOE + IV 筛选", BLUE_L, BLUE_D),
        ("④ 模型", "两个模型互相验证", "评分卡（可解释）+ 梯度提升", TEAL_L, TEAL_D),
        ("⑤ 评估与决策", "从数字落到业务动作", "OOT AUC 0.746 · 好坏档差 24 倍", TEAL_L, TEAL_D),
    ]
    h, gap, y0 = 1.42, 0.42, 8.15
    for i, (title, sub, detail, fc, ec) in enumerate(steps):
        y = y0 - i * (h + gap)
        _rounded(ax, 1.0, y, 8.0, h, fc, ec)
        ax.text(1.35, y + h * 0.66, title, fontsize=13, fontweight="bold", color=ec, va="center")
        ax.text(3.55, y + h * 0.68, sub, fontsize=11, color=INK, va="center")
        ax.text(3.55, y + h * 0.30, detail, fontsize=9.5, color=MUT, va="center")
        if i < len(steps) - 1:
            _arrow(ax, (5.0, y - 0.03), (5.0, y - gap + 0.03), color="#9AA3AD")
    fig.savefig(OUT / "01_pipeline.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 02
def fig_join() -> None:
    fig, ax = _blank((9.2, 5.4))

    def table(x, y, w, h, rows, title, sub, fc, ec):
        _rounded(ax, x, y, w, h, fc, ec)
        ax.text(x + 0.18, y + h - 0.30, title, fontsize=11, fontweight="bold", color=ec, va="center")
        ax.text(x + 0.18, y + h - 0.66, sub, fontsize=9, color=MUT, va="center")
        top, bottom = y + h - 0.95, y + 0.18
        step = (top - bottom) / rows
        for r in range(rows + 1):
            yy = bottom + r * step
            ax.plot([x + 0.15, x + w - 0.15], [yy, yy], color=LINE, lw=0.7, zorder=3)
        for c in (0.42, 0.68):
            ax.plot([x + w * c, x + w * c], [bottom, top], color=LINE, lw=0.7, zorder=3)

    table(0.35, 5.55, 3.5, 3.3, 3, "sample_orig（放款）", "一笔贷款 = 一行", BLUE_L, BLUE_D)
    table(0.35, 1.15, 3.5, 3.3, 7, "sample_perf（还款）", "一笔贷款 = 几十行", AMBER_L, AMBER_D)
    table(5.85, 2.55, 3.8, 5.3, 4, "统一宽表（建模用）", "一笔贷款一行：特征 + 标签", TEAL_L, TEAL_D)

    _arrow(ax, (3.95, 6.9), (5.7, 6.9), color=BLUE_D)
    _arrow(ax, (3.95, 2.9), (5.7, 4.2), color=AMBER_D)
    ax.text(4.18, 7.15, "按「贷款编号」关联", fontsize=9.5, color=BLUE_D)
    ax.text(4.18, 3.05, "同一笔贷款", fontsize=9.5, color=AMBER_D, va="top")
    ax.text(4.18, 2.55, "的多个月记录", fontsize=9.5, color=AMBER_D, va="top")
    fig.savefig(OUT / "02_join.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 03
def fig_window() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.set_xlim(-2, 48)
    ax.set_ylim(0.85, 3.35)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)

    ax.plot([0, 36], [2.62, 2.62], lw=13, color=TEAL, solid_capstyle="butt", zorder=2)
    ax.text(18, 2.62, "全程正常  →  标签 = 0", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)

    ax.plot([0, 20], [1.78, 1.78], lw=13, color=BLUE, solid_capstyle="butt", zorder=2)
    ax.plot([20, 36], [1.78, 1.78], lw=13, color=AMBER, solid_capstyle="butt", zorder=2)
    ax.text(10, 1.78, "正常还款", ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(28, 1.78, "拖欠恶化", ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", zorder=3)

    ax.annotate("第 20 个月\n开始拖欠", xy=(20, 1.72), xytext=(19.2, 1.24),
                ha="right", va="top", fontsize=9, color=AMBER_D,
                arrowprops=dict(arrowstyle="-", color=AMBER_D, lw=0.9))
    ax.annotate("第 23 个月\n触及逾期 90 天", xy=(23, 1.72), xytext=(24.0, 1.24),
                ha="left", va="top", fontsize=9, color=AMBER_D,
                arrowprops=dict(arrowstyle="-", color=AMBER_D, lw=0.9))

    ax.axvline(36, color=RED, lw=1.4, ls="--", zorder=1)
    ax.text(37.2, 2.62, "观察窗结束（第 36 个月）", color=RED, fontsize=9.5, va="center")
    ax.text(37.2, 1.78, "之后发生什么都不算", color=MUT, fontsize=9, va="center")

    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("放款后的月数", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.annotate("", xy=(36, 3.22), xytext=(0, 3.22),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
    ax.text(18, 3.28, "每笔贷款只被考察 36 个月：所有贷款站在同一起跑线",
            ha="center", fontsize=9.5, color=MUT)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.15)
    fig.savefig(OUT / "03_window.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 04
def fig_leakage() -> None:
    fig, ax = _blank((9.4, 5.2))
    ax.plot([5.0, 5.0], [1.4, 8.6], color=INK, lw=1.6, ls="--")
    ax.text(5.0, 9.0, "放款时点", ha="center", fontsize=11, fontweight="bold", color=INK)

    _rounded(ax, 0.4, 4.4, 4.3, 3.9, TEAL_L, TEAL_D)
    ax.text(0.7, 7.85, "放款之前：可以做特征", fontsize=11.5, fontweight="bold", color=TEAL_D)
    for i, t in enumerate(["信用分 FICO", "贷款金额 / 利率", "首付比例 LTV / CLTV",
                           "收入负债比 DTI", "州 / 房屋类型 / 用途"]):
        ax.text(0.75, 7.15 - i * 0.55, "·  " + t, fontsize=10, color=INK)

    _rounded(ax, 5.3, 4.4, 4.3, 3.9, RED_L, RED_D)
    ax.text(5.6, 7.85, "放款之后：只能用来造标签", fontsize=11.5, fontweight="bold", color=RED_D)
    for i, t in enumerate(["当前逾期月数", "剩余未还本金", "核销 / 结清代码",
                           "追回金额、处置费用", "是否已经违约"]):
        ax.text(5.65, 7.15 - i * 0.55, "·  " + t, fontsize=10, color=INK)

    _rounded(ax, 0.4, 1.0, 9.2, 2.4, GREY_L, LINE)
    ax.text(0.7, 2.85, "把右边的字段喂给模型会发生什么？", fontsize=11, fontweight="bold", color=RED_D)
    ax.text(0.7, 2.25, "模型会拿到「未来的答案」，训练时 AUC 冲到 0.95，看着完美；", fontsize=10, color=INK)
    ax.text(0.7, 1.75, "但上线那天这些字段根本还不存在 → 预测失效。这叫数据泄漏（leakage）。",
            fontsize=10, color=INK)
    fig.savefig(OUT / "04_leakage.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 05
def fig_split(df: pd.DataFrame) -> None:
    order = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    role = {2015: "train", 2016: "train", 2017: "train", 2018: "train",
            2019: "valid", 2020: "oot", 2021: "oot"}
    color = {"train": BLUE, "valid": AMBER, "oot": TEAL}
    label = {"train": "训练集 train", "valid": "验证集 valid", "oot": "时间外测试 OOT"}

    grp = df.groupby("vintage_year")["target"].agg(n="size", rate="mean").reindex(order)

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    bars = ax.bar([str(y) for y in order], grp["n"], color=[color[role[y]] for y in order],
                  width=0.66, zorder=2)
    for b, y in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 900,
                f"{int(grp.loc[y, 'n']):,}", ha="center", fontsize=9, color=INK)

    ax2 = ax.twinx()
    ax2.plot([str(y) for y in order], grp["rate"] * 100, color=RED, marker="o",
             lw=2, ms=6, zorder=3)
    for x, y in zip([str(v) for v in order], grp["rate"] * 100):
        ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=9, color=RED, zorder=4,
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
    ax2.set_ylabel("36 个月违约率（%）", color=RED, fontsize=10)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(0, grp["rate"].max() * 100 * 1.55)
    ax2.grid(False)

    ax.set_ylabel("贷款笔数", fontsize=10)
    ax.set_xlabel("放款年份（vintage）", fontsize=10)
    ax.set_ylim(0, grp["n"].max() * 1.18)
    ax.grid(axis="x", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in ("train", "valid", "oot")]
    ax.legend(handles, [label[k] for k in ("train", "valid", "oot")],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.26)
    fig.savefig(OUT / "05_oot_split.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 06
def fig_woe(df: pd.DataFrame) -> None:
    train = df[df["vintage_year"].between(2015, 2018)].copy()
    train = train[train["fico"].notna()]

    edges = np.unique(np.quantile(train["fico"], np.linspace(0, 1, 11)))
    train["bin"] = pd.cut(train["fico"], bins=edges, include_lowest=True, duplicates="drop")
    g = train.groupby("bin", observed=True)["target"].agg(n="size", bad="sum")
    g["rate"] = g["bad"] / g["n"]
    g["bad_dist"] = g["bad"] / g["bad"].sum()
    g["good_dist"] = (g["n"] - g["bad"]) / (g["n"] - g["bad"]).sum()
    g["woe"] = np.log(g["bad_dist"] / g["good_dist"])

    labels = [f"{int(iv.left)}–{int(iv.right)}" for iv in g.index]
    x = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x, g["rate"] * 100, color=BLUE, width=0.62, zorder=2)
    for xi, v in zip(x, g["rate"] * 100):
        ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("该分箱的违约率（%）", fontsize=10, color=BLUE_D)
    ax.tick_params(axis="y", colors=BLUE_D)
    ax.set_xlabel("信用分分箱（用训练集 2015–2018 的十分位切分）", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, g["rate"].max() * 100 * 1.25)
    ax.grid(axis="x", visible=False)

    ax2 = ax.twinx()
    ax2.plot(x, g["woe"], color=RED, marker="o", lw=2, ms=5.5, zorder=3)
    ax2.axhline(0, color=RED, lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel("WOE 值（对数几率）", fontsize=10, color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.grid(False)

    ax.set_title("信用分越高 → 违约率越低 → WOE 越小：这条单调性就是「变量可信」的证据",
                 fontsize=11, fontweight="bold", color=INK, pad=12)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.86, bottom=0.24)
    fig.savefig(OUT / "06_woe_fico.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 07
def fig_ks(df: pd.DataFrame) -> None:
    d = df[df["fico"].notna()]
    bad = np.sort(d.loc[d["target"] == 1, "fico"].to_numpy())
    good = np.sort(d.loc[d["target"] == 0, "fico"].to_numpy())
    xs = np.linspace(300, 900, 601)
    cdf_bad = np.searchsorted(bad, xs, side="right") / len(bad)
    cdf_good = np.searchsorted(good, xs, side="right") / len(good)
    gap = np.abs(cdf_bad - cdf_good)
    k = int(np.argmax(gap))

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(xs, cdf_good, color=TEAL, lw=2.2, label="好客户（未违约）累计占比")
    ax.plot(xs, cdf_bad, color=RED, lw=2.2, label="坏客户（违约）累计占比")
    ax.fill_between(xs, cdf_good, cdf_bad, color=AMBER, alpha=0.18)

    xk, y1, y2 = xs[k], cdf_good[k], cdf_bad[k]
    ax.annotate("", xy=(xk, y2), xytext=(xk, y1),
                arrowprops=dict(arrowstyle="<|-|>", color=AMBER_D, lw=1.6))
    ax.text(xk + 12, (y1 + y2) / 2, f"KS = {gap[k]:.3f}\n（两条曲线的最大垂直距离）",
            fontsize=10, color=AMBER_D, va="center")

    ax.set_xlabel("信用分", fontsize=10)
    ax.set_ylabel("累计占比", fontsize=10)
    ax.set_xlim(300, 900)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("KS 在说什么：好坏两群人能被分得多开",
                 fontsize=11, fontweight="bold", color=INK, pad=12)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13)
    fig.savefig(OUT / "07_ks_intuition.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 08
def fig_vintage_rate(df: pd.DataFrame) -> None:
    order = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    g = df.groupby("vintage_year")["target"].agg(n="size", rate="mean").reindex(order)
    covid = {2017, 2018, 2019, 2020}  # 36-month window touches 2020-2021

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    colors = [AMBER if y in covid else BLUE for y in order]
    bars = ax.bar([str(y) for y in order], g["rate"] * 100, color=colors, width=0.64, zorder=2)
    for b, y in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.10,
                f"{g.loc[y, 'rate'] * 100:.2f}%", ha="center", fontsize=9.5, color=INK)
    ax.set_ylabel("36 个月违约率（%）", fontsize=10)
    ax.set_xlabel("放款年份（vintage）", fontsize=10)
    ax.set_ylim(0, g["rate"].max() * 100 * 1.28)
    ax.grid(axis="x", visible=False)

    ax.axvspan(1.6, 4.4, color=AMBER, alpha=0.10, zorder=0)
    ax.text(3.0, g["rate"].max() * 100 * 1.17,
            "这四档的 36 个月窗口撞上 2020–2021 疫情冲击",
            ha="center", fontsize=9.5, color=AMBER_D)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=AMBER)]
    ax.legend(handles, ["窗口内无疫情冲击", "窗口覆盖疫情期"],
              loc="upper right", frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.13)
    fig.savefig(OUT / "08_vintage_default.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 09
def fig_ml_vs_programming() -> None:
    fig, ax = _blank((9.4, 4.6))

    ax.text(0.35, 9.35, "传统编程", fontsize=12.5, fontweight="bold", color=BLUE_D)
    ax.text(0.35, 8.75, "规则由人来写", fontsize=9.5, color=MUT)
    ax.text(5.45, 9.35, "机器学习", fontsize=12.5, fontweight="bold", color=TEAL_D)
    ax.text(5.45, 8.75, "规则由数据算出来", fontsize=9.5, color=MUT)

    ax.plot([5.0, 5.0], [3.05, 9.55], color=LINE, lw=1.0, ls="--")

    rows = [
        (7.05, "数据（历史贷款）", "数据 + 历史答案"),
        (5.15, "规则：人写 if-else", "规则：电脑算出的权重"),
        (3.25, "程序执行 → 得到答案", "拿去预测新贷款"),
    ]
    for i, (y, lt, rt) in enumerate(rows):
        _rounded(ax, 0.35, y, 4.2, 1.15, GREY_L, LINE, radius=0.10)
        ax.text(2.45, y + 0.58, lt, ha="center", va="center", fontsize=12, color=INK)

        if i == 1:
            _rounded(ax, 5.45, y, 4.2, 1.15, AMBER_L, AMBER_D, radius=0.10)
            ax.text(7.55, y + 0.72, rt, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=AMBER_D)
            ax.text(7.55, y + 0.30, "这一步就是「训练」", ha="center", va="center",
                    fontsize=9.5, color=AMBER_D)
        else:
            fc, ec = (BLUE_L, BLUE_D) if i == 0 else (TEAL_L, TEAL_D)
            _rounded(ax, 5.45, y, 4.2, 1.15, fc, ec, radius=0.10)
            ax.text(7.55, y + 0.58, rt, ha="center", va="center", fontsize=12, color=INK)

    for y in (7.05, 5.15):
        for x in (2.45, 7.55):
            _arrow(ax, (x, y - 0.02), (x, y - 0.73), color="#9AA3AD")

    ax.text(2.45, 2.55, "左边：规则是人想出来的", ha="center", fontsize=11, color=MUT)
    ax.text(7.55, 2.55, "右边：规则是数据算出来的", ha="center", fontsize=11, color=MUT)
    ax.set_ylim(2.1, 9.85)
    fig.savefig(OUT / "09_ml_vs_programming.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 10
def fig_loan_to_pd() -> None:
    fig, ax = _blank((9.4, 3.6))
    ax.set_xlim(0, 10.2)

    boxes = [
        ("① 贷款信息", "14 个放款时字段", "FICO 720 · LTV 80%", BLUE_L, BLUE_D),
        ("② 乘权重求和", "每个值 × 一个权重再相加", "z = −3.85", BLUE_L, BLUE_D),
        ("③ 压成 0 到 1", "S 形函数保证落在区间内", "PD = 1 / (1 + e^(-z))", BLUE_L, BLUE_D),
        ("④ 违约概率", "和审批阈值比大小", "2.08%　< 5% → 批准", TEAL_L, TEAL_D),
    ]
    w, h, y = 2.15, 3.2, 4.0
    xs = [0.30, 2.85, 5.40, 7.95]
    for (title, sub, val, fc, ec), x in zip(boxes, xs):
        _rounded(ax, x, y, w, h, fc, ec, radius=0.10)
        ax.text(x + w / 2, y + h - 0.62, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 1.45, sub, ha="center", va="center",
                fontsize=9, color=MUT)
        ax.text(x + w / 2, y + 0.75, val, ha="center", va="center",
                fontsize=10.5, color=INK, fontweight="bold")

    for x in xs[:-1]:
        _arrow(ax, (x + w + 0.03, y + h / 2), (x + w + 0.38, y + h / 2), color="#9AA3AD")

    ax.text(5.1, 2.7, "这四步全部是四则运算，不需要历史数据——所以预测一笔新贷款只要几毫秒",
            ha="center", fontsize=9.5, color=MUT)
    ax.set_ylim(2.2, 7.8)
    fig.savefig(OUT / "10_loan_to_pd.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 11
def fig_tree_forest_gbm() -> None:
    fig, ax = _blank((9.8, 5.2))

    heads = [(0.35, "① 决策树", "一棵 = 一串 if-else", BLUE_D),
             (3.55, "② 随机森林", "很多棵树各投一票", TEAL_D),
             (6.85, "③ 梯度提升 GBM", "很多棵树串行纠错", AMBER_D)]
    for x, t, s, c in heads:
        ax.text(x, 9.45, t, fontsize=12, fontweight="bold", color=c)
        ax.text(x, 8.95, s, fontsize=9, color=MUT)

    for x in (3.35, 6.65):
        ax.plot([x, x], [3.25, 9.7], color=LINE, lw=1.0, ls="--")

    _rounded(ax, 1.45, 7.15, 1.55, 0.62, BLUE_L, BLUE_D, radius=0.08)
    ax.text(2.225, 7.46, "FICO < 680 ?", ha="center", va="center", fontsize=9.5, color=BLUE_D)

    _rounded(ax, 0.35, 5.75, 1.45, 0.60, BLUE_L, BLUE_D, radius=0.08)
    ax.text(1.075, 6.05, "LTV > 90% ?", ha="center", va="center", fontsize=9.5, color=BLUE_D)
    _rounded(ax, 2.10, 5.75, 1.10, 0.60, GREY_L, LINE, radius=0.08)
    ax.text(2.65, 6.05, "低风险", ha="center", va="center", fontsize=9.5, color=INK)

    _rounded(ax, 0.35, 4.35, 1.45, 0.60, RED_L, RED_D, radius=0.08)
    ax.text(1.075, 4.65, "高风险", ha="center", va="center", fontsize=9.5, color=RED_D)
    _rounded(ax, 2.10, 4.35, 1.10, 0.60, AMBER_L, AMBER_D, radius=0.08)
    ax.text(2.65, 4.65, "中风险", ha="center", va="center", fontsize=9.5, color=AMBER_D)

    _arrow(ax, (2.225, 7.13), (1.10, 6.40), color="#9AA3AD", lw=1.0)
    _arrow(ax, (2.30, 7.13), (2.62, 6.40), color="#9AA3AD", lw=1.0)
    _arrow(ax, (1.075, 5.73), (1.075, 4.98), color="#9AA3AD", lw=1.0)
    _arrow(ax, (1.35, 5.73), (2.45, 4.98), color="#9AA3AD", lw=1.0)

    ax.text(1.75, 3.60, "一路问到底，末端给出一个分数", ha="center", fontsize=9, color=MUT)

    for i in range(6):
        cx = 3.62 + (i % 3) * 0.92
        cy = 7.55 - (i // 3) * 0.95
        _rounded(ax, cx, cy, 0.78, 0.72, TEAL_L, TEAL_D, radius=0.06)
        trunk_x, fork_y = cx + 0.39, cy + 0.36
        ax.plot([trunk_x, trunk_x], [cy + 0.12, fork_y], color=TEAL_D, lw=1.3, zorder=4)
        ax.plot([cx + 0.19, trunk_x, cx + 0.59], [cy + 0.60, fork_y, cy + 0.60],
                color=TEAL_D, lw=1.3, zorder=4)

    _arrow(ax, (5.05, 6.55), (5.05, 5.95), color="#9AA3AD")
    _rounded(ax, 4.05, 5.15, 2.05, 0.70, TEAL_L, TEAL_D, radius=0.08)
    ax.text(5.075, 5.50, "平均 / 投票 → 一个分数", ha="center", va="center",
            fontsize=9.5, color=TEAL_D)

    ax.text(5.05, 4.55, "六棵树长得都不一样：", ha="center", fontsize=9, color=MUT)
    ax.text(5.05, 4.15, "各看一部分数据、一部分特征", ha="center", fontsize=9, color=MUT)

    ys = [7.30, 6.05, 4.80]
    labels = ["第 1 棵树", "第 2 棵树", "第 3 棵树"]
    subs = ["先给一个粗糙的分数", "专门修正上一步的错", "再修正剩下的错"]
    for y, lb, sb in zip(ys, labels, subs):
        _rounded(ax, 6.95, y, 2.75, 0.95, AMBER_L, AMBER_D, radius=0.08)
        ax.text(7.15, y + 0.60, lb, fontsize=9.5, fontweight="bold", color=AMBER_D)
        ax.text(7.15, y + 0.26, sb, fontsize=8.5, color=MUT)

    for y in ys[:-1]:
        _arrow(ax, (8.32, y - 0.02), (8.32, y - 0.28), color="#9AA3AD")

    ax.text(8.32, 4.30, "…… 共 400 棵", ha="center", fontsize=9, color=AMBER_D)
    ax.text(8.32, 3.95, "每棵只信 4%", ha="center", fontsize=9, color=MUT)

    ax.text(5.0, 2.75, "单棵树很弱　→　很多棵树很强，但没人能读懂这 400 条规则的合集",
            ha="center", fontsize=11, color=INK)
    ax.set_ylim(2.3, 9.8)
    fig.savefig(OUT / "11_tree_forest_gbm.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 12
def fig_overfit() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.6))

    rows = [("梯度提升 GBM（400 棵树）", 0.7993, 0.7381, 0),
            ("评分卡（14 个权重）", 0.7805, 0.7462, 1)]
    for label, tr, oo, y in rows:
        ax.plot([oo, tr], [y, y], color=LINE, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([tr], [y], s=130, color=BLUE, zorder=3, edgecolor="white", linewidth=1.2)
        ax.scatter([oo], [y], s=130, color=TEAL, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(tr, y + 0.17, f"训练 {tr:.4f}", ha="center", fontsize=9, color=BLUE_D)
        ax.text(oo, y - 0.24, f"时间外 {oo:.4f}", ha="center", fontsize=9, color=TEAL_D)
        ax.text((tr + oo) / 2, y + 0.03, f"跌 {tr - oo:.3f}", ha="center", va="bottom",
                fontsize=9, color=RED_D, fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0.725, 0.815)
    ax.set_xlabel("AUC（时间外 = 没见过的贷款）", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=9, label="训练集（它见过的）"),
        plt.Line2D([], [], marker="o", ls="", color=TEAL, markersize=9, label="时间外（它没见过的）"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)

    ax.set_title("GBM 在训练集上更高，换到没见过的年份反而更低",
                 fontsize=11, color=INK, pad=10)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.16)
    fig.savefig(OUT / "12_overfit.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 13
def _auc_from_scores(good: np.ndarray, bad: np.ndarray) -> float:
    """AUC = P(random bad scores higher than random good), via the rank formula."""
    merged = np.concatenate([good, bad])
    order = np.argsort(merged, kind="mergesort")
    ranks = np.empty(len(merged), dtype=float)
    ranks[order] = np.arange(1, len(merged) + 1)
    n_good, n_bad = len(good), len(bad)
    return (ranks[:n_good].sum() - n_good * (n_good + 1) / 2) / (n_good * n_bad)


def fig_auc_intuition(df: pd.DataFrame) -> None:
    fico = df["fico"].where(df["fico"].between(350, 850))
    good = fico[(df["target"] == 0) & fico.notna()].to_numpy()
    bad = fico[(df["target"] == 1) & fico.notna()].to_numpy()
    auc = _auc_from_scores(good, bad)

    def density(values: np.ndarray, lo=350, hi=850, bins=100, window=9):
        h, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        kernel = np.ones(window) / window
        return centers, np.convolve(h, kernel, mode="same")

    cx, dg = density(good)
    _, db = density(bad)

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.fill_between(cx, dg, color=TEAL, alpha=0.28, zorder=1)
    ax.fill_between(cx, db, color=RED, alpha=0.30, zorder=2)
    ax.fill_between(cx, np.minimum(dg, db), color="#6B4A8C", alpha=0.45, zorder=3)

    ax.plot(cx, dg, color=TEAL_D, lw=1.6, zorder=4)
    ax.plot(cx, db, color=RED_D, lw=1.6, zorder=4)

    top = max(dg.max(), db.max())
    ax.set_ylim(0, top * 1.38)
    ax.set_xlim(578, 862)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("FICO 信用分", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)

    peak_good = cx[np.argmax(dg)]
    peak_bad = cx[np.argmax(db)]

    ax.annotate("实际没违约的客户", xy=(peak_good, dg.max() * 1.03),
                xytext=(peak_good - 12, top * 1.27),
                ha="center", fontsize=10, color=TEAL_D, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL_D, lw=0.9))
    ax.annotate("实际违约的客户", xy=(peak_bad, db.max() * 1.03),
                xytext=(peak_bad - 48, top * 1.08),
                ha="right", fontsize=10, color=RED_D, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED_D, lw=0.9))

    ax.annotate("紫色 = 重叠区\n这里的两个人\n模型分不清谁更危险",
                xy=(peak_bad + 18, top * 0.36), xytext=(598, top * 0.80),
                ha="left", va="center", fontsize=9.5, color="#4B2C63",
                arrowprops=dict(arrowstyle="-|>", color="#6B4A8C", lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))

    ax.set_title(
        f"只看 FICO 一个变量时：AUC = {auc:.2f}（把 14 个变量合起来才升到 0.7462）",
        fontsize=10.5, color=INK, pad=8,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(OUT / "13_auc_intuition.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"   fico-only AUC = {auc:.4f}")


# --------------------------------------------------------------------------- 14
def fig_roc_curve() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    x0746 = np.array([0, .05, .10, .20, .30, .50, .70, .85, 1.0])
    y0746 = np.array([0, .30, .50, .68, .78, .88, .94, .98, 1.0])
    x090 = np.array([0, .02, .05, .10, .20, .35, .60, 1.0])
    y090 = np.array([0, .35, .60, .75, .87, .95, .99, 1.0])

    ax.fill_between(x0746, y0746, color=BLUE, alpha=0.16, zorder=1)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#9AA3AD", zorder=2)
    ax.plot(x0746, y0746, color=BLUE_D, lw=2.4, zorder=4)
    ax.plot(x090, y090, color=TEAL_D, lw=1.6, ls=(0, (5, 3)), zorder=3)

    ax.annotate("对角线 = 瞎猜（AUC 0.50）", xy=(.52, .52), xytext=(.80, .16),
                ha="center", fontsize=9.5, color=MUT,
                arrowprops=dict(arrowstyle="-|>", color="#9AA3AD", lw=0.9))
    ax.annotate("本项目 AUC 0.746\n（这块面积）", xy=(.32, .65), xytext=(.46, .36),
                fontsize=10, color=BLUE_D, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=BLUE_D, lw=1.0))
    ax.annotate("AUC 0.90\n极强的模型", xy=(.14, .80), xytext=(.045, .925),
                fontsize=9.5, color=TEAL_D, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=TEAL_D, lw=1.0))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.07)
    ax.set_xlabel("误伤的好人比例（假阳性率）", fontsize=10.5)
    ax.set_ylabel("抓到的坏人比例（真阳性率）", fontsize=10.5)
    ax.set_title("ROC 曲线：曲线下的面积 = AUC", fontsize=11.5, color=INK, pad=10)
    ax.text(0.02, -0.155, "曲线形状为示意图，数值取自本项目实测",
            fontsize=8.5, color=MUT, transform=ax.transAxes)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.17)
    fig.savefig(OUT / "14_roc_curve.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_join()
    fig_window()
    fig_leakage()
    fig_ml_vs_programming()
    fig_loan_to_pd()
    fig_tree_forest_gbm()
    fig_overfit()
    fig_roc_curve()
    print(f"schematic figures written to {OUT}")

    if not PROCESSED.exists():
        print(f"!! {PROCESSED} not found - run the pipeline first (data-driven charts skipped)")
        return 0

    df = pd.read_csv(PROCESSED, usecols=["vintage_year", "fico", "target"])
    print(f"loaded {len(df):,} rows from {PROCESSED.name}")
    fig_split(df)
    fig_woe(df)
    fig_ks(df)
    fig_vintage_rate(df)
    fig_auc_intuition(df)

    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name:<28}{p.stat().st_size / 1024:>7.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
