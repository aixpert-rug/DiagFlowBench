"""Generate publication figures for DiagFlowBench paper.

Produces:
  fig1b_diverging.png  — diverging bars with group labels, sorted within groups by CA

Run from repo root: python3 docs/figures/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory, ScaledTranslation
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LANCZOS = Image.Resampling.LANCZOS

LOGOS_DIR = Path(__file__).parent.parent / "docs" / "figures" / "logos"
OUT_DIR   = Path(__file__).parent.parent / "docs" / "figures"

# Each model: name, subtitle (arch type), provider, group, FA, FM, CA (as fractions)
MODELS = [
    {"name": "Gemini 2.5 Flash",    "subtitle": "",          "provider": "google",  "group": "Commercial",     "fa": 0.024, "fm": 0.256, "ca": 0.720},
    {"name": "GPT-4o Mini",         "subtitle": "",          "provider": "openai",  "group": "Commercial",     "fa": 0.053, "fm": 0.364, "ca": 0.583},
    {"name": "Nemotron 3 Super 120B","subtitle": "MoE",      "provider": "nvidia",  "group": "Open-weight",    "fa": 0.041, "fm": 0.221, "ca": 0.739},
    {"name": "Mistral Small 24B",   "subtitle": "Instruct",  "provider": "mistral", "group": "Open-weight",    "fa": 0.053, "fm": 0.276, "ca": 0.671},
    {"name": "Qwen3 235B Thinking", "subtitle": "Reasoning", "provider": "qwen",    "group": "Open-weight",    "fa": 0.063, "fm": 0.288, "ca": 0.649},
    {"name": "GPT-OSS 120B",        "subtitle": "MoE",       "provider": "openai",  "group": "Open-weight",    "fa": 0.022, "fm": 0.431, "ca": 0.547},
    {"name": "Qwen3 30B Thinking",  "subtitle": "Reasoning", "provider": "qwen",    "group": "Open-weight",    "fa": 0.051, "fm": 0.674, "ca": 0.274},
    {"name": "Llama 3.3 70B",       "subtitle": "Instruct",  "provider": "meta",    "group": "Scalability Test","fa": 0.030, "fm": 0.157, "ca": 0.813},
    {"name": "Llama 4 Maverick",    "subtitle": "MoE",       "provider": "meta",    "group": "Scalability Test","fa": 0.050, "fm": 0.293, "ca": 0.657},
    {"name": "Llama 4 Scout",       "subtitle": "MoE",       "provider": "meta",    "group": "Scalability Test","fa": 0.086, "fm": 0.519, "ca": 0.394},
]

GROUPS = ["Commercial", "Open-weight", "Scalability Test"]

LOGO_FILE = {
    "google":  "gemini.png",
    "openai":  "gpt.png",
    "meta":    "llama.png",
    "mistral": "mistral.png",
    "qwen":    "qwen.png",
}

BRAND_COLORS = {
    "google":  "#4285F4",
    "openai":  "#1a1a1a",
    "meta":    "#0866FF",
    "mistral": "#FF7000",
    "qwen":    "#6C47FF",
    "nvidia":  "#76B900",
}

PROVIDER_INITIALS = {
    "google":  "G", "openai": "O", "meta": "M",
    "mistral": "Mi", "qwen": "Q", "nvidia": "N",
}

C_FAB    = "#D62728"
C_FORCED = "#FF7F0E"
C_ABST   = "#2CA02C"
C_SEP    = "#cccccc"

_FA_DISPLAY_MIN = 1.2
_FA_DISPLAY_MAX = 4.5


def _fa_display(fa_raw: list[float]) -> list[float]:
    nonzero = [v for v in fa_raw if v > 0]
    if not nonzero:
        return [0.0] * len(fa_raw)
    fa_max = max(nonzero)
    return [
        0.0 if v == 0 else _FA_DISPLAY_MIN + (v / fa_max) * (_FA_DISPLAY_MAX - _FA_DISPLAY_MIN)
        for v in fa_raw
    ]


def _make_chip(provider: str, size: int = 100) -> Image.Image:
    fname    = LOGO_FILE.get(provider, f"{provider}.png")
    png_path = LOGOS_DIR / fname
    if png_path.exists():
        img = Image.open(png_path).convert("RGBA")
        return img.resize((size, size), _LANCZOS)
    img   = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw  = ImageDraw.Draw(img)
    color = BRAND_COLORS.get(provider, "#888888")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    draw.ellipse((0, 0, size - 1, size - 1), fill=(r, g, b, 255))
    initial   = PROVIDER_INITIALS.get(provider, "?")
    font_size = size // 2
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initial, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2 - 2), initial,
              fill=(255, 255, 255, 255), font=font)
    return img


def _pil_to_mpl(img: Image.Image) -> np.ndarray:
    return np.array(img)


def make_diverging_bars():
    sorted_models = []
    group_boundaries = {}   # group -> (start_y, end_y) inclusive indices
    for g in GROUPS:
        members = sorted([m for m in MODELS if m["group"] == g],
                         key=lambda m: m["ca"], reverse=True)
        group_boundaries[g] = (len(sorted_models), len(sorted_models) + len(members) - 1)
        sorted_models.extend(members)

    n     = len(sorted_models)
    y     = np.arange(n)
    bar_h = 0.55

    fig_h = max(4.0, n * 0.55 + 1.0)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fm_pct = [m["fm"] * 100 for m in sorted_models]
    fa_raw = [m["fa"] * 100 for m in sorted_models]
    ca_pct = [m["ca"] * 100 for m in sorted_models]
    fa_display = _fa_display(fa_raw)

    ax.barh(y, ca_pct, bar_h, color=C_ABST,   zorder=3)
    ax.barh(y, [-v for v in fm_pct], bar_h, color=C_FORCED, zorder=3)
    ax.barh(y, [-v for v in fa_display], bar_h, left=[-v for v in fm_pct],
            color=C_FAB, zorder=4)

    ax.axvline(0, color="#222222", linewidth=1.8, zorder=5)

    for i in range(n):
        ca  = ca_pct[i]
        fm  = fm_pct[i]
        fa  = fa_raw[i]
        fad = fa_display[i]

        if ca >= 10:
            ax.text(ca / 2, i, f"{ca:.1f}%", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white", zorder=6)
        elif ca > 0:
            ax.text(ca + 1.2, i, f"{ca:.1f}%", ha="left", va="center",
                    fontsize=7.5, fontweight="bold", color=C_ABST, zorder=6)

        if fm >= 10:
            ax.text(-(fm / 2), i, f"{fm:.1f}%", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white", zorder=6)
        elif fm > 0:
            ax.text(-(fm + 1.2), i, f"{fm:.1f}%", ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color=C_FORCED, zorder=6)

        if fa > 0:
            ax.text(-(fm + fad) - 1.0, i, f"{fa:.1f}%", ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color=C_FAB, zorder=7)

    for g in GROUPS[:-1]:
        end_y = group_boundaries[g][1]
        ax.axhline(end_y + 0.5, color=C_SEP, linewidth=1.0, linestyle="--", zorder=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlim(-90, 90)
    ax.set_ylim(-0.65, n - 0.35)
    ax.invert_yaxis()

    ticks = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(int(v))}%"))
    ax.tick_params(axis="x", labelsize=9, color="#aaaaaa")
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)

    ax.text(-44, 1.04, "← Forced mapping / Fabrication",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=C_FORCED, transform=blended_transform_factory(ax.transData, ax.transAxes),
            clip_on=False)
    ax.text(36, 1.04, "Correct abstention →",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=C_ABST, transform=blended_transform_factory(ax.transData, ax.transAxes),
            clip_on=False)

    ytrans = blended_transform_factory(ax.transAxes, ax.transData)

    LOGO_OFFSET  = -46   # pts left of axes left edge
    LABEL_OFFSET = -68   # pts left of axes left edge
    GROUP_OFFSET = -125  # pts left of axes left edge

    chip_size = 100
    for i, m in enumerate(sorted_models):
        chip = _make_chip(m["provider"], chip_size)
        oi   = OffsetImage(_pil_to_mpl(chip), zoom=0.18, resample=True)
        ab   = AnnotationBbox(
            oi, (0, i),
            xycoords=("axes fraction", "data"),
            xybox=(LOGO_OFFSET, 0),
            boxcoords="offset points",
            frameon=False, clip_on=False, zorder=6,
        )
        ax.add_artist(ab)

        name_trans = ytrans + ScaledTranslation(LABEL_OFFSET / 72, 4 / 72, fig.dpi_scale_trans)
        t = ax.text(0, i, m["name"], transform=name_trans,
                    ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color="#111111")
        t.set_clip_on(False)

        if m["subtitle"]:
            sub_trans = ytrans + ScaledTranslation(LABEL_OFFSET / 72, -6 / 72, fig.dpi_scale_trans)
            t2 = ax.text(0, i, m["subtitle"], transform=sub_trans,
                         ha="right", va="center",
                         fontsize=7, style="italic", color="#666666")
            t2.set_clip_on(False)

    for g in GROUPS:
        start_y, end_y = group_boundaries[g]
        mid_y = (start_y + end_y) / 2.0
        g_trans = ytrans + ScaledTranslation(GROUP_OFFSET / 72, 0, fig.dpi_scale_trans)
        tg = ax.text(0, mid_y, g, transform=g_trans,
                     ha="center", va="center", rotation=90,
                     fontsize=7, fontweight="bold", color="#999999")
        tg.set_clip_on(False)

    fig.subplots_adjust(left=0.33, right=0.97, top=0.93, bottom=0.12)

    out_png = OUT_DIR / "fig1b_diverging.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    make_diverging_bars()
    print("Done.")
