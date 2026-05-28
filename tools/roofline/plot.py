"""Roofline plot generation using matplotlib.

Produces a log-log plot with:
  - Sloped memory-bandwidth ceiling
  - Flat compute ceiling(s) for different dtypes
  - One marker per layer:
      ◆  roofline predicted throughput
      ●  actual RTL throughput (if available)

Optionally overlays multiple hardware configs for STCO "what-if" analysis.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .hardware import CoralNPUConfig
from .model import WorkloadResult


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        return matplotlib, plt, np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for plotting.\n"
            "Install with: pip install matplotlib numpy"
        )


LAYER_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

DTYPE_COLORS = {
    "int8":   "#e74c3c",
    "int16":  "#e67e22",
    "int32":  "#f39c12",
    "scalar": "#95a5a6",
}


def plot_roofline(
    results: List[WorkloadResult],
    title: str = "CoralNPU Roofline Model",
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (10, 7),
    ai_range: Tuple[float, float] = (0.1, 1000),
):
    """Generate the roofline plot.

    Args:
        results:      one or more WorkloadResult objects (one per HW config or workload)
        title:        plot title
        output_path:  save to PNG/PDF if provided
        show:         call plt.show() (use False in headless/CI environments)
        figsize:      (width, height) in inches
        ai_range:     (min, max) arithmetic intensity range for the x-axis
    """
    _, plt, np = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ai_min, ai_max = ai_range
    ai_arr = np.logspace(math.log10(ai_min), math.log10(ai_max), 400)

    drawn_configs = set()

    for wr_idx, wr in enumerate(results):
        hw = wr.hw
        config_key = hw.name

        if config_key not in drawn_configs:
            drawn_configs.add(config_key)
            _draw_ceilings(ax, hw, ai_arr, np, wr_idx, len(results))

        # Draw layer operating points
        for li, lr in enumerate(wr.layers):
            color = LAYER_COLORS[li % len(LAYER_COLORS)]
            ai = lr.arithmetic_intensity
            roof_tp = lr.roofline_macs_cycle

            ax.scatter(
                [ai], [roof_tp],
                marker="D", s=80, color=color, zorder=5,
                label=f"{lr.layer.name} [roofline]",
            )
            ax.annotate(
                lr.layer.name,
                (ai, roof_tp),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=color,
            )

            if lr.rtl_cycles is not None:
                rtl_tp = lr.rtl_throughput
                ax.scatter(
                    [ai], [rtl_tp],
                    marker="o", s=80, color=color, zorder=5,
                    edgecolors="black", linewidths=0.8,
                    label=f"{lr.layer.name} [RTL]",
                )
                # Arrow from predicted to actual
                ax.annotate(
                    "", xy=(ai, rtl_tp), xytext=(ai, roof_tp),
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=1.2
                    ),
                )

        # RTL total operating point (estimated per-layer split)
        if wr.rtl_total_cycles is not None:
            # Plot as a large star at overall AI
            total_ai = wr.workload.overall_arithmetic_intensity
            total_tp = wr.workload.total_macs / wr.rtl_total_cycles
            ax.scatter(
                [total_ai], [total_tp],
                marker="*", s=250, color="black", zorder=6,
                label=f"RTL total ({wr.rtl_total_cycles/1e6:.1f}M cycles)",
            )

    # Labels and formatting
    ax.set_xlabel("Arithmetic Intensity  [MACs / byte]", fontsize=12)
    ax.set_ylabel("Attainable Throughput  [MACs / cycle]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(ai_min, ai_max)

    # Secondary x-axis: OPS/byte (1 MAC = 2 ops)
    def macs_to_ops(x):
        return x * 2
    def ops_to_macs(x):
        return x / 2
    secax = ax.secondary_xaxis("top", functions=(macs_to_ops, ops_to_macs))
    secax.set_xlabel("Arithmetic Intensity  [OPS / byte]", fontsize=10)

    # Secondary y-axis: MACs/s @ 1 GHz
    def macs_to_gmacs(y):
        return y  # 1 MAC/cycle @ 1 GHz = 1 GMAC/s
    ax.secondary_yaxis("right", functions=(macs_to_gmacs, macs_to_gmacs)).set_ylabel(
        "Throughput  [GMACs/s @ 1 GHz]", fontsize=10
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    return fig


def _draw_ceilings(ax, hw: CoralNPUConfig, ai_arr, np, idx: int, n_configs: int):
    """Draw compute and memory ceilings for one hardware config."""
    linestyle = ["-", "--", "-."][idx % 3]
    alpha = 1.0 if n_configs == 1 else 0.8

    for dtype, color in DTYPE_COLORS.items():
        if dtype == "scalar":
            continue  # skip scalar on the ceiling lines
        peak = hw.peak_macs_per_cycle(dtype)
        bw = hw.lsu_bw_bytes_per_cycle
        roof = np.minimum(peak, ai_arr * bw)

        label = f"{hw.name}: {dtype} ({peak:.0f} MACs/cyc)"
        ax.loglog(ai_arr, roof, linestyle=linestyle, color=color,
                  linewidth=1.8, alpha=alpha, label=label)

        # Mark ridge point
        ridge = hw.ridge_point(dtype)
        ax.axvline(ridge, color=color, linestyle=":", alpha=0.3, linewidth=1)
        ax.text(ridge * 1.05, peak * 0.95, f"ridge\n{ridge:.1f}", fontsize=7,
                color=color, alpha=0.7)

    # Memory-bandwidth slope (only label once)
    bw = hw.lsu_bw_bytes_per_cycle
    mem_roof = ai_arr * bw
    ax.loglog(ai_arr, mem_roof, linestyle=linestyle, color="#2c3e50",
              linewidth=1.8, alpha=alpha * 0.6,
              label=f"{hw.name}: memory ({bw:.0f} B/cyc)")


def ascii_roofline(wr: WorkloadResult, width: int = 70, height: int = 20) -> str:
    """Render a simple ASCII roofline chart (no matplotlib dependency)."""
    hw = wr.hw
    ai_min, ai_max = 0.1, 100.0
    peak = hw.peak_macs_per_cycle("int16")
    bw = hw.lsu_bw_bytes_per_cycle

    grid = [[" "] * width for _ in range(height)]

    def x_to_col(ai):
        import math
        frac = (math.log10(ai) - math.log10(ai_min)) / (math.log10(ai_max) - math.log10(ai_min))
        return int(frac * (width - 1))

    def y_to_row(tp):
        import math
        tp_min, tp_max = 0.01, peak * 2
        frac = (math.log10(tp) - math.log10(tp_min)) / (math.log10(tp_max) - math.log10(tp_min))
        return height - 1 - int(frac * (height - 1))

    # Draw roofline
    import math
    for col in range(width):
        ai = ai_min * 10 ** (col / (width - 1) * math.log10(ai_max / ai_min))
        tp = min(peak, bw * ai)
        row = y_to_row(tp)
        if 0 <= row < height:
            grid[row][col] = "─"

    # Draw layer points
    markers = {"scalar_conv2d": "S", "depthwise_conv2d": "D"}
    for li, lr in enumerate(wr.layers):
        ai = lr.arithmetic_intensity
        tp = lr.roofline_macs_cycle
        col = x_to_col(max(ai_min, min(ai_max, ai)))
        row = y_to_row(max(0.01, min(peak * 2, tp)))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = markers.get(lr.layer.op_type, "●")

        if lr.rtl_throughput:
            row_rtl = y_to_row(max(0.01, min(peak * 2, lr.rtl_throughput)))
            if 0 <= row_rtl < height and 0 <= col < width:
                grid[row_rtl][col] = "R"

    lines = ["=" * (width + 4)]
    lines.append(f"  Roofline: {hw.name}  (int16 peak={peak:.1f} MACs/cyc, BW={bw:.0f}B/cyc)")
    lines.append("  " + "─" * width)
    for row in grid:
        lines.append("  |" + "".join(row) + "|")
    lines.append("  " + "─" * width)
    lines.append(f"  AI: {ai_min} → {ai_max} MACs/byte (log scale)")
    lines.append(f"  Symbols: ─ = roofline, S = scalar_conv2d, D = depthwise_conv2d, R = RTL actual")
    lines.append("=" * (width + 4))
    return "\n".join(lines)
