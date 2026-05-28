#!/usr/bin/env python3
"""CoralNPU Roofline Analysis Tool.

Usage
-----
  # Analytical model only (no RTL sim needed):
  python3 tools/roofline/run_roofline.py

  # With RTL cycle count from sim:
  python3 tools/roofline/run_roofline.py --rtl-cycles 127500000

  # Parse a .tflite model automatically:
  python3 tools/roofline/run_roofline.py \\
      --tflite tests/cocotb/tutorial/tfmicro/models/mobilenet_v1_025_partial_layers.tflite

  # Save plot to file (headless):
  python3 tools/roofline/run_roofline.py --output /tmp/roofline.png --no-show

  # STCO: compare two hardware configs:
  python3 tools/roofline/run_roofline.py --compare-wide

  # Full validation with RTL sim (rebuilds sim if needed):
  python3 tools/roofline/run_roofline.py --run-sim --elf <path-to-elf>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add repo root to path so 'tools.roofline' is importable from any cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.roofline.hardware import CORAL_CORE_MINI, CORAL_CORE_WIDE
from tools.roofline.model import RooflineModel
from tools.roofline.rtl import find_latest_sim_output, parse_sim_output
from tools.roofline.workload import WorkloadSpec, mobilenet_v1_025_partial, parse_tflite


def _banner(text: str):
    print()
    print("─" * 70)
    print(f"  {text}")
    print("─" * 70)


def run_analysis(args) -> None:
    # ── 1. Build workload spec ────────────────────────────────────────────────
    if args.tflite:
        print(f"Parsing TFLite model: {args.tflite}")
        workload = parse_tflite(args.tflite)
    else:
        workload = mobilenet_v1_025_partial()

    _banner("Workload")
    print(workload.summary())

    # ── 2. Hardware configs ───────────────────────────────────────────────────
    configs = [CORAL_CORE_MINI]
    if args.compare_wide:
        configs.append(CORAL_CORE_WIDE)

    _banner("Hardware")
    for hw in configs:
        print(hw.summary())
        print()

    # ── 3. RTL cycle count ────────────────────────────────────────────────────
    rtl_total = None
    rtl_layers = None

    if args.rtl_cycles is not None:
        rtl_total = float(args.rtl_cycles)
        print(f"Using provided RTL cycle count: {rtl_total:,.0f}")
    else:
        # Try to find a recent sim output file
        sim_file = find_latest_sim_output()
        if sim_file:
            result = parse_sim_output(sim_file)
            if result:
                rtl_total = result.total_cycles
                rtl_layers = result.layer_cycles or None
                print(f"Found RTL sim output in {sim_file}: {rtl_total:,.0f} cycles")

    if args.run_sim:
        rtl_total, rtl_layers = _run_sim(args, workload)

    # ── 4. Roofline analysis ──────────────────────────────────────────────────
    _banner("Roofline Analysis")

    results = []
    for hw in configs:
        model = RooflineModel(hw)
        wr = model.analyse(
            workload,
            rtl_layer_cycles=rtl_layers,
            rtl_total_cycles=rtl_total,
        )
        results.append(wr)
        print(wr.summary())
        print()

    # ── 5. Derived metrics ────────────────────────────────────────────────────
    _banner("Key Metrics")

    hw = CORAL_CORE_MINI
    for layer in workload.layers:
        ai = layer.arithmetic_intensity
        ridge = hw.ridge_point(layer.compute_dtype if layer.compute_dtype != "scalar" else "int8")
        bound = "MEMORY-BOUND" if ai < ridge else "COMPUTE-BOUND"
        print(f"  {layer.name:<40s}  AI={ai:.2f}  ridge={ridge:.2f}  → {bound}")

    print()
    print("  Runtime at 1 GHz:")
    for wr in results:
        roof = wr.total_roofline_cycles / 1e9 * 1e3
        real = wr.total_realistic_cycles / 1e9 * 1e3
        line = f"    [{wr.hw.name}]  roofline lower bound = {roof:.2f}ms  realistic = {real:.2f}ms"
        if rtl_total:
            rtl_ms = rtl_total / 1e9 * 1e3
            line += f"  RTL = {rtl_ms:.2f}ms"
        print(line)

    # ── 6. ASCII roofline ─────────────────────────────────────────────────────
    _banner("ASCII Roofline (int16 ceiling)")
    from tools.roofline.plot import ascii_roofline
    print(ascii_roofline(results[0]))

    # ── 7. Matplotlib plot ────────────────────────────────────────────────────
    if args.output or (not args.no_show):
        try:
            from tools.roofline.plot import plot_roofline
            fig = plot_roofline(
                results,
                title=f"CoralNPU Roofline — {workload.name}",
                output_path=args.output,
                show=not args.no_show,
            )
        except ImportError as e:
            print(f"\n[plot] {e}")


def _run_sim(args, workload):
    """Build and run the RTL simulator, return (total_cycles, layer_cycles)."""
    sim_bin = args.sim_bin or "/tmp/rvv_core_mini_highmem_axi_sim"
    elf = args.elf
    if not elf:
        # Try to find the ELF in Bazel cache
        import glob
        patterns = [
            "/root/.cache/bazel/**/run_mobilenet_v1_025_partial_binary.elf",
        ]
        for pat in patterns:
            matches = glob.glob(pat, recursive=True)
            if matches:
                elf = matches[0]
                print(f"Found ELF: {elf}")
                break
    if not elf:
        print("ERROR: --elf required for --run-sim")
        return None, None

    if not Path(sim_bin).exists():
        print(f"ERROR: sim binary not found at {sim_bin}")
        print("Rebuild with: see previous session or doc/build_highmem_sim.sh")
        return None, None

    output_file = "/tmp/sim_cycles_out.txt"
    print(f"Running simulation: {sim_bin} --binary={elf}")
    print("This typically takes 5–10 minutes...")

    try:
        result = subprocess.run(
            [sim_bin, f"--binary={elf}", "--cycles=200000000"],
            capture_output=True, text=True, timeout=900
        )
        output = result.stdout + result.stderr
        with open(output_file, "w") as f:
            f.write(output)

        sim_result = parse_sim_output(output_file)
        if sim_result:
            print(f"Simulation complete: {sim_result.total_cycles:,} cycles")
            return sim_result.total_cycles, sim_result.layer_cycles
        else:
            print("Simulation output:")
            print(output[-500:])
            return None, None
    except subprocess.TimeoutExpired:
        print("Simulation timed out after 15 minutes")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="CoralNPU Roofline Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tflite",
        help="Path to .tflite model (default: hardcoded mobilenet_v1_025_partial)",
    )
    parser.add_argument(
        "--rtl-cycles", type=int,
        help="Total RTL simulation cycle count (skips auto-detection)",
    )
    parser.add_argument(
        "--run-sim", action="store_true",
        help="Build and run the RTL simulator to get actual cycle count",
    )
    parser.add_argument(
        "--sim-bin", default=None,
        help="Path to RTL simulator binary (default: /tmp/rvv_core_mini_highmem_axi_sim)",
    )
    parser.add_argument(
        "--elf", default=None,
        help="Path to ELF binary for the simulator",
    )
    parser.add_argument(
        "--compare-wide", action="store_true",
        help="Also plot the 256-bit bus (wider LSU) variant for comparison",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save plot to file (e.g. /tmp/roofline.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (use in headless environments)",
    )
    args = parser.parse_args()

    run_analysis(args)


if __name__ == "__main__":
    main()
